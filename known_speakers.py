"""Known-speaker transcription on Modal: WhisperX words + pyannote diarization + enrolled voiceprints.

Like WhisperX with diarization, except the speakers come out with real names instead of SPEAKER_00/01,
because their voices were enrolled once (`enroll`) and every later recording is matched against them.

    # enroll: one recording + timing-only labels -> voiceprints (optionally pushed to a private HF model repo)
    modal run known_speakers.py::enroll --audio rec.mp3 --labels rec.labels.json --out voiceprints/ [--publish user/repo]

    # transcribe: any recording -> named transcript (JSON, Markdown, SRT)
    modal run known_speakers.py::transcribe --audio new.mp3 --voiceprints hf:user/repo|local_dir --out-dir out/ \
        [--language en] [--min-speakers 2] [--max-speakers 4] [--asr-model large-v3] [--no-normalize]

labels.json: [{"start": s, "end": s, "speaker": "Name"}, ...]; any diarized transcript (AssemblyAI etc.) whose
generic labels you mapped to names once will do. Voiceprints: a directory or HF repo holding voiceprints.npz
(names, L2-normalised WeSpeaker centroids), classifier.json, metrics.json, config.json.

Loudness: recordings where the gain changed part-way (or one person sits nearer the mic) are levelled before
anything else: enrollment scales every labelled turn to the same RMS; transcription runs ffmpeg dynaudnorm over the
file. Disable with --no-normalize.

Secrets: a Modal secret named `huggingface` holding HF_TOKEN (gated pyannote models; private voiceprint repos).
"""
import io, itertools, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import modal

APP = "known-speaker-transcription"
SR = 16000
EMB = "pyannote/wespeaker-voxceleb-resnet34-LM"
DIAR = "pyannote/speaker-diarization-community-1"
WIN, STEP, TRIM = 2.0, 1.0, 0.25
TARGET_DBFS = -20.0
GPUS = ["L4", "A10G", "T4"]  # fallback order

app = modal.App(APP)
hf = modal.Secret.from_name("huggingface")
cache = modal.Volume.from_name(f"{APP}-cache", create_if_missing=True)

# Two images so WhisperX's pyannote pin never meets pyannote 4 (needed for community-1).
asr_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("whisperx", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
spk_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("pyannote.audio>=4.0", "numpy", "soundfile", "scikit-learn", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf"})
)


# ---------- shared helpers (run inside containers) ----------

def decode(audio: bytes, normalize: bool):
    """Any format -> 16 kHz mono float32, optionally levelled with dynaudnorm."""
    import numpy as np
    af = ["-af", "dynaudnorm=f=250:g=15:p=0.9"] if normalize else []
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *af, "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "pipe:1"], input=audio, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def level_turns(y, labels):
    """Scale each labelled turn to TARGET_DBFS RMS (gain changes, uneven mic distance)."""
    import numpy as np
    y = y.copy()
    for g in labels:
        a, b = int(g["start"] * SR), int(g["end"] * SR)
        seg = y[a:b]
        if len(seg) < SR // 4:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        y[a:b] = np.clip(seg * (10 ** (TARGET_DBFS / 20) / rms), -1, 1)
    return y


def plain(x):
    """Results go back to a client that may have no numpy: numpy scalars -> Python floats/ints."""
    return json.loads(json.dumps(x, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def unit(m):
    import numpy as np
    m = np.asarray(m, dtype=np.float64)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def load_voiceprints(ref, tok: str):
    """ref: 'hf:user/repo' (private repos need HF_TOKEN), or {filename: bytes} read from a local directory."""
    import numpy as np
    if isinstance(ref, dict):
        vp, cfg = np.load(io.BytesIO(ref["voiceprints.npz"])), json.loads(ref["config.json"])
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(ref[3:], token=tok, allow_patterns=["voiceprints.npz", "config.json"])
        vp, cfg = np.load(os.path.join(d, "voiceprints.npz")), json.load(open(os.path.join(d, "config.json")))
    return [str(n) for n in vp["names"]], vp["centroids"].astype(np.float64), cfg


# ---------- enroll ----------

@app.function(image=spk_image, gpu=GPUS, timeout=3600, secrets=[hf], volumes={"/cache": cache})
def enroll_remote(audios: list[bytes], labels: list[list], names_in: list[str], normalize: bool = True) -> dict:
    import numpy as np, torch
    from pyannote.audio import Model, Inference
    from sklearn.linear_model import LogisticRegression
    tok = os.environ["HF_TOKEN"]
    inf = Inference(Model.from_pretrained(EMB, token=tok), window="sliding", duration=WIN, step=STEP,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    Xs, Ns, Ts, per_file = [], [], [], []
    for k, (au, lb) in enumerate(zip(audios, labels)):
        y = decode(au, normalize=False)
        if normalize:
            y = level_turns(y, lb)
        feats = inf({"waveform": torch.from_numpy(y)[None], "sample_rate": SR})
        X, N, T = [], [], []
        for i, e in enumerate(feats.data):
            s = feats.sliding_window[i].start
            for g in lb:
                if g["start"] + TRIM <= s and s + WIN <= g["end"] - TRIM:
                    if np.all(np.isfinite(e)):
                        X.append(e); N.append(g["speaker"]); T.append(s + k * 1e6)
                    break
        per_file.append({"file": names_in[k], "windows": {n: N.count(n) for n in sorted(set(N))}})
        Xs += X; Ns += N; Ts += T
    X, N, T = unit(np.array(Xs)), np.array(Ns), np.array(Ts)
    names = sorted(set(N))

    tr = np.zeros(len(N), bool)  # chronological 75/25 per speaker
    for n in names:
        idx = np.where(N == n)[0]
        idx = idx[np.argsort(T[idx])]
        tr[idx[: int(0.75 * len(idx))]] = True
    te = ~tr
    cent = unit(np.stack([X[tr & (N == n)].mean(0) for n in names]))
    pred_c = np.array(names)[np.argmax(X[te] @ cent.T, axis=1)]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], N[tr])
    pred_l = clf.predict(X[te])

    def recall(p):
        return {n: round(float((p[N[te] == n] == n).mean()), 4) for n in names}

    metrics = {"windows_total": {n: int((N == n).sum()) for n in names},
               "windows_test": {n: int((te & (N == n)).sum()) for n in names},
               "centroid": {"recall": recall(pred_c), "balanced_accuracy": round(float(np.mean(list(recall(pred_c).values()))), 4)},
               "logreg": {"recall": recall(pred_l), "balanced_accuracy": round(float(np.mean(list(recall(pred_l).values()))), 4)},
               "centroid_cosine": {f"{names[i]}~{names[j]}": round(float(cent[i] @ cent[j]), 4)
                                   for i in range(len(names)) for j in range(i + 1, len(names))},
               "split": "chronological per speaker, 75/25", "per_file": per_file}
    cent = unit(np.stack([X[N == n].mean(0) for n in names]))  # refit on everything
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, N)
    buf = io.BytesIO()
    np.savez(buf, names=np.array(names), centroids=cent.astype(np.float32))
    return {"voiceprints.npz": buf.getvalue(),
            "classifier.json": json.dumps({"classes": clf.classes_.tolist(), "coef": clf.coef_.tolist(),
                                           "intercept": clf.intercept_.tolist(),
                                           "input": "L2-normalised WeSpeaker embedding"}).encode(),
            "metrics.json": json.dumps(metrics, indent=1).encode(),
            "config.json": json.dumps({"embedding_model": EMB, "window_s": WIN, "step_s": STEP, "trim_s": TRIM,
                                       "sample_rate": SR, "speakers": names, "level_turns_dbfs": TARGET_DBFS if normalize else None,
                                       "trained": time.strftime("%Y-%m-%d"), "sources": names_in}, indent=1).encode()}


@app.function(image=spk_image, secrets=[hf])
def publish_remote(files: dict, repo: str, readme: str | None = None):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, private=True, exist_ok=True)
    for name, data in files.items():
        api.upload_file(path_or_fileobj=data, path_in_repo=name, repo_id=repo)
    if readme:
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def enroll(audio: str, labels: str, out: str, publish: str = "", no_normalize: bool = False):
    """--audio/--labels accept comma-separated lists for several recordings."""
    auds, labs = audio.split(","), labels.split(",")
    assert len(auds) == len(labs), "--audio and --labels must pair up"
    files = enroll_remote.remote([Path(a).read_bytes() for a in auds], [json.load(open(l)) for l in labs],
                                 [Path(a).name for a in auds], not no_normalize)
    Path(out).mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        Path(out, n).write_bytes(b)
    print(files["metrics.json"].decode())
    if publish:
        print(publish_remote.remote(files, publish))


# ---------- transcribe ----------

@app.function(image=asr_image, gpu=GPUS, timeout=3 * 3600, secrets=[hf], volumes={"/cache": cache})
def asr_remote(audio: bytes, language: str | None, model: str, normalize: bool) -> dict:
    import whisperx
    y = decode(audio, normalize)
    dev = "cuda"
    m = whisperx.load_model(model, dev, compute_type="float16", language=language, download_root="/cache/whisper")
    res = m.transcribe(y, batch_size=16, language=language)
    lang = res["language"]
    am, meta = whisperx.load_align_model(language_code=lang, device=dev)
    al = whisperx.align(res["segments"], am, meta, y, dev, return_char_alignments=False)
    words = [{"word": w["word"], "start": w.get("start"), "end": w.get("end"), "score": w.get("score")}
             for s in al["segments"] for w in s.get("words", [])]
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in al["segments"]]
    return plain({"language": lang, "words": words, "segments": segs, "duration_s": round(len(y) / SR, 2)})


@app.function(image=spk_image, gpu=GPUS, timeout=3 * 3600, secrets=[hf], volumes={"/cache": cache})
def speakers_remote(audio: bytes, voiceprints, normalize: bool, min_speakers: int | None,
                    max_speakers: int | None, num_speakers: int | None, min_cos: float) -> dict:
    import numpy as np, torch
    from pyannote.audio import Pipeline, Model, Inference
    tok = os.environ["HF_TOKEN"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, cent, vcfg = load_voiceprints(voiceprints, tok)
    y = decode(audio, normalize)
    wav = {"waveform": torch.from_numpy(y)[None], "sample_rate": SR}
    pipe = Pipeline.from_pretrained(DIAR, token=tok).to(dev)
    kw = {"num_speakers": num_speakers} if num_speakers else {
        "min_speakers": min_speakers or len(names), **({"max_speakers": max_speakers} if max_speakers else {})}
    res = pipe(wav, **kw)
    dia = getattr(res, "exclusive_speaker_diarization", None) or getattr(res, "speaker_diarization", res)
    labels = sorted(dia.labels())

    inf = Inference(Model.from_pretrained(vcfg.get("embedding_model", EMB), token=tok), window="whole", device=dev)
    emb = {}
    for s in labels:
        es = []
        for t, _, l in dia.itertracks(yield_label=True):
            if l == s and t.duration >= 1.0:
                e = np.asarray(inf.crop(wav, t)).ravel()
                if np.all(np.isfinite(e)):
                    es.append(e)
        emb[s] = unit(np.mean(unit(np.array(es)), axis=0)) if es else None
    S = np.array([[float(emb[s] @ c) if emb[s] is not None else -1.0 for c in cent] for s in labels])

    # Each enrolled name goes to at most one cluster (best total similarity); extra clusters (a third person,
    # a TV, a voice assistant) stay "unknown-N" unless above min_cos AND clearly closer to one name.
    mapping = {}
    k = min(len(labels), len(names))
    best = max(((rows, cols) for rows in itertools.combinations(range(len(labels)), k)
                for cols in itertools.permutations(range(len(names)), k)),
               key=lambda rc: sum(S[i, j] for i, j in zip(*rc)))
    for i, j in zip(*best):
        if S[i, j] >= min_cos:
            mapping[labels[i]] = names[j]
    n_unk = 0
    for s in labels:
        if s not in mapping:
            n_unk += 1
            mapping[s] = f"unknown-{n_unk}"
    turns = [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": mapping[l], "cluster": l}
             for t, _, l in dia.itertracks(yield_label=True)]
    return plain({"mapping": mapping,
            "similarity": {s: dict(zip(names, (round(float(v), 3) for v in S[i]))) for i, s in enumerate(labels)},
            "turns": turns})


def assign(words, turns):
    """Word -> speaker by largest time overlap; words without timestamps take their neighbour's."""
    out = []
    for w in words:
        if w["start"] is None:
            out.append(dict(w, speaker=None)); continue
        best, ov = None, 0.0
        for t in turns:
            o = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if o > ov:
                best, ov = t["speaker"], o
        if best is None:  # in a gap: nearest turn
            best = min(turns, key=lambda t: min(abs(w["start"] - t["end"]), abs(w["end"] - t["start"])))["speaker"]
        out.append(dict(w, speaker=best))
    for i, w in enumerate(out):
        if w["speaker"] is None:
            nb = next((out[j]["speaker"] for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(out)))
                       if out[j]["speaker"]), "unknown")
            w["speaker"] = nb
            w["start"] = w["start"] or (out[i - 1]["end"] if i else 0.0)
            w["end"] = w["end"] or w["start"]
    return out


def utterances(words, gap=1.5):
    us = []
    for w in words:
        if us and us[-1]["speaker"] == w["speaker"] and w["start"] - us[-1]["end"] <= gap:
            us[-1]["text"] += " " + w["word"]; us[-1]["end"] = w["end"]
        else:
            us.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": w["word"]})
    return us


def ts(s, srt=False):
    h, r = divmod(float(s), 3600); m, sec = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",") if srt else f"{int(h):02d}:{int(m):02d}:{int(sec):02d}"


@app.local_entrypoint()
def transcribe(audio: str, voiceprints: str, out_dir: str = ".", language: str = "", asr_model: str = "large-v3",
               min_speakers: int = 0, max_speakers: int = 0, num_speakers: int = 0, min_cos: float = 0.3,
               no_normalize: bool = False):
    t0 = time.time()
    data, norm = Path(audio).read_bytes(), not no_normalize
    vref = voiceprints if voiceprints.startswith("hf:") else {
        n: Path(voiceprints, n).read_bytes() for n in ("voiceprints.npz", "config.json")}
    a = asr_remote.spawn(data, language or None, asr_model, norm)
    s = speakers_remote.spawn(data, vref, norm, min_speakers or None, max_speakers or None,
                              num_speakers or None, min_cos)
    asr, spk = a.get(), s.get()
    words = assign(asr["words"], spk["turns"])
    us = utterances(words)
    stem = Path(out_dir, Path(audio).stem)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    talk = {}
    for t in spk["turns"]:
        talk[t["speaker"]] = round(talk.get(t["speaker"], 0) + t["end"] - t["start"], 1)
    doc = {"audio": Path(audio).name, "language": asr["language"], "duration_s": asr["duration_s"],
           "asr_model": asr_model, "voiceprints": voiceprints, "normalized": norm,
           "mapping": spk["mapping"], "similarity": spk["similarity"], "talk_time_s": talk,
           "elapsed_s": round(time.time() - t0, 1), "utterances": us, "words": words, "diarization": spk["turns"]}
    json.dump(doc, open(f"{stem}.speakers.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{stem}.speakers.md", "w") as f:
        f.write(f"# {Path(audio).name}\n\n")
        for u in us:
            f.write(f"**{u['speaker']}** [{ts(u['start'])}]: {u['text'].strip()}\n\n")
    with open(f"{stem}.speakers.srt", "w") as f:
        for i, u in enumerate(us, 1):
            f.write(f"{i}\n{ts(u['start'], True)} --> {ts(u['end'], True)}\n{u['speaker']}: {u['text'].strip()}\n\n")
    print(json.dumps({k: doc[k] for k in ("mapping", "similarity", "talk_time_s", "language", "duration_s", "elapsed_s")}, indent=1))
    print(f"wrote {stem}.speakers.json / .md / .srt", file=sys.stderr)
