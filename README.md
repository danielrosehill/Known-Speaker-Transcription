# Known-Speaker-Transcription

WhisperX-style transcription on [Modal](https://modal.com) where the speakers come out **with their real
names**, not `SPEAKER_00` / `SPEAKER_01`. Enroll each regular voice once; every later recording is diarized,
transcribed and matched against those voiceprints. Built for households, podcasts with fixed hosts and
anyone who keeps relabelling the same two or three people by hand.

| Stage | Model | Image |
|---|---|---|
| Transcription + word timestamps | WhisperX (`large-v3` by default) + its wav2vec2 aligner | CUDA 12.4 / cuDNN |
| Who spoke when | `pyannote/speaker-diarization-community-1` | pyannote.audio 4 |
| Who is who | `pyannote/wespeaker-voxceleb-resnet34-LM` embeddings vs enrolled voiceprints | pyannote.audio 4 |

The two stages run in separate images, in parallel, so WhisperX's pinned pyannote never meets pyannote 4.
Words get the speaker whose diarized turn they overlap most.

## Enroll

You need one recording where you know who spoke when: `labels.json` =
`[{"start": 12.3, "end": 18.9, "speaker": "Alice"}, ...]`. The easy source is any diarized transcript
(AssemblyAI, WhisperX) whose `Speaker A/B` you have mapped to names once.

```bash
modal run known_speakers.py::enroll --audio rec.mp3 --labels rec.labels.json --out voiceprints/ \
  [--publish you/your-private-voiceprints]       # comma-separate several --audio / --labels
```

2 s windows inside labelled turns (0.25 s trimmed each end) are embedded. Evaluation is chronological (first
75 % of each person's windows train, last 25 % test); the shipped voiceprints are refit on everything.
Output: `voiceprints.npz` (names, L2-normalised centroids), `classifier.json` (logistic regression as JSON, no
pickle), `metrics.json`, `config.json`. `--publish` pushes them to a **private** Hugging Face model repo.

## Transcribe

```bash
modal run known_speakers.py::transcribe --audio new.mp3 --voiceprints hf:you/your-private-voiceprints \
  --out-dir out/ [--language en] [--min-speakers 2] [--max-speakers 4] [--num-speakers N] [--min-cos 0.3]
```

Writes `<stem>.speakers.json` (mapping, similarities, talk time, utterances, words, diarization),
`<stem>.speakers.md` and `<stem>.speakers.srt`. Each enrolled name goes to at most one diarized cluster (best
total cosine). Clusters left over, such as a guest, a TV or a voice assistant, become `unknown-1`, `unknown-2`.
A cluster whose best similarity is under `--min-cos` stays unknown.

## Uneven levels

When the gain changed part-way through, or one person sat nearer the phone, levels are evened out first.
Enrollment scales each labelled turn to −20 dBFS RMS. Transcription runs `ffmpeg dynaudnorm` over the whole
file before both stages. Turn it off with `--no-normalize`.

## Tested

2026-11-22, 10 minutes of a two-person phone recording where a voice assistant also spoke, gain raised part-way
through. Voiceprints were enrolled from the full recording, so this checks the pipeline rather than accuracy on
unseen audio. The two people matched their clusters at cosine 0.94 and 0.92. Both assistant-voice clusters stayed
`unknown` (cosine 0.24 or lower). 98.4 % of words agreed with an independent AssemblyAI diarization; the misses sat
at turn boundaries. 52 s wall time with warm images.

## Setup

- A Modal secret `huggingface` with `HF_TOKEN`, from an account that has accepted the terms for
  `pyannote/speaker-diarization-community-1` and `pyannote/wespeaker-voxceleb-resnet34-LM`.
- GPU fallback order: L4, A10G, T4. Model weights are cached on the Modal volume `known-speaker-transcription-cache`.
- Voiceprints are personal biometric data: keep them in a private repo, never in this one.

## License

MIT
