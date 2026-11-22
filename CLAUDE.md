# Known-Speaker-Transcription

Public, generic. Read README.md. One file: `known_speakers.py` (Modal app; `enroll` and `transcribe` entrypoints).

- Never commit voiceprints, audio, labels or names of real people: user data lives in private HF model repos
  (e.g. passed as `--voiceprints hf:<user>/<repo>`).
- Daniel runs it on Modal workspace `dsrhmodeltraining` via `~/.local/bin/modal-dsr` on residencehome (has the
  `huggingface` secret); stage files to `residencehome:~/tmp/kst/` first.
- Results must be plain JSON types (`plain()`): the local client may not have numpy (first pilot failed on that).
