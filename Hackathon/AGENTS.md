# DeepTrace engineering brief

## PRODUCT

DeepTrace detects manipulated face videos by MEASURING physical and biological invariants instead of using a black-box classifier. No model training. It outputs a Forensic Authenticity Index (FAI), a verdict (AUTHENTIC / SUSPECT / MANIPULATED / INCONCLUSIVE), a timestamped anomaly timeline, and a SHA-256-hashed PDF forensic report.

## SIGNALS

A) Multi-site rPPG spatial coherence: forehead and both cheeks should share a synchronized pulse in real faces.

B) Bilabial closure check: during P/B/M phonemes the lips must close; lip-sync fakes leave them open.

C) Arbiter fuses A and B with explicit uncertainty and a quality gate.

D) Chain-of-custody docket (PDF) with hash, metadata, plots, evidence frames.

E) Streamlit dark-mode forensic terminal UI.

## RULES

- Python 3.10, pinned dependencies in requirements.txt. Use mediapipe==0.10.14 (legacy mp.solutions.face_mesh API).
- CPU must work; GPU (CUDA) is optional and auto-detected only for faster-whisper.
- All thresholds and weights live in config.yaml, never hardcoded.
- Never assume a fixed FPS; use real per-frame timestamps.
- Type hints, docstrings, dataclasses for results, logging (not print), deterministic outputs.
- Every module exposes a pure function with a clear dataclass return type, plus pytest tests using synthetic signals.
- Be honest: never claim "court-admissible" (use "court-ready format"), never claim immunity to compression, never claim 100% accuracy.
- Do not add features I did not ask for. Do not implement webcam mode.

