---
inclusion: auto
name: deeptrace-project-rules
description: Core product, design system, and coding rules for the DeepTrace forensic deepfake-evidence tool. Activate whenever working on any DeepTrace feature, UI, analysis logic, config, or tests.
---

# DeepTrace — Project Steering

DeepTrace is a forensic deepfake-evidence tool built for a hackathon judged by NVIDIA and Meta engineers.
It detects manipulation through two physical signals only — it trains no model.

---

## 1. Signal Detection (What the System Measures)

DeepTrace measures exactly two physical signals:

1. **Pulse coherence** — rPPG via the CHROM method. Pearson r correlation across forehead and both cheeks.
2. **Lip closure on P / B / M sounds** — lip aperture ratio. Any violation (mouth open when it should be closed) is flagged.

No other signals are used. No ML model is trained.

---

## 2. Verdict System

Verdicts are exactly four words. No others are permitted.

| Verdict | Meaning |
|---|---|
| `AUTHENTIC` | No anomalies detected. **Always paired with the phrase "no manipulation found."** |
| `SUSPECT` | One or more signals are borderline. |
| `MANIPULATED` | Signals indicate synthetic content. |
| `INCONCLUSIVE` | Quality gate failed — cannot produce a reliable verdict. |

Rules:
- Never claim certainty or 100% accuracy anywhere in the UI, docs, or code comments.
- `INCONCLUSIVE` is a designed, first-class outcome — not a failure. It is returned when: clip is too short, face is not visible, SNR is too low, or too few P/B/M speech events are detected.
- The word "Authentic" may only appear alongside the phrase "no manipulation found."

---

## 3. SHA-256 Hash Semantics

The SHA-256 hash proves a file is **unchanged since analysis**. It does **not** prove authenticity.
This distinction must be stated wherever the hash appears — in the UI, reports, tooltips, and docs.

---

## 4. Thresholds and FAI Weights (Config-Only)

All numeric thresholds and weights live **only** in `config.yaml`. They are never hardcoded in application logic.

```yaml
# config.yaml — initial values, adjust as needed
thresholds:
  r_authentic: 0.60   # Pearson r above this → authentic pulse signal
  r_fake: 0.35        # Pearson r below this → manipulated pulse signal

fai_weights:
  pulse: 0.55
  speech: 0.35
  snr: 0.10
```

Mark them as "initial values" in comments. Any module that reads thresholds must read them from config.

---

## 5. Explanation Text

Explanation text shown to the user must be **generated from result fields at runtime** — never hardcoded strings.
This ensures the explanation can never contradict the displayed numbers.

---

## 6. Network Access

Processing is fully local. The **only** permitted network activity is downloading pre-trained models (e.g., face detector weights). Document any such download clearly.

---

## 7. Design System

### Color Palette

| Token | Hex | Usage |
|---|---|---|
| Background | `#0B0F14` | App background |
| Panel | `#121923` | Cards, panels |
| Border | `#1E2935` | Dividers, outlines |
| Text | `#E6EDF3` | Primary text |
| Muted | `#8B98A5` | Secondary / label text |
| Teal | `#2DD4BF` | Healthy / normal signal |
| Amber | `#F5A524` | Suspect signal |
| Red | `#F43F5E` | Manipulated signal |
| Slate | `#64748B` | Inconclusive state |
| Green | `#22C55E` | Authentic verdict |

### Typography

- **Inter** — all prose, labels, UI text.
- **JetBrains Mono** — all numbers, hashes, and timestamps.

### Layout & Accessibility

- Every metric card shows the metric value **plus one short explanatory line** beneath it (what the number means).
- Minimum font size: **14px** — no exceptions.
- Layout must be legible on both a laptop screen and a projector: large numbers, high contrast.
- Use the accent colors above for verdict and signal state — never use color as the only differentiator (pair with an icon or label).

---

## 8. Coding Standards

| Rule | Detail |
|---|---|
| Language | Python 3.10 |
| Type hints | Required on every function signature and dataclass field |
| Result types | Use `@dataclass` for all analysis result structures |
| Logging | Use the `logging` module — never `print()` in application code |
| Outputs | Deterministic: same input must always produce identical output |
| Tests | Every module must have a corresponding test module |
| Dependencies | All dependencies pinned with exact versions in `requirements.txt` |
| Config reading | Thresholds/weights loaded from `config.yaml` only — no magic numbers in code |

---

## 9. Quality Gate Criteria

The quality gate runs before analysis. `INCONCLUSIVE` is returned immediately if any of these fail:

- Clip is too short (below minimum duration defined in config)
- Face is not visible or detected for sufficient frames
- SNR is too low for reliable rPPG extraction
- Too few P / B / M phoneme events detected for lip-closure analysis

Document the specific thresholds for each gate condition in `config.yaml`.

---

## 10. Hackathon Context

Judges are NVIDIA and Meta engineers. Emphasise:
- Physical signal integrity (no black-box ML)
- Transparent, reproducible forensic methodology
- Clean architecture and readable code
- Honest uncertainty communication (INCONCLUSIVE is a feature, not a bug)
