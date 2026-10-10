"""Shared immutable result types for DeepTrace evidence processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Verdict = Literal["AUTHENTIC", "SUSPECT", "MANIPULATED", "INCONCLUSIVE"]


@dataclass(frozen=True)
class VideoMeta:
    """Verified metadata for one input video."""

    source_path: Path
    sha256: str
    duration_s: float
    frame_count: int
    codec: str = "unknown"
    resolution: tuple[int, int] = (0, 0)
    fps: float = 0.0
    bitrate_bps: int | None = None


@dataclass(frozen=True)
class FrameData:
    """One timestamped frame and its eventual tracking state."""

    timestamp_s: float
    frame_index: int
    face_detected: bool


@dataclass(frozen=True)
class RPPGTraces:
    """Uniformly sampled, band-passed rPPG signals for each facial ROI."""

    time_s: tuple[float, ...]
    forehead: tuple[float, ...]
    left_cheek: tuple[float, ...]
    right_cheek: tuple[float, ...]


@dataclass(frozen=True)
class CoherenceWindow:
    """Pairwise rPPG coherence in one timestamped analysis window."""

    start_s: float
    end_s: float
    forehead_left_r: float | None
    forehead_right_r: float | None
    left_right_r: float | None
    incoherent: bool
    is_artifact: bool = False


@dataclass(frozen=True)
class RPPGResult:
    """Explainable rPPG extraction, spectral, coherence, and quality evidence."""

    score: float | None
    uncertainty: float | None
    explanation: str
    usable: bool = False
    traces: RPPGTraces | None = None
    windowed_r: tuple[CoherenceWindow, ...] = field(default_factory=tuple)
    mean_r: float | None = None
    global_r: tuple[float | None, float | None, float | None] = (None, None, None)
    bpm_per_roi: tuple[float | None, float | None, float | None] = (None, None, None)
    snr_db_per_roi: tuple[float | None, float | None, float | None] = (None, None, None)
    bpm_agreement_bpm: float | None = None
    peak_frequency_difference_hz: float | None = None
    incoherent_windows: tuple[CoherenceWindow, ...] = field(default_factory=tuple)
    quality: "QualityReport | None" = None
    clean_window_fraction: float | None = None
    artifact_windows: tuple[CoherenceWindow, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PhoneticResult:
    """Explainable evidence from timestamped bilabial-closure checks."""

    score: float | None
    bilabial_events_checked: int
    violations: int
    violation_rate: float | None
    lag_ms: float | None
    anomalies: tuple["Anomaly", ...]
    usable: bool
    explanation: str
    reason: str | None = None
    closure_threshold: float | None = None
    transcript_snippets: tuple[str, ...] = field(default_factory=tuple)
    has_audio: bool = False
    speech_detected: bool = False


@dataclass(frozen=True)
class QualityReport:
    """Input-quality assessment used to gate a verdict."""

    passed: bool
    face_fraction: float | None
    snr_db: float | None
    explanation: str
    motion_level: float | None = None
    duration_ok: bool = False
    snr_ok: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Anomaly:
    """A timestamped, explainable forensic observation."""

    start_s: float
    end_s: float
    kind: str
    severity: float
    explanation: str
    evidence: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ArbiterResult:
    """Fusion result containing a provisional verdict and evidence."""

    fai: float | None
    verdict: Verdict
    uncertainty: float | None
    anomalies: tuple[Anomaly, ...] = field(default_factory=tuple)
    explanation: str = "Scaffold only; no verdict has been computed."
    confidence: str = "LOW"
    explain: str = ""
    weights_used: dict[str, float] = field(default_factory=dict)
    modules_contributed: tuple[str, ...] = field(default_factory=tuple)
    rppg_score: float | None = None
    phonetic_score: float | None = None
    spectral_snr_score: float | None = None
    bootstrap_ci_mean_r: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "fai": float(self.fai) if self.fai is not None else None,
            "verdict": self.verdict,
            "uncertainty": float(self.uncertainty) if self.uncertainty is not None else None,
            "confidence": self.confidence,
            "explain": self.explain or self.explanation,
            "explanation": self.explanation or self.explain,
            "weights_used": {k: float(v) for k, v in self.weights_used.items()},
            "modules_contributed": list(self.modules_contributed),
            "rppg_score": float(self.rppg_score) if self.rppg_score is not None else None,
            "phonetic_score": float(self.phonetic_score) if self.phonetic_score is not None else None,
            "spectral_snr_score": float(self.spectral_snr_score) if self.spectral_snr_score is not None else None,
            "bootstrap_ci_mean_r": list(self.bootstrap_ci_mean_r) if self.bootstrap_ci_mean_r is not None else None,
            "anomalies": [
                {
                    "start_s": float(a.start_s),
                    "end_s": float(a.end_s),
                    "kind": a.kind,
                    "severity": float(a.severity),
                    "explanation": a.explanation,
                    "evidence": [[k, str(v)] for k, v in a.evidence],
                }
                for a in self.anomalies
            ],
        }
