"""Timestamp-aware CHROM rPPG extraction and spatial-coherence assessment.

Assumptions and known weaknesses: CHROM depends on stable skin illumination,
adequate face tracking, and a pulse-visible frame rate. It is not robust to
large head motion, occlusion, beauty filters, or severe compression. Coherence
is evidence, not a deepfake verdict or a medical measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch
import yaml

from engine.types import CoherenceWindow, QualityReport, RPPGResult, RPPGTraces, VideoMeta

if TYPE_CHECKING:
    from engine.face_tracker import FaceTrackingResult


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RPPGConfig:
    """Analysis parameters loaded from config.yaml."""

    band_hz: tuple[float, float]
    filter_order: int
    window_s: float
    chrom_window_s: float
    target_hz: float
    welch_segment_s: float
    snr_peak_half_width_hz: float
    corr_real: float
    corr_fake: float
    min_duration_s: float
    min_face_fraction: float
    min_snr_db: float
    max_motion_level: float
    discard_initial_s: float = 1.5
    max_lag_ms: float = 150.0
    method: str = "chrom"
    artifact_amp_ratio: float = 3.0


@dataclass(frozen=True)
class RPPGInput:
    """Timestamped spatially averaged RGB traces from three facial ROIs."""

    timestamps_s: tuple[float, ...]
    forehead_rgb: tuple[tuple[float, float, float], ...]
    left_cheek_rgb: tuple[tuple[float, float, float], ...]
    right_cheek_rgb: tuple[tuple[float, float, float], ...]
    face_fraction: float = 1.0
    motion_level: float | None = 0.0
    per_frame_motion: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GroundTruthValidationResult:
    """BPM agreement between analysed video and timestamped reference BPM."""

    result: RPPGResult
    ground_truth_mean_bpm: float | None
    mae_bpm_per_roi: tuple[float | None, float | None, float | None]
    explanation: str


@dataclass(frozen=True)
class VideoRPPGAnalysis:
    """One-pass video provenance, tracker output, and rPPG evidence."""

    metadata: VideoMeta
    tracking: "FaceTrackingResult"
    rppg: RPPGResult


def load_rppg_config(config_path: str | Path | None = None) -> RPPGConfig:
    """Load every rPPG threshold and gate from config.yaml."""
    path = Path(config_path) if config_path is not None else Path(__file__).parents[1] / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    rppg, quality = values["rppg"], values["quality"]
    return RPPGConfig(
        band_hz=tuple(float(value) for value in rppg["band_hz"]), filter_order=int(rppg["filter_order"]),
        window_s=float(rppg["window_s"]), chrom_window_s=float(rppg["chrom_window_s"]), target_hz=float(rppg["target_hz"]),
        welch_segment_s=float(rppg["welch_segment_s"]), snr_peak_half_width_hz=float(rppg["snr_peak_half_width_hz"]),
        corr_real=float(rppg["corr_real"]), corr_fake=float(rppg["corr_fake"]), min_duration_s=float(rppg["min_duration_s"]),
        min_face_fraction=float(quality["min_face_fraction"]), min_snr_db=float(quality["min_snr_db"]), max_motion_level=float(quality["max_motion_level"]),
        discard_initial_s=float(rppg.get("discard_initial_s", 1.5)),
        max_lag_ms=float(rppg.get("max_lag_ms", 150.0)),
        method=str(rppg.get("method", "chrom")).lower(),
        artifact_amp_ratio=float(rppg.get("artifact_amp_ratio", 3.0)),
    )


def rppg_input_from_tracking(tracking: "FaceTrackingResult") -> RPPGInput:
    """Convert valid one-pass tracking output into rPPG RGB observations."""
    valid = [frame for frame in tracking.frames if frame.valid and frame.forehead.rgb and frame.left_cheek.rgb and frame.right_cheek.rgb]
    return RPPGInput(
        timestamps_s=tuple(frame.timestamp_s for frame in valid),
        forehead_rgb=tuple(frame.forehead.rgb for frame in valid if frame.forehead.rgb is not None),
        left_cheek_rgb=tuple(frame.left_cheek.rgb for frame in valid if frame.left_cheek.rgb is not None),
        right_cheek_rgb=tuple(frame.right_cheek.rgb for frame in valid if frame.right_cheek.rgb is not None),
        face_fraction=tracking.face_fraction,
        motion_level=tracking.mean_motion_level,
        per_frame_motion=tuple(float(frame.motion_level or 0.0) for frame in valid),
    )


def _empty_result(reasons: Sequence[str], face_fraction: float | None = None, motion_level: float | None = None) -> RPPGResult:
    """Return an explicit unusable result rather than inventing a signal score."""
    message = "; ".join(reasons)
    quality = QualityReport(False, face_fraction, None, message, motion_level=motion_level, reasons=tuple(reasons))
    return RPPGResult(score=None, uncertainty=None, explanation=message, usable=False, quality=quality)


def _as_valid_arrays(signal: RPPGInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate finite equal-length samples without reordering timestamps."""
    timestamps = np.asarray(signal.timestamps_s, dtype=np.float64)
    forehead = np.asarray(signal.forehead_rgb, dtype=np.float64)
    left = np.asarray(signal.left_cheek_rgb, dtype=np.float64)
    right = np.asarray(signal.right_cheek_rgb, dtype=np.float64)
    lengths = (timestamps.size, forehead.shape[0] if forehead.ndim else 0, left.shape[0] if left.ndim else 0, right.shape[0] if right.ndim else 0)
    if len(set(lengths)) != 1 or timestamps.size < 2 or any(values.shape[1:] != (3,) for values in (forehead, left, right)):
        return None
    if not all(np.isfinite(values).all() for values in (timestamps, forehead, left, right)) or np.any(np.diff(timestamps) <= 0.0):
        return None
    return timestamps, forehead, left, right


def _resample(timestamps: np.ndarray, rgb: np.ndarray, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample irregular RGB samples to a common uniform timebase."""
    step = 1.0 / target_hz
    uniform_time = np.arange(timestamps[0], timestamps[-1] + step * 0.5, step, dtype=np.float64)
    return uniform_time, np.column_stack([np.interp(uniform_time, timestamps, rgb[:, index]) for index in range(3)])


def _chrom(rgb: np.ndarray, sample_rate_hz: float, window_s: float) -> np.ndarray:
    """Compute overlap-added CHROM pulse signal from normalized RGB windows.

    Each window is independently detrended and mean-normalized to isolate the
    pulsatile component.  The overlap-add with Hann tapering produces a smooth
    output even when individual windows are noisy.
    """
    window_size = max(2, int(round(window_s * sample_rate_hz)))
    hop_size = max(1, window_size // 2)
    output = np.zeros(rgb.shape[0])
    weights = np.zeros(rgb.shape[0])
    taper = np.hanning(window_size)
    eps = np.finfo(float).eps

    for start in range(0, rgb.shape[0], hop_size):
        end = min(start + window_size, rgb.shape[0])
        segment = rgb[start:end]
        if segment.shape[0] < 4:
            continue

        # Per-segment temporal mean normalization
        seg_mean = np.mean(segment, axis=0)
        if np.any(np.abs(seg_mean) < eps):
            continue

        # Detrend each channel before normalization to remove slow drift
        from scipy.signal import detrend as _dt
        normed = np.column_stack([
            _dt(segment[:, ch] / seg_mean[ch], type='linear')
            for ch in range(3)
        ])

        red, green, blue = normed.T
        xs = 3.0 * red - 2.0 * green
        ys = 1.5 * red + green - 1.5 * blue

        std_xs = np.std(xs)
        std_ys = np.std(ys)
        alpha = float(std_xs / std_ys) if std_ys > eps else 0.0

        local_taper = taper[:end - start]
        output[start:end] += (xs - alpha * ys) * local_taper
        weights[start:end] += local_taper

    return np.divide(output, weights, out=np.zeros_like(output), where=weights > eps)


def _pos(rgb: np.ndarray, sample_rate_hz: float, window_s: float) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (POS) rPPG extraction (Wang et al. 2017).

    Projects temporal normalized RGB signals onto the plane orthogonal to the
    skin tone vector, removing specular reflectance and intensity fluctuations.
    """
    window_size = max(2, int(round(window_s * sample_rate_hz)))
    hop_size = max(1, window_size // 2)
    output = np.zeros(rgb.shape[0])
    weights = np.zeros(rgb.shape[0])
    taper = np.hanning(window_size)
    eps = np.finfo(float).eps

    for start in range(0, rgb.shape[0], hop_size):
        end = min(start + window_size, rgb.shape[0])
        segment = rgb[start:end]
        if segment.shape[0] < 4:
            continue

        seg_mean = np.mean(segment, axis=0)
        if np.any(np.abs(seg_mean) < eps):
            continue

        normed = np.column_stack([
            detrend(segment[:, ch] / seg_mean[ch], type="linear")
            for ch in range(3)
        ])

        red, green, blue = normed.T
        s1 = green - blue
        s2 = green + blue - 2.0 * red

        std_s1 = np.std(s1)
        std_s2 = np.std(s2)
        alpha = float(std_s1 / std_s2) if std_s2 > eps else 0.0

        h = s1 + alpha * s2
        h = h - np.mean(h)

        local_taper = taper[:end - start]
        output[start:end] += h * local_taper
        weights[start:end] += local_taper

    return np.divide(output, weights, out=np.zeros_like(output), where=weights > eps)


def _extract_pulse(rgb: np.ndarray, settings: RPPGConfig) -> np.ndarray:
    """Extract raw rPPG signal using the method specified in config (CHROM or POS)."""
    method = getattr(settings, "method", "chrom").lower()
    if method == "pos":
        return _pos(rgb, settings.target_hz, settings.chrom_window_s)
    return _chrom(rgb, settings.target_hz, settings.chrom_window_s)


def _bandpass(values: np.ndarray, settings: RPPGConfig) -> np.ndarray | None:
    """Detrend then apply the configured zero-phase Butterworth bandpass."""
    b, a = butter(settings.filter_order, settings.band_hz, btype="bandpass", fs=settings.target_hz)
    if values.size < 3 * max(len(a), len(b)) + 1:
        return None
    return filtfilt(b, a, detrend(values, type="linear"))


def _bpm_and_snr(values: np.ndarray, settings: RPPGConfig) -> tuple[float | None, float | None, float | None]:
    """Estimate BPM and SNR from the peak plus its second harmonic.

    Zero-padding via nfft improves frequency resolution from ~6 BPM to ~1.8 BPM,
    critical for accurate cross-ROI BPM agreement on short clips.
    """
    nperseg = min(values.size, max(2, int(round(settings.welch_segment_s * settings.target_hz))))
    nfft = max(nperseg, 1024)  # zero-pad for finer frequency bins
    frequencies, power = welch(values, fs=settings.target_hz, nperseg=nperseg, nfft=nfft)
    in_band = (frequencies >= settings.band_hz[0]) & (frequencies <= settings.band_hz[1])
    if not np.any(in_band) or np.all(power[in_band] <= 0.0):
        return None, None, None
    peak_index = np.flatnonzero(in_band)[int(np.argmax(power[in_band]))]
    peak_hz = float(frequencies[peak_index])
    near_peak = np.abs(frequencies - peak_hz) <= settings.snr_peak_half_width_hz
    harmonic = 2.0 * peak_hz
    near_harmonic = np.abs(frequencies - harmonic) <= settings.snr_peak_half_width_hz if harmonic <= settings.band_hz[1] else np.zeros_like(in_band)
    signal_power = float(np.sum(power[in_band & (near_peak | near_harmonic)]))
    noise_power = float(np.sum(power[in_band & ~(near_peak | near_harmonic)]))
    snr_db = float("inf") if noise_power <= np.finfo(float).eps and signal_power > 0.0 else float(10.0 * np.log10(signal_power / max(noise_power, np.finfo(float).eps)))
    return peak_hz * 60.0, snr_db, peak_hz


def _max_lag_cross_correlation(left: np.ndarray, right: np.ndarray, max_lag_samples: int = 0) -> float | None:
    """Return max normalized cross-correlation within [-max_lag_samples, +max_lag_samples]."""
    if left.size < 2 or right.size < 2 or np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return None
    if max_lag_samples <= 0:
        val = float(np.corrcoef(left, right)[0, 1])
        return val if np.isfinite(val) else None

    left_c = left - np.mean(left)
    right_c = right - np.mean(right)
    norm = np.sqrt(np.sum(left_c ** 2) * np.sum(right_c ** 2))
    if norm <= np.finfo(float).eps:
        return None
    cc = np.correlate(left_c, right_c, mode="full") / norm
    zero_idx = len(right_c) - 1
    start_idx = max(0, zero_idx - max_lag_samples)
    end_idx = min(len(cc), zero_idx + max_lag_samples + 1)
    window_cc = cc[start_idx:end_idx]
    if window_cc.size == 0:
        return None
    val = float(np.max(window_cc))
    return val if np.isfinite(val) else None


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a finite Pearson correlation only for non-constant signals."""
    if left.size < 2 or np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _coherence_windows(
    time_s: np.ndarray,
    traces: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: RPPGConfig,
    motion_series: np.ndarray | None = None,
) -> tuple[CoherenceWindow, ...]:
    """Compute configured 50%-overlap pairwise-coherence windows with artifact rejection."""
    window_size = max(2, int(round(settings.window_s * settings.target_hz)))
    hop_size = max(1, window_size // 2)
    max_lag_samples = int(round((getattr(settings, "max_lag_ms", 150.0) / 1000.0) * settings.target_hz))
    amp_ratio_thresh = getattr(settings, "artifact_amp_ratio", 3.0)

    window_slices: list[tuple[int, int]] = []
    window_stds: list[tuple[float, float, float]] = []
    for start in range(0, time_s.size - window_size + 1, hop_size):
        end = start + window_size
        window_slices.append((start, end))
        stds = (
            float(np.std(traces[0][start:end])),
            float(np.std(traces[1][start:end])),
            float(np.std(traces[2][start:end])),
        )
        window_stds.append(stds)

    if window_stds:
        clip_med_stds = (
            float(np.median([s[0] for s in window_stds])),
            float(np.median([s[1] for s in window_stds])),
            float(np.median([s[2] for s in window_stds])),
        )
    else:
        clip_med_stds = (0.0, 0.0, 0.0)

    windows: list[CoherenceWindow] = []
    for idx, (start, end) in enumerate(window_slices):
        stds = window_stds[idx]
        local_med_std = float(np.median(stds))

        # Artifact rejection rule 1: single-ROI amplitude > 3x median
        is_amp_artifact = False
        if local_med_std > np.finfo(float).eps and max(stds) > amp_ratio_thresh * local_med_std:
            is_amp_artifact = True
        for roi_idx in range(3):
            if clip_med_stds[roi_idx] > np.finfo(float).eps and stds[roi_idx] > amp_ratio_thresh * clip_med_stds[roi_idx]:
                is_amp_artifact = True

        # Artifact rejection rule 2: high head motion in window
        is_motion_artifact = False
        if motion_series is not None and motion_series.size >= end:
            win_motion = float(np.mean(motion_series[start:end]))
            if win_motion > settings.max_motion_level:
                is_motion_artifact = True

        is_artifact = is_amp_artifact or is_motion_artifact

        values = (
            _max_lag_cross_correlation(traces[0][start:end], traces[1][start:end], max_lag_samples),
            _max_lag_cross_correlation(traces[0][start:end], traces[2][start:end], max_lag_samples),
            _max_lag_cross_correlation(traces[1][start:end], traces[2][start:end], max_lag_samples),
        )
        valid = [value for value in values if value is not None]

        # Artifact windows are rejected and not counted as physical desynchronization anomalies
        if is_artifact:
            is_incoherent = False
        else:
            is_incoherent = bool(valid and (np.mean(valid) < settings.corr_fake or sum(v < settings.corr_fake for v in valid) >= 2))

        windows.append(
            CoherenceWindow(
                float(time_s[start]),
                float(time_s[end - 1]),
                values[0],
                values[1],
                values[2],
                is_incoherent,
                is_artifact=is_artifact,
            )
        )
    return tuple(windows)


def analyze_rppg(signal: RPPGInput, config: RPPGConfig | None = None) -> RPPGResult:
    """Extract CHROM rPPG, spectra, coherence, and quality gates from RGB traces."""
    settings = config or load_rppg_config()
    arrays = _as_valid_arrays(signal)
    if arrays is None:
        return _empty_result(("Timestamps and all three RGB traces must be finite, equally sized, and strictly increasing.",), signal.face_fraction, signal.motion_level)
    timestamps, forehead_rgb, left_rgb, right_rgb = arrays
    duration_s = float(timestamps[-1] - timestamps[0])
    if duration_s < settings.min_duration_s:
        return _empty_result((f"Duration {duration_s:.2f}s is below configured minimum {settings.min_duration_s:.2f}s.",), signal.face_fraction, signal.motion_level)

    # Discard the first 1.5s to remove camera auto-exposure and settling transients
    discard_s = getattr(settings, "discard_initial_s", 1.5)
    if duration_s > discard_s and (duration_s - discard_s) >= settings.min_duration_s:
        mask = timestamps >= (timestamps[0] + discard_s)
        timestamps = timestamps[mask]
        forehead_rgb = forehead_rgb[mask]
        left_rgb = left_rgb[mask]
        right_rgb = right_rgb[mask]
        if signal.per_frame_motion and len(signal.per_frame_motion) == len(mask):
            raw_motion = np.asarray(signal.per_frame_motion)[mask]
        else:
            raw_motion = None
        duration_s = float(timestamps[-1] - timestamps[0])
    else:
        raw_motion = np.asarray(signal.per_frame_motion) if signal.per_frame_motion else None

    time_s, forehead_uniform = _resample(timestamps, forehead_rgb, settings.target_hz)
    _, left_uniform = _resample(timestamps, left_rgb, settings.target_hz)
    _, right_uniform = _resample(timestamps, right_rgb, settings.target_hz)
    motion_series = np.interp(time_s, timestamps, raw_motion) if raw_motion is not None and len(raw_motion) == len(timestamps) else None

    filtered = tuple(_bandpass(_extract_pulse(values, settings), settings) for values in (forehead_uniform, left_uniform, right_uniform))
    if any(values is None for values in filtered):
        return _empty_result(("Signal is too short for the configured zero-phase bandpass filter.",), signal.face_fraction, signal.motion_level)
    forehead, left, right = filtered
    assert forehead is not None and left is not None and right is not None
    metrics = tuple(_bpm_and_snr(values, settings) for values in (forehead, left, right))
    bpm, snr, peak_hz = tuple(metric[0] for metric in metrics), tuple(metric[1] for metric in metrics), tuple(metric[2] for metric in metrics)

    max_lag_samples = int(round((getattr(settings, "max_lag_ms", 150.0) / 1000.0) * settings.target_hz))
    global_r = (
        _max_lag_cross_correlation(forehead, left, max_lag_samples),
        _max_lag_cross_correlation(forehead, right, max_lag_samples),
        _max_lag_cross_correlation(left, right, max_lag_samples),
    )
    global_values = [value for value in global_r if value is not None]
    mean_r = float(np.mean(global_values)) if global_values else None
    windows = _coherence_windows(time_s, (forehead, left, right), settings, motion_series=motion_series)

    artifact_windows = tuple(w for w in windows if w.is_artifact)
    clean_windows = tuple(w for w in windows if not w.is_artifact)
    clean_window_fraction = float(len(clean_windows) / len(windows)) if windows else 1.0

    finite_bpm = [value for value in bpm if value is not None]
    finite_peak = [value for value in peak_hz if value is not None]
    finite_snr = [value for value in snr if value is not None and np.isfinite(value)]
    median_snr = float(np.median(finite_snr)) if finite_snr else None
    best_2 = tuple(sorted(finite_snr, reverse=True)[:2]) if len(finite_snr) >= 2 else ()
    best_2_mean = float(np.mean(best_2)) if best_2 else None

    reasons: list[str] = []

    # 1. Duration check
    duration_ok = duration_s >= settings.min_duration_s
    LOGGER.info(
        "Quality check [duration]: measured=%.2fs, threshold=%.2fs -> %s",
        duration_s, settings.min_duration_s, "PASS" if duration_ok else "FAIL",
    )
    if not duration_ok:
        reasons.append(
            f"Duration {duration_s:.2f}s is below minimum threshold {settings.min_duration_s:.2f}s."
        )

    # 2. Face fraction check
    face_ok = signal.face_fraction >= settings.min_face_fraction
    LOGGER.info(
        "Quality check [face_fraction]: measured=%.2f, threshold=%.2f -> %s",
        signal.face_fraction, settings.min_face_fraction, "PASS" if face_ok else "FAIL",
    )
    if not face_ok:
        reasons.append(
            f"Face fraction {signal.face_fraction:.2f} is below minimum threshold {settings.min_face_fraction:.2f}."
        )

    # 3. Motion level check
    motion_val = signal.motion_level if signal.motion_level is not None else 0.0
    motion_ok = signal.motion_level is None or signal.motion_level <= settings.max_motion_level
    LOGGER.info(
        "Quality check [motion_level]: measured=%.4f, threshold=%.4f -> %s",
        motion_val, settings.max_motion_level, "PASS" if motion_ok else "FAIL",
    )
    if not motion_ok:
        reasons.append(
            f"Motion level {motion_val:.3f} exceeds maximum threshold {settings.max_motion_level:.3f}."
        )

    # 4. Spectral BPM peak estimation
    bpm_ok = len(finite_bpm) == 3
    LOGGER.info(
        "Quality check [spectral_bpm]: peaks=%s (3 required) -> %s",
        bpm, "PASS" if bpm_ok else "FAIL",
    )
    if not bpm_ok:
        reasons.append(
            f"Unable to estimate a spectral BPM peak for all 3 ROIs (found {len(finite_bpm)}/3)."
        )

    # 5. SNR check: Gate on median ROI SNR, report per-ROI and best 2 of 3
    snr_forehead = snr[0] if len(snr) > 0 and snr[0] is not None else float("nan")
    snr_left = snr[1] if len(snr) > 1 and snr[1] is not None else float("nan")
    snr_right = snr[2] if len(snr) > 2 and snr[2] is not None else float("nan")
    snr_ok = len(finite_snr) == 3 and median_snr is not None and median_snr >= settings.min_snr_db
    LOGGER.info(
        "Quality check [snr_db]: Forehead=%.2fdB, LeftCheek=%.2fdB, RightCheek=%.2fdB | median=%.2fdB, best_2_of_3=%s (mean=%.2fdB), threshold=%.2fdB -> %s",
        snr_forehead, snr_left, snr_right,
        median_snr if median_snr is not None else float("nan"),
        f"({best_2[0]:.2f}, {best_2[1]:.2f})" if len(best_2) == 2 else str(best_2),
        best_2_mean if best_2_mean is not None else float("nan"),
        settings.min_snr_db,
        "PASS" if snr_ok else "FAIL",
    )
    if not snr_ok:
        med_str = f"{median_snr:.2f}" if median_snr is not None else "unavailable"
        reasons.append(
            f"Median ROI SNR {med_str} dB is below minimum threshold {settings.min_snr_db:.2f} dB (per-ROI: Forehead={snr_forehead:.2f} dB, Left={snr_left:.2f} dB, Right={snr_right:.2f} dB)."
        )

    quality = QualityReport(
        passed=not reasons,
        face_fraction=signal.face_fraction,
        snr_db=median_snr,
        explanation="Quality gate passed." if not reasons else " ".join(reasons),
        motion_level=signal.motion_level,
        duration_ok=duration_ok,
        snr_ok=snr_ok,
        reasons=tuple(reasons),
    )
    result = RPPGResult(
        score=mean_r, uncertainty=None, explanation="rPPG extraction completed." if quality.passed else quality.explanation,
        usable=quality.passed,
        traces=RPPGTraces(tuple(float(value) for value in time_s), tuple(float(value) for value in forehead), tuple(float(value) for value in left), tuple(float(value) for value in right)),
        windowed_r=windows, mean_r=mean_r, global_r=global_r, bpm_per_roi=bpm, snr_db_per_roi=snr,
        bpm_agreement_bpm=float(max(finite_bpm) - min(finite_bpm)) if len(finite_bpm) == 3 else None,
        peak_frequency_difference_hz=float(max(finite_peak) - min(finite_peak)) if len(finite_peak) == 3 else None,
        incoherent_windows=tuple(window for window in windows if window.incoherent and not window.is_artifact),
        quality=quality,
        clean_window_fraction=clean_window_fraction,
        artifact_windows=artifact_windows,
    )
    LOGGER.info("rPPG BPM=%s mean_r=%s SNR=%s quality=%s", result.bpm_per_roi, result.mean_r, result.snr_db_per_roi, result.usable)
    return result


def analyze_video(video_path: str | Path, config: RPPGConfig | None = None) -> VideoRPPGAnalysis:
    """Decode and track a video once, then report its rPPG evidence.

    Configure logging in the calling application to see the BPM, coherence, SNR,
    and quality summary emitted by :func:`analyze_rppg`.
    """
    from engine.face_tracker import track_video
    from engine.video_io import load_video

    loaded = load_video(video_path)
    tracking = track_video(loaded.frames)
    result = analyze_rppg(rppg_input_from_tracking(tracking), config=config)
    return VideoRPPGAnalysis(metadata=loaded.metadata, tracking=tracking, rppg=result)


def validate_against_ground_truth(video_path: str | Path, gt_bpm_series: Sequence[tuple[float, float]], config: RPPGConfig | None = None) -> GroundTruthValidationResult:
    """Analyse one video once and return per-ROI MAE against BPM labels.

    A video-level peak is compared to each finite ground-truth sample; it is not
    a time-varying BPM estimator, so fast HR changes are a known limitation.
    """
    analysis = analyze_video(video_path, config=config)
    result = analysis.rppg
    reference = np.asarray([bpm for _, bpm in gt_bpm_series], dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    reference_mean = float(np.mean(reference)) if reference.size else None
    errors = tuple(float(np.mean(np.abs(reference - estimate))) if estimate is not None and reference.size else None for estimate in result.bpm_per_roi)
    return GroundTruthValidationResult(result, reference_mean, errors, "MAE compares each video-level ROI BPM estimate with every finite reference BPM sample.")


def main() -> None:
    """Run the single-pass video analysis and log its human-readable summary."""
    parser = argparse.ArgumentParser(description="Run DeepTrace rPPG analysis on one video.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument("--debug", action="store_true", help="Print debug diagnostics: BPM, SNR, and r per ROI pair.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if arguments.debug else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    analysis = analyze_video(arguments.video_path)
    result = analysis.rppg

    if arguments.debug:
        meta = analysis.metadata
        quality = result.quality
        fh_bpm, left_bpm, right_bpm = result.bpm_per_roi
        fh_snr, left_snr, right_snr = result.snr_db_per_roi
        r_fl, r_fr, r_lr = result.global_r if result.global_r else (None, None, None)

        finite_snr = [s for s in result.snr_db_per_roi if s is not None and np.isfinite(s)]
        med_snr = float(np.median(finite_snr)) if finite_snr else None
        best_2 = sorted(finite_snr, reverse=True)[:2] if len(finite_snr) >= 2 else []
        best_2_mean = float(np.mean(best_2)) if best_2 else None

        print("=" * 60)
        print(" DeepTrace rPPG Debug Analysis")
        print("=" * 60)
        print(f"Video File:      {arguments.video_path}")
        print(f"Resolution:      {meta.resolution[0]}x{meta.resolution[1]}")
        print(f"Duration:        {meta.duration_s:.2f} s ({meta.frame_count} frames @ {meta.fps:.2f} fps)")
        print("-" * 60)
        print("QUALITY GATE:")
        if quality:
            print(f"  Passed:          {quality.passed}")
            print(f"  Face Fraction:   {quality.face_fraction:.2%}")
            print(f"  Motion Level:    {quality.motion_level:.4f}")
            print(f"  Duration OK:     {quality.duration_ok}")
            print(f"  SNR OK:          {quality.snr_ok}")
            if quality.reasons:
                print("  Failing Reasons:")
                for r in quality.reasons:
                    print(f"    - {r}")
        print("-" * 60)
        print("HEART RATE (BPM per ROI):")
        print(f"  Forehead:        {f'{fh_bpm:.1f} BPM' if fh_bpm is not None else 'None'}")
        print(f"  Left Cheek:      {f'{left_bpm:.1f} BPM' if left_bpm is not None else 'None'}")
        print(f"  Right Cheek:     {f'{right_bpm:.1f} BPM' if right_bpm is not None else 'None'}")
        if result.bpm_agreement_bpm is not None:
            print(f"  Spread / Diff:   {result.bpm_agreement_bpm:.2f} BPM")
        print("-" * 60)
        print("SIGNAL-TO-NOISE RATIO (SNR dB per ROI):")
        print(f"  Forehead:        {f'{fh_snr:.2f} dB' if fh_snr is not None else 'None'}")
        print(f"  Left Cheek:      {f'{left_snr:.2f} dB' if left_snr is not None else 'None'}")
        print(f"  Right Cheek:     {f'{right_snr:.2f} dB' if right_snr is not None else 'None'}")
        print(f"  Median SNR:      {f'{med_snr:.2f} dB' if med_snr is not None else 'None'}")
        if best_2:
            print(f"  Best 2 of 3:     [{best_2[0]:.2f} dB, {best_2[1]:.2f} dB] (mean: {best_2_mean:.2f} dB)")
        print("-" * 60)
        print("-" * 60)
        print("SPATIAL COHERENCE (Pearson r per ROI pair with +/-150ms max lag):")
        print(f"  Forehead <-> Left Cheek:    {f'{r_fl:.4f}' if r_fl is not None else 'None'}")
        print(f"  Forehead <-> Right Cheek:   {f'{r_fr:.4f}' if r_fr is not None else 'None'}")
        print(f"  Left Cheek <-> Right Cheek: {f'{r_lr:.4f}' if r_lr is not None else 'None'}")
        print(f"  Mean Spatial r:             {f'{result.mean_r:.4f}' if result.mean_r is not None else 'None'}")

        # Compute both CHROM and POS for comparison
        cfg_base = load_rppg_config()
        rppg_inp = rppg_input_from_tracking(analysis.tracking)
        cfg_chrom = RPPGConfig(**{**cfg_base.__dict__, "method": "chrom"})
        cfg_pos = RPPGConfig(**{**cfg_base.__dict__, "method": "pos"})
        res_chrom = analyze_rppg(rppg_inp, config=cfg_chrom)
        res_pos = analyze_rppg(rppg_inp, config=cfg_pos)

        print("-" * 60)
        print("METHOD COMPARISON (CHROM vs POS):")
        print(f"  Configured Active Method:   {getattr(cfg_base, 'method', 'chrom').upper()}")
        print(f"  CHROM Mean Spatial r:       {f'{res_chrom.mean_r:.4f}' if res_chrom.mean_r is not None else 'None'}")
        print(f"  POS Mean Spatial r:         {f'{res_pos.mean_r:.4f}' if res_pos.mean_r is not None else 'None'}")
        print("-" * 60)
        print(f"COHERENCE WINDOWS:     {len(result.windowed_r)} total, {len(result.incoherent_windows)} anomalous/incoherent, {len(result.artifact_windows)} rejected artifact(s)")
        clean_str = f"{result.clean_window_fraction:.2%}" if result.clean_window_fraction is not None else "N/A"
        print(f"CLEAN WINDOW FRACTION: {clean_str}")
        print(f"OVERALL USABLE:        {result.usable}")
        print("=" * 60)
    else:
        LOGGER.info("BPM (forehead, left cheek, right cheek): %s", result.bpm_per_roi)
        LOGGER.info("Mean spatial r: %s; SNR dB: %s", result.mean_r, result.snr_db_per_roi)
        LOGGER.info("Quality usable: %s; reasons: %s", result.usable, result.quality.reasons if result.quality else ())


if __name__ == "__main__":
    main()
