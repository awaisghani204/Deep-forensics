"""Audio-guided bilabial closure evidence without a second face-mesh pass.

The lip measure is a dimensionless image-space ratio, not millimetres: both
lip aperture and inter-ocular distance are projected through the camera. Its
limitations include accents and out-of-dictionary words, fast speech,
word-level timestamp imprecision, lip occlusion, and inaccurate landmarks.
It is forensic evidence rather than a deepfake verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import argparse
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import correlate, correlation_lags
import yaml

from engine.types import Anomaly, PhoneticResult

if TYPE_CHECKING:
    from engine.face_tracker import FaceTrackingResult, LandmarkPoint


LOGGER = logging.getLogger(__name__)
_BILABIAL_PHONEMES = frozenset(("P", "B", "M"))


@dataclass(frozen=True)
class PhoneticConfig:
    """All calibrated phonetic gates read from ``config.yaml``."""

    tolerance_ms: float
    model_size: str
    min_speech_s: float
    closure_percentile: float
    closure_margin: float
    min_bilabial_events: int
    aperture_target_hz: float
    aperture_smoothing_s: float
    min_landmark_fraction: float
    burst_window_ms: float
    burst_rms_percentile: float
    max_lag_ms: float


@dataclass(frozen=True)
class ApertureSample:
    """One valid normalized lip-aperture observation on the video timeline."""

    timestamp_s: float
    normalized_aperture_ratio: float


@dataclass(frozen=True)
class ApertureTrace:
    """Lightly smoothed, uniformly sampled dimensionless aperture signal."""

    time_s: tuple[float, ...]
    normalized_aperture_ratio: tuple[float, ...]
    landmark_fraction: float


@dataclass(frozen=True)
class WordTiming:
    """One word-level transcription timing in seconds from video start."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class BilabialEvent:
    """A CMU-dictionary word containing at least one P, B, or M phoneme."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AudioFeatures:
    """Acoustic secondary-evidence signals aligned to audio/video start."""

    time_s: tuple[float, ...]
    rms: tuple[float, ...]
    onset_s: tuple[float, ...]


@dataclass(frozen=True)
class AudioExtractionResult:
    """Temporary mono WAV path or a human-readable extraction failure reason."""

    audio_path: Path | None
    reason: str | None


@dataclass(frozen=True)
class TranscriptResult:
    """Word timings and compact segment snippets from faster-whisper."""

    words: tuple[WordTiming, ...]
    snippets: tuple[str, ...]
    reason: str | None


def load_phonetic_config(config_path: str | Path | None = None) -> PhoneticConfig:
    """Load all phonetic thresholds and sizes from the project config."""
    path = Path(config_path) if config_path is not None else Path(__file__).parents[1] / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)["phonetics"]
    return PhoneticConfig(
        tolerance_ms=float(values["tolerance_ms"]),
        model_size=str(values["model_size"]),
        min_speech_s=float(values["min_speech_s"]),
        closure_percentile=float(values["closure_percentile"]),
        closure_margin=float(values["closure_margin"]),
        min_bilabial_events=int(values["min_bilabial_events"]),
        aperture_target_hz=float(values["aperture_target_hz"]),
        aperture_smoothing_s=float(values["aperture_smoothing_s"]),
        min_landmark_fraction=float(values["min_landmark_fraction"]),
        burst_window_ms=float(values["burst_window_ms"]),
        burst_rms_percentile=float(values["burst_rms_percentile"]),
        max_lag_ms=float(values["max_lag_ms"]),
    )


def normalized_aperture_ratio(
    upper_lip: "LandmarkPoint", lower_lip: "LandmarkPoint", left_eye: "LandmarkPoint", right_eye: "LandmarkPoint"
) -> float | None:
    """Return vertical lip aperture normalized by inter-ocular distance.

    This ratio deliberately has no physical length unit. Camera distance,
    perspective, and face size affect both projected measurements, so reporting
    millimetres would be unsupported without calibrated camera geometry.
    """
    interocular = float(np.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y))
    if interocular <= np.finfo(float).eps:
        return None
    return abs(upper_lip.y - lower_lip.y) / interocular


def _smooth(values: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth with edge padding so the first/last aperture are not depressed."""
    if window_size <= 1:
        return values.copy()
    left = window_size // 2
    right = window_size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window_size) / window_size, mode="valid")


def build_aperture_trace(
    samples: Sequence[ApertureSample], landmark_fraction: float, config: PhoneticConfig | None = None
) -> ApertureTrace:
    """Resample irregular valid landmark observations and smooth them lightly."""
    settings = config or load_phonetic_config()
    if len(samples) < 2:
        return ApertureTrace((), (), landmark_fraction)
    timestamps = np.asarray([sample.timestamp_s for sample in samples], dtype=np.float64)
    apertures = np.asarray([sample.normalized_aperture_ratio for sample in samples], dtype=np.float64)
    if not np.isfinite(timestamps).all() or not np.isfinite(apertures).all() or np.any(np.diff(timestamps) <= 0.0):
        return ApertureTrace((), (), landmark_fraction)
    step = 1.0 / settings.aperture_target_hz
    timeline = np.arange(timestamps[0], timestamps[-1] + step * 0.5, step)
    resampled = np.interp(timeline, timestamps, apertures)
    window_size = max(1, int(round(settings.aperture_smoothing_s * settings.aperture_target_hz)))
    smoothed = _smooth(resampled, window_size)
    return ApertureTrace(
        time_s=tuple(float(value) for value in timeline),
        normalized_aperture_ratio=tuple(float(value) for value in smoothed),
        landmark_fraction=landmark_fraction,
    )


def aperture_from_tracking(tracking: "FaceTrackingResult", config: PhoneticConfig | None = None) -> ApertureTrace:
    """Reuse the compact tracker landmarks; this function never invokes Face Mesh."""
    observations: list[ApertureSample] = []
    for frame in tracking.frames:
        if not frame.face_found or frame.landmarks is None:
            continue
        ratio = normalized_aperture_ratio(
            frame.landmarks.upper_lip,
            frame.landmarks.lower_lip,
            frame.landmarks.left_eye_corner,
            frame.landmarks.right_eye_corner,
        )
        if ratio is not None:
            observations.append(ApertureSample(frame.timestamp_s, ratio))
    fraction = len(observations) / len(tracking.frames) if tracking.frames else 0.0
    return build_aperture_trace(observations, fraction, config)


def calibrate_closure_threshold(aperture: ApertureTrace, config: PhoneticConfig | None = None) -> float | None:
    """Calibrate a per-video closure threshold from the speaker's own aperture."""
    settings = config or load_phonetic_config()
    values = np.asarray(aperture.normalized_aperture_ratio, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(np.percentile(values, settings.closure_percentile) + settings.closure_margin)


def extract_audio(video_path: str | Path, output_path: str | Path) -> AudioExtractionResult:
    """Extract a mono 16 kHz WAV with ffmpeg, without claiming audio exists."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return AudioExtractionResult(None, "ffmpeg is unavailable; cannot extract an audio track.")
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return AudioExtractionResult(None, f"Audio extraction failed: {error}")
    if completed.returncode != 0 or not Path(output_path).is_file():
        detail = completed.stderr.strip() or "no audio track or ffmpeg decoding failure"
        return AudioExtractionResult(None, f"No usable audio track: {detail}")
    return AudioExtractionResult(Path(output_path), None)


def _whisper_device() -> tuple[str, str]:
    """Choose CUDA float16 only when CTranslate2 reports an available device."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except (ImportError, AttributeError, RuntimeError):
        pass
    return "cpu", "int8"


def transcribe_audio(audio_path: str | Path, config: PhoneticConfig | None = None) -> TranscriptResult:
    """Transcribe word timings with faster-whisper and log the selected device."""
    settings = config or load_phonetic_config()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return TranscriptResult((), (), "faster-whisper is not installed.")
    device, compute_type = _whisper_device()
    LOGGER.info("faster-whisper device=%s compute_type=%s model=%s", device, compute_type, settings.model_size)
    try:
        model = WhisperModel(settings.model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(str(audio_path), word_timestamps=True)
        words: list[WordTiming] = []
        snippets: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                snippets.append(text)
            for word in segment.words or ():
                if word.start is not None and word.end is not None:
                    words.append(WordTiming(str(word.word).strip(), float(word.start), float(word.end)))
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("Whisper transcription failed: %s", error)
        return TranscriptResult((), (), f"Whisper transcription failed: {error}")
    if not words:
        lang = getattr(info, "language", "unknown") if "info" in locals() else "unknown"
        lang_prob = getattr(info, "language_probability", 0.0) if "info" in locals() else 0.0
        if not snippets:
            reason = f"No speech detected in audio track (silent or non-vocal audio, detected_language={lang}, p={lang_prob:.2f})."
        else:
            reason = f"Whisper transcribed snippets without word timestamps (snippets={len(snippets)}, language={lang}, p={lang_prob:.2f})."
        LOGGER.info("Audio track present but yielded 0 words: %s", reason)
        return TranscriptResult((), tuple(snippets), reason)
    speech_duration = max(word.end_s for word in words) - min(word.start_s for word in words)
    if speech_duration < settings.min_speech_s:
        return TranscriptResult(tuple(words), tuple(snippets), f"Speech duration {speech_duration:.2f}s is below configured minimum {settings.min_speech_s:.2f}s.")
    return TranscriptResult(tuple(words), tuple(snippets), None)


def _normalized_word(word: str) -> str:
    """Strip punctuation for CMU-dictionary lookup without changing timings."""
    return re.sub(r"[^a-z']", "", word.lower())


def bilabial_events(words: Sequence[WordTiming]) -> tuple[BilabialEvent, ...]:
    """Mark English CMU-dictionary words containing P, B, or M phonemes."""
    try:
        import cmudict
    except ImportError:
        LOGGER.warning("cmudict is unavailable; no bilabial words can be mapped")
        return ()
    pronunciations = cmudict.dict()
    events: list[BilabialEvent] = []
    for timing in words:
        variants = pronunciations.get(_normalized_word(timing.word), ())
        has_bilabial = any(any(phone.rstrip("012") in _BILABIAL_PHONEMES for phone in pronunciation) for pronunciation in variants)
        if has_bilabial:
            events.append(BilabialEvent(timing.word, timing.start_s, timing.end_s))
    return tuple(events)


def extract_audio_features(audio_path: str | Path) -> AudioFeatures | None:
    """Build librosa RMS/onset secondary evidence; it never decides closure alone."""
    try:
        import librosa
    except ImportError:
        LOGGER.warning("librosa is unavailable; acoustic burst confirmation is omitted")
        return None
    try:
        audio, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        rms = librosa.feature.rms(y=audio)[0]
        times = librosa.times_like(rms, sr=sample_rate)
        onset = librosa.onset.onset_detect(y=audio, sr=sample_rate, units="time")
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("Could not extract optional acoustic features: %s", error)
        return None
    return AudioFeatures(tuple(float(value) for value in times), tuple(float(value) for value in rms), tuple(float(value) for value in onset))


def acoustic_burst_confirmation(events: Sequence[BilabialEvent], features: AudioFeatures | None, config: PhoneticConfig | None = None) -> tuple[bool | None, ...]:
    """Return secondary RMS/onset burst evidence near each word start."""
    if features is None or not features.rms:
        return tuple(None for _ in events)
    settings = config or load_phonetic_config()
    time_s, rms = np.asarray(features.time_s), np.asarray(features.rms)
    threshold = float(np.percentile(rms, settings.burst_rms_percentile))
    radius = settings.burst_window_ms / 1000.0
    return tuple(
        bool(
            (np.any(np.abs(time_s - event.start_s) <= radius) and np.max(rms[np.abs(time_s - event.start_s) <= radius]) >= threshold)
            or any(abs(onset - event.start_s) <= radius for onset in features.onset_s)
        )
        for event in events
    )


def estimate_av_lag_ms(aperture: ApertureTrace, features: AudioFeatures | None, config: PhoneticConfig | None = None) -> float | None:
    """Estimate lag by RMS/aperture cross-correlation; positive means video lags audio."""
    if features is None or len(aperture.time_s) < 2 or len(features.time_s) < 2:
        return None
    settings = config or load_phonetic_config()
    start = max(aperture.time_s[0], features.time_s[0])
    end = min(aperture.time_s[-1], features.time_s[-1])
    if end <= start:
        return None
    step = 1.0 / settings.aperture_target_hz
    timeline = np.arange(start, end + step * 0.5, step)
    if timeline.size < 2:
        return None
    visual = np.interp(timeline, aperture.time_s, aperture.normalized_aperture_ratio)
    acoustic = np.interp(timeline, features.time_s, features.rms)
    if np.std(visual) <= np.finfo(float).eps or np.std(acoustic) <= np.finfo(float).eps:
        return None
    visual = (visual - np.mean(visual)) / np.std(visual)
    acoustic = (acoustic - np.mean(acoustic)) / np.std(acoustic)
    lags = correlation_lags(visual.size, acoustic.size, mode="full")
    values = correlate(visual, acoustic, mode="full")
    allowed = np.abs(lags / settings.aperture_target_hz * 1000.0) <= settings.max_lag_ms
    if not np.any(allowed):
        return None
    return float(lags[allowed][int(np.argmax(values[allowed]))] / settings.aperture_target_hz * 1000.0)


def _unusable(
    reason: str,
    snippets: tuple[str, ...] = (),
    has_audio: bool = False,
    speech_detected: bool = False,
) -> PhoneticResult:
    """Return a complete negative result for missing audio, speech, or landmarks."""
    return PhoneticResult(
        score=None,
        bilabial_events_checked=0,
        violations=0,
        violation_rate=None,
        lag_ms=None,
        anomalies=(),
        usable=False,
        explanation=reason,
        reason=reason,
        transcript_snippets=snippets,
        has_audio=has_audio,
        speech_detected=speech_detected,
    )


def analyze_bilabial_events(
    aperture: ApertureTrace,
    events: Sequence[BilabialEvent],
    config: PhoneticConfig | None = None,
    acoustic_confirmation: Sequence[bool | None] | None = None,
) -> PhoneticResult:
    """Purely testable aperture-vs-word-window closure analysis.

    Acoustic confirmation is evidence attached to an anomaly only. It never
    substitutes for the required visual aperture dip.
    """
    settings = config or load_phonetic_config()
    if aperture.landmark_fraction < settings.min_landmark_fraction:
        return _unusable(f"Landmark fraction {aperture.landmark_fraction:.2f} is below configured minimum {settings.min_landmark_fraction:.2f}.")
    threshold = calibrate_closure_threshold(aperture, settings)
    if threshold is None:
        return _unusable("No valid lip-aperture samples are available.")
    times = np.asarray(aperture.time_s)
    values = np.asarray(aperture.normalized_aperture_ratio)
    tolerance_s = settings.tolerance_ms / 1000.0
    anomalies: list[Anomaly] = []
    checked = 0
    unavailable = 0
    confirmations = acoustic_confirmation or tuple(None for _ in events)
    for index, event in enumerate(events):
        start, end = event.start_s - tolerance_s, event.end_s + tolerance_s
        in_window = (times >= start) & (times <= end)
        if not np.any(in_window):
            unavailable += 1
            continue
        checked += 1
        minimum = float(np.min(values[in_window]))
        if minimum > threshold:
            evidence = (
                ("word", event.word),
                ("min_aperture_ratio", f"{minimum:.5f}"),
                ("closure_threshold", f"{threshold:.5f}"),
                ("acoustic_burst_confirmed", str(confirmations[index] if index < len(confirmations) else None)),
            )
            severity = min(1.0, max(0.0, (minimum - threshold) / max(threshold, np.finfo(float).eps)))
            anomalies.append(
                Anomaly(
                    start_s=float(start),
                    end_s=float(end),
                    kind="lip_closure_missing",
                    severity=float(severity),
                    explanation=f"No normalized lip-aperture dip below the calibrated closure threshold was observed for bilabial word '{event.word}'.",
                    evidence=evidence,
                )
            )
    violations = len(anomalies)
    rate = violations / checked if checked else None
    reasons: list[str] = []
    if unavailable:
        reasons.append(f"{unavailable} bilabial word window(s) had no usable face landmarks.")
    if checked < settings.min_bilabial_events:
        reasons.append(f"Only {checked} bilabial event(s) were checked; configured minimum is {settings.min_bilabial_events}.")
    usable = not reasons
    explanation = "Bilabial closure analysis completed." if usable else " ".join(reasons)
    return PhoneticResult(
        score=1.0 - rate if usable and rate is not None else None,
        bilabial_events_checked=checked,
        violations=violations,
        violation_rate=rate,
        lag_ms=None,
        anomalies=tuple(anomalies),
        usable=usable,
        explanation=explanation,
        reason=None if usable else explanation,
        closure_threshold=threshold,
        has_audio=True,
        speech_detected=True,
    )


def analyze_phonetics(video_path: str | Path, tracking: "FaceTrackingResult", config: PhoneticConfig | None = None) -> PhoneticResult:
    """Analyse audio and pre-extracted tracker landmarks without rerunning Face Mesh."""
    settings = config or load_phonetic_config()
    aperture = aperture_from_tracking(tracking, settings)
    if aperture.landmark_fraction < settings.min_landmark_fraction:
        return _unusable(f"Landmark fraction {aperture.landmark_fraction:.2f} is below configured minimum {settings.min_landmark_fraction:.2f}.")
    with tempfile.TemporaryDirectory(prefix="deeptrace-audio-") as temporary_directory:
        extraction = extract_audio(video_path, Path(temporary_directory) / "audio.wav")
        if extraction.reason or extraction.audio_path is None:
            return _unusable(extraction.reason or "Audio extraction failed.", has_audio=False, speech_detected=False)
        transcript = transcribe_audio(extraction.audio_path, settings)
        if transcript.reason or not transcript.words:
            return _unusable(transcript.reason or "No speech words transcribed.", transcript.snippets, has_audio=True, speech_detected=False)
        events = bilabial_events(transcript.words)
        features = extract_audio_features(extraction.audio_path)
        confirmation = acoustic_burst_confirmation(events, features, settings)
        result = analyze_bilabial_events(aperture, events, settings, confirmation)
        lag = estimate_av_lag_ms(aperture, features, settings)
    result = replace(result, lag_ms=lag, transcript_snippets=transcript.snippets, has_audio=True, speech_detected=True)
    LOGGER.info("Transcript snippets: %s", transcript.snippets[:3])
    LOGGER.info("Bilabial events=%d violations=%d lag_ms=%s usable=%s", result.bilabial_events_checked, result.violations, result.lag_ms, result.usable)
    return result


def main() -> None:
    """Track a video once, then log phonetic evidence from those landmarks."""
    parser = argparse.ArgumentParser(description="Run DeepTrace bilabial-closure analysis on one video.")
    parser.add_argument("video_path", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from engine.face_tracker import track_video
    from engine.video_io import load_video

    loaded = load_video(arguments.video_path)
    tracking = track_video(loaded.frames)
    result = analyze_phonetics(arguments.video_path, tracking)
    LOGGER.info("Transcript snippets: %s", result.transcript_snippets[:3])
    LOGGER.info("P/B/M checked=%d violations=%d rate=%s lag_ms=%s", result.bilabial_events_checked, result.violations, result.violation_rate, result.lag_ms)


if __name__ == "__main__":
    main()
