"""Synthetic-signal unit tests for bilabial closure and phonetic analysis.

No real video, camera, microphone, or external model download is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.face_tracker import LandmarkPoint
from engine.phonetics import (
    ApertureSample,
    ApertureTrace,
    AudioFeatures,
    BilabialEvent,
    PhoneticConfig,
    acoustic_burst_confirmation,
    analyze_bilabial_events,
    build_aperture_trace,
    calibrate_closure_threshold,
    estimate_av_lag_ms,
    normalized_aperture_ratio,
)


def _config() -> PhoneticConfig:
    return PhoneticConfig(
        tolerance_ms=80.0,
        model_size="base",
        min_speech_s=1.0,
        closure_percentile=10.0,
        closure_margin=0.015,
        min_bilabial_events=3,
        aperture_target_hz=30.0,
        aperture_smoothing_s=0.08,
        min_landmark_fraction=0.60,
        burst_window_ms=100.0,
        burst_rms_percentile=75.0,
        max_lag_ms=500.0,
    )


def test_normalized_aperture_ratio_valid_and_degenerate() -> None:
    """Normalized aperture ratio must scale by interocular distance and reject zero distance."""
    upper = LandmarkPoint(x=0.5, y=0.50, z=0.0)
    lower = LandmarkPoint(x=0.5, y=0.55, z=0.0)
    left_eye = LandmarkPoint(x=0.4, y=0.4, z=0.0)
    right_eye = LandmarkPoint(x=0.6, y=0.4, z=0.0)

    ratio = normalized_aperture_ratio(upper, lower, left_eye, right_eye)
    assert ratio is not None
    # Lip gap is 0.05, interocular is 0.20 -> ratio is 0.25
    assert pytest.approx(ratio, rel=1e-3) == 0.25

    # Degenerate eye distance
    same_eye = LandmarkPoint(x=0.5, y=0.4, z=0.0)
    assert normalized_aperture_ratio(upper, lower, same_eye, same_eye) is None


def test_build_aperture_trace_resampling_and_smoothing() -> None:
    """Aperture trace must uniformly resample and preserve timeline range."""
    cfg = _config()
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    samples = [ApertureSample(t, 0.2 + 0.05 * np.sin(2 * np.pi * t)) for t in times]

    trace = build_aperture_trace(samples, landmark_fraction=0.95, config=cfg)
    assert trace.landmark_fraction == 0.95
    assert len(trace.time_s) == len(trace.normalized_aperture_ratio)
    assert len(trace.time_s) > 20
    assert pytest.approx(trace.time_s[0], abs=1e-3) == 0.0
    assert pytest.approx(trace.time_s[-1], abs=0.05) == 1.0


def test_calibrate_closure_threshold() -> None:
    """Threshold must compute 10th percentile plus calibrated margin."""
    cfg = _config()
    values = np.linspace(0.05, 0.25, 100)
    trace = ApertureTrace(
        time_s=tuple(float(v) for v in np.linspace(0.0, 3.0, 100)),
        normalized_aperture_ratio=tuple(float(v) for v in values),
        landmark_fraction=1.0,
    )
    threshold = calibrate_closure_threshold(trace, cfg)
    assert threshold is not None
    expected = float(np.percentile(values, 10.0) + cfg.closure_margin)
    assert pytest.approx(threshold, rel=1e-4) == expected


def test_bilabial_closure_detects_authentic_sync() -> None:
    """When mouth closes during bilabial phoneme windows, no violations occur."""
    cfg = _config()
    # 5 seconds timeline at 30 Hz
    times = np.arange(0.0, 5.0, 1.0 / 30.0)
    # Baseline aperture is open (~0.15), dips to ~0.02 during bilabial words
    apertures = np.full_like(times, 0.15)

    events = [
        BilabialEvent(word="peter", start_s=1.0, end_s=1.3),
        BilabialEvent(word="baker", start_s=2.2, end_s=2.5),
        BilabialEvent(word="maker", start_s=3.4, end_s=3.7),
    ]

    for event in events:
        mask = (times >= event.start_s) & (times <= event.end_s)
        apertures[mask] = 0.02  # Lip closure dip

    trace = ApertureTrace(
        time_s=tuple(float(t) for t in times),
        normalized_aperture_ratio=tuple(float(a) for a in apertures),
        landmark_fraction=1.0,
    )

    result = analyze_bilabial_events(trace, events, cfg)
    assert result.usable
    assert result.bilabial_events_checked == 3
    assert result.violations == 0
    assert result.violation_rate == 0.0
    assert result.score is not None and result.score > 0.99
    assert len(result.anomalies) == 0


def test_bilabial_closure_detects_violations_in_manipulated_video() -> None:
    """When mouth remains wide open during bilabial words, anomalies are flagged."""
    cfg = _config()
    times = np.arange(0.0, 5.0, 1.0 / 30.0)
    # Baseline is fluctuating between 0.04 and 0.20 (so 10th percentile is low ~0.05)
    apertures = 0.10 + 0.06 * np.sin(2 * np.pi * 0.5 * times)

    # All 3 bilabial words stay wide open (0.22)
    events = [
        BilabialEvent(word="peter", start_s=1.0, end_s=1.3),
        BilabialEvent(word="baker", start_s=2.2, end_s=2.5),
        BilabialEvent(word="maker", start_s=3.4, end_s=3.7),
    ]

    for event in events:
        mask = (times >= event.start_s - 0.1) & (times <= event.end_s + 0.1)
        apertures[mask] = 0.22  # Open mouth during P/B/M

    trace = ApertureTrace(
        time_s=tuple(float(t) for t in times),
        normalized_aperture_ratio=tuple(float(a) for a in apertures),
        landmark_fraction=1.0,
    )

    result = analyze_bilabial_events(trace, events, cfg)
    assert result.usable
    assert result.bilabial_events_checked == 3
    assert result.violations == 3
    assert result.violation_rate == 1.0
    assert result.score is not None and result.score < 0.01
    assert len(result.anomalies) == 3
    for anomaly in result.anomalies:
        assert anomaly.kind == "lip_closure_missing"
        assert anomaly.severity > 0.0


def test_bilabial_closure_gates_low_landmarks_and_few_events() -> None:
    """Low landmark fraction or fewer than minimum events must mark result unusable."""
    cfg = _config()
    times = np.arange(0.0, 3.0, 1.0 / 30.0)
    apertures = np.full_like(times, 0.1)
    trace = ApertureTrace(
        time_s=tuple(float(t) for t in times),
        normalized_aperture_ratio=tuple(float(a) for a in apertures),
        landmark_fraction=0.40,  # Below cfg.min_landmark_fraction (0.60)
    )
    events = [BilabialEvent(word="peter", start_s=1.0, end_s=1.3)]

    result_low_lm = analyze_bilabial_events(trace, events, cfg)
    assert not result_low_lm.usable
    assert "below configured minimum" in result_low_lm.explanation

    # High landmark fraction but only 1 event (< min 3)
    trace_good = ApertureTrace(
        time_s=tuple(float(t) for t in times),
        normalized_aperture_ratio=tuple(float(a) for a in apertures),
        landmark_fraction=0.90,
    )
    result_few_events = analyze_bilabial_events(trace_good, events, cfg)
    assert not result_few_events.usable
    assert "Only 1 bilabial event(s) were checked" in result_few_events.explanation


def test_estimate_av_lag_ms_synthetic() -> None:
    """Cross-correlation should identify injected temporal lag between visual and acoustic signals."""
    cfg = _config()
    timeline = np.arange(0.0, 5.0, 1.0 / 30.0)
    # Acoustic burst pulse at t = 2.0s
    acoustic_rms = np.exp(-((timeline - 2.0) ** 2) / 0.02)
    # Visual mouth aperture pulse at t = 2.1s (video lags audio by +100 ms)
    visual_aperture = np.exp(-((timeline - 2.1) ** 2) / 0.02)

    aperture = ApertureTrace(
        time_s=tuple(float(t) for t in timeline),
        normalized_aperture_ratio=tuple(float(v) for v in visual_aperture),
        landmark_fraction=1.0,
    )
    features = AudioFeatures(
        time_s=tuple(float(t) for t in timeline),
        rms=tuple(float(v) for v in acoustic_rms),
        onset_s=(2.0,),
    )

    lag_ms = estimate_av_lag_ms(aperture, features, cfg)
    assert lag_ms is not None
    # Should be close to +100 ms (within 1-2 frames at 30fps is ~33ms)
    assert abs(lag_ms - 100.0) <= 40.0


def test_acoustic_burst_confirmation() -> None:
    """Secondary acoustic burst confirmation marks events with nearby RMS peaks."""
    cfg = _config()
    times = np.arange(0.0, 3.0, 0.01)
    rms = np.linspace(0.001, 0.02, len(times))
    # Peak at 1.0s
    rms[np.abs(times - 1.0) < 0.02] = 0.50

    features = AudioFeatures(
        time_s=tuple(float(t) for t in times),
        rms=tuple(float(r) for r in rms),
        onset_s=(1.0,),
    )
    events = [
        BilabialEvent(word="pack", start_s=1.02, end_s=1.3),  # Near burst
        BilabialEvent(word="back", start_s=2.0, end_s=2.3),   # No burst
    ]

    confirmations = acoustic_burst_confirmation(events, features, cfg)
    assert confirmations[0] is True
    assert confirmations[1] is False
