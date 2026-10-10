"""Synthetic rPPG tests; no real video, camera, or model asset is required."""

import numpy as np

from engine.rppg import RPPGInput, analyze_rppg


def _rgb_trace(timestamps: np.ndarray, phase: float, noise: float, seed: int) -> tuple[tuple[float, float, float], ...]:
    """Create a CHROM-visible 1.2 Hz RGB pulse with deterministic noise."""
    random = np.random.default_rng(seed)
    pulse = np.sin(2.0 * np.pi * 1.2 * timestamps + phase)
    rgb = np.column_stack((100.0 + 0.2 * pulse, 120.0 + 2.5 * pulse, 90.0 + 0.1 * pulse))
    return tuple(tuple(float(value) for value in row) for row in rgb + random.normal(0.0, noise, rgb.shape))


def _signal(phases: tuple[float, float, float], noise: float = 0.03, irregular: bool = False) -> RPPGInput:
    """Build a deterministic, three-site synthetic trace longer than 30 seconds."""
    times = np.arange(0.0, 30.0, 1.0 / 29.0)
    if irregular:
        times = times + 0.004 * np.sin(np.arange(times.size) * 0.17)
    return RPPGInput(
        timestamps_s=tuple(float(value) for value in times),
        forehead_rgb=_rgb_trace(times, phases[0], noise, 1),
        left_cheek_rgb=_rgb_trace(times, phases[1], noise, 2),
        right_cheek_rgb=_rgb_trace(times, phases[2], noise, 3),
        face_fraction=1.0,
        motion_level=0.0,
    )


def test_rppg_recovers_approximately_72_bpm_from_non_uniform_timestamps() -> None:
    """Interpolation plus CHROM should recover the injected 1.2 Hz pulse."""
    result = analyze_rppg(_signal((0.0, 0.02, -0.01), irregular=True))

    assert result.usable
    assert all(bpm is not None and abs(bpm - 72.0) <= 3.0 for bpm in result.bpm_per_roi)
    assert result.traces is not None and len(result.traces.time_s) > 800


def test_shared_pulse_has_high_spatial_coherence() -> None:
    """Near-phase-aligned ROIs must produce high global coherence."""
    result = analyze_rppg(_signal((0.0, 0.02, -0.01)))

    assert result.mean_r is not None and result.mean_r > 0.85
    assert not result.incoherent_windows


def test_independent_roi_noise_has_low_spatial_coherence() -> None:
    """Independent RGB noise must not masquerade as a shared rPPG pulse."""
    random = np.random.default_rng(12)
    times = np.arange(0.0, 30.0, 1.0 / 30.0)
    traces = [tuple(tuple(float(value) for value in row) for row in 100.0 + random.normal(0.0, 1.0, (times.size, 3))) for _ in range(3)]
    result = analyze_rppg(RPPGInput(tuple(float(value) for value in times), traces[0], traces[1], traces[2]))

    assert result.mean_r is not None and result.mean_r < 0.35
    assert result.incoherent_windows
