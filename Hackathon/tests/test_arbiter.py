"""Synthetic-evidence unit tests for the DeepTrace arbiter and evidence fusion.

No real video or external models required.
"""

from __future__ import annotations

import json
import pytest

from engine.arbiter import ArbiterConfig, ArbiterInput, arbitrate, load_arbiter_config, merge_anomalies
from engine.types import Anomaly, CoherenceWindow, PhoneticResult, QualityReport, RPPGResult, RPPGTraces


def _test_config() -> ArbiterConfig:
    return ArbiterConfig(
        weight_hemodynamic=0.55,
        weight_phonetic=0.35,
        weight_spectral_snr=0.10,
        cutoff_authentic=0.50,
        cutoff_suspect=0.35,
        cutoff_manipulated=0.35,
        corr_real=0.35,
        corr_fake=0.10,
        max_bpm_diff=15.0,
        snr_min_db=-3.0,
        snr_max_db=15.0,
        shrinkage_prior_events=5.0,
        min_duration_high_s=12.0,
        min_duration_med_s=6.0,
        min_events_high=5,
        min_events_med=2,
    )


def test_all_good_yields_authentic() -> None:
    """Authentic rPPG coherence and clean bilabial closures yield AUTHENTIC with HIGH confidence."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.98, snr_db=12.0, explanation="Quality passed.")

    # 20-second traces
    time_s = tuple(float(t) for t in range(21))
    traces = RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s)

    rppg = RPPGResult(
        score=0.68,
        uncertainty=None,
        explanation="Strong multi-site rPPG coherence",
        usable=True,
        mean_r=0.68,
        traces=traces,
        windowed_r=(
            CoherenceWindow(0.0, 10.0, 0.68, 0.65, 0.70, False),
            CoherenceWindow(10.0, 20.0, 0.67, 0.69, 0.66, False),
        ),
        snr_db_per_roi=(12.0, 11.5, 12.5),
        bpm_agreement_bpm=1.5,
        incoherent_windows=(),
    )

    phonetics = PhoneticResult(
        score=1.0,
        bilabial_events_checked=8,
        violations=0,
        violation_rate=0.0,
        lag_ms=10.0,
        anomalies=(),
        usable=True,
        explanation="All 8 bilabial closures verified",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    assert result.verdict == "AUTHENTIC"
    assert result.fai is not None and result.fai >= 0.70
    assert result.confidence == "HIGH"
    assert len(result.anomalies) == 0

    # Weights must sum to 1.0 and record all 3 modules
    assert pytest.approx(sum(result.weights_used.values()), abs=1e-6) == 1.0
    assert set(result.modules_contributed) == {"hemodynamic", "phonetic", "spectral_snr"}
    assert result.rppg_score is not None and result.rppg_score > 0.90
    assert result.phonetic_score is not None and result.phonetic_score > 0.80
    assert result.spectral_snr_score is not None and result.spectral_snr_score > 0.70


def test_low_r_and_violations_yields_manipulated() -> None:
    """Low rPPG correlation and bilabial violations yield MANIPULATED with merged anomalies."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.92, snr_db=8.0, explanation="Quality passed.")

    incoh_win = CoherenceWindow(2.0, 8.0, 0.10, 0.12, 0.08, True)
    rppg = RPPGResult(
        score=0.15,
        uncertainty=None,
        explanation="Spatial desynchronization",
        usable=True,
        mean_r=0.15,
        windowed_r=(incoh_win,),
        snr_db_per_roi=(5.0, 4.0, 6.0),
        bpm_agreement_bpm=15.0,
        incoherent_windows=(incoh_win,),
    )

    phon_anom = Anomaly(
        start_s=5.0,
        end_s=6.0,
        kind="lip_closure_missing",
        severity=0.8,
        explanation="Lips open during 'peter'",
        evidence=(("word", "peter"),),
    )
    phonetics = PhoneticResult(
        score=0.0,
        bilabial_events_checked=8,
        violations=8,
        violation_rate=1.0,
        lag_ms=0.0,
        anomalies=(phon_anom,),
        usable=True,
        explanation="8 violations detected",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    assert result.verdict == "MANIPULATED"
    assert result.fai is not None and result.fai < 0.40
    assert pytest.approx(sum(result.weights_used.values()), abs=1e-6) == 1.0

    # Anomalies merged across overlapping intervals:
    # rPPG desync is [2.0, 8.0], phonetic anomaly is [5.0, 6.0] -> overlapping!
    assert len(result.anomalies) == 1
    merged = result.anomalies[0]
    assert merged.start_s == 2.0
    assert merged.end_s == 8.0
    assert "rppg_spatial_desync" in merged.kind and "lip_closure_missing" in merged.kind


def test_poor_quality_yields_inconclusive() -> None:
    """Failed quality gate immediately yields INCONCLUSIVE with no FAI score."""
    cfg = _test_config()
    quality = QualityReport(
        passed=False,
        face_fraction=0.40,
        snr_db=-3.0,
        explanation="Severe camera motion and face lost in 60% of frames.",
    )
    rppg = RPPGResult(score=0.70, uncertainty=None, explanation="rppg ok", usable=True)
    phonetics = PhoneticResult(score=1.0, bilabial_events_checked=5, violations=0, violation_rate=0.0, lag_ms=0.0, anomalies=(), usable=True, explanation="phonetics ok")

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    assert result.verdict == "INCONCLUSIVE"
    assert result.fai is None
    assert result.confidence == "LOW"
    assert "Quality gate failed" in result.explain


def test_audio_missing_renormalizes_weights_and_produces_verdict_with_low_or_medium_confidence() -> None:
    """When audio is missing, phonetic module is dropped, remaining weights renormalize, and confidence is LOW/MEDIUM."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=10.0, explanation="Quality passed.")

    time_s = tuple(float(t) for t in range(18))
    traces = RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s)
    rppg = RPPGResult(
        score=0.65,
        uncertainty=None,
        explanation="rPPG available",
        usable=True,
        mean_r=0.65,
        traces=traces,
        windowed_r=(CoherenceWindow(0.0, 10.0, 0.65, 0.66, 0.64, False),),
        snr_db_per_roi=(10.0, 9.0, 11.0),
        bpm_agreement_bpm=1.0,
        incoherent_windows=(),
    )

    phonetics = PhoneticResult(
        score=None,
        bilabial_events_checked=0,
        violations=0,
        violation_rate=None,
        lag_ms=None,
        anomalies=(),
        usable=False,
        explanation="No audio track found in video container.",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    # Verdict still produced
    assert result.verdict in ("AUTHENTIC", "SUSPECT")
    assert result.fai is not None

    # Phonetic module must not contribute
    assert "phonetic" not in result.modules_contributed
    assert "hemodynamic" in result.modules_contributed
    assert "spectral_snr" in result.modules_contributed

    # Weights must sum to 1.0
    assert pytest.approx(sum(result.weights_used.values()), abs=1e-6) == 1.0

    # Hemodynamic base weight 0.55, spectral_snr 0.10 -> renormalized 0.55/0.65 and 0.10/0.65
    expected_hemo_w = 0.55 / 0.65
    assert pytest.approx(result.weights_used["hemodynamic"], abs=1e-4) == expected_hemo_w

    # Confidence must be LOW or MEDIUM (never HIGH without audio)
    assert result.confidence in ("LOW", "MEDIUM")


def test_weights_always_sum_to_one() -> None:
    """Weights used by arbitrate must always sum to 1.0 regardless of which modules are active."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=10.0, explanation="Quality passed.")

    # Case A: Only phonetic module active
    rppg_unusable = RPPGResult(score=None, uncertainty=None, explanation="unusable", usable=False)
    phonetics_usable = PhoneticResult(
        score=0.9, bilabial_events_checked=5, violations=0, violation_rate=0.0, lag_ms=0.0, anomalies=(), usable=True, explanation="ok"
    )

    result_phon_only = arbitrate(ArbiterInput(rppg_unusable, phonetics_usable, quality), config=cfg)
    assert result_phon_only.fai is not None
    assert pytest.approx(sum(result_phon_only.weights_used.values()), abs=1e-6) == 1.0
    assert result_phon_only.modules_contributed == ("phonetic",)
    assert result_phon_only.weights_used["phonetic"] == 1.0


def test_anomaly_merging_overlapping_intervals() -> None:
    """Overlapping anomaly intervals from distinct modalities must merge cleanly."""
    anoms = [
        Anomaly(start_s=1.0, end_s=3.0, kind="lip_closure_missing", severity=0.6, explanation="Lip open", evidence=(("w", "pop"),)),
        Anomaly(start_s=2.5, end_s=5.0, kind="rppg_spatial_desync", severity=0.9, explanation="Desync", evidence=(("r", "0.1"),)),
        Anomaly(start_s=7.0, end_s=9.0, kind="lip_closure_missing", severity=0.4, explanation="Lip open 2", evidence=()),
    ]

    merged = merge_anomalies(anoms)
    assert len(merged) == 2
    # First merged span [1.0, 5.0]
    assert merged[0].start_s == 1.0
    assert merged[0].end_s == 5.0
    assert merged[0].severity == 0.9
    assert len(merged[0].evidence) == 2
    # Second span [7.0, 9.0]
    assert merged[1].start_s == 7.0
    assert merged[1].end_s == 9.0


def test_explain_string_non_expert_readable() -> None:
    """Explain string must be 3 to 5 clear sentences readable by a non-expert."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=10.0, explanation="Quality passed.")
    rppg = RPPGResult(score=0.65, uncertainty=None, explanation="ok", usable=True, mean_r=0.65)
    phonetics = PhoneticResult(score=1.0, bilabial_events_checked=6, violations=0, violation_rate=0.0, lag_ms=0.0, anomalies=(), usable=True, explanation="ok")

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    assert result.explain
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[a-zA-Z\)])\.\s+", result.explain) if s.strip()]
    assert 3 <= len(sentences) <= 5
    assert result.verdict in result.explain


def test_arbiter_result_is_json_serializable() -> None:
    """ArbiterResult.to_dict() must return a JSON-serializable structure with all required fields."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=10.0, explanation="Quality passed.")
    rppg = RPPGResult(score=0.65, uncertainty=None, explanation="ok", usable=True, mean_r=0.65, windowed_r=(CoherenceWindow(0.0, 10.0, 0.65, 0.65, 0.65, False),))
    phonetics = PhoneticResult(score=1.0, bilabial_events_checked=6, violations=0, violation_rate=0.0, lag_ms=0.0, anomalies=(), usable=True, explanation="ok")

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)

    data = result.to_dict()
    assert isinstance(data, dict)
    assert "fai" in data
    assert "verdict" in data
    assert "confidence" in data
    assert "weights_used" in data
    assert "modules_contributed" in data
    assert "explain" in data
    assert "anomalies" in data

    # Verify JSON dumps does not raise TypeError
    json_bytes = json.dumps(data)
    assert len(json_bytes) > 50


def test_single_module_low_confidence_yields_inconclusive() -> None:
    """If only one module is usable and confidence is LOW, arbiter must return INCONCLUSIVE."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.90, snr_db=5.0, explanation="Quality passed.")
    # Short 7s clip without phonetics -> LOW confidence
    time_s = tuple(float(t) for t in range(8))
    rppg = RPPGResult(
        score=0.20,
        uncertainty=None,
        explanation="Low rPPG coherence",
        usable=True,
        mean_r=0.20,
        traces=RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s),
        windowed_r=(CoherenceWindow(0.0, 6.0, 0.20, 0.20, 0.20, True),),
        snr_db_per_roi=(5.0, 4.0, 6.0),
        incoherent_windows=(CoherenceWindow(0.0, 6.0, 0.20, 0.20, 0.20, True),),
        clean_window_fraction=1.0,
    )
    phonetics = PhoneticResult(
        score=None,
        bilabial_events_checked=0,
        violations=0,
        violation_rate=None,
        lag_ms=None,
        anomalies=(),
        usable=False,
        explanation="No speech detected.",
        has_audio=True,
        speech_detected=False,
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)
    assert result.verdict == "INCONCLUSIVE"
    assert result.confidence == "LOW"
    assert "LOW confidence" in result.explain or "Single-module" in result.explain


def test_rppg_alone_short_clip_cannot_be_manipulated() -> None:
    """Never output MANIPULATED from rPPG module alone on a clip shorter than 15s."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=8.0, explanation="Quality passed.")
    # 10s clip
    time_s = tuple(float(t) for t in range(11))
    rppg = RPPGResult(
        score=0.05,
        uncertainty=None,
        explanation="Very low rPPG coherence",
        usable=True,
        mean_r=-0.15,
        traces=RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s),
        windowed_r=(CoherenceWindow(0.0, 6.0, -0.15, -0.15, -0.15, True),),
        snr_db_per_roi=(8.0, 7.0, 9.0),
        incoherent_windows=(CoherenceWindow(0.0, 6.0, -0.15, -0.15, -0.15, True),),
        clean_window_fraction=1.0,
    )
    phonetics = PhoneticResult(
        score=None,
        bilabial_events_checked=0,
        violations=0,
        violation_rate=None,
        lag_ms=None,
        anomalies=(),
        usable=False,
        explanation="Speech not detected.",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)
    assert result.verdict != "MANIPULATED"
    assert result.verdict == "INCONCLUSIVE"


def test_manipulated_requires_clean_window_fraction_above_threshold() -> None:
    """When clean_window_fraction < 0.5 due to artifacts, verdict must not be MANIPULATED."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=8.0, explanation="Quality passed.")
    time_s = tuple(float(t) for t in range(21))
    # 2 windows, both artifacts -> clean_window_fraction = 0.0
    rppg = RPPGResult(
        score=0.10,
        uncertainty=None,
        explanation="Artifact corrupted signal",
        usable=True,
        mean_r=0.10,
        traces=RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s),
        windowed_r=(
            CoherenceWindow(0.0, 10.0, 0.10, 0.10, 0.10, False, is_artifact=True),
            CoherenceWindow(10.0, 20.0, 0.10, 0.10, 0.10, False, is_artifact=True),
        ),
        clean_window_fraction=0.0,
    )
    phonetics = PhoneticResult(
        score=0.0,
        bilabial_events_checked=8,
        violations=8,
        violation_rate=1.0,
        lag_ms=0.0,
        anomalies=(),
        usable=True,
        explanation="Violations detected.",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)
    assert result.verdict == "INCONCLUSIVE"
    assert "Clean window fraction" in result.explain


def test_explanation_does_not_contradict_metrics() -> None:
    """Explanation must not claim desynchronized when r is above threshold, or no anomalies when anomalies exist."""
    cfg = _test_config()
    quality = QualityReport(passed=True, face_fraction=0.95, snr_db=10.0, explanation="Quality passed.")
    time_s = tuple(float(t) for t in range(21))
    rppg = RPPGResult(
        score=0.55,
        uncertainty=None,
        explanation="Synchronized",
        usable=True,
        mean_r=0.42,  # Well above corr_fake (0.10)
        traces=RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s),
        windowed_r=(CoherenceWindow(0.0, 10.0, 0.42, 0.42, 0.42, False),),
    )
    anom = Anomaly(start_s=5.0, end_s=6.0, kind="lip_closure_missing", severity=0.7, explanation="open", evidence=())
    phonetics = PhoneticResult(
        score=0.5,
        bilabial_events_checked=6,
        violations=1,
        violation_rate=1 / 6,
        lag_ms=0.0,
        anomalies=(anom,),
        usable=True,
        explanation="1 violation",
    )

    result = arbitrate(ArbiterInput(rppg, phonetics, quality), config=cfg)
    # 1 anomaly exists -> explanation must NOT say 'No physical desynchronization or phonetic timing anomalies'
    assert "No physical desynchronization or phonetic timing anomalies" not in result.explain
    assert "1 distinct temporal anomaly" in result.explain
    # r = 0.42 > 0.10 -> must NOT say 'desynchronized'
    assert "desynchronized" not in result.explain
