"""Synthetic unit tests for DeepTrace evaluation runners, plots, and compression utilities."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import numpy as np
import pytest

from engine.arbiter import ArbiterConfig, ArbiterInput, arbitrate
from engine.types import (
    CoherenceWindow,
    PhoneticResult,
    QualityReport,
    RPPGResult,
    RPPGTraces,
)
from eval.compress import COMPRESSION_PRESETS, compress_for_evaluation, compress_video
from eval.plots import (
    build_evaluation_plot,
    generate_all_plots,
    plot_compression_robustness,
    plot_confusion_matrix,
    plot_per_fake_type_breakdown,
    plot_r_distribution,
    plot_roc_curve,
)
from eval.run_eval import (
    ItemEvaluation,
    calibrate_thresholds,
    compute_held_out_metrics,
    generate_summary_markdown,
    run_evaluation,
    save_results_csv,
    validate_pulse_ground_truth,
)


def _make_mock_item(
    path: Path,
    label: str,
    fake_type: str = "none",
    mean_r: float = 0.80,
    violations: int = 0,
    snr: float = 12.0,
    bpm_diff: float = 1.0,
    pred_bpm: float = 72.0,
) -> ItemEvaluation:
    """Create a fully populated deterministic ItemEvaluation with synthetic evidence."""
    time_s = tuple(float(t) for t in range(16))
    traces = RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s)
    win = CoherenceWindow(0.0, 10.0, mean_r, mean_r, mean_r, mean_r < 0.35)

    rppg = RPPGResult(
        score=mean_r,
        uncertainty=0.1,
        explanation="Synthetic rPPG test evidence",
        usable=True,
        mean_r=mean_r,
        traces=traces,
        windowed_r=(win,),
        snr_db_per_roi=(snr, snr, snr),
        bpm_agreement_bpm=bpm_diff,
        bpm_per_roi=(pred_bpm, pred_bpm + bpm_diff, pred_bpm - bpm_diff),
        incoherent_windows=(win,) if mean_r < 0.35 else (),
    )
    phonetics = PhoneticResult(
        score=max(0.0, 1.0 - (violations / 6.0)),
        bilabial_events_checked=6,
        violations=violations,
        violation_rate=violations / 6.0,
        lag_ms=10.0,
        anomalies=(),
        usable=True,
        explanation=f"{violations}/6 violations",
    )
    quality = QualityReport(passed=True, face_fraction=0.98, snr_db=snr, explanation="Quality passed.")
    arb_in = ArbiterInput(rppg=rppg, phonetics=phonetics, quality=quality)
    arb_res = arbitrate(arb_in)

    return ItemEvaluation(
        file_path=path,
        label=label,
        fake_type=fake_type,
        duration=15.0,
        mean_r=mean_r,
        bpm_agreement=bpm_diff,
        snr=snr,
        phonetic_score=arb_res.phonetic_score,
        violations=violations,
        fai=arb_res.fai,
        verdict=arb_res.verdict,
        confidence=arb_res.confidence,
        runtime=0.05,
        explanation=arb_res.explanation,
        passed_quality=True,
        arbiter_input=arb_in,
    )


def test_build_evaluation_plot_synthetic() -> None:
    """Evaluation plot must be generated from synthetic FAI lists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_path = Path(tmp_dir) / "fai_plot.png"
        res = build_evaluation_plot(
            authentic_fai=[0.82, 0.78, 0.91, 0.88],
            manipulated_fai=[0.12, 0.25, 0.18, 0.30],
            output_path=plot_path,
        )

        assert res.output_path is not None
        assert res.output_path.is_file()
        assert res.series_count == 8
        assert res.output_path.stat().st_size > 500


def test_generate_all_plots_synthetic() -> None:
    """generate_all_plots must render all 5 presentation-ready dark theme charts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        metrics = {
            "y_true": [0, 0, 1, 1],
            "y_scores": [0.15, 0.20, 0.85, 0.90],
            "real_r": [0.75, 0.82, 0.79],
            "fake_r": [0.18, 0.22, 0.30],
            "corr_real": 0.60,
            "corr_fake": 0.35,
            "confusion_matrix": [[10, 0, 0, 0], [0, 1, 9, 0]],
            "compression_robustness": {
                "Raw": {"accuracy": 0.95, "inconclusive_rate": 0.02},
                "crf18": {"accuracy": 0.94, "inconclusive_rate": 0.03},
                "crf28": {"accuracy": 0.88, "inconclusive_rate": 0.08},
                "whatsapp": {"accuracy": 0.75, "inconclusive_rate": 0.22},
            },
            "fake_type_breakdown": {
                "faceswap": {"accuracy": 0.92, "inconclusive_rate": 0.05},
                "lipsync": {"accuracy": 0.89, "inconclusive_rate": 0.08},
                "diffusion": {"accuracy": 0.80, "inconclusive_rate": 0.15},
            },
        }

        results = generate_all_plots(metrics, out_dir)
        expected_plots = [
            "roc_curve",
            "r_distribution",
            "confusion_matrix",
            "compression_robustness",
            "per_fake_type_breakdown",
        ]
        for key in expected_plots:
            assert key in results
            assert results[key].output_path is not None
            assert results[key].output_path.is_file()
            assert results[key].output_path.stat().st_size > 1000


def test_run_evaluation_empty_directory() -> None:
    """Evaluation on an empty directory gracefully returns zero evaluated items without generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = run_evaluation(Path(tmp_dir), generate_if_empty=False)
        assert res.total_evaluated == 0
        assert res.accuracy is None
        assert "No MP4 video files found" in res.explanation


def test_compress_presets_definitions() -> None:
    """Compression module must define required CRF presets and WhatsApp social preset."""
    assert "crf18" in COMPRESSION_PRESETS
    assert "crf23" in COMPRESSION_PRESETS
    assert "crf28" in COMPRESSION_PRESETS
    assert "crf35" in COMPRESSION_PRESETS
    assert "whatsapp" in COMPRESSION_PRESETS
    assert COMPRESSION_PRESETS["crf18"]["crf"] == 18
    assert COMPRESSION_PRESETS["crf35"]["crf"] == 35


def test_compress_missing_file_handling() -> None:
    """Compression runner handles missing source video gracefully."""
    fake_source = Path("does_not_exist_xyz.mp4")
    res = compress_for_evaluation(fake_source)
    assert not res.success
    assert res.output_path is None or not res.output_path.is_file()


def test_calibration_and_held_out_metrics() -> None:
    """Threshold calibration operates on calibration split; metrics computed strictly on held-out split."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # 4 Calibration items
        c_real1 = _make_mock_item(tmp_p / "c_r1.mp4", "AUTHENTIC", "none", mean_r=0.82, violations=0)
        c_real2 = _make_mock_item(tmp_p / "c_r2.mp4", "AUTHENTIC", "none", mean_r=0.78, violations=0)
        c_fake1 = _make_mock_item(tmp_p / "c_f1.mp4", "MANIPULATED", "faceswap", mean_r=0.20, violations=4)
        c_fake2 = _make_mock_item(tmp_p / "c_f2.mp4", "MANIPULATED", "lipsync", mean_r=0.58, violations=5)
        calib_items = [c_real1, c_real2, c_fake1, c_fake2]

        cfg_file = tmp_p / "test_config.yaml"
        # Seed a dummy config
        cfg_file.write_text(
            "rppg:\n  corr_real: 0.60\n  corr_fake: 0.35\n"
            "arbiter:\n  verdict_cutoffs:\n    authentic: 0.70\n    suspect: 0.40\n    manipulated: 0.40\n",
            encoding="utf-8",
        )

        params, best_j = calibrate_thresholds(calib_items, config_path=cfg_file)
        assert "corr_real" in params
        assert "corr_fake" in params
        assert params["corr_real"] > params["corr_fake"]
        assert best_j > 0.0

        # 4 Held-Out items
        h_real1 = _make_mock_item(tmp_p / "h_r1.mp4", "AUTHENTIC", "none", mean_r=0.85, violations=0)
        h_real2 = _make_mock_item(tmp_p / "h_r2.mp4", "AUTHENTIC", "none", mean_r=0.75, violations=0)
        h_fake1 = _make_mock_item(tmp_p / "h_f1.mp4", "MANIPULATED", "faceswap", mean_r=0.18, violations=5)
        h_fake2 = _make_mock_item(tmp_p / "h_f2.mp4", "MANIPULATED", "diffusion", mean_r=0.22, violations=6)
        held_out_items = [h_real1, h_real2, h_fake1, h_fake2]

        test_cfg = ArbiterConfig(
            corr_real=params["corr_real"],
            corr_fake=params["corr_fake"],
            cutoff_authentic=params["cutoff_authentic"],
            cutoff_suspect=params["cutoff_manipulated"],
            cutoff_manipulated=params["cutoff_manipulated"],
        )
        held_out = compute_held_out_metrics(held_out_items, config=test_cfg)

        assert held_out.total_evaluated == 4
        assert held_out.conclusive_count == 4
        assert held_out.inconclusive_rate == 0.0
        assert held_out.accuracy_conclusive == 1.0
        assert held_out.accuracy_with_abstain == 1.0
        assert held_out.balanced_accuracy == 1.0
        assert held_out.precision == 1.0
        assert held_out.recall == 1.0
        assert held_out.false_positive_rate == 0.0
        assert held_out.false_negative_rate == 0.0
        assert len(held_out.confusion_matrix_2x4) == 2
        assert len(held_out.confusion_matrix_2x4[0]) == 4


def test_save_results_csv_columns() -> None:
    """Saved results.csv must include all 12 specified columns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "results.csv"
        item = _make_mock_item(Path("sample.mp4"), "AUTHENTIC", "none", mean_r=0.82)
        save_results_csv([item], csv_path)

        assert csv_path.is_file()
        with csv_path.open("r", encoding="utf-8") as h:
            reader = csv.reader(h)
            header = next(reader)

        expected_columns = [
            "label",
            "fake_type",
            "duration",
            "mean_r",
            "bpm agreement",
            "snr",
            "phonetic score",
            "violations",
            "FAI",
            "verdict",
            "confidence",
            "runtime",
        ]
        for col in expected_columns:
            assert col in header


def test_summary_markdown_generation() -> None:
    """generate_summary_markdown must produce formatted tables and failure cases."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        md_path = Path(tmp_dir) / "summary.md"
        item = _make_mock_item(Path("clip1.mp4"), "AUTHENTIC", "none", mean_r=0.80)
        held_out = compute_held_out_metrics([item])

        generate_summary_markdown(
            held_out=held_out,
            calib_params={"corr_real": 0.60, "corr_fake": 0.35},
            calib_j=0.95,
            compression_metrics={"Raw": {"accuracy": 1.0, "inconclusive_rate": 0.0, "description": "Raw"}},
            pulse_mae=1.5,
            pulse_samples=1,
            output_path=md_path,
            seed=42,
        )

        assert md_path.is_file()
        content = md_path.read_text(encoding="utf-8")
        assert "# DeepTrace Forensic Evaluation & Benchmark Report" in content
        assert "Calibration Protocol" in content
        assert "Conclusive Accuracy" in content
        assert "Full Accuracy (with Abstain)" in content
        assert "Confusion Matrix" in content
        assert "Video Compression Degradation Robustness" in content
        assert "1.50 BPM" in content
        assert "Failure Cases Analysis" in content


def test_pulse_ground_truth_validation() -> None:
    """validate_pulse_ground_truth computes accurate MAE against ground_truth.csv."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = Path(tmp_dir)
        gt_csv = gt_dir / "ground_truth.csv"
        gt_csv.write_text("filename,bpm\nclip_a.mp4,70.0\nclip_b.mp4,80.0\n", encoding="utf-8")

        item_a = _make_mock_item(Path("clip_a.mp4"), "AUTHENTIC", pred_bpm=72.0)
        item_b = _make_mock_item(Path("clip_b.mp4"), "AUTHENTIC", pred_bpm=78.0)

        mae, count = validate_pulse_ground_truth([item_a, item_b], gt_dir)
        assert count == 2
        # Error for clip_a: |72 - 70| = 2.0; Error for clip_b: |78 - 80| = 2.0; MAE = 2.0
        assert mae == pytest.approx(2.0, rel=1e-3)
