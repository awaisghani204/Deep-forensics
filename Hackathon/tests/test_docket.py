"""Synthetic unit tests for the DeepTrace court-ready PDF forensic docket generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

import pytest

from engine.docket import DocketResult, build_docket, generate_docket
from engine.types import (
    Anomaly,
    ArbiterResult,
    CoherenceWindow,
    PhoneticResult,
    QualityReport,
    RPPGResult,
    RPPGTraces,
    VideoMeta,
)


def _count_pdf_pages(pdf_path: Path) -> int:
    """Count page objects in a PDF by parsing /Type /Page tags."""
    content = pdf_path.read_bytes()
    pages = re.findall(rb"/Type\s*/Page\b", content)
    return len(pages)


def test_generate_docket_returns_valid_2_page_pdf() -> None:
    """Docket generation produces an uncorrupted, exactly 2-page PDF and matching SHA-256."""
    finding = ArbiterResult(
        fai=0.88,
        verdict="AUTHENTIC",
        uncertainty=0.12,
        confidence="HIGH",
        explain="All physiological and phonetic invariants were verified to be consistent with a genuine recording.",
        weights_used={"hemodynamic": 0.55, "phonetic": 0.35, "spectral_snr": 0.10},
        modules_contributed=("hemodynamic", "phonetic", "spectral_snr"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = generate_docket(None, finding, out_dir=tmp_dir, analyst_name="Forensic Investigator #402")
        pdf_path, pdf_sha = res

        assert pdf_path.is_file()
        assert pdf_path.stat().st_size > 5000
        assert len(pdf_sha) == 64

        # Verify cryptographic hash matches file on disk
        hasher = hashlib.sha256()
        with pdf_path.open("rb") as f:
            hasher.update(f.read())
        assert pdf_sha == hasher.hexdigest()

        # Verify page count is EXACTLY 2 pages
        assert _count_pdf_pages(pdf_path) == 2

        # Verify sidecar JSON is created and matches
        assert res.sidecar_json_path is not None
        assert res.sidecar_json_path.is_file()
        data = json.loads(res.sidecar_json_path.read_text(encoding="utf-8"))
        assert data["report_integrity"]["docket_sha256"] == pdf_sha
        assert data["report_integrity"]["analyst"] == "Forensic Investigator #402"
        assert data["verdict_assessment"]["verdict"] == "AUTHENTIC"


def test_generate_docket_with_inconclusive_results() -> None:
    """Inconclusive findings must render a clean 2-page report showing gating reasons without errors."""
    finding = ArbiterResult(
        fai=None,
        verdict="INCONCLUSIVE",
        uncertainty=1.0,
        confidence="LOW",
        explain="Quality gate failed: Face fraction was below minimum requirement (0.42 < 0.70).",
        weights_used={},
        modules_contributed=(),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = generate_docket(None, finding, out_dir=tmp_dir)
        pdf_path, pdf_sha = res

        assert pdf_path.is_file()
        assert len(pdf_sha) == 64
        # Verify page count is EXACTLY 2 pages
        assert _count_pdf_pages(pdf_path) == 2

        sidecar = json.loads(res.sidecar_json_path.read_text(encoding="utf-8"))
        assert sidecar["verdict_assessment"]["verdict"] == "INCONCLUSIVE"
        assert sidecar["verdict_assessment"]["fai"] is None


def test_generate_docket_with_full_signals_and_anomalies() -> None:
    """Full multimodal evidence (rPPG traces, phonetic violations) generates a 2-page report with plots and table."""
    time_s = tuple(float(t) for t in range(21))
    traces = RPPGTraces(time_s=time_s, forehead=time_s, left_cheek=time_s, right_cheek=time_s)
    incoh_win = CoherenceWindow(3.0, 9.0, 0.12, 0.08, 0.15, True)
    rppg = RPPGResult(
        score=0.18,
        uncertainty=None,
        explanation="Low spatial coherence",
        usable=True,
        mean_r=0.18,
        traces=traces,
        windowed_r=(incoh_win,),
        incoherent_windows=(incoh_win,),
    )
    phonetics = PhoneticResult(
        score=0.10,
        bilabial_events_checked=6,
        violations=5,
        violation_rate=0.83,
        lag_ms=15.0,
        anomalies=(
            Anomaly(start_s=4.0, end_s=4.5, kind="lip_closure_missing", severity=0.85, explanation="Open lips during 'peter'"),
        ),
        usable=True,
        explanation="5 violations observed",
    )
    finding = ArbiterResult(
        fai=0.15,
        verdict="MANIPULATED",
        uncertainty=0.15,
        confidence="HIGH",
        explain="Severe multi-site desynchronization and bilabial lip violations detected.",
        weights_used={"hemodynamic": 0.55, "phonetic": 0.35, "spectral_snr": 0.10},
        modules_contributed=("hemodynamic", "phonetic", "spectral_snr"),
        anomalies=(
            Anomaly(start_s=3.0, end_s=9.0, kind="rppg_spatial_desync+lip_closure_missing", severity=0.85, explanation="Simultaneous rPPG and phonetic violation"),
        ),
    )

    results_dict = {
        "rppg": rppg,
        "phonetics": phonetics,
        "arbiter": finding,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = generate_docket(None, results_dict, out_dir=tmp_dir)
        pdf_path, pdf_sha = res

        assert pdf_path.is_file()
        assert _count_pdf_pages(pdf_path) == 2


def test_build_docket_backwards_compatible() -> None:
    """Legacy build_docket API returns DocketResult with report_path and sha256."""
    meta = VideoMeta(
        source_path=Path("sample.mp4"),
        sha256="abc123def456",
        duration_s=15.0,
        frame_count=450,
        fps=30.0,
    )
    finding = ArbiterResult(
        fai=0.75,
        verdict="AUTHENTIC",
        uncertainty=0.15,
        confidence="HIGH",
        explain="Evidence verified.",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_pdf = Path(tmp_dir) / "legacy_report.pdf"
        res = build_docket(meta, finding, output_path=out_pdf)

        assert isinstance(res, DocketResult)
        assert res.report_path.is_file()
        assert len(res.sha256) == 64
        assert _count_pdf_pages(res.report_path) == 2
