"""Uncertainty-aware evidence-fusion and quality gating for DeepTrace.

Fuses:
1. Multi-site rPPG spatial coherence & BPM agreement (Hemodynamic invariant)
2. Audio-guided bilabial closure during P/B/M phonemes (Phonetic invariant)
3. Normalized spectral SNR (Confidence modifier)

Outputs:
- Forensic Authenticity Index (FAI)
- Deterministic Verdict (AUTHENTIC / SUSPECT / MANIPULATED / INCONCLUSIVE)
- Confidence Level (LOW / MEDIUM / HIGH)
- Chronologically sorted & merged anomaly timeline
- Explainable non-expert finding summary
- JSON-serializable ArbiterResult via to_dict()
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from engine.types import Anomaly, ArbiterResult, PhoneticResult, QualityReport, RPPGResult, Verdict

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbiterConfig:
    """All calibrated fusion weights, cutoffs, and bounds from config.yaml."""

    weight_hemodynamic: float = 0.55
    weight_phonetic: float = 0.35
    weight_spectral_snr: float = 0.10
    cutoff_authentic: float = 0.70
    cutoff_suspect: float = 0.40
    cutoff_manipulated: float = 0.40
    corr_real: float = 0.60
    corr_fake: float = 0.35
    max_bpm_diff: float = 10.0
    snr_min_db: float = 0.0
    snr_max_db: float = 15.0
    shrinkage_prior_events: float = 5.0
    min_duration_high_s: float = 15.0
    min_duration_med_s: float = 8.0
    min_events_high: int = 6
    min_events_med: int = 3
    manipulated_min_confidence: str = "MEDIUM"
    manipulated_min_clean_window_fraction: float = 0.5
    min_duration_rppg_alone_s: float = 15.0


@dataclass(frozen=True)
class ArbiterInput:
    """Evidence passed to the calibrated fusion arbiter."""

    rppg: RPPGResult
    phonetics: PhoneticResult
    quality: QualityReport


def load_arbiter_config(config_path: str | Path | None = None) -> ArbiterConfig:
    """Load fusion weights, verdict cutoffs, SNR bounds, and shrinkage parameters from config.yaml."""
    path = Path(config_path) if config_path is not None else Path(__file__).parents[1] / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    arb = cfg.get("arbiter", {})
    rppg_cfg = cfg.get("rppg", {})
    weights = arb.get("weights", {})
    cutoffs = arb.get("verdict_cutoffs", {})
    snr_bounds = arb.get("snr_bounds", {})
    bpm_aggr = arb.get("bpm_agreement", {})
    shrink = arb.get("shrinkage", {})
    conf = arb.get("confidence", {})
    rules = arb.get("rules", {})

    w_hemo = float(weights.get("hemodynamic", weights.get("rppg", 0.55)))
    w_phon = float(weights.get("phonetic", weights.get("phonetics", 0.35)))
    w_snr = float(weights.get("spectral_snr", weights.get("snr", 0.10)))

    corr_real = float(rppg_cfg.get("corr_real", 0.60))
    corr_fake = float(rppg_cfg.get("corr_fake", 0.35))

    return ArbiterConfig(
        weight_hemodynamic=w_hemo,
        weight_phonetic=w_phon,
        weight_spectral_snr=w_snr,
        cutoff_authentic=float(cutoffs.get("authentic", 0.70)),
        cutoff_suspect=float(cutoffs.get("suspect", 0.40)),
        cutoff_manipulated=float(cutoffs.get("manipulated", 0.40)),
        corr_real=corr_real,
        corr_fake=corr_fake,
        max_bpm_diff=float(bpm_aggr.get("max_bpm_diff", 10.0)),
        snr_min_db=float(snr_bounds.get("min_db", 0.0)),
        snr_max_db=float(snr_bounds.get("max_db", 15.0)),
        shrinkage_prior_events=float(shrink.get("prior_events", 5.0)),
        min_duration_high_s=float(conf.get("min_duration_high_s", 15.0)),
        min_duration_med_s=float(conf.get("min_duration_med_s", 8.0)),
        min_events_high=int(conf.get("min_events_high", 6)),
        min_events_med=int(conf.get("min_events_med", 3)),
        manipulated_min_confidence=str(rules.get("manipulated_min_confidence", "MEDIUM")),
        manipulated_min_clean_window_fraction=float(rules.get("manipulated_min_clean_window_fraction", 0.5)),
        min_duration_rppg_alone_s=float(rules.get("min_duration_rppg_alone_s", 15.0)),
    )


def compute_rppg_score(rppg: RPPGResult, config: ArbiterConfig) -> float | None:
    """Map mean windowed r using corr_real and corr_fake thresholds, combined with BPM agreement."""
    # 1. Determine mean correlation from windowed r if available, else mean_r / score
    corr_samples: list[float] = []
    if rppg.windowed_r:
        for win in rppg.windowed_r:
            for pair_r in (win.forehead_left_r, win.forehead_right_r, win.left_right_r):
                if pair_r is not None and np.isfinite(pair_r):
                    corr_samples.append(float(pair_r))

    mean_corr: float | None = float(np.mean(corr_samples)) if corr_samples else rppg.mean_r
    if mean_corr is None:
        mean_corr = rppg.score

    if mean_corr is None or not np.isfinite(mean_corr):
        return None

    # Piecewise-linear mapping to [0, 1]
    cf = config.corr_fake
    cr = config.corr_real
    if mean_corr <= cf:
        linear_score = 0.0
    elif mean_corr >= cr:
        linear_score = 1.0
    else:
        linear_score = float((mean_corr - cf) / max(cr - cf, 1e-6))

    # Combine with BPM agreement across ROIs if available
    if rppg.bpm_agreement_bpm is not None and np.isfinite(rppg.bpm_agreement_bpm):
        bpm_factor = float(np.clip(1.0 - (rppg.bpm_agreement_bpm / max(config.max_bpm_diff * 1.5, 1e-6)), 0.0, 1.0))
        # 50% linear correlation mapping, 50% pulse frequency agreement
        score = 0.50 * linear_score + 0.50 * bpm_factor
    else:
        score = linear_score

    return float(np.clip(score, 0.0, 1.0))


def compute_phonetic_score(phonetics: PhoneticResult, config: ArbiterConfig) -> float | None:
    """Calculate 1 - violation_rate, shrunk toward 0.5 when few events were checked."""
    events_checked = phonetics.bilabial_events_checked
    if events_checked == 0:
        return 0.5

    if phonetics.violation_rate is not None and np.isfinite(phonetics.violation_rate):
        raw_violation_rate = phonetics.violation_rate
    else:
        raw_violation_rate = phonetics.violations / max(events_checked, 1)

    raw_score = 1.0 - raw_violation_rate

    # Shrink toward 0.5 with prior events weight
    m = config.shrinkage_prior_events
    shrunk_score = (events_checked * raw_score + m * 0.5) / (events_checked + m)
    return float(np.clip(shrunk_score, 0.0, 1.0))


def compute_spectral_snr_score(
    rppg: RPPGResult, quality: QualityReport, config: ArbiterConfig
) -> float | None:
    """Normalize mean rPPG spectral SNR (dB) across ROIs to [0, 1]."""
    snrs: list[float] = []
    if rppg.snr_db_per_roi:
        for val in rppg.snr_db_per_roi:
            if val is not None and np.isfinite(val):
                snrs.append(float(val))

    mean_snr: float | None = float(np.mean(snrs)) if snrs else quality.snr_db
    if mean_snr is None or not np.isfinite(mean_snr):
        return None

    denom = max(config.snr_max_db - config.snr_min_db, 1e-6)
    score = (mean_snr - config.snr_min_db) / denom
    return float(np.clip(score, 0.0, 1.0))


def bootstrap_ci_mean_r(rppg: RPPGResult, n_bootstrap: int = 200) -> tuple[float, float] | None:
    """Optionally provide a 95% bootstrap confidence interval for windowed mean r."""
    samples: list[float] = []
    if rppg.windowed_r:
        for win in rppg.windowed_r:
            for pair_r in (win.forehead_left_r, win.forehead_right_r, win.left_right_r):
                if pair_r is not None and np.isfinite(pair_r):
                    samples.append(float(pair_r))

    if len(samples) < 3:
        return None

    rng = np.random.default_rng(42)
    sample_arr = np.asarray(samples, dtype=np.float64)
    boot_means = [
        float(np.mean(rng.choice(sample_arr, size=len(sample_arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def merge_anomalies(anomalies: Sequence[Anomaly]) -> tuple[Anomaly, ...]:
    """Sort and merge overlapping anomaly intervals across rPPG and phonetics into a unified timeline."""
    if not anomalies:
        return ()

    # Sort primarily by start_s, secondarily by end_s
    sorted_anoms = sorted(anomalies, key=lambda a: (a.start_s, a.end_s))
    merged: list[Anomaly] = []
    current = sorted_anoms[0]

    for nxt in sorted_anoms[1:]:
        if nxt.start_s <= current.end_s:
            # Overlapping or contiguous interval -> merge
            new_start = current.start_s
            new_end = max(current.end_s, nxt.end_s)
            new_sev = max(current.severity, nxt.severity)
            if current.kind == nxt.kind:
                new_kind = current.kind
            else:
                new_kind = f"{current.kind}+{nxt.kind}"
            new_explanation = f"{current.explanation} Simultaneously, {nxt.explanation}"
            new_evidence = current.evidence + nxt.evidence
            current = Anomaly(
                start_s=float(new_start),
                end_s=float(new_end),
                kind=new_kind,
                severity=float(new_sev),
                explanation=new_explanation,
                evidence=new_evidence,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return tuple(merged)


def _collect_raw_anomalies(evidence: ArbiterInput) -> list[Anomaly]:
    """Gather raw anomaly records from both phonetic closures and rPPG desynchronizations."""
    anoms: list[Anomaly] = list(evidence.phonetics.anomalies)

    for win in evidence.rppg.incoherent_windows:
        r_vals = [r for r in (win.forehead_left_r, win.forehead_right_r, win.left_right_r) if r is not None]
        mean_win_r = float(np.mean(r_vals)) if r_vals else 0.0
        anoms.append(
            Anomaly(
                start_s=win.start_s,
                end_s=win.end_s,
                kind="rppg_spatial_desync",
                severity=float(np.clip(1.0 - max(0.0, mean_win_r), 0.0, 1.0)),
                explanation=(
                    f"Facial ROI pulse traces desynchronized in window [{win.start_s:.1f}s, {win.end_s:.1f}s] "
                    f"(mean r: {mean_win_r:.2f})."
                ),
                evidence=(
                    ("forehead_left_r", f"{win.forehead_left_r:.3f}" if win.forehead_left_r is not None else "None"),
                    ("forehead_right_r", f"{win.forehead_right_r:.3f}" if win.forehead_right_r is not None else "None"),
                    ("left_right_r", f"{win.left_right_r:.3f}" if win.left_right_r is not None else "None"),
                ),
            )
        )
    return anoms


def _determine_confidence(
    quality_passed: bool,
    hemo_usable: bool,
    phon_usable: bool,
    duration_s: float,
    events_checked: int,
    snr_score: float | None,
    config: ArbiterConfig,
) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """Compute confidence level (LOW / MEDIUM / HIGH) from quality, clip duration, SNR, and event counts."""
    if not quality_passed:
        return "LOW"

    snr_ok = (snr_score is not None and snr_score >= 0.30)

    # Audio missing -> weights renormalized, produced with LOW or MEDIUM confidence
    if hemo_usable and not phon_usable:
        if duration_s >= config.min_duration_high_s and snr_ok:
            return "MEDIUM"
        return "LOW"

    # Both modules usable
    if hemo_usable and phon_usable:
        if (
            duration_s >= config.min_duration_high_s
            and events_checked >= config.min_events_high
            and snr_ok
        ):
            return "HIGH"
        if (
            duration_s >= config.min_duration_med_s
            and events_checked >= config.min_events_med
        ):
            return "MEDIUM"
        return "LOW"

    # Phonetics only or insufficient evidence
    if phon_usable and events_checked >= config.min_events_high:
        return "MEDIUM"
    return "LOW"


def _build_explain_string(
    verdict: Verdict,
    fai: float | None,
    confidence: str,
    rppg_score: float | None,
    phonetic_score: float | None,
    duration_s: float,
    events_checked: int,
    anomaly_count: int,
    reasons: Sequence[str],
    mean_r: float | None = None,
    corr_fake_threshold: float = 0.10,
) -> str:
    """Generate articulate sentences strictly faithful to the underlying result metrics."""
    dur_int = max(1, int(round(duration_s)))
    if verdict == "INCONCLUSIVE":
        s1 = "The automated forensic analysis concluded with an INCONCLUSIVE finding due to degraded input quality or insufficient signal data."
        s2 = f"Primary limitation: {'; '.join(reasons) if reasons else 'signals did not satisfy calibrated admissibility thresholds'}."
        s3 = f"The confidence in this outcome is {confidence} over a recording duration of {dur_int} second(s) with {events_checked} phonetic checkpoints."
        if anomaly_count > 0:
            s4 = f"Note: {anomaly_count} potential anomaly interval(s) were flagged but could not be definitively attributed due to inconclusive evidentiary quality."
        else:
            s4 = "No verified anomalies were established across the inspected intervals."
        s5 = "Further review with higher resolution footage or clearer speech audio is recommended."
        return f"{s1} {s2} {s3} {s4} {s5}"

    fai_str = f"{fai:.2f}" if fai is not None else "N/A"
    s1 = f"DeepTrace concluded with an official verdict of {verdict}, yielding a Forensic Authenticity Index (FAI) of {fai_str}."

    # Invariant A: Hemodynamic description
    if rppg_score is not None:
        pct = int(round(rppg_score * 100))
        if mean_r is not None and mean_r >= corr_fake_threshold:
            h_desc = f"consistent with genuine biological blood circulation ({pct}% coherence, mean r = {mean_r:+.2f})"
        elif rppg_score >= 0.50:
            h_desc = f"consistent with genuine biological blood circulation ({pct}% coherence)"
        else:
            h_desc = f"inconsistent or desynchronized across face regions ({pct}% coherence" + (f", mean r = {mean_r:+.2f})" if mean_r is not None else ")")
    else:
        h_desc = "not evaluable due to signal quality gating"

    # Invariant B: Phonetic description
    if phonetic_score is not None:
        p_pct = int(round(phonetic_score * 100))
        if phonetic_score >= 0.60:
            p_desc = f"consistent with natural lip closures during spoken words ({p_pct}% compliance)"
        else:
            p_desc = f"flagged with unnatural closure violations during speech ({p_pct}% compliance)"
    else:
        p_desc = "unavailable in the audio track"

    s2 = f"Facial pulse synchronization across the forehead and cheeks was {h_desc}, while bilabial speech articulation was {p_desc}."
    s3 = f"This finding is reported with {confidence} confidence based on a {dur_int}-second recording and {events_checked} phonetic checkpoints."

    if anomaly_count == 0:
        s4 = "No physical desynchronization or phonetic timing anomalies were observed across the verified intervals."
    else:
        s4 = f"A total of {anomaly_count} distinct temporal anomaly interval(s) were flagged in the evidence timeline."

    s5 = "All measurements reflect physical invariant verification in a court-ready format without claiming absolute certainty."
    return f"{s1} {s2} {s3} {s4} {s5}"


def arbitrate(evidence: ArbiterInput, config: ArbiterConfig | None = None) -> ArbiterResult:
    """Fuse rPPG spatial coherence, bilabial closure, and spectral SNR into a final verdict.

    Renormalizes weights dynamically when modules are unusable, merges overlapping
    anomaly intervals, and produces an explainable, JSON-serializable ArbiterResult.
    """
    settings = config or load_arbiter_config()
    raw_anomalies = _collect_raw_anomalies(evidence)
    anomalies = merge_anomalies(raw_anomalies)

    # Clip duration
    duration_s = 0.0
    if evidence.rppg.traces and evidence.rppg.traces.time_s:
        duration_s = float(evidence.rppg.traces.time_s[-1] - evidence.rppg.traces.time_s[0])
    elif evidence.rppg.windowed_r:
        duration_s = float(max(w.end_s for w in evidence.rppg.windowed_r))

    events_checked = evidence.phonetics.bilabial_events_checked if evidence.phonetics.usable else 0

    # 1. Quality Gate Check
    if not evidence.quality.passed:
        reasons = [f"Quality gate failed ({evidence.quality.explanation})"]
        explain_str = _build_explain_string(
            verdict="INCONCLUSIVE",
            fai=None,
            confidence="LOW",
            rppg_score=None,
            phonetic_score=None,
            duration_s=duration_s,
            events_checked=events_checked,
            anomaly_count=len(anomalies),
            reasons=reasons,
            mean_r=evidence.rppg.mean_r,
            corr_fake_threshold=settings.corr_fake,
        )
        return ArbiterResult(
            fai=None,
            verdict="INCONCLUSIVE",
            uncertainty=1.0,
            confidence="LOW",
            anomalies=anomalies,
            explanation=explain_str,
            explain=explain_str,
            weights_used={},
            modules_contributed=(),
            rppg_score=None,
            phonetic_score=None,
            spectral_snr_score=None,
            bootstrap_ci_mean_r=None,
        )

    # 2. Compute individual module scores in [0, 1]
    hemo_usable = evidence.rppg.usable and (
        evidence.rppg.score is not None or evidence.rppg.mean_r is not None or bool(evidence.rppg.windowed_r)
    )
    phon_usable = evidence.phonetics.usable and (
        evidence.phonetics.score is not None or evidence.phonetics.violation_rate is not None
    )

    rppg_score = compute_rppg_score(evidence.rppg, settings) if hemo_usable else None
    phonetic_score = compute_phonetic_score(evidence.phonetics, settings) if phon_usable else None
    snr_score = compute_spectral_snr_score(evidence.rppg, evidence.quality, settings) if hemo_usable else None

    # Check if total usable evidence is too weak
    if rppg_score is None and phonetic_score is None:
        reasons = [
            f"rPPG unusable ({evidence.rppg.explanation})",
            f"phonetics unusable ({evidence.phonetics.explanation})",
        ]
        explain_str = _build_explain_string(
            verdict="INCONCLUSIVE",
            fai=None,
            confidence="LOW",
            rppg_score=None,
            phonetic_score=None,
            duration_s=duration_s,
            events_checked=events_checked,
            anomaly_count=len(anomalies),
            reasons=reasons,
            mean_r=evidence.rppg.mean_r,
            corr_fake_threshold=settings.corr_fake,
        )
        return ArbiterResult(
            fai=None,
            verdict="INCONCLUSIVE",
            uncertainty=1.0,
            confidence="LOW",
            anomalies=anomalies,
            explanation=explain_str,
            explain=explain_str,
            weights_used={},
            modules_contributed=(),
            rppg_score=None,
            phonetic_score=None,
            spectral_snr_score=None,
            bootstrap_ci_mean_r=None,
        )

    # 3. Dynamic Weight Renormalization
    active_candidates: list[tuple[str, float, float]] = []
    if rppg_score is not None:
        active_candidates.append(("hemodynamic", rppg_score, settings.weight_hemodynamic))
    if phonetic_score is not None:
        active_candidates.append(("phonetic", phonetic_score, settings.weight_phonetic))
    if snr_score is not None and rppg_score is not None:
        active_candidates.append(("spectral_snr", snr_score, settings.weight_spectral_snr))

    total_weight = sum(w for _, _, w in active_candidates)
    if total_weight <= 0.0:
        total_weight = 1.0

    weights_used: dict[str, float] = {}
    fai_val = 0.0
    modules_contributed: list[str] = []

    for name, sc, raw_w in active_candidates:
        norm_w = raw_w / total_weight
        weights_used[name] = float(norm_w)
        modules_contributed.append(name)
        fai_val += norm_w * sc

    fai = float(np.clip(fai_val, 0.0, 1.0))

    # 4. Confidence and Uncertainty
    confidence = _determine_confidence(
        quality_passed=evidence.quality.passed,
        hemo_usable=(rppg_score is not None),
        phon_usable=(phonetic_score is not None),
        duration_s=duration_s,
        events_checked=events_checked,
        snr_score=snr_score,
        config=settings,
    )
    conf_uncertainty_map = {"HIGH": 0.15, "MEDIUM": 0.35, "LOW": 0.65}
    uncertainty = conf_uncertainty_map.get(confidence, 0.50)

    # Clean window fraction from rPPG
    clean_win_frac = getattr(evidence.rppg, "clean_window_fraction", None)
    if clean_win_frac is None:
        if evidence.rppg.windowed_r:
            clean_w = [w for w in evidence.rppg.windowed_r if not getattr(w, "is_artifact", False)]
            clean_win_frac = float(len(clean_w) / len(evidence.rppg.windowed_r))
        else:
            clean_win_frac = 1.0 if evidence.rppg.usable else 0.0

    usable_module_count = sum(1 for sc in (rppg_score, phonetic_score) if sc is not None)
    inconclusive_reasons: list[str] = []

    # Rule A: If only one module is usable and confidence is LOW, return INCONCLUSIVE
    single_module_low_conf = (usable_module_count == 1 and confidence == "LOW")
    if single_module_low_conf:
        active_mod = "hemodynamic rPPG" if rppg_score is not None else "bilabial phonetics"
        inconclusive_reasons.append(
            f"Only {active_mod} was usable with LOW confidence; cross-modal verification is required for conclusive attribution."
        )

    # 5. Verdict based on FAI cutoffs and calibrated safeguards
    verdict: Verdict
    if single_module_low_conf:
        verdict = "INCONCLUSIVE"
    elif fai >= settings.cutoff_authentic:
        verdict = "AUTHENTIC"
    elif fai >= settings.cutoff_suspect:
        verdict = "SUSPECT"
    else:
        # Potential MANIPULATED verdict: evaluate strict safeguards
        conf_ok = confidence in ("MEDIUM", "HIGH")
        qual_ok = evidence.quality.passed
        clean_ok = clean_win_frac >= settings.manipulated_min_clean_window_fraction
        rppg_alone_too_short = (hemo_usable and not phon_usable and duration_s < settings.min_duration_rppg_alone_s)

        if conf_ok and qual_ok and clean_ok and not rppg_alone_too_short:
            verdict = "MANIPULATED"
        else:
            verdict = "INCONCLUSIVE"
            if rppg_alone_too_short:
                inconclusive_reasons.append(
                    f"Single-module rPPG requires duration >= {settings.min_duration_rppg_alone_s:.1f}s for manipulation attribution (current: {duration_s:.1f}s)."
                )
            if not conf_ok:
                inconclusive_reasons.append(
                    f"Manipulation verdict requires confidence >= {settings.manipulated_min_confidence} (current: {confidence})."
                )
            if not clean_ok:
                inconclusive_reasons.append(
                    f"Clean window fraction {clean_win_frac:.2%} is below required {settings.manipulated_min_clean_window_fraction:.2%}."
                )
            if not qual_ok:
                inconclusive_reasons.append(f"Quality gate not passed ({evidence.quality.explanation}).")

    final_fai = fai if verdict != "INCONCLUSIVE" else None
    final_uncertainty = uncertainty if verdict != "INCONCLUSIVE" else 1.0

    # Optional bootstrap CI for mean r
    boot_ci = bootstrap_ci_mean_r(evidence.rppg)

    explain_str = _build_explain_string(
        verdict=verdict,
        fai=fai,
        confidence=confidence,
        rppg_score=rppg_score,
        phonetic_score=phonetic_score,
        duration_s=duration_s,
        events_checked=events_checked,
        anomaly_count=len(anomalies),
        reasons=inconclusive_reasons,
        mean_r=evidence.rppg.mean_r,
        corr_fake_threshold=settings.corr_fake,
    )

    LOGGER.info(
        "Arbiter: verdict=%s FAI=%s confidence=%s weights=%s",
        verdict,
        f"{final_fai:.3f}" if final_fai is not None else "None",
        confidence,
        weights_used,
    )

    return ArbiterResult(
        fai=final_fai,
        verdict=verdict,
        uncertainty=final_uncertainty,
        confidence=confidence,
        anomalies=anomalies,
        explanation=explain_str,
        explain=explain_str,
        weights_used=weights_used,
        modules_contributed=tuple(modules_contributed),
        rppg_score=rppg_score,
        phonetic_score=phonetic_score,
        spectral_snr_score=snr_score,
        bootstrap_ci_mean_r=boot_ci,
    )
