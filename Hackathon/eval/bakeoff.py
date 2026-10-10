"""Pretrained deepfake detection model bakeoff and benchmark against DeepTrace invariants.

Evaluates pretrained image, video, and audio detectors against existing DeepTrace
physiological invariants (rPPG spatial coherence, spectral SNR, BPM agreement, and
bilabial lip-sync phonetics) across authentic and manipulated clips.

Adheres strictly to AGENTS.md:
- Reads model IDs from config.yaml
- CPU friendly with optional GPU auto-detection
- Single-pass face tracking: reuses track_video() landmarks and face crops (no second detection pass)
- Full LOO-CV (leave-one-out cross-validation) so no clip sets its own threshold
- Outputs results to outputs/bakeoff/results.csv and saves dark-theme ROC plots
- Pure functions, type hints, dataclasses, docstrings, and robust logging
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import logging
from pathlib import Path
import tempfile
from typing import Any, Sequence
import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
import soundfile as sf

from engine.face_tracker import FaceTrackingResult, TrackedFrame, track_video
from engine.phonetics import analyze_phonetics, extract_audio
from engine.rppg import analyze_rppg, rppg_input_from_tracking
from engine.types import PhoneticResult, QualityReport, RPPGResult
from engine.video_io import VideoFrame, VideoLoadResult, load_video

LOGGER = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModelForAudioClassification,
        AutoModelForImageClassification,
        AutoModelForVideoClassification,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    LOGGER.warning("torch or transformers is not available in the current environment.")


# -----------------------------------------------------------------------------
# Configuration & Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BakeoffConfig:
    """Bakeoff benchmark configuration loaded from config.yaml."""

    image_model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    video_model_id: str = "SoraExplora/VideoMae"
    audio_model_id: str | None = "MelodyMachine/Deepfake-audio-detection"
    sample_frames: int = 16
    video_crop_size: int = 224
    audio_sample_rate_hz: int = 16000
    audio_chunk_s: float = 10.0


@dataclass
class PretrainedDetectors:
    """Container for loaded pretrained neural detection models."""

    device: str
    image_processor: Any = None
    image_model: Any = None
    image_fake_idx: int = 1
    video_processor: Any = None
    video_model: Any = None
    video_fake_idx: int = 1
    audio_processor: Any = None
    audio_model: Any = None
    audio_fake_idx: int = 0


@dataclass(frozen=True)
class ClipBakeoffResult:
    """Individual video evaluation row with both pretrained scores and invariants."""

    label: str  # "AUTHENTIC" or "MANIPULATED"
    clip: str  # Relative path or filename
    duration: float
    image_fake_median: float | None
    image_fake_p90: float | None
    video_fake: float | None
    audio_fake: float | None
    mean_r: float | None
    snr: float | None
    bpm_spread: float | None
    phonetic_score: float | None


@dataclass(frozen=True)
class FeatureEvaluationMetric:
    """ROC-AUC and LOO-CV balanced accuracy performance for one feature."""

    feature_name: str
    direction: int  # +1 if higher value indicates fake, -1 if lower indicates fake
    roc_auc: float | None
    best_threshold: float | None
    loo_balanced_accuracy: float | None
    sample_count: int
    wrong_clips: tuple[str, ...] = field(default_factory=tuple)
    fpr: tuple[float, ...] = field(default_factory=tuple)
    tpr: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BakeoffSummary:
    """Complete summary of the bakeoff benchmark run."""

    results: tuple[ClipBakeoffResult, ...]
    metrics: dict[str, FeatureEvaluationMetric]
    csv_path: Path
    plot_path: Path


# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------


def load_bakeoff_config(config_path: str | Path | None = None) -> BakeoffConfig:
    """Load bakeoff parameters from config.yaml with sensible fallbacks."""
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not path.is_file():
        LOGGER.warning("config.yaml not found at %s; using default bakeoff config", path)
        return BakeoffConfig()

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    b_cfg = data.get("bakeoff", {})
    models = b_cfg.get("models", {})
    return BakeoffConfig(
        image_model_id=models.get("image", "prithivMLmods/Deep-Fake-Detector-v2-Model"),
        video_model_id=models.get("video", "SoraExplora/VideoMae"),
        audio_model_id=models.get("audio", "MelodyMachine/Deepfake-audio-detection"),
        sample_frames=int(b_cfg.get("sample_frames", 16)),
        video_crop_size=int(b_cfg.get("video_crop_size", 224)),
        audio_sample_rate_hz=int(b_cfg.get("audio_sample_rate_hz", 16000)),
        audio_chunk_s=float(b_cfg.get("audio_chunk_s", 10.0)),
    )


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------


def _resolve_model_path(model_id: str, alias: str, models_dir: Path | None = None) -> str:
    """Return local path in models/ if cached offline, else the remote repo ID."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parents[1] / "models"
    local_path = models_dir / alias
    has_weights = (local_path / "model.safetensors").exists() or (local_path / "pytorch_model.bin").exists()
    if local_path.is_dir() and (local_path / "config.json").exists() and has_weights:
        LOGGER.info("Using cached offline %s model at %s", alias, local_path)
        return str(local_path)
    return model_id


def _find_fake_class_index(id2label: dict[Any, Any] | None, default_idx: int = 1) -> int:
    """Inspect model label mapping to robustly find the fake/deepfake class index."""
    if not id2label:
        return default_idx
    for key, label in id2label.items():
        label_str = str(label).strip().lower()
        if any(term in label_str for term in ("fake", "deepfake", "spoof", "manipulated")):
            try:
                return int(key)
            except ValueError:
                pass
    return default_idx


def load_pretrained_detectors(
    config: BakeoffConfig,
    models_dir: Path | None = None,
) -> PretrainedDetectors:
    """Load neural detectors safely on CPU/CUDA, logging errors gracefully."""
    device = "cuda" if HAS_TRANSFORMERS and torch.cuda.is_available() else "cpu"
    detectors = PretrainedDetectors(device=device)

    if not HAS_TRANSFORMERS:
        LOGGER.warning("Transformers library is not available; pretrained neural models will be skipped.")
        return detectors

    # 1. Image Model
    try:
        model_source = _resolve_model_path(config.image_model_id, "image", models_dir)
        LOGGER.info("Loading image model from %s on %s...", model_source, device)
        detectors.image_processor = AutoImageProcessor.from_pretrained(model_source)
        detectors.image_model = AutoModelForImageClassification.from_pretrained(model_source).to(device)
        detectors.image_model.eval()
        detectors.image_fake_idx = _find_fake_class_index(
            getattr(detectors.image_model.config, "id2label", None), default_idx=1
        )
        LOGGER.info("Image model loaded successfully (fake_class_idx=%d).", detectors.image_fake_idx)
    except Exception as exc:
        LOGGER.warning("Could not load image model [%s]: %s", config.image_model_id, exc)

    # 2. Video Model
    try:
        model_source = _resolve_model_path(config.video_model_id, "video", models_dir)
        LOGGER.info("Loading video model from %s on %s...", model_source, device)
        detectors.video_processor = AutoImageProcessor.from_pretrained(model_source)
        detectors.video_model = AutoModelForVideoClassification.from_pretrained(model_source).to(device)
        detectors.video_model.eval()
        detectors.video_fake_idx = _find_fake_class_index(
            getattr(detectors.video_model.config, "id2label", None), default_idx=1
        )
        LOGGER.info("Video model loaded successfully (fake_class_idx=%d).", detectors.video_fake_idx)
    except Exception as exc:
        LOGGER.warning("Could not load video model [%s]: %s", config.video_model_id, exc)

    # 3. Audio Model (Configurable; skip gracefully if fails)
    if config.audio_model_id:
        try:
            model_source = _resolve_model_path(config.audio_model_id, "audio", models_dir)
            LOGGER.info("Loading audio model from %s on %s...", model_source, device)
            detectors.audio_processor = AutoFeatureExtractor.from_pretrained(model_source)
            detectors.audio_model = AutoModelForAudioClassification.from_pretrained(model_source).to(device)
            detectors.audio_model.eval()
            detectors.audio_fake_idx = _find_fake_class_index(
                getattr(detectors.audio_model.config, "id2label", None), default_idx=0
            )
            LOGGER.info("Audio model loaded successfully (fake_class_idx=%d).", detectors.audio_fake_idx)
        except Exception as exc:
            LOGGER.warning("Could not load audio model [%s]: %s", config.audio_model_id, exc)
    else:
        LOGGER.info("No audio model configured; skipping audio neural detector.")

    return detectors


# -----------------------------------------------------------------------------
# Single-Pass Face Crop Sampling
# -----------------------------------------------------------------------------


def sample_face_crops_from_tracking(
    frames: Sequence[VideoFrame],
    tracking: FaceTrackingResult,
    num_samples: int = 16,
    crop_size: int = 224,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Sample evenly-spaced face crops and 224x224 resized frames using existing face tracker.

    NO second detection pass is performed: face locations are directly extracted
    from the tracked frames produced by MediaPipe Face Mesh.
    """
    total_frames = len(frames)
    if total_frames == 0:
        return (), ()

    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    crops_rgb: list[np.ndarray] = []
    video_frames_224: list[np.ndarray] = []

    for idx in indices:
        v_frame: VideoFrame = frames[idx]
        t_frame: TrackedFrame = tracking.frames[idx]
        bgr = v_frame.bgr
        height, width = bgr.shape[:2]

        crop: np.ndarray | None = None
        if t_frame.face_box is not None:
            x0, y0, x1, y1 = t_frame.face_box
            xmin = max(0, min(width - 1, int(round(x0 * width))))
            ymin = max(0, min(height - 1, int(round(y0 * height))))
            xmax = max(xmin + 1, min(width, int(round(x1 * width))))
            ymax = max(ymin + 1, min(height, int(round(y1 * height))))
            crop = bgr[ymin:ymax, xmin:xmax]

        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            # Fallback to center square crop of frame if face box was not detected
            side = min(height, width)
            sy = (height - side) // 2
            sx = (width - side) // 2
            crop = bgr[sy : sy + side, sx : sx + side]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_224 = cv2.resize(crop_rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        crops_rgb.append(crop_rgb)
        video_frames_224.append(crop_224)

    return tuple(crops_rgb), tuple(video_frames_224)


# -----------------------------------------------------------------------------
# Pretrained Neural Model Inference
# -----------------------------------------------------------------------------


def predict_image_model(
    detectors: PretrainedDetectors,
    face_crops_rgb: Sequence[np.ndarray],
) -> tuple[float | None, float | None]:
    """Inference for image detector: fake-probability per crop, returning median and 90th percentile."""
    if detectors.image_model is None or detectors.image_processor is None or not face_crops_rgb:
        return None, None

    try:
        pil_images = [Image.fromarray(crop) for crop in face_crops_rgb]
        inputs = detectors.image_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(detectors.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = detectors.image_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        fake_idx = min(detectors.image_fake_idx, probs.shape[-1] - 1)
        fake_probs = probs[:, fake_idx].detach().cpu().numpy()

        median_prob = float(np.median(fake_probs))
        p90_prob = float(np.percentile(fake_probs, 90))
        return median_prob, p90_prob
    except Exception as exc:
        LOGGER.warning("Image model inference error: %s", exc)
        return None, None


def predict_video_model(
    detectors: PretrainedDetectors,
    video_frames_224: Sequence[np.ndarray],
) -> float | None:
    """Inference for video detector: one fake-probability from 16 frames at 224x224."""
    if detectors.video_model is None or detectors.video_processor is None or len(video_frames_224) < 16:
        return None

    try:
        pil_frames = [Image.fromarray(f) for f in video_frames_224[:16]]
        inputs = detectors.video_processor(images=list(pil_frames), return_tensors="pt")
        inputs = {k: v.to(detectors.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = detectors.video_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        fake_idx = min(detectors.video_fake_idx, probs.shape[-1] - 1)
        return float(probs[fake_idx].item())
    except Exception as exc:
        LOGGER.warning("Video model inference error: %s", exc)
        return None


def predict_audio_model(
    detectors: PretrainedDetectors,
    clip_path: Path,
    sample_rate_hz: int = 16000,
    chunk_s: float = 10.0,
) -> float | None:
    """Inference for audio detector: extract 16 kHz mono audio and average fake-probability over 10 s chunks."""
    if detectors.audio_model is None or detectors.audio_processor is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav = Path(tmp_file.name)

    try:
        extraction = extract_audio(clip_path, tmp_wav)
        if extraction.audio_path is None or not tmp_wav.is_file():
            LOGGER.info("Audio extraction not available for %s: %s", clip_path.name, extraction.reason)
            return None

        audio_data, sr = sf.read(str(tmp_wav))
        if len(audio_data) == 0:
            LOGGER.info("Audio track for %s is empty", clip_path.name)
            return None

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=-1)

        chunk_samples = int(sample_rate_hz * chunk_s)
        chunk_probs: list[float] = []

        for offset in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[offset : offset + chunk_samples]
            if len(chunk) < sample_rate_hz:  # Less than 1 second
                if chunk_probs:
                    continue  # We already have longer chunks

            inputs = detectors.audio_processor(
                chunk,
                sampling_rate=sample_rate_hz,
                return_tensors="pt",
            )
            inputs = {k: v.to(detectors.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = detectors.audio_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            fake_idx = min(detectors.audio_fake_idx, probs.shape[-1] - 1)
            chunk_probs.append(float(probs[fake_idx].item()))

        return float(np.mean(chunk_probs)) if chunk_probs else None
    except Exception as exc:
        LOGGER.warning("Audio model inference error for %s: %s", clip_path.name, exc)
        return None
    finally:
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Single Clip Evaluation
# -----------------------------------------------------------------------------


def evaluate_clip(
    clip_path: Path,
    label: str,
    detectors: PretrainedDetectors,
    config: BakeoffConfig,
) -> ClipBakeoffResult:
    """Evaluate one clip headlessly, computing both pretrained detector scores and DeepTrace invariants."""
    LOGGER.info("Evaluating clip [%s] -> %s", label, clip_path.name)
    loaded = load_video(clip_path)
    duration = loaded.metadata.duration_s
    frames = tuple(loaded.frames)

    # 1. Single-pass Face Tracking (records ROI colors AND face bounding box)
    tracking = track_video(frames)

    # 2. Sample 16 face crops from tracking (zero second detection pass)
    face_crops, video_crops_224 = sample_face_crops_from_tracking(
        frames=frames,
        tracking=tracking,
        num_samples=config.sample_frames,
        crop_size=config.video_crop_size,
    )

    # 3. Pretrained Neural Model Scores
    img_median, img_p90 = predict_image_model(detectors, face_crops)
    vid_fake = predict_video_model(detectors, video_crops_224)
    aud_fake = predict_audio_model(
        detectors,
        clip_path,
        sample_rate_hz=config.audio_sample_rate_hz,
        chunk_s=config.audio_chunk_s,
    )

    # 4. Existing DeepTrace Invariant Signals
    rppg_res: RPPGResult = analyze_rppg(rppg_input_from_tracking(tracking))
    phon_res: PhoneticResult = analyze_phonetics(clip_path, tracking)

    valid_snrs = [s for s in rppg_res.snr_db_per_roi if s is not None and np.isfinite(s)]
    mean_snr = float(np.mean(valid_snrs)) if valid_snrs else None

    return ClipBakeoffResult(
        label=label,
        clip=str(clip_path.name),
        duration=round(duration, 2),
        image_fake_median=round(img_median, 4) if img_median is not None else None,
        image_fake_p90=round(img_p90, 4) if img_p90 is not None else None,
        video_fake=round(vid_fake, 4) if vid_fake is not None else None,
        audio_fake=round(aud_fake, 4) if aud_fake is not None else None,
        mean_r=round(rppg_res.mean_r, 4) if rppg_res.mean_r is not None else None,
        snr=round(mean_snr, 4) if mean_snr is not None else None,
        bpm_spread=round(rppg_res.bpm_agreement_bpm, 4) if rppg_res.bpm_agreement_bpm is not None else None,
        phonetic_score=round(phon_res.score, 4) if phon_res.usable and phon_res.score is not None else None,
    )


# -----------------------------------------------------------------------------
# Leave-One-Out Cross-Validation (LOO-CV) & Metrics
# -----------------------------------------------------------------------------


def compute_feature_loo_metric(
    feature_name: str,
    values: Sequence[float | None],
    labels: Sequence[str],
    clip_names: Sequence[str],
    direction: int,
) -> FeatureEvaluationMetric:
    """Compute ROC-AUC and LOO-CV Balanced Accuracy for a single signal without data leakage."""
    valid_data: list[tuple[float, int, str]] = []
    for val, lbl, name in zip(values, labels, clip_names):
        if val is not None and np.isfinite(val):
            binary_lbl = 1 if lbl == "MANIPULATED" else 0
            valid_data.append((float(val), binary_lbl, name))

    n_samples = len(valid_data)
    if n_samples < 2:
        return FeatureEvaluationMetric(
            feature_name=feature_name,
            direction=direction,
            roc_auc=None,
            best_threshold=None,
            loo_balanced_accuracy=None,
            sample_count=n_samples,
        )

    vals = np.array([item[0] for item in valid_data], dtype=np.float64)
    y_true = np.array([item[1] for item in valid_data], dtype=np.int32)
    names = [item[2] for item in valid_data]

    # Check for presence of both classes
    if len(np.unique(y_true)) < 2:
        return FeatureEvaluationMetric(
            feature_name=feature_name,
            direction=direction,
            roc_auc=None,
            best_threshold=None,
            loo_balanced_accuracy=None,
            sample_count=n_samples,
        )

    # 1. Standardized score for ROC: higher must mean fake (label=1)
    std_scores = vals if direction == 1 else -vals
    auc_val = float(roc_auc_score(y_true, std_scores))
    fpr_arr, tpr_arr, _ = roc_curve(y_true, std_scores)

    # 2. Leave-One-Out Cross-Validation for Balanced Accuracy
    loo_preds: list[int] = []

    for i in range(n_samples):
        # Training split: all samples except sample i
        train_indices = [j for j in range(n_samples) if j != i]
        train_vals = vals[train_indices]
        train_y = y_true[train_indices]

        # Candidate thresholds: midpoints between unique sorted training values
        unique_vals = np.sort(np.unique(train_vals))
        if len(unique_vals) > 1:
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            candidates = np.concatenate(([unique_vals[0] - 1.0], midpoints, [unique_vals[-1] + 1.0]))
        else:
            candidates = np.array([unique_vals[0]])

        best_train_bal_acc = -1.0
        best_tau = float(candidates[0])

        for tau in candidates:
            preds_train = (train_vals >= tau).astype(int) if direction == 1 else (train_vals <= tau).astype(int)
            if len(np.unique(train_y)) >= 2:
                bal_acc = float(balanced_accuracy_score(train_y, preds_train))
            else:
                bal_acc = float(np.mean(preds_train == train_y))
            if bal_acc > best_train_bal_acc:
                best_train_bal_acc = bal_acc
                best_tau = float(tau)

        # Predict held-out sample i strictly using training threshold best_tau
        pred_i = int(vals[i] >= best_tau) if direction == 1 else int(vals[i] <= best_tau)
        loo_preds.append(pred_i)

    loo_bal_acc = float(balanced_accuracy_score(y_true, loo_preds))

    # Determine overall best threshold on full dataset for reporting
    unique_all = np.sort(np.unique(vals))
    midpoints_all = (unique_all[:-1] + unique_all[1:]) / 2.0 if len(unique_all) > 1 else unique_all
    best_overall_tau = float(midpoints_all[0])
    best_overall_score = -1.0
    for tau in midpoints_all:
        p = (vals >= tau).astype(int) if direction == 1 else (vals <= tau).astype(int)
        score = float(balanced_accuracy_score(y_true, p))
        if score > best_overall_score:
            best_overall_score = score
            best_overall_tau = float(tau)

    # Identify which clips this feature misclassifies under LOO-CV
    wrong_clips = [names[k] for k in range(n_samples) if loo_preds[k] != y_true[k]]

    return FeatureEvaluationMetric(
        feature_name=feature_name,
        direction=direction,
        roc_auc=round(auc_val, 4),
        best_threshold=round(best_overall_tau, 4),
        loo_balanced_accuracy=round(loo_bal_acc, 4),
        sample_count=n_samples,
        wrong_clips=tuple(wrong_clips),
        fpr=tuple(float(v) for v in fpr_arr),
        tpr=tuple(float(v) for v in tpr_arr),
    )


# -----------------------------------------------------------------------------
# ROC Plotting
# -----------------------------------------------------------------------------


def plot_roc_curves(
    metrics: dict[str, FeatureEvaluationMetric],
    output_path: Path,
) -> None:
    """Generate high-contrast dark-mode ROC curve comparison plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    colors = [
        "#00d2ff", "#ff007f", "#39ff14", "#ffaa00",
        "#a855f7", "#38bdf8", "#4ade80", "#f43f5e",
    ]

    color_idx = 0
    for feat_name, metric in metrics.items():
        if metric.roc_auc is None or not metric.fpr:
            continue
        c = colors[color_idx % len(colors)]
        label_text = f"{feat_name} (AUC={metric.roc_auc:.3f}, LOO-Acc={metric.loo_balanced_accuracy:.3f})"
        ax.plot(metric.fpr, metric.tpr, label=label_text, color=c, lw=2.2, alpha=0.9)
        color_idx += 1

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6e7681", lw=1.5, alpha=0.7, label="Chance (AUC=0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#e6edf3", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", color="#e6edf3", fontsize=11, fontweight="bold")
    ax.set_title("DeepTrace // Pretrained Detector Bakeoff ROC Curves", color="#e6edf3", fontsize=13, fontweight="bold", pad=12)

    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.grid(color="#30363d", linestyle=":", linewidth=0.8, alpha=0.6)
    legend = ax.legend(loc="lower right", facecolor="#0d1117", edgecolor="#30363d", fontsize=9)
    for text in legend.get_texts():
        text.set_color("#e6edf3")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    LOGGER.info("Saved ROC plot to %s", output_path)


# -----------------------------------------------------------------------------
# Main Benchmark Pipeline
# -----------------------------------------------------------------------------


def load_results_from_csv(csv_path: Path) -> list[ClipBakeoffResult]:
    """Parse an existing results.csv into a list of ClipBakeoffResult dataclasses."""
    results: list[ClipBakeoffResult] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            def _parse_flt(val: str | None) -> float | None:
                if val is None or not val.strip():
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            results.append(
                ClipBakeoffResult(
                    label=row["label"],
                    clip=row["clip"],
                    duration=float(row["duration"]),
                    image_fake_median=_parse_flt(row.get("image_fake_median")),
                    image_fake_p90=_parse_flt(row.get("image_fake_p90")),
                    video_fake=_parse_flt(row.get("video_fake")),
                    audio_fake=_parse_flt(row.get("audio_fake")),
                    mean_r=_parse_flt(row.get("mean_r")),
                    snr=_parse_flt(row.get("snr")),
                    bpm_spread=_parse_flt(row.get("bpm_spread")),
                    phonetic_score=_parse_flt(row.get("phonetic_score")),
                )
            )
    return results


def run_bakeoff(
    data_dir: Path,
    out_dir: Path,
    config_path: Path | None = None,
    models_dir: Path | None = None,
    from_csv: Path | None = None,
) -> BakeoffSummary:
    """Discover real and fake video clips, run bakeoff evaluation, and compute LOO metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    if from_csv is not None and from_csv.is_file():
        LOGGER.info("Loading existing results from %s (skipping re-inference)", from_csv)
        results = load_results_from_csv(from_csv)
    else:
        cfg = load_bakeoff_config(config_path)
        detectors = load_pretrained_detectors(cfg, models_dir=models_dir)

        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"

        real_clips = sorted(real_dir.rglob("*.mp4")) if real_dir.exists() else []
        fake_clips = sorted(fake_dir.rglob("*.mp4")) if fake_dir.exists() else []

        all_clips: list[tuple[Path, str]] = []
        for clip in real_clips:
            all_clips.append((clip, "AUTHENTIC"))
        for clip in fake_clips:
            all_clips.append((clip, "MANIPULATED"))

        LOGGER.info("Discovered %d clips (%d authentic, %d manipulated).", len(all_clips), len(real_clips), len(fake_clips))

        results = []
        for clip_path, label in all_clips:
            res = evaluate_clip(clip_path, label, detectors, cfg)
            results.append(res)

        # Save results.csv
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "label",
                "clip",
                "duration",
                "image_fake_median",
                "image_fake_p90",
                "video_fake",
                "audio_fake",
                "mean_r",
                "snr",
                "bpm_spread",
                "phonetic_score",
            ])
            for r in results:
                writer.writerow([
                    r.label,
                    r.clip,
                    r.duration,
                    r.image_fake_median if r.image_fake_median is not None else "",
                    r.image_fake_p90 if r.image_fake_p90 is not None else "",
                    r.video_fake if r.video_fake is not None else "",
                    r.audio_fake if r.audio_fake is not None else "",
                    r.mean_r if r.mean_r is not None else "",
                    r.snr if r.snr is not None else "",
                    r.bpm_spread if r.bpm_spread is not None else "",
                    r.phonetic_score if r.phonetic_score is not None else "",
                ])
        LOGGER.info("Saved bakeoff results to %s", csv_path)

    # Feature definitions and their physiological/statistical direction:
    # direction = +1 if higher values indicate MANIPULATED; -1 if lower values indicate MANIPULATED
    feature_specs: list[tuple[str, int]] = [
        ("image_fake_median", 1),
        ("image_fake_p90", 1),
        ("video_fake", 1),
        ("audio_fake", 1),
        ("mean_r", -1),
        ("snr", -1),
        ("bpm_spread", 1),
        ("phonetic_score", -1),
    ]

    labels = [r.label for r in results]
    clip_names = [r.clip for r in results]
    metrics: dict[str, FeatureEvaluationMetric] = {}

    for feat_name, direction in feature_specs:
        values = [getattr(r, feat_name) for r in results]
        metric = compute_feature_loo_metric(
            feature_name=feat_name,
            values=values,
            labels=labels,
            clip_names=clip_names,
            direction=direction,
        )
        metrics[feat_name] = metric

    # Save ROC curves plot
    plot_path = out_dir / "roc_curves.png"
    plot_roc_curves(metrics, plot_path)

    return BakeoffSummary(
        results=tuple(results),
        metrics=metrics,
        csv_path=csv_path,
        plot_path=plot_path,
    )


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI runner printing ranked performance table and misclassified clips."""
    parser = argparse.ArgumentParser(description="DeepTrace pretrained detector bakeoff benchmark.")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="outputs/bakeoff", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models-dir", type=str, default="models", help="Cached models directory")
    parser.add_argument("--from-csv", type=str, default=None, help="Optional path to existing results.csv to skip re-inference")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    summary = run_bakeoff(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config),
        models_dir=Path(args.models_dir),
        from_csv=Path(args.from_csv) if args.from_csv else None,
    )

    # 1. Print Ranked Results Table
    sorted_metrics = sorted(
        summary.metrics.items(),
        key=lambda item: (item[1].roc_auc if item[1].roc_auc is not None else -1.0),
        reverse=True,
    )

    print("\n" + "=" * 90)
    print(" DEEPTRACE DETECTOR BAKEOFF BENCHMARK // RANKED PERFORMANCE TABLE")
    print("=" * 90)
    print(f"{'Feature / Model':<22} | {'ROC-AUC':<9} | {'LOO Bal Acc':<11} | {'Best Thresh':<11} | {'Direction':<12} | {'Clips'}")
    print("-" * 90)

    for name, m in sorted_metrics:
        auc_str = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A"
        bal_str = f"{m.loo_balanced_accuracy:.4f}" if m.loo_balanced_accuracy is not None else "N/A"
        tau_str = f"{m.best_threshold:.4f}" if m.best_threshold is not None else "N/A"
        dir_str = "Higher=Fake" if m.direction == 1 else "Lower=Fake"
        print(f"{name:<22} | {auc_str:<9} | {bal_str:<11} | {tau_str:<11} | {dir_str:<12} | {m.sample_count}")
    print("=" * 90)

    # 2. Print Clips Each Feature Gets Wrong
    print("\n" + "=" * 90)
    print(" MISCLASSIFIED CLIPS PER FEATURE (Under Leave-One-Out Cross-Validation)")
    print("=" * 90)
    for name, m in sorted_metrics:
        print(f"\n>> {name} (Error Count: {len(m.wrong_clips)} / {m.sample_count}):")
        if m.wrong_clips:
            for clip in m.wrong_clips:
                print(f"   [FAIL] {clip}")
        else:
            print("   [PERFECT] No errors under LOO-CV.")
    print("\n" + "=" * 90)
    print(f"Results CSV: {summary.csv_path}")
    print(f"ROC Curves:  {summary.plot_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

