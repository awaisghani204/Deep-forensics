"""Unit tests for pretrained detector bakeoff and LOO-CV metric calculation.

Adheres strictly to AGENTS.md:
- Deterministic synthetic signals
- Pure function verification
- Verifies LOO-CV (no clip sets its own threshold)
- Type safety and dataclass assertions
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from engine.face_tracker import FaceTrackingResult, LandmarkPoint, LandmarkSubset, ROIColor, TrackedFrame
from engine.video_io import VideoFrame, VideoMeta
from eval.bakeoff import (
    BakeoffConfig,
    ClipBakeoffResult,
    FeatureEvaluationMetric,
    compute_feature_loo_metric,
    load_bakeoff_config,
    sample_face_crops_from_tracking,
)
from scripts.download_models import load_bakeoff_config as script_load_bakeoff_config


def test_load_bakeoff_config_reads_defaults_and_yaml() -> None:
    """Config loader reads values from config.yaml or provides valid dataclass defaults."""
    cfg = load_bakeoff_config()
    assert isinstance(cfg, BakeoffConfig)
    assert cfg.sample_frames == 16
    assert cfg.video_crop_size == 224
    assert cfg.audio_sample_rate_hz == 16000
    assert "Deep-Fake" in cfg.image_model_id
    assert "VideoMae" in cfg.video_model_id


def test_download_models_script_config_loading() -> None:
    """scripts/download_models.py loads models dictionary properly."""
    cfg_data = script_load_bakeoff_config()
    assert isinstance(cfg_data, dict)
    assert "models" in cfg_data
    assert "image" in cfg_data["models"]
    assert "video" in cfg_data["models"]


def test_sample_face_crops_from_tracking_synthetic() -> None:
    """16 face crops are sampled using single-pass face tracker without second detection pass."""
    num_frames = 30
    h, w = 100, 100
    frames: list[VideoFrame] = []
    tracked: list[TrackedFrame] = []

    dummy_point = LandmarkPoint(0.5, 0.5, 0.0)
    dummy_subset = LandmarkSubset(dummy_point, dummy_point, dummy_point, dummy_point)
    roi_ok = ROIColor((120.0, 100.0, 90.0), inside_frame=True)

    for i in range(num_frames):
        # Frame with a central face patch
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[20:70, 30:80] = 150  # Face area
        frames.append(VideoFrame(bgr=img, timestamp_s=i / 30.0, frame_index=i, used_fallback_timestamp=False))

        # Normalized bounding box: (0.3, 0.2, 0.8, 0.7)
        face_box = (0.3, 0.2, 0.8, 0.7)
        tracked.append(
            TrackedFrame(
                frame_index=i,
                timestamp_s=i / 30.0,
                face_found=True,
                valid=True,
                forehead=roi_ok,
                left_cheek=roi_ok,
                right_cheek=roi_ok,
                landmarks=dummy_subset,
                motion_level=0.01,
                face_box=face_box,
            )
        )

    tracking_res = FaceTrackingResult(
        frames=tuple(tracked),
        face_fraction=1.0,
        mean_motion_level=0.01,
        explanation="Synthetic test tracking",
    )

    face_crops, video_224 = sample_face_crops_from_tracking(
        frames=tuple(frames),
        tracking=tracking_res,
        num_samples=16,
        crop_size=224,
    )

    assert len(face_crops) == 16
    assert len(video_224) == 16
    for crop in face_crops:
        assert isinstance(crop, np.ndarray)
        assert crop.shape[2] == 3
        # Dimensions match bbox cropping: 50x50
        assert crop.shape[0] == 50
        assert crop.shape[1] == 50
    for frame_224 in video_224:
        assert frame_224.shape == (224, 224, 3)


def test_loo_cv_perfect_separation() -> None:
    """Synthetic feature with perfect separation yields ROC-AUC=1.0 and 100% LOO balanced accuracy."""
    # 5 authentic (low values) and 5 manipulated (high values)
    values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9]
    labels = ["AUTHENTIC"] * 5 + ["MANIPULATED"] * 5
    clips = [f"clip_{i}.mp4" for i in range(10)]

    metric: FeatureEvaluationMetric = compute_feature_loo_metric(
        feature_name="test_feature",
        values=values,
        labels=labels,
        clip_names=clips,
        direction=1,  # higher is fake
    )

    assert metric.roc_auc == 1.0
    assert metric.loo_balanced_accuracy == 1.0
    assert len(metric.wrong_clips) == 0
    assert metric.best_threshold is not None
    assert 0.3 < metric.best_threshold < 0.7


def test_loo_cv_with_known_misclassification() -> None:
    """Under LOO-CV, an outlier clip cannot set its own threshold and is identified as misclassified."""
    # 4 authentic and 4 manipulated, with one manipulated outlier scoring low (0.1)
    values = [0.1, 0.2, 0.25, 0.3, 0.1, 0.8, 0.85, 0.9]  # clip_4 is an outlier fake
    labels = ["AUTHENTIC", "AUTHENTIC", "AUTHENTIC", "AUTHENTIC", "MANIPULATED", "MANIPULATED", "MANIPULATED", "MANIPULATED"]
    clips = [f"clip_{i}.mp4" for i in range(8)]

    metric: FeatureEvaluationMetric = compute_feature_loo_metric(
        feature_name="noisy_feature",
        values=values,
        labels=labels,
        clip_names=clips,
        direction=1,
    )

    assert metric.roc_auc is not None
    assert metric.roc_auc < 1.0  # Impaired by the outlier
    assert metric.loo_balanced_accuracy is not None
    # Outlier clip_4 must be in the misclassified list
    assert "clip_4.mp4" in metric.wrong_clips


def test_loo_cv_inverted_direction() -> None:
    """rPPG spatial correlation r where lower is fake (direction = -1) evaluates correctly."""
    # Reals have high r (0.7-0.9), fakes have low r (0.0-0.2)
    values = [0.75, 0.80, 0.85, 0.90, 0.05, 0.10, 0.15, 0.20]
    labels = ["AUTHENTIC"] * 4 + ["MANIPULATED"] * 4
    clips = [f"clip_{i}.mp4" for i in range(8)]

    metric: FeatureEvaluationMetric = compute_feature_loo_metric(
        feature_name="mean_r",
        values=values,
        labels=labels,
        clip_names=clips,
        direction=-1,  # lower is fake
    )

    assert metric.roc_auc == 1.0
    assert metric.loo_balanced_accuracy == 1.0
    assert len(metric.wrong_clips) == 0
