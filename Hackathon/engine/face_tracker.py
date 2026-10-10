"""Single-pass MediaPipe Face Mesh tracking and skin-patch colour extraction.

The head-pose gate is a conservative proxy, not a calibrated pose estimate.
Occlusion, makeup, colour grading, and poor landmark placement can still
contaminate skin traces and must be reflected in the quality assessment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

from engine.video_io import VideoFrame


_FOREHEAD = 151
_LEFT_CHEEK = 117
_RIGHT_CHEEK = 346
_LEFT_EYE = 33
_RIGHT_EYE = 263
_UPPER_LIP = 13
_LOWER_LIP = 14


@dataclass(frozen=True)
class FaceTrackerConfig:
    """Configured geometric gates for one-face tracking."""

    roi_scale_interocular: float
    max_eye_depth_ratio: float
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class LandmarkPoint:
    """A normalized MediaPipe landmark retained for later phonetics reuse."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class LandmarkSubset:
    """Only lips and eye corners are retained to constrain memory use."""

    upper_lip: LandmarkPoint
    lower_lip: LandmarkPoint
    left_eye_corner: LandmarkPoint
    right_eye_corner: LandmarkPoint


@dataclass(frozen=True)
class ROIColor:
    """Mean RGB colour from a polygonal skin patch."""

    rgb: tuple[float, float, float] | None
    inside_frame: bool


@dataclass(frozen=True)
class TrackedFrame:
    """One timestamped face-tracking and ROI-colour observation."""

    frame_index: int
    timestamp_s: float
    face_found: bool
    valid: bool
    forehead: ROIColor
    left_cheek: ROIColor
    right_cheek: ROIColor
    landmarks: LandmarkSubset | None
    motion_level: float | None
    face_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FaceTrackingResult:
    """One-video tracker output, suitable for direct rPPG conversion."""

    frames: tuple[TrackedFrame, ...]
    face_fraction: float
    mean_motion_level: float | None
    explanation: str

    @property
    def face_found_count(self) -> int:
        return sum(1 for f in self.frames if f.face_found)

    @property
    def face_found_fraction(self) -> float:
        return self.face_fraction


def load_face_tracker_config(config_path: str | Path | None = None) -> FaceTrackerConfig:
    """Read tracking thresholds from config.yaml rather than hard-coding them."""
    path = Path(config_path) if config_path is not None else Path(__file__).parents[1] / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)["face_tracker"]
    return FaceTrackerConfig(**values)


def _point(landmark: object) -> LandmarkPoint:
    """Convert a MediaPipe landmark to an immutable compact value object."""
    return LandmarkPoint(x=float(landmark.x), y=float(landmark.y), z=float(landmark.z))


def _pixel(point: LandmarkPoint, width: int, height: int) -> np.ndarray:
    """Convert a normalized point to an image-space coordinate."""
    return np.array((point.x * width, point.y * height), dtype=np.float64)


def _patch_vertices(center: np.ndarray, radius: float) -> np.ndarray:
    """Return a rectangular patch for robust skin-area sampling.

    A rectangle covers ~4× more pixels than a diamond of the same radius,
    producing a more stable mean-RGB trace for rPPG extraction.
    """
    half_w = radius
    half_h = radius * 0.6  # slightly shorter vertically to stay on skin
    return np.array((
        (center[0] - half_w, center[1] - half_h),
        (center[0] + half_w, center[1] - half_h),
        (center[0] + half_w, center[1] + half_h),
        (center[0] - half_w, center[1] + half_h),
    ), dtype=np.float64)


def _mean_patch_rgb(bgr: np.ndarray, vertices: np.ndarray) -> ROIColor:
    """Average BGR pixels inside a fully in-frame polygon and return RGB."""
    height, width = bgr.shape[:2]
    inside = bool(np.all(vertices[:, 0] >= 0.0) and np.all(vertices[:, 0] < width) and np.all(vertices[:, 1] >= 0.0) and np.all(vertices[:, 1] < height))
    if not inside:
        return ROIColor(rgb=None, inside_frame=False)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(vertices).astype(np.int32), 255)
    blue, green, red, _ = cv2.mean(bgr, mask=mask)
    return ROIColor(rgb=(float(red), float(green), float(blue)), inside_frame=True)


def _empty_frame(frame: VideoFrame) -> TrackedFrame:
    """Build a consistent invalid output for a missed face."""
    invalid_roi = ROIColor(rgb=None, inside_frame=False)
    return TrackedFrame(frame.frame_index, frame.timestamp_s, False, False, invalid_roi, invalid_roi, invalid_roi, None, None, None)


def track_video(frames: Iterable[VideoFrame], config: FaceTrackerConfig | None = None) -> FaceTrackingResult:
    """Track one face and extract all three skin patches in the same decode pass."""
    settings = config or load_face_tracker_config()
    tracked: list[TrackedFrame] = []
    previous_eye_midpoint: np.ndarray | None = None
    previous_eye_distance: float | None = None
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    ) as mesh:
        for frame in frames:
            result = mesh.process(cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                tracked.append(_empty_frame(frame))
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            height, width = frame.bgr.shape[:2]
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            face_box = (
                max(0.0, x_min - pad_x),
                max(0.0, y_min - pad_y),
                min(1.0, x_max + pad_x),
                min(1.0, y_max + pad_y),
            )
            subset = LandmarkSubset(_point(landmarks[_UPPER_LIP]), _point(landmarks[_LOWER_LIP]), _point(landmarks[_LEFT_EYE]), _point(landmarks[_RIGHT_EYE]))
            left_eye = _pixel(subset.left_eye_corner, width, height)
            right_eye = _pixel(subset.right_eye_corner, width, height)
            eye_distance = float(np.linalg.norm(right_eye - left_eye))
            eye_midpoint = (left_eye + right_eye) / 2.0
            motion = None if previous_eye_midpoint is None or previous_eye_distance is None or previous_eye_distance <= 0.0 else float(np.linalg.norm(eye_midpoint - previous_eye_midpoint) / previous_eye_distance)
            previous_eye_midpoint, previous_eye_distance = eye_midpoint, eye_distance
            patch_radius = settings.roi_scale_interocular * eye_distance
            forehead = _mean_patch_rgb(frame.bgr, _patch_vertices(_pixel(_point(landmarks[_FOREHEAD]), width, height), patch_radius))
            left_cheek = _mean_patch_rgb(frame.bgr, _patch_vertices(_pixel(_point(landmarks[_LEFT_CHEEK]), width, height), patch_radius))
            right_cheek = _mean_patch_rgb(frame.bgr, _patch_vertices(_pixel(_point(landmarks[_RIGHT_CHEEK]), width, height), patch_radius))
            eye_depth_ratio = abs(subset.left_eye_corner.z - subset.right_eye_corner.z) / max(eye_distance / width, np.finfo(float).eps)
            pose_ok = eye_depth_ratio <= settings.max_eye_depth_ratio
            valid = bool(pose_ok and forehead.inside_frame and left_cheek.inside_frame and right_cheek.inside_frame)
            tracked.append(TrackedFrame(frame.frame_index, frame.timestamp_s, True, valid, forehead, left_cheek, right_cheek, subset, motion, face_box))
    face_fraction = sum(frame.valid for frame in tracked) / len(tracked) if tracked else 0.0
    motion_values = [frame.motion_level for frame in tracked if frame.motion_level is not None]
    return FaceTrackingResult(tuple(tracked), face_fraction, float(np.mean(motion_values)) if motion_values else None, "Face Mesh tracked one face; colour patches are valid only when fully in-frame and pose proxy passes.")
