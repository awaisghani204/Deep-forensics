"""Timestamp-preserving video ingestion for DeepTrace.

The decoder trusts container presentation timestamps when OpenCV exposes them.
Some codecs/backends provide missing or non-monotonic values; those streams fall
back to ``frame_index / fps`` and are explicitly marked in logs.  Bitrate is
optional because ``ffprobe`` is not a Python dependency and may be unavailable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2

from engine.types import VideoMeta


LOGGER = logging.getLogger(__name__)
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class VideoFrame:
    """One decoded BGR frame with its best available presentation timestamp."""

    frame_index: int
    timestamp_s: float
    bgr: Any
    used_fallback_timestamp: bool


@dataclass(frozen=True)
class VideoLoadResult:
    """Metadata plus a one-pass frame iterator.

    Iterate ``frames`` once. A new call to :func:`load_video` creates a fresh
    decoder, which avoids retaining an entire video in memory.
    """

    metadata: VideoMeta
    frames: Iterator[VideoFrame]


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot be decoded with reliable fallback timing."""


def compute_sha256(path: str | Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Compute a deterministic SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _fourcc_to_text(value: float) -> str:
    """Convert OpenCV's numeric FOURCC property to a printable codec label."""
    number = int(value)
    return "".join(chr((number >> (8 * offset)) & 0xFF) for offset in range(4)).strip() or "unknown"


def _ffprobe_bitrate(path: Path) -> int | None:
    """Return container bitrate from ffprobe when the optional binary exists."""
    executable = shutil.which("ffprobe")
    if executable is None:
        LOGGER.info("ffprobe unavailable; video bitrate will be omitted")
        return None
    command = [executable, "-v", "error", "-show_entries", "format=bit_rate:stream=bit_rate", "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        payload = json.loads(completed.stdout)
        candidates = [payload.get("format", {}).get("bit_rate")]
        candidates.extend(stream.get("bit_rate") for stream in payload.get("streams", []))
        return next((int(value) for value in candidates if value not in (None, "N/A")), None)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        LOGGER.warning("Unable to obtain bitrate with ffprobe: %s", error)
        return None


def _iter_frames(path: Path, fps: float) -> Iterator[VideoFrame]:
    """Decode frames once and attach monotonic container/fallback timestamps."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError(f"Unable to open video: {path}")
    previous_timestamp = -1.0
    use_fallback = False
    index = 0
    try:
        while True:
            success, bgr = capture.read()
            if not success:
                break
            raw_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            container_timestamp = raw_msec / 1000.0
            invalid_timestamp = not use_fallback and (
                raw_msec < 0.0 or (index > 0 and raw_msec <= 0.0) or (index > 0 and container_timestamp <= previous_timestamp)
            )
            if invalid_timestamp:
                use_fallback = True
                LOGGER.warning("Non-monotonic or missing PTS at frame %d; using frame_index/fps for remaining stream.", index)
            timestamp = index / fps if use_fallback else container_timestamp
            yield VideoFrame(index, timestamp, bgr, use_fallback)
            previous_timestamp = timestamp
            index += 1
    finally:
        capture.release()


def load_video(path: str | Path) -> VideoLoadResult:
    """Load metadata and a one-pass timestamped frame iterator.

    Assumption: a positive decoder-reported FPS exists when PTS fallback is
    needed. DeepTrace never invents a default frame rate for a broken container.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"Unable to open video: {source_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            raise VideoDecodeError("Video has no usable FPS for timestamp fallback")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = _fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    metadata = VideoMeta(
        source_path=source_path,
        sha256=compute_sha256(source_path),
        duration_s=frame_count / fps if frame_count > 0 else 0.0,
        frame_count=frame_count,
        codec=codec,
        resolution=(width, height),
        fps=fps,
        bitrate_bps=_ffprobe_bitrate(source_path),
    )
    return VideoLoadResult(metadata=metadata, frames=_iter_frames(source_path, fps))
