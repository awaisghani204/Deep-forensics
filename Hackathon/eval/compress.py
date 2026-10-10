"""Controlled video compression utility for forensic degradation testing.

Evaluates how lossy H.264 compression and social media transcoding affect
physiological and phonetic invariants without making false claims of immunity
to compression.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess

LOGGER = logging.getLogger(__name__)

COMPRESSION_PRESETS: dict[str, dict[str, str | int]] = {
    "crf18": {
        "name": "crf18",
        "description": "High fidelity, visually near-lossless (CRF 18)",
        "crf": 18,
        "type": "crf",
    },
    "crf23": {
        "name": "crf23",
        "description": "Standard x264 default web distribution (CRF 23)",
        "crf": 23,
        "type": "crf",
    },
    "crf28": {
        "name": "crf28",
        "description": "Noticeable lossy web compression (CRF 28)",
        "crf": 28,
        "type": "crf",
    },
    "crf35": {
        "name": "crf35",
        "description": "Heavy lossy degradation artifacts (CRF 35)",
        "crf": 35,
        "type": "crf",
    },
    "whatsapp": {
        "name": "whatsapp",
        "description": "WhatsApp-like preset (480-720p, ~800 kbps H.264, 64 kbps AAC)",
        "type": "social",
    },
}


@dataclass(frozen=True)
class CompressionResult:
    """Result of running controlled video re-encoding."""

    output_path: Path | None
    preset_name: str
    crf: int | None
    bitrate_kbps: int | None
    file_size_bytes: int | None
    explanation: str
    success: bool = True


def estimate_bitrate_kbps(file_path: Path, duration_s: float | None = None) -> int | None:
    """Estimate average container bitrate in kbps from file size and duration."""
    if not file_path.is_file():
        return None
    size_bytes = file_path.stat().st_size
    if duration_s is not None and duration_s > 0.0:
        return int((size_bytes * 8) / (duration_s * 1000.0))
    return None


def compress_video(
    source_path: Path | str,
    output_dir: Path | str | None = None,
    preset: str = "crf28",
    crf: int | None = None,
    ffmpeg_bin: str | None = None,
) -> CompressionResult:
    """Re-encode a single video according to specified preset or CRF setting."""
    source = Path(source_path)
    if not source.is_file():
        return CompressionResult(
            output_path=None,
            preset_name=preset,
            crf=crf,
            bitrate_kbps=None,
            file_size_bytes=None,
            explanation=f"Source video file not found: {source}",
            success=False,
        )

    binary = ffmpeg_bin or shutil.which("ffmpeg")
    if binary is None:
        return CompressionResult(
            output_path=None,
            preset_name=preset,
            crf=crf,
            bitrate_kbps=None,
            file_size_bytes=None,
            explanation="ffmpeg executable unavailable on system PATH.",
            success=False,
        )

    out_dir = Path(output_dir) if output_dir is not None else source.parent / "compressed"
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_key = preset.lower()
    if preset_key not in COMPRESSION_PRESETS and crf is None:
        preset_key = "crf28"

    effective_crf = crf if crf is not None else COMPRESSION_PRESETS.get(preset_key, {}).get("crf")

    if preset_key == "whatsapp":
        out_name = f"{source.stem}_whatsapp{source.suffix}"
        out_path = out_dir / out_name
        # Social media preset: scale to <= 720p height, 800 kbps H.264, 64k AAC
        cmd = [
            binary,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vf",
            "scale=-2:min(720\\,ih)",
            "-c:v",
            "libx264",
            "-b:v",
            "800k",
            "-maxrate",
            "1000k",
            "-bufsize",
            "1600k",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(out_path),
        ]
    else:
        target_crf = effective_crf if effective_crf is not None else 28
        out_name = f"{source.stem}_crf{target_crf}{source.suffix}"
        out_path = out_dir / out_name
        cmd = [
            binary,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-crf",
            str(target_crf),
            "-preset",
            "medium",
            "-c:a",
            "copy",
            str(out_path),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
    except Exception as exc:
        LOGGER.error("Transcoding failed with exception for %s: %s", source, exc)
        return CompressionResult(
            output_path=None,
            preset_name=preset_key,
            crf=effective_crf if isinstance(effective_crf, int) else None,
            bitrate_kbps=None,
            file_size_bytes=None,
            explanation=f"Transcoding execution error: {exc}",
            success=False,
        )

    if proc.returncode != 0 or not out_path.is_file():
        err_msg = proc.stderr.strip()[-200:] if proc.stderr else "Unknown error"
        LOGGER.warning("ffmpeg exited with code %d for %s: %s", proc.returncode, source, err_msg)
        return CompressionResult(
            output_path=None,
            preset_name=preset_key,
            crf=effective_crf if isinstance(effective_crf, int) else None,
            bitrate_kbps=None,
            file_size_bytes=None,
            explanation=f"ffmpeg transcoding failed: {err_msg}",
            success=False,
        )

    file_size = out_path.stat().st_size
    LOGGER.info("Successfully re-encoded %s -> %s (preset=%s, size=%d bytes)", source.name, out_path.name, preset_key, file_size)
    return CompressionResult(
        output_path=out_path,
        preset_name=preset_key,
        crf=effective_crf if isinstance(effective_crf, int) else None,
        bitrate_kbps=None,
        file_size_bytes=file_size,
        explanation=f"Successfully re-encoded via {preset_key} preset.",
        success=True,
    )


def compress_for_evaluation(
    source_path: Path,
    output_dir: Path | str | None = None,
    crf: int = 28,
    preset: str = "medium",
) -> CompressionResult:
    """Backwards-compatible wrapper re-encoding a video at a specified CRF."""
    return compress_video(
        source_path=source_path,
        output_dir=output_dir,
        preset=f"crf{crf}",
        crf=crf,
    )


def compress_dataset(
    video_paths: Sequence[Path | str],
    output_dir: Path | str,
    presets: Sequence[str] = ("crf18", "crf23", "crf28", "crf35", "whatsapp"),
) -> dict[str, list[CompressionResult]]:
    """Transcode a list of videos across multiple compression presets."""
    out_root = Path(output_dir)
    results: dict[str, list[CompressionResult]] = {p: [] for p in presets}

    for preset in presets:
        preset_dir = out_root / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        for vp in video_paths:
            res = compress_video(vp, output_dir=preset_dir, preset=preset)
            results[preset].append(res)

    return results
