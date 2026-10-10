"""Precompute and disk-cache DeepTrace verification results for demo clips.

Ensures that live demo presentations load instantaneously with zero runtime delay.
Generates full pipeline metrics, anomalies, and court-ready PDF dockets ahead of time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import run_pipeline_for_clip
from eval.run_eval import generate_synthetic_benchmark_dataset

LOGGER = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of video file on disk."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def precompute_demo_cache(
    data_dir: Path | str = "data",
    cache_dir: Path | str = "outputs/cache",
    whisper_model: str = "base",
    window_s: float = 10.0,
) -> list[Path]:
    """Run verification and cache results for all MP4 files in data directory."""
    d_dir = Path(data_dir)
    c_dir = Path(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    videos = list(d_dir.rglob("*.mp4")) if d_dir.is_dir() else []
    if not videos:
        LOGGER.info("No videos found in %s. Generating synthetic demo clips...", d_dir)
        generate_synthetic_benchmark_dataset(d_dir)
        videos = list(d_dir.rglob("*.mp4"))

    cached_files: list[Path] = []
    LOGGER.info("Precomputing forensic analysis for %d demo videos...", len(videos))

    for vid in videos:
        sha = compute_file_sha256(vid)
        cache_path = c_dir / f"{sha}.json"

        if cache_path.is_file():
            LOGGER.info("✓ Already cached: %s (SHA-256: %s)", vid.name, sha[:12])
            cached_files.append(cache_path)
            continue

        try:
            LOGGER.info("Processing %s -> %s...", vid.name, cache_path.name)
            payload = run_pipeline_for_clip(vid, whisper_model=whisper_model, window_s=window_s)
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            LOGGER.info("✓ Precomputed & cached %s in %s", vid.name, cache_path)
            cached_files.append(cache_path)
        except Exception as exc:
            LOGGER.error("Failed precomputing %s: %s", vid, exc)

    return cached_files


def main() -> None:
    """CLI entrypoint for precompute mode."""
    parser = argparse.ArgumentParser(description="Precompute DeepTrace forensic caches for demo clips.")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to demo data directory.")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache", help="Path to output cache directory.")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size.")
    parser.add_argument("--window-s", type=float, default=10.0, help="rPPG sliding window in seconds.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cached = precompute_demo_cache(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        whisper_model=args.whisper_model,
        window_s=args.window_s,
    )
    LOGGER.info("Precompute complete. Total %d demo clips cached.", len(cached))


if __name__ == "__main__":
    main()

