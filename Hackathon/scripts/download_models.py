"""Download and cache pretrained deepfake detection models for offline bakeoff use.

Adheres to DeepTrace AGENTS.md:
- Reads model IDs from config.yaml
- Graceful error handling and informative logging
- Type hints, dataclasses, docstrings
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import urllib.request
import yaml

from huggingface_hub import HfApi, hf_hub_download

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedModel:
    """Metadata for a downloaded pretrained model."""

    name: str
    repo_id: str
    local_dir: Path
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class DownloadSummary:
    """Overall status of model downloads."""

    models: tuple[DownloadedModel, ...]
    all_success: bool


def load_bakeoff_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read bakeoff model IDs and configurations from config.yaml."""
    cfg_path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not cfg_path.exists():
        LOGGER.warning("Configuration file %s does not exist; using defaults", cfg_path)
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("bakeoff", {})


def _download_file(repo_id: str, filename: str, output_path: Path) -> bool:
    """Download one file directly via HTTP stream or hf_hub_download."""
    if output_path.exists() and output_path.stat().st_size > 0:
        LOGGER.info("File %s already exists; skipping download", output_path.name)
        return True

    # 1. Direct HTTP stream download for large files (avoids httpx CAS timeouts)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    try:
        LOGGER.info("Streaming %s/%s to %s ...", repo_id, filename, output_path.name)
        req = urllib.request.Request(url, headers={"User-Agent": "DeepTrace/1.0"})
        with urllib.request.urlopen(req, timeout=300) as response, output_path.open("wb") as out_f:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                out_f.write(chunk)
                downloaded += len(chunk)
        LOGGER.info("Download succeeded for %s/%s (%d bytes)", repo_id, filename, downloaded)
        return True
    except Exception as stream_exc:
        LOGGER.info("Direct stream encountered issue for %s/%s (%s); trying hf_hub_download...", repo_id, filename, stream_exc)
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

    # 2. Fallback to hf_hub_download
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(output_path.parent))
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception as exc:
        LOGGER.warning("hf_hub_download also failed for %s/%s: %s", repo_id, filename, exc)
    return False


def download_model(
    name: str,
    repo_id: str | None,
    output_dir: Path,
) -> DownloadedModel:
    """Download a single model repository from Hugging Face Hub into target directory."""
    if not repo_id or not repo_id.strip():
        return DownloadedModel(
            name=name,
            repo_id="",
            local_dir=output_dir,
            success=False,
            error="No model repository ID specified",
        )

    LOGGER.info("Downloading %s model [%s] to %s ...", name, repo_id, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = HfApi()
        info = api.model_info(repo_id)
        filenames = [f.rfilename for f in info.siblings]
        priority_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
        downloaded_count = 0
        for fn in priority_files:
            if fn in filenames:
                LOGGER.info("Fetching %s/%s -> %s", repo_id, fn, output_dir)
                if _download_file(repo_id, fn, output_dir / fn):
                    downloaded_count += 1
            else:
                LOGGER.info("File %s not listed in repository siblings for %s", fn, repo_id)

        if downloaded_count == 0:
            return DownloadedModel(
                name=name,
                repo_id=repo_id,
                local_dir=output_dir,
                success=False,
                error="No compatible model weights found",
            )

        LOGGER.info("Successfully downloaded %s to %s", name, output_dir)
        return DownloadedModel(name=name, repo_id=repo_id, local_dir=output_dir, success=True)
    except Exception as exc:
        LOGGER.warning("Failed to download %s model [%s]: %s", name, repo_id, exc)
        return DownloadedModel(
            name=name,
            repo_id=repo_id,
            local_dir=output_dir,
            success=False,
            error=str(exc),
        )


def download_all_models(
    config_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    skip_audio: bool = False,
) -> DownloadSummary:
    """Download all configured bakeoff models to the local models cache."""
    cfg = load_bakeoff_config(config_path)
    model_ids = cfg.get("models", {})

    image_id = model_ids.get("image", "prithivMLmods/Deep-Fake-Detector-v2-Model")
    video_id = model_ids.get("video", "SoraExplora/VideoMae")
    audio_id = None if skip_audio else model_ids.get("audio", "MelodyMachine/Deepfake-audio-detection")

    base_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1] / "models"
    base_dir.mkdir(parents=True, exist_ok=True)

    results: list[DownloadedModel] = []

    # 1. Image Model
    results.append(download_model("image", image_id, base_dir / "image"))

    # 2. Video Model
    results.append(download_model("video", video_id, base_dir / "video"))

    # 3. Audio Model
    if audio_id:
        results.append(download_model("audio", audio_id, base_dir / "audio"))
    else:
        LOGGER.info("Audio model skipped or not configured.")

    all_success = all(m.success for m in results)
    return DownloadSummary(models=tuple(results), all_success=all_success)


def main() -> None:
    """CLI entrypoint for scripts/download_models.py."""
    parser = argparse.ArgumentParser(description="Download and cache pretrained deepfake models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models-dir", type=str, default="models", help="Destination directory")
    parser.add_argument("--skip-audio", action="store_true", help="Skip downloading audio model")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    summary = download_all_models(
        config_path=args.config,
        models_dir=args.models_dir,
        skip_audio=args.skip_audio,
    )
    for model in summary.models:
        status = "OK" if model.success else f"FAILED ({model.error})"
        LOGGER.info("[%s] %s -> %s: %s", model.name.upper(), model.repo_id, model.local_dir, status)


if __name__ == "__main__":
    main()

