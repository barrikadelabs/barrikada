"""Runtime artifact bootstrap for Barrikade SDK installs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from core.settings import Settings

log = logging.getLogger(__name__)

DEFAULT_GCS_BUCKET = "barrikade-bundles"
GCS_MODELS_PREFIX = "models"


class ArtifactDownloadError(RuntimeError):
    """Raised when the runtime artifact bundle cannot be prepared."""


def _artifact_bucket(bucket_name: str | None = None) -> str:
    return bucket_name or os.getenv("BARRIKADA_GCS_BUCKET") or DEFAULT_GCS_BUCKET


def _artifact_targets(settings: Settings) -> list[tuple[str, Path]]:
    return [
        ("layer_b", Path(settings.layer_b_signatures_dirname)),
        ("layer_c", Path(settings.layer_c_model_pathname)),
        ("layer_d", Path(settings.layer_d_model_dirname)),
        ("layer_e", Path(settings.layer_e_model_dirname)),
    ]


def _layer_candidates(settings: Settings) -> dict[str, list[Path]]:
    return {
        "layer_b": settings.layer_b_signatures_candidates,
        "layer_c": settings.layer_c_model_candidates,
        "layer_d": settings.layer_d_model_candidates,
        "layer_e": settings.layer_e_model_candidates,
    }


def _missing_layers(settings: Settings) -> list[str]:
    missing = []
    for layer_name, candidates in _layer_candidates(settings).items():
        if not any(candidate.exists() for candidate in candidates):
            missing.append(layer_name)
    return missing


def artifacts_available(settings: Settings | None = None) -> bool:
    runtime_settings = settings or Settings()
    return not _missing_layers(runtime_settings)


def _list_gcs_layer_files(bucket_name: str, layer_name: str) -> list[str]:
    try:
        from google.auth.credentials import AnonymousCredentials
        from google.cloud import storage
    except ImportError as exc:
        raise ArtifactDownloadError(
            "Artifact download requires google-cloud-storage. "
            "Install the SDK with its default dependencies."
        ) from exc

    client = storage.Client(
        credentials=AnonymousCredentials(),
        project="anonymous-project",
    )
    prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/"
    files = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/") or "/archives/" in blob.name:
            continue
        files.append(blob.name)
    return files


def _download_gcs_file(bucket_name: str, blob_name: str, local_path: Path) -> None:
    try:
        import requests
    except ImportError as exc:
        raise ArtifactDownloadError(
            "Artifact download requires requests. Install the SDK with its default dependencies."
        ) from exc

    url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise ArtifactDownloadError(
            f"Failed to download {blob_name} from bucket {bucket_name}: "
            f"HTTP {response.status_code} {response.reason}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)


def _layer_root(layer_name: str, settings: Settings) -> Path:
    return Path(settings.artifacts_root_dir) / layer_name


def _iter_download_targets(
    layer_name: str,
    files: Iterable[str],
    settings: Settings,
) -> Iterable[tuple[str, Path]]:
    layer_root = _layer_root(layer_name, settings)
    for blob_name in files:
        parts = blob_name.split("/")
        relative_path = Path(*parts[2:])
        yield blob_name, layer_root / relative_path


def download_runtime_artifacts(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    runtime_settings = settings or Settings()
    resolved_bucket = _artifact_bucket(bucket_name)
    summary: dict[str, object] = {"bucket": resolved_bucket, "downloaded_layers": []}
    downloaded_layers: list[str] = []

    for layer_name in sorted(set(_missing_layers(runtime_settings) if not force else [name for name, _ in _artifact_targets(runtime_settings)])):
        log.info("Downloading Barrikade artifacts for %s", layer_name)
        files = _list_gcs_layer_files(resolved_bucket, layer_name)
        if not files:
            raise ArtifactDownloadError(
                f"No runtime artifacts found for {layer_name} in gs://{resolved_bucket}/{GCS_MODELS_PREFIX}/"
            )

        for blob_name, local_path in _iter_download_targets(layer_name, files, runtime_settings):
            _download_gcs_file(resolved_bucket, blob_name, local_path)

        downloaded_layers.append(layer_name)

    summary["downloaded_layers"] = downloaded_layers
    summary["artifacts_dir"] = runtime_settings.artifacts_root_dir
    return summary


def ensure_runtime_artifacts(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    auto_download: bool | None = None,
) -> None:
    runtime_settings = settings or Settings()
    missing_layers = _missing_layers(runtime_settings)
    if not missing_layers:
        return

    if auto_download is None:
        auto_download = os.getenv("BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS", "1") != "0"

    if not auto_download:
        missing = ", ".join(sorted(missing_layers))
        raise ArtifactDownloadError(
            "Barrikade runtime artifacts are missing for "
            f"{missing}. Run `python -m barrikade download-artifacts` or set "
            "BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS=1."
        )

    try:
        download_runtime_artifacts(settings=runtime_settings, bucket_name=bucket_name)
    except Exception as exc:
        missing = ", ".join(sorted(missing_layers))
        raise ArtifactDownloadError(
            "Barrikade could not prepare runtime artifacts for "
            f"{missing}. Run `python -m barrikade download-artifacts` to retry."
        ) from exc
