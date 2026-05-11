"""Public SDK for Barrikade."""

import os

from barrikade.sdk import PIPipeline
from core.artifacts import (
    ArtifactDownloadError,
    download_runtime_bundle,
    download_runtime_artifacts,
    ensure_runtime_bundle,
    ensure_runtime_artifacts,
)

if os.getenv("BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK", "0") == "0":
    ensure_runtime_bundle()

__all__ = [
    "ArtifactDownloadError",
    "PIPipeline",
    "download_runtime_bundle",
    "download_runtime_artifacts",
    "ensure_runtime_bundle",
    "ensure_runtime_artifacts",
]

