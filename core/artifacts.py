from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def fetch_artifacts(
    *,
    base_url: str,
    target_dir: Path,
    manifest_name: str = "manifest.json",
    force: bool = False,
) -> dict:
    if not base_url:
        raise ValueError("base_url is required")

    normalized_base = base_url.rstrip("/") + "/"
    manifest_url = urljoin(normalized_base, manifest_name)
    with urlopen(manifest_url) as response:
        manifest = json.loads(response.read().decode("utf-8"))

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Artifact manifest must contain a non-empty 'files' list")

    downloaded = 0
    skipped = 0

    for entry in files:
        rel_path = entry.get("path")
        sha256 = entry.get("sha256")
        if not rel_path or not sha256:
            raise ValueError("Each manifest entry must contain 'path' and 'sha256'")

        destination = target_dir / rel_path
        if destination.exists() and not force:
            if _sha256(destination) == sha256:
                skipped += 1
                continue

        artifact_url = urljoin(normalized_base, rel_path)
        _download_file(artifact_url, destination)

        actual = _sha256(destination)
        if actual != sha256:
            raise ValueError(
                f"SHA mismatch for {rel_path}: expected {sha256}, got {actual}"
            )
        downloaded += 1

    return {
        "manifest_url": manifest_url,
        "target_dir": str(target_dir),
        "downloaded": downloaded,
        "skipped": skipped,
        "total": len(files),
    }
