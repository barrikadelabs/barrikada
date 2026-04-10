from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_FILES = [
    "layer_b/signatures/embeddings/metadata.json",
    "layer_b/signatures/embeddings/cluster_radii.json",
    "layer_b/signatures/embeddings/faiss_index.bin",
    "layer_b/signatures/embeddings/benign_faiss_index.bin",
    "layer_b/signatures/embeddings/centroids.npy",
    "layer_b/signatures/embeddings/benign_centroids.npy",
    "layer_c/outputs/classifier.joblib",
]

DEFAULT_GLOBS = [
    "layer_b/signatures/extracted/**/*",
    "layer_b/signatures/embeddings/prompt_encoder/**/*",
    "layer_b/signatures/embeddings/signature_encoder/**/*",
    "layer_c/outputs/releases/**/classifier.joblib",
    "layer_d/outputs/model/**/*",
]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(core_dir: Path, paths: list[str]) -> dict:
    files = []
    seen: set[str] = set()

    for rel in paths:
        file_path = core_dir / rel
        if not file_path.exists():
            continue
        rel_norm = rel.replace("\\", "/")
        if rel_norm in seen:
            continue
        seen.add(rel_norm)
        files.append(
            {
                "path": rel_norm,
                "sha256": _sha256(file_path),
                "size": file_path.stat().st_size,
            }
        )

    for pattern in DEFAULT_GLOBS:
        for file_path in sorted(core_dir.glob(pattern)):
            if not file_path.is_file():
                continue
            rel_norm = file_path.relative_to(core_dir).as_posix()
            if rel_norm in seen:
                continue
            seen.add(rel_norm)
            files.append(
                {
                    "path": rel_norm,
                    "sha256": _sha256(file_path),
                    "size": file_path.stat().st_size,
                }
            )

    files.sort(key=lambda item: item["path"])
    return {"files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build artifact manifest for external hosting.")
    parser.add_argument("--core-dir", default="core", help="Path to core directory.")
    parser.add_argument(
        "--out",
        default="artifact-manifest.json",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="Additional relative path under core/ to include (repeatable).",
    )
    args = parser.parse_args()

    core_dir = Path(args.core_dir).expanduser().resolve()
    requested = DEFAULT_FILES + (args.path or [])
    manifest = build_manifest(core_dir, requested)

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path} with {len(manifest['files'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
