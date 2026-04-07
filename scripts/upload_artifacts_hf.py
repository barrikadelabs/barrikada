import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def _normalize_repo_id(repo_id: str) -> str:
    cleaned = repo_id.strip().strip("/")
    if not cleaned or "/" not in cleaned:
        raise ValueError("repo-id must be in the format '<namespace>/<name>'")
    return cleaned


def _load_manifest(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain a non-empty 'files' array")
    return files


def _resolve_local_file(core_dir: Path, rel_path: str) -> Path:
    candidate = core_dir / rel_path
    if not candidate.exists():
        raise FileNotFoundError(f"Missing local artifact file: {candidate}")
    return candidate


def _build_hf_base_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"


def upload_manifest_and_artifacts(
    *,
    repo_id: str,
    revision: str,
    manifest_path: Path,
    core_dir: Path,
    token: str | None,
    private: bool,
    create_repo: bool,
    dry_run: bool,
) -> dict:
    files = _load_manifest(manifest_path)
    api = HfApi(token=token)

    if create_repo and not dry_run:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    uploaded = []
    for entry in files:
        rel_path = entry.get("path")
        if not rel_path:
            raise ValueError("Each manifest file entry must include 'path'")

        local_file = _resolve_local_file(core_dir, rel_path)
        if dry_run:
            uploaded.append({"path": rel_path, "local": str(local_file), "uploaded": False})
            continue

        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=f"Upload artifact: {rel_path}",
        )
        uploaded.append({"path": rel_path, "local": str(local_file), "uploaded": True})

    manifest_target_name = "manifest.json"
    if dry_run:
        manifest_uploaded = False
    else:
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=manifest_target_name,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message="Upload artifact manifest",
        )
        manifest_uploaded = True

    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "manifest": str(manifest_path),
        "manifest_uploaded": manifest_uploaded,
        "files": uploaded,
        "base_url": _build_hf_base_url(repo_id, revision),
        "fetch_command": f"barrikada fetch-artifacts --base-url {_build_hf_base_url(repo_id, revision)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Barrikada artifacts to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. Ishaan005/barrikada-artifacts")
    parser.add_argument("--revision", default="main", help="Branch or tag to upload to.")
    parser.add_argument("--manifest", default="artifact-manifest.json", help="Path to artifact manifest JSON.")
    parser.add_argument("--core-dir", default="core", help="Path to local core directory containing artifact files.")
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private (only used with --create-repo).",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repository if it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and local paths without uploading.",
    )

    args = parser.parse_args()

    repo_id = _normalize_repo_id(args.repo_id)
    manifest_path = Path(args.manifest).expanduser().resolve()
    core_dir = Path(args.core_dir).expanduser().resolve()

    payload = upload_manifest_and_artifacts(
        repo_id=repo_id,
        revision=args.revision,
        manifest_path=manifest_path,
        core_dir=core_dir,
        token=args.token,
        private=args.private,
        create_repo=args.create_repo,
        dry_run=args.dry_run,
    )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
