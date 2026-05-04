#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Pull Barrikada model/signature artifacts and datasets from a private Git repo
and place them into the expected runtime paths in this codebase.

Usage:
  ./scripts/pull_artifacts.sh [--repo-url URL] [--ref BRANCH_OR_TAG] [--source-prefix PATH] [--delete]

Options:
  --repo-url URL       Artifact repo clone URL.
                       Default: https://github.com/barrikadelabs/barrikade-artifacts.git
  --ref REF            Git ref to fetch from artifact repo (branch/tag/commit).
                       Default: main
  --source-prefix DIR  Optional path inside artifact repo that contains artifact tree.
                       Example: artifacts
  --delete             Delete files in destination that do not exist in source.
  -h, --help           Show this help text.

Examples:
  ./scripts/pull_artifacts.sh
  ./scripts/pull_artifacts.sh --ref feat/new-artifacts
  ./scripts/pull_artifacts.sh --source-prefix artifacts --delete
EOF
}

REPO_URL="https://github.com/barrikadelabs/barrikade-artifacts.git"
REF="main"
SOURCE_PREFIX=""
DELETE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --source-prefix)
      SOURCE_PREFIX="$2"
      shift 2
      ;;
    --delete)
      DELETE_FLAG="--delete"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required but was not found in PATH." >&2
  echo "Install it first: brew install git-lfs" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in PATH." >&2
  exit 1
fi

echo "[pull-artifacts] Cloning $REPO_URL @ $REF ..."
git clone --depth 1 --branch "$REF" "$REPO_URL" "$TMP_DIR/repo"

pushd "$TMP_DIR/repo" >/dev/null
git lfs install --local >/dev/null
echo "[pull-artifacts] Fetching LFS objects..."
git lfs fetch origin "$REF"
echo "[pull-artifacts] Pulling LFS objects..."
git lfs pull
popd >/dev/null

map_path() {
  local rel="$1"
  if [[ -n "$SOURCE_PREFIX" ]]; then
    printf "%s/%s" "$SOURCE_PREFIX" "$rel"
  else
    printf "%s" "$rel"
  fi
}

# Source paths in the artifact repo map 1:1 to destination paths in this repo.
PATHS=(
  "datasets"
  "core/layer_b/signatures"
  "core/layer_c/outputs"
  "core/layer_c/train/outputs"
  "core/layer_d/outputs"
  "core/layer_e/outputs/merged_teacher"
)

for rel in "${PATHS[@]}"; do
  src="$TMP_DIR/repo/$(map_path "$rel")"
  dst="$REPO_ROOT/$rel"

  if [[ ! -e "$src" ]]; then
    echo "[pull-artifacts] Skipping missing source path: $(map_path "$rel")"
    continue
  fi

  mkdir -p "$dst"
  echo "[pull-artifacts] Syncing $rel"
  rsync -a $DELETE_FLAG "$src/" "$dst/"
done

echo "[pull-artifacts] Completed. Artifacts and datasets are now synced into local directories."
