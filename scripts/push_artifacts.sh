#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Push Barrikada model/signature artifacts and datasets to the private artifacts repo.

Usage:
  ./scripts/push_artifacts.sh [options]

Options:
  --repo-url URL       Artifact repo clone URL.
                       Default: https://github.com/barrikadelabs/barrikade-artifacts.git
  --base-ref REF       Ref to clone from in artifact repo before creating/updating branch.
                       Default: main
  --branch NAME        Branch to commit/push in artifact repo.
                       Default: current local git branch (or artifact-update-YYYYmmdd-HHMMSS)
  --source-prefix DIR  Optional path inside artifact repo that contains artifact tree.
                       Example: artifacts
  --message MSG        Commit message.
                       Default: Update Barrikada artifacts and datasets
  --path REL_PATH      Relative path to sync/push. Repeat to pass multiple paths.
                       If omitted, defaults to all Barrikada artifact paths.
  --max-file-size-bytes N
                       Fail early if any source file is larger than N bytes.
                       Default: 2147483648 (2 GiB, common GitHub LFS per-file limit).
                       Set to 0 to disable this check.
  --delete             Delete files in destination path that are not present locally.
  --no-push            Commit locally in temp clone but do not push to origin.
  -h, --help           Show this help text.

Examples:
  ./scripts/push_artifacts.sh --branch feat/update-artifacts
  ./scripts/push_artifacts.sh --branch feat/layer-e --path core/layer_e/outputs
  ./scripts/push_artifacts.sh --source-prefix artifacts --branch feat/artifacts-sync --delete
EOF
}

REPO_URL="https://github.com/barrikadelabs/barrikade-artifacts.git"
BASE_REF="main"
SOURCE_PREFIX=""
COMMIT_MESSAGE="Update Barrikada artifacts and datasets"
DELETE_FLAG=""
DO_PUSH="1"
BRANCH=""
MAX_FILE_SIZE_BYTES="2147483648"

DEFAULT_PATHS=(
  "datasets"
  "core/layer_b/signatures"
  "core/layer_c/outputs"
  "core/layer_c/train/outputs"
  "core/layer_d/outputs"
  "core/layer_e/outputs"
)

CUSTOM_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --base-ref)
      BASE_REF="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --source-prefix)
      SOURCE_PREFIX="$2"
      shift 2
      ;;
    --message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --max-file-size-bytes)
      MAX_FILE_SIZE_BYTES="$2"
      shift 2
      ;;
    --path)
      CUSTOM_PATHS+=("$2")
      shift 2
      ;;
    --delete)
      DELETE_FLAG="--delete"
      shift
      ;;
    --no-push)
      DO_PUSH="0"
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
  echo "Install it first, then rerun this script." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in PATH." >&2
  exit 1
fi

if [[ -z "$BRANCH" ]]; then
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    guessed_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD || true)"
    if [[ -n "$guessed_branch" && "$guessed_branch" != "HEAD" ]]; then
      BRANCH="$guessed_branch"
    fi
  fi
fi

if [[ -z "$BRANCH" ]]; then
  BRANCH="artifact-update-$(date +%Y%m%d-%H%M%S)"
fi

PATHS=("${DEFAULT_PATHS[@]}")
if [[ ${#CUSTOM_PATHS[@]} -gt 0 ]]; then
  PATHS=("${CUSTOM_PATHS[@]}")
fi

map_path() {
  local rel="$1"
  if [[ -n "$SOURCE_PREFIX" ]]; then
    printf "%s/%s" "$SOURCE_PREFIX" "$rel"
  else
    printf "%s" "$rel"
  fi
}

validate_file_sizes() {
  local src_dir="$1"
  local rel_path="$2"
  local oversized="0"

  while IFS=' ' read -r size file_path; do
    if [[ -n "$size" ]] && (( size > MAX_FILE_SIZE_BYTES )); then
      if [[ "$oversized" == "0" ]]; then
        echo "[push-artifacts] ERROR: found files larger than --max-file-size-bytes=$MAX_FILE_SIZE_BYTES" >&2
        oversized="1"
      fi
      echo "  $size bytes: $file_path" >&2
    fi
  done < <(find "$src_dir" -type f -printf '%s %p\n')

  if [[ "$oversized" == "1" ]]; then
    echo "[push-artifacts] Path: $rel_path" >&2
    echo "[push-artifacts] Hint: shard model weights (e.g. max_shard_size=1900MB) or push a smaller artifact set." >&2
    exit 1
  fi
}

echo "[push-artifacts] Cloning $REPO_URL @ $BASE_REF ..."
git clone --depth 1 --branch "$BASE_REF" "$REPO_URL" "$TMP_DIR/repo"

pushd "$TMP_DIR/repo" >/dev/null
git lfs install --local >/dev/null
git checkout -B "$BRANCH"

staged_any="0"

for rel in "${PATHS[@]}"; do
  src="$REPO_ROOT/$rel"
  dst="$TMP_DIR/repo/$(map_path "$rel")"

  if [[ ! -e "$src" ]]; then
    echo "[push-artifacts] Skipping missing local path: $rel"
    continue
  fi

  if [[ "$MAX_FILE_SIZE_BYTES" != "0" ]]; then
    validate_file_sizes "$src" "$rel"
  fi

  mkdir -p "$dst"
  echo "[push-artifacts] Syncing $rel"
  rsync -a $DELETE_FLAG "$src/" "$dst/"

  git add "$(map_path "$rel")"
  staged_any="1"
done

if [[ "$staged_any" != "1" || -z "$(git diff --cached --name-only)" ]]; then
  echo "[push-artifacts] No changes detected. Nothing to commit."
  popd >/dev/null
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"

if [[ "$DO_PUSH" == "1" ]]; then
  git push -u origin "$BRANCH"
  echo "[push-artifacts] Pushed branch '$BRANCH' to origin."
else
  echo "[push-artifacts] Commit created in temp clone but push was skipped (--no-push)."
fi

popd >/dev/null

echo "[push-artifacts] Completed."