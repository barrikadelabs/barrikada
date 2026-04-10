from pathlib import Path
from zipfile import ZipFile

import pytest


REQUIRED_WHEEL_PATHS = [
    "core/layer_b/signatures/extracted/safe_allow_signatures.yar",
    "core/layer_b/signatures/embeddings/metadata.json",
    "core/layer_c/outputs/tf_idf_logreg.metadata.json",
]

EXCLUDED_WHEEL_PATHS = [
    "core/layer_c/outputs/classifier.joblib",
    "core/layer_d/outputs/model/model.safetensors",
    "core/layer_b/signatures/embeddings/prompt_encoder/model.safetensors",
    "core/layer_b/signatures/embeddings/signature_encoder/model.safetensors",
]


def _latest_wheel(dist_dir: Path) -> Path | None:
    wheels = sorted(dist_dir.glob("barrikada-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def test_built_wheel_contains_required_model_artifacts() -> None:
    dist_dir = Path(__file__).resolve().parents[1] / "dist"
    wheel_path = _latest_wheel(dist_dir)
    if wheel_path is None:
        pytest.skip("No wheel found in dist/. Build one first with: python -m build --wheel")

    repo_root = Path(__file__).resolve().parents[1]
    policy_inputs = [
        repo_root / "pyproject.toml",
        repo_root / "MANIFEST.in",
    ]
    latest_policy_mtime = max(path.stat().st_mtime for path in policy_inputs)
    if wheel_path.stat().st_mtime < latest_policy_mtime:
        pytest.skip("Latest wheel is stale. Rebuild with: python -m build --wheel")

    with ZipFile(wheel_path) as whl:
        names = set(whl.namelist())

    missing = [path for path in REQUIRED_WHEEL_PATHS if path not in names]
    assert not missing, f"Missing required packaged artifacts: {missing}"

    unexpectedly_present = [path for path in EXCLUDED_WHEEL_PATHS if path in names]
    assert not unexpectedly_present, (
        f"Heavy artifacts must not be bundled in slim wheel: {unexpectedly_present}"
    )
