"""Tests for scripts/bundle_models.py.

Pins the per-layer required_patterns so a future regression that re-broadens
them (e.g. back to "*.joblib") can't silently re-ship vestigial training
artifacts. Currently focused on Layer C, which had ~160 MB of dead-weight
joblibs (pca_reducer, rf_model, tf_idf_*, feature_selector, meta_features)
left over from earlier classifier experiments — none consumed at runtime by
core/layer_c/classifier.py.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bundle_models import LAYER_CONFIGS, get_model_files


def test_layer_c_pattern_excludes_dead_weight(tmp_path):
    """Layer C bundling must match only classifier.joblib files (current +
    pinned releases), not other joblibs from earlier experimental pipelines."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # Files that must be bundled
    (outputs / "classifier.joblib").write_bytes(b"current")
    release_dir = outputs / "releases" / "v0.1"
    release_dir.mkdir(parents=True)
    (release_dir / "classifier.joblib").write_bytes(b"v0.1")

    # Vestigial files that must NOT be bundled
    for name in (
        "pca_reducer.joblib",
        "rf_model.joblib",
        "tf_idf_logreg.joblib",
        "tf_idf_vectorizer.joblib",
        "feature_selector.joblib",
        "meta_features.joblib",
    ):
        (outputs / name).write_bytes(b"dead")

    patterns = LAYER_CONFIGS["layer_c"]["required_patterns"]
    matched = get_model_files(outputs, patterns)
    matched_relative = {p.relative_to(outputs).as_posix() for p in matched if p.is_file()}

    assert matched_relative == {
        "classifier.joblib",
        "releases/v0.1/classifier.joblib",
    }
