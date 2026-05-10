"""
Bundle models from scattered layer_*/outputs/ directories into core/models/ structure.

This script consolidates trained models from each layer's outputs directory into a
centralized core/models/ directory tree, organizing them by layer with support for
versioning via archives.

Usage:
    python scripts/bundle_models.py [--dry-run] [--archive-old]
    
    --dry-run:       Show what would be bundled without making changes
    --archive-old:   Move current models to archives before bundling new ones
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"

# Layer configurations: (layer_name, outputs_dir, required_files_pattern)
LAYER_CONFIGS = {
    "layer_b": {
        "outputs_dir": CORE_DIR / "layer_b" / "signatures",
        "target_dir": MODELS_DIR / "layer_b",
        "required_patterns": [
            "embeddings/centroids.npy",
            "embeddings/benign_centroids.npy",
            "embeddings/cluster_radii.json",
            "embeddings/metadata.json",
            "embeddings/faiss_index.bin",
            "embeddings/benign_faiss_index.bin",
            "embeddings/prompt_encoder/",
        ],
        "description": "Signature Engine (FAISS indices, embeddings)",
    },
    "layer_c": {
        "outputs_dir": CORE_DIR / "layer_c" / "outputs",
        "target_dir": MODELS_DIR / "layer_c",
        "required_patterns": ["classifier.joblib"],
        "description": "ML Classifier (XGBoost/sklearn models)",
    },
    "layer_d": {
        "outputs_dir": CORE_DIR / "layer_d" / "outputs",
        "target_dir": MODELS_DIR / "layer_d",
        "required_patterns": ["model/", "tokenizer.json", "*.safetensors"],
        "description": "ModernBERT (Hugging Face model)",
    },
    "layer_e": {
        "outputs_dir": CORE_DIR / "layer_e" / "outputs",
        "target_dir": MODELS_DIR / "layer_e",
        "required_patterns": ["qwen3guard-barrikade/", "*.json"],
        "description": "LLM Judge (bundled Hugging Face checkpoint)",
    },
}


def get_model_files(source_dir: Path, patterns: List[str]) -> List[Path]:
    """Find all files matching the given patterns."""
    files = []
    if not source_dir.exists():
        return files
    
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            dir_name = pattern.rstrip("/")
            dir_path = source_dir / dir_name
            if dir_path.is_dir():
                files.extend(dir_path.rglob("*"))
        else:
            # File pattern
            files.extend(source_dir.glob(pattern))
            files.extend(source_dir.rglob(pattern))
    
    return list(set(files))


def archive_existing_models(target_dir: Path) -> bool:
    """Move existing models in target_dir to archives subfolder."""
    if not target_dir.exists() or not list(target_dir.glob("*")):
        return False
    
    archives_dir = target_dir / "archives"
    archives_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subdir = archives_dir / f"backup_{timestamp}"
    
    logger.info(f"Archiving existing models to {archive_subdir.relative_to(REPO_ROOT)}")
    
    # Move all non-archive items to archive
    for item in target_dir.iterdir():
        if item.is_dir() and item.name == "archives":
            continue
        
        archive_subdir.mkdir(parents=True, exist_ok=True)
        dest = archive_subdir / item.name
        
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            shutil.rmtree(item)
        else:
            shutil.copy2(item, dest)
            item.unlink()
    
    return True


def bundle_layer(layer_name: str, config: Dict, dry_run: bool = False, archive_old: bool = False) -> Tuple[bool, str]:
    """
    Bundle models for a single layer.
    
    Returns:
        (success, message)
    """
    outputs_dir = config["outputs_dir"]
    target_dir = config["target_dir"]
    patterns = config["required_patterns"]
    description = config["description"]
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    model_files = get_model_files(outputs_dir, patterns)
    
    if not model_files:
        return False, f"No model files found in {outputs_dir.relative_to(REPO_ROOT)}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Layer {layer_name.upper()}: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Found {len(model_files)} file(s) in {outputs_dir.relative_to(REPO_ROOT)}")
    
    if archive_old:
        if not dry_run:
            archive_existing_models(target_dir)
        else:
            logger.info(f"[DRY RUN] Would archive existing models in {target_dir.relative_to(REPO_ROOT)}")
    
    # Copy/link model files
    copied_files = []
    for src_file in model_files:
        if src_file.is_file():
            # Preserve directory structure relative to outputs_dir
            rel_path = src_file.relative_to(outputs_dir)
            dest_file = target_dir / rel_path
            
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            if dry_run:
                logger.info(f"[DRY RUN] Would copy: {src_file.relative_to(REPO_ROOT)}")
                logger.info(f"          to: {dest_file.relative_to(REPO_ROOT)}")
            else:
                if not dest_file.exists() or dest_file.stat().st_mtime < src_file.stat().st_mtime:
                    shutil.copy2(src_file, dest_file)
                    logger.info(f"Copied: {rel_path}")
                    copied_files.append(str(rel_path))
                else:
                    logger.info(f"Skipped (up-to-date): {rel_path}")
        elif src_file.is_dir() and src_file != target_dir / "archives":
            # Copy entire directory
            rel_path = src_file.relative_to(outputs_dir)
            dest_dir = target_dir / rel_path
            
            if dry_run:
                logger.info(f"[DRY RUN] Would copy directory: {src_file.relative_to(REPO_ROOT)}")
                logger.info(f"          to: {dest_dir.relative_to(REPO_ROOT)}")
            else:
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_file, dest_dir)
                logger.info(f"Copied directory: {rel_path}")
    
    return True, f"Bundled {len(model_files)} file(s) for {layer_name}"


def validate_bundle() -> bool:
    """Validate that bundled models are present and accessible."""
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION")
    logger.info(f"{'='*60}")
    
    all_valid = True
    for layer_name, config in LAYER_CONFIGS.items():
        target_dir = config["target_dir"]
        
        if not target_dir.exists():
            logger.warning(f"{layer_name}: Target directory does not exist")
            all_valid = False
            continue
        
        files = list(target_dir.glob("**/*"))
        files = [f for f in files if f.is_file() and f.parent.name != "archives"]
        
        if files:
            logger.info(f"{layer_name}: OK ({len(files)} file(s))")
        else:
            logger.warning(f"{layer_name}: No files found in {target_dir}")
            all_valid = False
    
    return all_valid


def generate_bundle_manifest() -> Dict:
    """Generate a manifest of bundled models."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "layers": {},
    }
    
    for layer_name, config in LAYER_CONFIGS.items():
        target_dir = config["target_dir"]
        
        if target_dir.exists():
            files = []
            for file_path in target_dir.rglob("*"):
                if file_path.is_file() and file_path.parent.name != "archives":
                    rel_path = file_path.relative_to(target_dir)
                    files.append({
                        "name": str(rel_path),
                        "size": file_path.stat().st_size,
                        "mtime": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    })
            
            manifest["layers"][layer_name] = {
                "directory": str(target_dir.relative_to(REPO_ROOT)),
                "file_count": len(files),
                "files": files,
            }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Bundle models from layer outputs into centralized core/models/ directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be bundled without making changes",
    )
    parser.add_argument(
        "--archive-old",
        action="store_true",
        help="Archive existing models before bundling new ones",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Save bundle manifest to this file (JSON)",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model bundling process...")
    logger.info(f"Repo root: {REPO_ROOT}")
    logger.info(f"Models directory: {MODELS_DIR.relative_to(REPO_ROOT)}")
    
    if args.dry_run:
        logger.info("[DRY RUN] No changes will be made")
    
    results = {}
    for layer_name, config in LAYER_CONFIGS.items():
        success, message = bundle_layer(layer_name, config, args.dry_run, args.archive_old)
        results[layer_name] = {"success": success, "message": message}
    
    # Validate bundle
    valid = validate_bundle()
    
    # Generate and save manifest
    if args.manifest:
        manifest = generate_bundle_manifest()
        manifest_path = Path(args.manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        try:
            logger.info(f"Manifest saved to: {manifest_path.relative_to(REPO_ROOT)}")
        except ValueError:
            logger.info(f"Manifest saved to: {manifest_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for layer_name, result in results.items():
        status = "✓" if result["success"] else "✗"
        logger.info(f"{status} {layer_name}: {result['message']}")
    
    if valid:
        logger.info("\n✓ Bundle validation passed")
        return 0
    else:
        logger.warning("\n✗ Bundle validation failed or incomplete")
        return 1


if __name__ == "__main__":
    exit(main())
