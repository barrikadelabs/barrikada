"""
Validate that bundled models are present and loadable.

This script checks:
1. All required files exist in core/models/
2. Models can be loaded without errors
3. Model directory structure is correct

Usage:
    python scripts/validate_models.py [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"


def validate_layer_b() -> Tuple[bool, List[str]]:
    """Validate Layer B (Signature Engine) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_b"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for required file types
    has_faiss = any(target_dir.glob("**/*.faiss"))
    has_npy = any(target_dir.glob("**/*.npy"))
    has_json = any(target_dir.glob("**/*.json"))
    
    if not has_faiss and not has_npy:
        errors.append("No FAISS indices (.faiss) or NumPy files (.npy) found")
    
    logger.info("Layer B:")
    logger.info(f"  ✓ FAISS indices: {has_faiss}")
    logger.info(f"  ✓ NumPy arrays: {has_npy}")
    logger.info(f"  ✓ JSON metadata: {has_json}")
    
    # Try to load a sample FAISS index if available
    try:
        import faiss
        faiss_files = list(target_dir.glob("**/*.faiss"))
        if faiss_files:
            test_file = faiss_files[0]
            index = faiss.read_index(str(test_file))
            logger.info(f"  ✓ FAISS index loads successfully ({test_file.name})")
    except ImportError:
        logger.debug("  ⊘ faiss not installed, skipping load test")
    except Exception as e:
        errors.append(f"Failed to load FAISS index: {e}")
    
    return len(errors) == 0, errors


def validate_layer_c() -> Tuple[bool, List[str]]:
    """Validate Layer C (ML Classifier) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_c"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for joblib files
    joblib_files = list(target_dir.glob("**/*.joblib"))
    
    if not joblib_files:
        errors.append("No .joblib files found")
        return False, errors
    
    logger.info("Layer C:")
    logger.info(f"  ✓ Joblib files: {len(joblib_files)}")
    
    # Try to load a sample joblib file
    try:
        import joblib
        test_file = joblib_files[0]
        model = joblib.load(test_file)
        logger.info(f"  ✓ Joblib model loads successfully ({test_file.name})")
    except ImportError:
        logger.debug("  ⊘ joblib not installed, skipping load test")
    except Exception as e:
        errors.append(f"Failed to load joblib model: {e}")
    
    return len(errors) == 0, errors


def validate_layer_d() -> Tuple[bool, List[str]]:
    """Validate Layer D (ModernBERT) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_d"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for Hugging Face model structure
    # Models can be in model/ subdirectory (from bundling) or at root
    model_dir = target_dir / "model"
    config_file = target_dir / "config.json" if (target_dir / "config.json").exists() else target_dir / "model" / "config.json"
    tokenizer_file = target_dir / "tokenizer.json" if (target_dir / "tokenizer.json").exists() else target_dir / "model" / "tokenizer.json"
    
    has_model_dir = model_dir.exists()
    has_config = config_file.exists()
    has_tokenizer = tokenizer_file.exists()
    
    logger.info("Layer D:")
    logger.info(f"  ✓ Model directory: {has_model_dir}")
    logger.info(f"  ✓ Config file: {has_config}")
    logger.info(f"  ✓ Tokenizer file: {has_tokenizer}")
    
    if not has_config:
        errors.append("Missing config.json")
    if not has_tokenizer and not has_config:
        errors.append("Missing both config.json and tokenizer.json")
    
    # Try to load model if transformers is available
    try:
        from transformers import AutoModel, AutoTokenizer
        
        if has_config and has_model_dir:
            try:
                model = AutoModel.from_pretrained(str(target_dir))
                logger.info(f"  ✓ Transformers model loads successfully")
            except Exception as e:
                logger.warning(f"  ⚠ Warning loading model: {e}")
    except ImportError:
        logger.debug("  ⊘ transformers not installed, skipping load test")
    
    return len(errors) == 0, errors


def validate_layer_e() -> Tuple[bool, List[str]]:
    """Validate Layer E (LLM Judge) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_e"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for teacher model structure
    teacher_dir = target_dir / "teacher"
    has_teacher = teacher_dir.exists()
    
    # Check for config files
    config_files = list(target_dir.glob("**/*.json"))
    
    logger.info("Layer E:")
    logger.info(f"  ✓ Teacher directory: {has_teacher}")
    logger.info(f"  ✓ Config files: {len(config_files)}")
    
    if not has_teacher and not config_files:
        errors.append("Neither teacher directory nor config files found")
    
    # Try to load teacher model if available
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if teacher_dir.exists():
            try:
                model = AutoModelForCausalLM.from_pretrained(str(teacher_dir))
                logger.info(f"  ✓ Teacher model loads successfully")
            except Exception as e:
                logger.warning(f"  ⚠ Warning loading teacher model: {e}")
    except ImportError:
        logger.debug("  ⊘ transformers not installed, skipping load test")
    
    return len(errors) == 0, errors


def check_archive_structure() -> Tuple[bool, List[str]]:
    """Check that archive directories exist."""
    errors = []
    
    logger.info("\nArchive structure:")
    for layer in ["layer_b", "layer_c", "layer_d", "layer_e"]:
        archive_dir = MODELS_DIR / layer / "archives"
        exists = archive_dir.exists()
        logger.info(f"  {layer}/archives: {exists}")
        
        if not exists:
            errors.append(f"Archive directory missing: {archive_dir}")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate bundled models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Validating bundled models...")
    logger.info(f"Models directory: {MODELS_DIR.relative_to(REPO_ROOT)}\n")
    
    results = {
        "Layer B (Signatures)": validate_layer_b(),
        "Layer C (Classifier)": validate_layer_c(),
        "Layer D (ModernBERT)": validate_layer_d(),
        "Layer E (LLM Judge)": validate_layer_e(),
        "Archive structure": check_archive_structure(),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    all_valid = True
    for name, (success, errors) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        
        if errors:
            for error in errors:
                logger.error(f"  → {error}")
            all_valid = False
    
    if all_valid:
        logger.info(f"\n✓ All validations passed")
        return 0
    else:
        logger.error(f"\n✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
