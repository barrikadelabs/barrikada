from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.agent import DEFAULT_MODEL_NAME, evaluate, interactive
from core.artifacts import fetch_artifacts
from core.orchestrator import PIPipeline
from core.settings import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barrikada",
        description="Run Barrikada security pipeline from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run the pipeline on one input string.",
    )
    detect_parser.add_argument("text", help="Input text to scan")
    detect_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON output.",
    )

    health_parser = subparsers.add_parser(
        "health",
        help="Show resolved local artifact paths.",
    )
    health_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON output.",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-artifacts",
        help="Download large model artifacts from external storage.",
    )
    fetch_parser.add_argument(
        "--base-url",
        default=os.getenv("BARRIKADA_ARTIFACTS_BASE_URL", ""),
        help="Base URL that serves manifest.json and artifact files.",
    )
    fetch_parser.add_argument(
        "--target-dir",
        default=None,
        help="Local target directory for artifact files.",
    )
    fetch_parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="Artifact manifest filename at base URL.",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when local hashes already match.",
    )
    fetch_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON output.",
    )

    chat_parser = subparsers.add_parser(
        "agent-chat",
        help="Start interactive Barrikada-wrapped agent chat.",
    )
    chat_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Ollama model name to use for responses.",
    )

    eval_parser = subparsers.add_parser(
        "agent-eval",
        help="Run CSV evaluation through the Barrikada-wrapped agent.",
    )
    eval_parser.add_argument(
        "--csv",
        default="datasets/barrikada_test.csv",
        help="Path to evaluation CSV with columns: text,label",
    )
    eval_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row limit for quick local checks.",
    )
    eval_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Ollama model name to use for responses.",
    )

    return parser


def _run_detect(text: str, pretty: bool) -> int:
    pipeline = PIPipeline()
    result = pipeline.detect(text)
    payload = result.to_dict()
    output = json.dumps(payload, indent=2 if pretty else None)
    print(output)
    return 0


def _run_health(pretty: bool) -> int:
    settings = Settings()
    payload = {
        "artifacts_root_dir": settings.artifacts_root_dir,
        "layer_b_signatures_dir": settings.layer_b_signatures_dir,
        "layer_c_model_path": settings.model_path,
        "layer_d_model_dir": settings.layer_d_output_dir,
        "layer_e_base_url": settings.layer_e_ollama_base_url,
    }
    output = json.dumps(payload, indent=2 if pretty else None)
    print(output)
    return 0


def _run_fetch_artifacts(
    base_url: str,
    target_dir: str | None,
    manifest: str,
    force: bool,
    pretty: bool,
) -> int:
    settings = Settings()
    destination = Path(target_dir).expanduser().resolve() if target_dir else Path(settings.artifacts_root_dir)
    payload = fetch_artifacts(
        base_url=base_url,
        target_dir=destination,
        manifest_name=manifest,
        force=force,
    )
    output = json.dumps(payload, indent=2 if pretty else None)
    print(output)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "detect":
        return _run_detect(args.text, args.pretty)
    if args.command == "health":
        return _run_health(args.pretty)
    if args.command == "fetch-artifacts":
        return _run_fetch_artifacts(
            base_url=args.base_url,
            target_dir=args.target_dir,
            manifest=args.manifest,
            force=args.force,
            pretty=args.pretty,
        )
    if args.command == "agent-chat":
        interactive(model_name=args.model)
        return 0
    if args.command == "agent-eval":
        evaluate(
            csv_path=args.csv,
            max_samples=args.max_samples,
            model_name=args.model,
        )
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
