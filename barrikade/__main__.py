"""CLI entry point for SDK setup utilities."""

from __future__ import annotations

import argparse
import json

from core.artifacts import download_runtime_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="barrikade")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-artifacts",
        help="Download the runtime artifact bundle used by the SDK.",
    )
    download_parser.add_argument(
        "--bucket",
        default=None,
        help="Override the public GCS bucket name.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download artifacts even if local copies already exist.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-artifacts":
        summary = download_runtime_artifacts(bucket_name=args.bucket, force=args.force)
        print(json.dumps(summary, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
