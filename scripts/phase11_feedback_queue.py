"""Local summary and private export commands for the Phase 11.0E feedback queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from Backend.rag_system.feedback import FeedbackStore, default_feedback_database_path


def _private_root() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation_data" / "private"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize or privately export user feedback records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary",
        help="Print aggregate counts only; no question or answer text is emitted.",
    )
    summary.add_argument("--db", type=Path, default=default_feedback_database_path())

    export = subparsers.add_parser(
        "export",
        help="Export the review queue as JSONL under evaluation_data/private/.",
    )
    export.add_argument("--db", type=Path, default=default_feedback_database_path())
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = FeedbackStore(args.db)

    if args.command == "summary":
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        return 0

    private_root = _private_root().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(private_root)
    except ValueError:
        print(
            "Refusing to export feedback outside evaluation_data/private/.",
            file=sys.stderr,
        )
        return 2

    if output.exists():
        print(f"Refusing to overwrite existing feedback export: {output}", file=sys.stderr)
        return 2

    records = store.list_records()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(
        json.dumps(
            {"exported_records": len(records), "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
