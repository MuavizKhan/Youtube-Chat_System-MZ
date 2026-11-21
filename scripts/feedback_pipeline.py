"""Local operator commands for the Phase 11 feedback pipeline.

Examples:
  python scripts/feedback_pipeline.py summary
  python scripts/feedback_pipeline.py review --feedback-id ID --status verified_issue --reviewer-id R01
  python scripts/feedback_pipeline.py export-candidates --output evaluation_data/private/feedback_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.rag_system import feedback_store  # noqa: E402


def _configure_database(database: str | None) -> None:
    if database:
        path = Path(database).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        feedback_store.FEEDBACK_DB_PATH = path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, review and export answer feedback locally."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary",
        help="Show vote, reason, review-state and affected-video counts.",
    )
    summary.add_argument("--db", help="Override the feedback SQLite database path.")

    list_command = subparsers.add_parser(
        "list",
        help="List feedback records for local triage.",
    )
    list_command.add_argument("--db", help="Override the feedback SQLite database path.")
    list_command.add_argument(
        "--status",
        nargs="*",
        choices=feedback_store.REVIEW_STATUSES,
        default=["new"],
        help="Review states to include (default: new).",
    )
    list_command.add_argument("--limit", type=int, default=50)
    list_command.add_argument(
        "--include-context",
        action="store_true",
        help="Include question/answer/source context that users explicitly shared.",
    )

    review = subparsers.add_parser(
        "review",
        help="Record a human triage outcome for one feedback item.",
    )
    review.add_argument("--db", help="Override the feedback SQLite database path.")
    review.add_argument("--feedback-id", required=True)
    review.add_argument(
        "--status",
        required=True,
        choices=tuple(
            value for value in feedback_store.REVIEW_STATUSES if value != "new"
        ),
    )
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--notes", default=None)

    export = subparsers.add_parser(
        "export-candidates",
        help="Export human-approved, context-consented evaluation candidates.",
    )
    export.add_argument("--db", help="Override the feedback SQLite database path.")
    export.add_argument(
        "--output",
        required=True,
        help="New JSONL file under evaluation_data/private/ (must not already exist).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_database(args.db)

    try:
        if args.command == "summary":
            print(json.dumps(feedback_store.summarize_feedback(), indent=2))
            return 0

        if args.command == "list":
            if args.limit < 1 or args.limit > 500:
                raise feedback_store.FeedbackStoreError(
                    "--limit must be between 1 and 500."
                )
            records = feedback_store.list_feedback_records(
                statuses=tuple(args.status) if args.status else None
            )[:args.limit]
            if not args.include_context:
                for item in records:
                    item.pop("context", None)
            print(json.dumps(records, indent=2, ensure_ascii=False))
            return 0

        if args.command == "review":
            record = feedback_store.update_feedback_review(
                feedback_id=args.feedback_id,
                status=args.status,
                reviewer_id=args.reviewer_id,
                notes=args.notes,
            )
            print(
                json.dumps(
                    {
                        "feedback_id": record["feedback_id"],
                        "review_status": record["review_status"],
                        "reviewer_id": record["reviewer_id"],
                        "reviewed_at": record["reviewed_at"],
                        "is_gold_label": False,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "export-candidates":
            count = feedback_store.export_evaluation_candidates(args.output)
            print(
                json.dumps(
                    {
                        "exported_candidates": count,
                        "output": str(Path(args.output)),
                        "is_phase11_0C_dataset": False,
                        "requires_manual_source_verification": True,
                    },
                    indent=2,
                )
            )
            return 0

        raise RuntimeError(f"Unsupported command: {args.command}")
    except (feedback_store.FeedbackStoreError, OSError) as error:
        print(f"Feedback pipeline error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
