"""Download ChatDoctor-HealthCareMagic-100k and save as data/chatdoctor.json."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
DEFAULT_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = Path("data/chatdoctor.json")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ChatDoctor from Hugging Face to data/chatdoctor.json.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Split to download (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap before saving.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON.",
    )
    return parser.parse_args()
def import_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets\n\n"
            "Install it with:\n"
            "    python -m pip install datasets\n\n"
            "Then run again:\n"
            "    python download_chatdoctor.py"
        ) from exc
    return load_dataset
def normalise_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "")).strip(),
        "output": str(row.get("output", "")).strip(),
    }
def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer when provided.")
    load_dataset = import_load_dataset()
    print(f"Downloading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    rows = [normalise_row(dict(row)) for row in dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.compact:
            json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows):,} rows to {args.output}")
    print("The project loader will now use this real dataset instead of mock data.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
