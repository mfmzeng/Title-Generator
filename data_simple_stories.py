"""
Load SimpleStories for BERT seq2seq title generation.

Dataset: SimpleStories/SimpleStories (Hugging Face)
  https://huggingface.co/datasets/SimpleStories/SimpleStories

Model-generated short stories annotated with theme, topic, style, etc.
Uses story text as input and theme label as the target title for learning
key ideas from short narrative text.
"""

from __future__ import annotations

import random
from pathlib import Path

import config as cfg
from data import _chunk_text, _format_input, dedupe_rows, export_jsonl
from safety import is_safe_pair

DATASET_ID = "SimpleStories/SimpleStories"
DEFAULT_MAX_ROWS = 20_000  # cap streaming scan (full dataset is ~2.1M)
MIN_STORY_CHARS = 80
MIN_WORD_COUNT = 40


def _theme_title_ok(title: str) -> bool:
    title = title.strip()
    return 2 <= len(title) <= 80


def load_simple_stories(max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    print(f"  loading {DATASET_ID}...", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    rows: list[dict] = []

    for i, row in enumerate(ds, 1):
        story = (row.get("story") or "").strip()
        theme = (row.get("theme") or "").strip()
        word_count = row.get("word_count") or 0

        if len(story) < MIN_STORY_CHARS or word_count < MIN_WORD_COUNT:
            continue
        if not _theme_title_ok(theme):
            continue

        chunk = _chunk_text(story)
        if not is_safe_pair(chunk, theme):
            continue

        rows.append(
            {
                "text": _format_input(story),
                "title": theme,
                "source": "simple_stories",
                "topic": (row.get("topic") or "").strip(),
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
        if i % 5000 == 0:
            print(f"    scanned {i:,}, kept {len(rows):,}", flush=True)

    return rows


def build_dataset(root: Path | None = None) -> tuple[list[dict], list[dict]]:
    root = root or Path(__file__).parent

    max_rows = cfg.MAX_ROWS if cfg.MAX_ROWS is not None else DEFAULT_MAX_ROWS
    print(f"Loading SimpleStories (cap {max_rows:,} rows)...")
    all_rows = load_simple_stories(max_rows)
    print(f"  {len(all_rows):,} rows after filtering")

    before = len(all_rows)
    all_rows = dedupe_rows(all_rows)
    if before != len(all_rows):
        print(f"  deduplicated {before:,} -> {len(all_rows):,} rows")

    if not all_rows:
        raise SystemExit(f"No rows loaded from {DATASET_ID}")

    random.seed(cfg.RANDOM_SEED)
    random.shuffle(all_rows)

    val_size = max(1, int(len(all_rows) * cfg.VAL_FRACTION))
    if cfg.MAX_VAL_SAMPLES:
        val_size = min(val_size, cfg.MAX_VAL_SAMPLES)
    val_rows = all_rows[:val_size]
    train_rows = all_rows[val_size:]

    if cfg.MAX_TRAIN_SAMPLES:
        train_rows = train_rows[: cfg.MAX_TRAIN_SAMPLES]

    return train_rows, val_rows


def prepare_data(root: Path | None = None) -> tuple[Path, Path]:
    root = root or Path(__file__).parent
    train_rows, val_rows = build_dataset(root)
    train_path = root / "data" / "seq2seq_train_simple_stories.jsonl"
    val_path = root / "data" / "seq2seq_val_simple_stories.jsonl"
    export_jsonl(train_rows, train_path)
    export_jsonl(val_rows, val_path)
    print(f"Wrote {len(train_rows):,} train -> {train_path}")
    print(f"Wrote {len(val_rows):,} val   -> {val_path}")
    return train_path, val_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare SimpleStories (SimpleStories/SimpleStories on Hugging Face)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Stop after this many usable rows (default: {DEFAULT_MAX_ROWS:,})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only 500 rows for a fast smoke test",
    )
    args = parser.parse_args()
    if args.quick:
        cfg.MAX_ROWS = 500
        cfg.MAX_TRAIN_SAMPLES = 450
        cfg.MAX_VAL_SAMPLES = 50
    else:
        cfg.MAX_ROWS = args.max_rows
        cfg.MAX_TRAIN_SAMPLES = None
        cfg.MAX_VAL_SAMPLES = None
    prepare_data()
