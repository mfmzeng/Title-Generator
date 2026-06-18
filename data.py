"""
Load CMU Book Summaries for BERT seq2seq title generation.

Dataset: textminr/cmu-book-summaries
  summary (plot text) → title (book title)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import config as cfg
from safety import is_safe_pair

PLAY_WORDS = ("scene", "act", "prologue", "epilogue")
GENERIC_CHAPTER = re.compile(r"^chapters?\s+(\d+|[ivxlcdm]+)$", re.I)
MIN_SUMMARY_CHARS = 100


def _chunk_text(text: str, max_chars: int | None = None) -> str:
    max_chars = max_chars if max_chars is not None else cfg.CHUNK_CHARS
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars]


def _title_ok(title: str) -> bool:
    title = title.strip()
    if len(title) < 2 or len(title) > 120:
        return False
    lower = title.lower()
    if any(w in lower for w in PLAY_WORDS):
        return False
    if GENERIC_CHAPTER.match(lower):
        return False
    if lower.startswith("chapter "):
        return False
    return True


def _format_input(text: str) -> str:
    return cfg.INPUT_PREFIX + _chunk_text(text)


def _row_key(row: dict) -> tuple[str, str]:
    return (row["text"].strip().lower(), row["title"].strip().lower())


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_cmu_book_summaries(max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    print(f"  loading {cfg.DATASET_ID}...", flush=True)
    ds = load_dataset(cfg.DATASET_ID, split="train")
    rows: list[dict] = []

    for i, row in enumerate(ds, 1):
        title = (row.get("title") or "").strip()
        summary = (row.get("summary") or "").strip()
        if len(summary) < MIN_SUMMARY_CHARS or not _title_ok(title):
            continue
        chunk = _chunk_text(summary)
        if not is_safe_pair(chunk, title):
            continue
        rows.append(
            {
                "text": _format_input(summary),
                "title": title,
                "source": "cmu_book_summaries",
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
        if i % 2000 == 0:
            print(f"    processed {i:,}, kept {len(rows):,}", flush=True)

    return rows


def build_dataset(root: Path | None = None) -> tuple[list[dict], list[dict]]:
    root = root or Path(__file__).parent

    print("Loading CMU Book Summaries...")
    all_rows = load_cmu_book_summaries(cfg.MAX_ROWS)
    print(f"  {len(all_rows):,} rows after filtering")

    before = len(all_rows)
    all_rows = dedupe_rows(all_rows)
    if before != len(all_rows):
        print(f"  deduplicated {before:,} -> {len(all_rows):,} rows")

    if not all_rows:
        raise SystemExit(f"No rows loaded from {cfg.DATASET_ID}")

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


def export_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_data(root: Path | None = None) -> tuple[Path, Path]:
    root = root or Path(__file__).parent
    train_rows, val_rows = build_dataset(root)
    train_path = root / "data" / "seq2seq_train.jsonl"
    val_path = root / "data" / "seq2seq_val.jsonl"
    export_jsonl(train_rows, train_path)
    export_jsonl(val_rows, val_path)
    print(f"Wrote {len(train_rows):,} train -> {train_path}")
    print(f"Wrote {len(val_rows):,} val   -> {val_path}")
    return train_path, val_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    prepare_data()
