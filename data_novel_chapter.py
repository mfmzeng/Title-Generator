"""
Load Novel Chapter Dataset for BERT seq2seq title generation.

Dataset: manestay/novel-chapter-dataset (GitHub)
  https://github.com/manestay/novel-chapter-dataset

Study-guide chapter summaries paired with Project Gutenberg chapter text
(Ladhak et al., ACL 2020). Summary pickle files are not hosted on GitHub;
by default this loader uses kmfoda/booksum on Hugging Face (BookSum,
Kryściński et al., 2021), which packages the same chapter-summary pairs.

For fully local loading, run make_data_splits.py in the upstream repo and pass
--pickle-dir pointing at the raw_splits/ folder.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import config as cfg
from data import _chunk_text, _format_input, dedupe_rows, export_jsonl
from safety import is_safe_pair

BOOKSUM_DATASET_ID = "kmfoda/booksum"
NOVEL_CHAPTER_REPO = "https://github.com/manestay/novel-chapter-dataset"
RAW_TEXTS_URL = (
    "https://github.com/manestay/novel-chapter-dataset/raw/main/pks/raw_texts.pk"
)
MIN_TEXT_CHARS = 100


def _section_title_ok(title: str) -> bool:
    title = title.strip()
    return 2 <= len(title) <= 120


def _summary_to_text(summary: object) -> str:
    if isinstance(summary, list):
        parts: list[str] = []
        for item in summary:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, list):
                parts.extend(str(x) for x in item)
        return " ".join(parts).strip()
    return str(summary or "").strip()


def _raw_text_to_str(raw_text: object) -> str:
    if isinstance(raw_text, list):
        return " ".join(str(x) for x in raw_text).strip()
    return str(raw_text or "").strip()


def _section_title_from_id(sect_id: str) -> str:
    return sect_id.rsplit(".", 1)[-1].strip()


def _append_row(
    rows: list[dict],
    narrative: str,
    title: str,
    source: str,
    max_rows: int | None,
) -> bool:
    narrative = narrative.strip()
    title = title.strip()
    if len(narrative) < MIN_TEXT_CHARS or not _section_title_ok(title):
        return False
    chunk = _chunk_text(narrative)
    if not is_safe_pair(chunk, title):
        return False
    rows.append(
        {
            "text": _format_input(narrative),
            "title": title,
            "source": source,
        }
    )
    if max_rows and len(rows) >= max_rows:
        return True
    return False


def load_from_booksum(
    max_rows: int | None = None,
    *,
    use_summary: bool = True,
) -> list[dict]:
    from datasets import load_dataset

    print(f"  loading {BOOKSUM_DATASET_ID} (Novel Chapter / BookSum)...", flush=True)
    ds = load_dataset(BOOKSUM_DATASET_ID, split="train")
    rows: list[dict] = []

    for i, row in enumerate(ds, 1):
        title = (row.get("summary_name") or "").strip()
        if use_summary:
            narrative = (row.get("summary_text") or "").strip()
        else:
            narrative = (row.get("chapter") or "").strip()
        if _append_row(rows, narrative, title, "novel_chapter_booksum", max_rows):
            break
        if i % 2000 == 0:
            print(f"    processed {i:,}, kept {len(rows):,}", flush=True)

    return rows


def _load_pickle(path: Path):
    import pickle

    try:
        import dill  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Pickle loading requires dill. Install with: pip install dill"
        ) from exc

    import dill

    with path.open("rb") as f:
        return dill.load(f)


def load_from_pickles(
    pickle_dir: Path,
    max_rows: int | None = None,
    *,
    use_summary: bool = True,
) -> list[dict]:
    pickle_dir = pickle_dir.resolve()
    pk_files = sorted(pickle_dir.glob("*.pk"))
    if not pk_files:
        raise SystemExit(f"No .pk files found in {pickle_dir}")

    rows: list[dict] = []
    for pk_path in pk_files:
        print(f"  loading {pk_path.name}...", flush=True)
        split_rows = _load_pickle(pk_path)
        for item in split_rows:
            sect_id = (item.get("id") or "").strip()
            title = _section_title_from_id(sect_id) if sect_id else ""
            if use_summary:
                summaries = item.get("summaries") or []
                if not summaries:
                    continue
                narrative = _summary_to_text(summaries[0].get("summary"))
            else:
                narrative = _raw_text_to_str(item.get("raw_text"))
            if _append_row(rows, narrative, title, "novel_chapter_pickle", max_rows):
                return rows

    return rows


def download_raw_texts_pk(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  downloading raw_texts.pk from {NOVEL_CHAPTER_REPO}...", flush=True)
    urllib.request.urlretrieve(RAW_TEXTS_URL, dest)
    return dest


def load_from_raw_texts(
    raw_texts_path: Path | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    root = Path(__file__).parent
    raw_texts_path = raw_texts_path or root / "data" / "novel-chapter" / "raw_texts.pk"
    download_raw_texts_pk(raw_texts_path)
    raw_texts = _load_pickle(raw_texts_path)
    rows: list[dict] = []

    for book_title, chapters in raw_texts.items():
        for chapter_name, chapter_lines in chapters.items():
            narrative = _raw_text_to_str(chapter_lines)
            title = chapter_name.strip()
            if _append_row(rows, narrative, title, "novel_chapter_gutenberg", max_rows):
                return rows

    return rows


def load_novel_chapter_dataset(
    max_rows: int | None = None,
    *,
    pickle_dir: Path | None = None,
    use_summary: bool = True,
    raw_texts_only: bool = False,
) -> list[dict]:
    if raw_texts_only:
        return load_from_raw_texts(max_rows=max_rows)
    if pickle_dir is not None:
        return load_from_pickles(pickle_dir, max_rows, use_summary=use_summary)
    return load_from_booksum(max_rows, use_summary=use_summary)


def build_dataset(
    root: Path | None = None,
    *,
    pickle_dir: Path | None = None,
    use_summary: bool = True,
    raw_texts_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    root = root or Path(__file__).parent

    print("Loading Novel Chapter Dataset...")
    all_rows = load_novel_chapter_dataset(
        cfg.MAX_ROWS,
        pickle_dir=pickle_dir,
        use_summary=use_summary,
        raw_texts_only=raw_texts_only,
    )
    print(f"  {len(all_rows):,} rows after filtering")

    before = len(all_rows)
    all_rows = dedupe_rows(all_rows)
    if before != len(all_rows):
        print(f"  deduplicated {before:,} -> {len(all_rows):,} rows")

    if not all_rows:
        raise SystemExit("No rows loaded from Novel Chapter Dataset")

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


def prepare_data(
    root: Path | None = None,
    *,
    pickle_dir: Path | None = None,
    use_summary: bool = True,
    raw_texts_only: bool = False,
) -> tuple[Path, Path]:
    root = root or Path(__file__).parent
    train_rows, val_rows = build_dataset(
        root,
        pickle_dir=pickle_dir,
        use_summary=use_summary,
        raw_texts_only=raw_texts_only,
    )
    train_path = root / "data" / "seq2seq_train_novel_chapter.jsonl"
    val_path = root / "data" / "seq2seq_val_novel_chapter.jsonl"
    export_jsonl(train_rows, train_path)
    export_jsonl(val_rows, val_path)
    print(f"Wrote {len(train_rows):,} train -> {train_path}")
    print(f"Wrote {len(val_rows):,} val   -> {val_path}")
    return train_path, val_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare Novel Chapter Dataset (manestay/novel-chapter-dataset)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only 500 rows for a fast smoke test",
    )
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        default=None,
        help="Path to manestay raw_splits/ folder with train.pk / val.pk / test.pk",
    )
    parser.add_argument(
        "--chapter-text",
        action="store_true",
        help="Use full chapter text as input instead of study-guide summaries",
    )
    parser.add_argument(
        "--raw-texts-only",
        action="store_true",
        help="Use only Project Gutenberg chapter text from raw_texts.pk (no summaries)",
    )
    args = parser.parse_args()
    if args.quick:
        cfg.MAX_ROWS = 500
        cfg.MAX_TRAIN_SAMPLES = 450
        cfg.MAX_VAL_SAMPLES = 50
    prepare_data(
        pickle_dir=args.pickle_dir,
        use_summary=not args.chapter_text,
        raw_texts_only=args.raw_texts_only,
    )
