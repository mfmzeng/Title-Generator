"""Shared helpers for dataset loaders and inference."""

from __future__ import annotations

import json
from pathlib import Path

import config as cfg


def chunk_text(text: str, max_chars: int | None = None) -> str:
    max_chars = max_chars if max_chars is not None else cfg.CHUNK_CHARS
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars]


def format_input(text: str) -> str:
    return cfg.INPUT_PREFIX + chunk_text(text)


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


def export_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
