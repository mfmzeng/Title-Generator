"""
Generate chapter title candidates from narrative text.

  python generate_title.py --text "Once upon a time..."
  python generate_title.py --chapter-file chapter.txt
  python generate_title.py --eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, EncoderDecoderModel, LogitsProcessor, LogitsProcessorList

from config import (
    CHECKPOINT_DIR,
    CHUNK_CHARS,
    INPUT_PREFIX,
    MAX_GEN_LENGTH,
    MAX_INPUT_LENGTH,
    NUM_BEAMS,
    NUM_CANDIDATES,
)
from data_cmu_book_summaries import VAL_JSONL
from data_utils import chunk_text
from safety import blocked_word_token_ids


class BlockedWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, blocked_ids: set[int]):
        self.blocked_ids = blocked_ids

    def __call__(self, input_ids, scores):
        if not self.blocked_ids:
            return scores
        for idx in self.blocked_ids:
            if idx < scores.shape[-1]:
                scores[:, idx] = float("-inf")
        return scores


def load_model_and_tokenizer(root: Path, checkpoint: Path | None):
    ckpt = checkpoint or (root / CHECKPOINT_DIR / "best")
    if not ckpt.exists():
        raise SystemExit(
            f"No checkpoint at {ckpt}. Run: python data_cmu_book_summaries.py && python train.py"
        )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def format_input(text: str) -> str:
    return INPUT_PREFIX + chunk_text(text.strip(), CHUNK_CHARS)


def generate_titles(
    model,
    tokenizer,
    text: str,
    device: str,
    *,
    num_candidates: int = NUM_CANDIDATES,
    num_beams: int = NUM_BEAMS,
) -> list[str]:
    inputs = tokenizer(
        format_input(text),
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    blocked = blocked_word_token_ids(tokenizer)
    processors = LogitsProcessorList([BlockedWordsLogitsProcessor(blocked)])

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_GEN_LENGTH,
            num_beams=max(num_beams, num_candidates),
            num_return_sequences=num_candidates,
            early_stopping=True,
            logits_processor=processors,
        )

    candidates: list[str] = []
    seen: set[str] = set()
    for seq in outputs:
        title = tokenizer.decode(seq, skip_special_tokens=True).strip()
        key = title.lower()
        if title and key not in seen:
            seen.add(key)
            candidates.append(title)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Generate chapter title candidates")
    parser.add_argument("--text", type=str, help="Chapter / story text")
    parser.add_argument("--chapter-file", type=Path, help="File with chapter text")
    parser.add_argument("--eval", action="store_true", help="Evaluate on val set")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES)
    args = parser.parse_args()

    root = Path(__file__).parent
    model, tokenizer, device = load_model_and_tokenizer(root, args.checkpoint)

    if args.eval:
        val_path = root / "data" / VAL_JSONL
        if not val_path.exists():
            raise SystemExit("No validation file. Run: python data_cmu_book_summaries.py")

        exact = total = 0
        with val_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                if text.startswith(INPUT_PREFIX):
                    text = text[len(INPUT_PREFIX) :]
                gold = row["title"]
                preds = generate_titles(
                    model, tokenizer, text, device, num_candidates=1
                )
                pred = preds[0] if preds else ""
                match = pred.strip().lower() == gold.strip().lower()
                exact += int(match)
                total += 1
                mark = "OK" if match else "  "
                print(f"{mark} {gold}")
                if not match:
                    print(f"     -> {pred}")
        print(f"\nExact match: {exact}/{total}")
        return

    if args.chapter_file:
        text = args.chapter_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide --text, --chapter-file, or --eval")

    titles = generate_titles(
        model, tokenizer, text, device, num_candidates=args.candidates
    )
    print("Suggested titles:")
    for i, t in enumerate(titles, 1):
        print(f"  {i}. {t}")


if __name__ == "__main__":
    main()
