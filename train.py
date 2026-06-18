"""
Fine-tune BERT Encoder–Decoder for chapter title generation.

  python data.py    # download & merge datasets
  python train.py   # fine-tune bert2bert
"""

from __future__ import annotations

import datasets  # noqa: F401 — must import before torch on Windows (pyarrow/CUDA)

import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DECODER_MODEL,
    ENCODER_MODEL,
    FP16,
    GRAD_ACCUM_STEPS,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def preprocess_rows(rows: list[dict], tokenizer, max_input: int, max_target: int):
    inputs = [r["text"] for r in rows]
    targets = [r["title"] for r in rows]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class TitleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def load_tokenizer():
    return AutoTokenizer.from_pretrained(ENCODER_MODEL)


def build_model(tokenizer) -> EncoderDecoderModel:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ENCODER_MODEL,
        DECODER_MODEL,
    )
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.num_beams = 4

    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def main():
    root = Path(__file__).parent
    train_path = root / "data" / "seq2seq_train.jsonl"
    val_path = root / "data" / "seq2seq_val.jsonl"

    if not train_path.exists():
        print("Preparing datasets...")
        from data import prepare_data

        prepare_data(root)

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    if not train_rows:
        raise SystemExit("No training data. Run: python data.py")

    if torch.cuda.is_available():
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu (CUDA not found — install GPU PyTorch)")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")

    print(f"Train: {len(train_rows):,}  Val: {len(val_rows):,}")

    tokenizer = load_tokenizer()
    model = build_model(tokenizer)

    train_enc = preprocess_rows(train_rows, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_enc = preprocess_rows(val_rows, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    train_ds = TitleDataset(train_enc)
    val_ds = TitleDataset(val_enc)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = FP16 and torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(root / CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting fine-tune...")
    trainer.train()
    best_dir = root / CHECKPOINT_DIR / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved to {best_dir}")


if __name__ == "__main__":
    main()
