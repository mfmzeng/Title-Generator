# Chapter Title Generator

**Michael Zeng, Andrew Wang, Lakshana Kathirkamaranjan, Peri Gandhi**

Supervised BERT Encoder–Decoder (seq2seq) system that suggests creative chapter titles from narrative text.

## Problem

Authors often struggle to name chapters in a way that fits tone, genre, and events. This tool reads a passage and proposes one or more title candidates so writers can stay in flow.

## Approach

We fine-tune a **BERT Encoder–Decoder** (`EncoderDecoderModel` with `google-bert/bert-base-uncased`) using Hugging Face `Seq2SeqTrainer`:

| Component | Role |
|-----------|------|
| **Encoder** | Reads instruction-tuned chapter text → content representation |
| **Decoder** | Generates the title token-by-token (beam search at inference) |

**Instruction format**

```
Input:  generate chapter title: <narrative text>
Output: <title>
```

At inference, beam search returns **multiple title candidates** (`NUM_CANDIDATES` in `config.py`).

## Dataset

[textminr/cmu-book-summaries](https://huggingface.co/datasets/textminr/cmu-book-summaries) — ~16.6k book plot summaries paired with book titles (`summary` → `title`). Summaries are chunked to `CHUNK_CHARS` before encoding.

## Safety

- `better-profanity` + regex blocklist scan on training pairs (`safety.py`)
- Blocked-word logits processor during decoder generation

## Setup

```powershell
cd C:\Users\joe\Desktop\llm
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`bert-base` needs **~6 GB+ VRAM** (batch size 1 + gradient accumulation). On a 4 GB GPU, set `ENCODER_MODEL` / `DECODER_MODEL` to `prajjwal1/bert-tiny` in `config.py`.

## Usage

```powershell
# 1. Download CMU Book Summaries → data/seq2seq_train.jsonl, seq2seq_val.jsonl
python data.py

# Fast smoke test (~500 rows)
python data.py --quick

# 2. Fine-tune BERT encoder–decoder
python train.py

# 3. Generate title candidates
python generate_title.py --text "Your chapter passage here..."
python generate_title.py --chapter-file my_chapter.txt
python generate_title.py --eval
```

Checkpoints are saved to `checkpoints/bert2bert-titles/best/`.

## Project layout

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters, dataset caps, blocked words |
| `data.py` | Dataset loading, profanity filter, JSONL export |
| `safety.py` | Profanity / blocked-word filters |
| `train.py` | BERT seq2seq fine-tuning |
| `generate_title.py` | Inference with multiple candidates |
