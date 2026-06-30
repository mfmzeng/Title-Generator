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

At inference, beam search returns **multiple title candidates**.

## Datasets

All three sources are loaded and merged in `titleGenerator.ipynb`:

| Source | Mapping | Approx. size |
|--------|---------|--------------|
| [CMU Book Summaries](https://huggingface.co/datasets/textminr/cmu-book-summaries) | plot summary → book title | ~16.6k |
| [Novel Chapter / BookSum](https://github.com/manestay/novel-chapter-dataset) | chapter summary → section title | ~9.6k |
| [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | short story → theme | 20k cap |

Hybrid JSONL is written to `data/seq2seq_train_hybrid.jsonl` and `data/seq2seq_val_hybrid.jsonl` after the train/val split cell.

## Safety

- `better-profanity` + regex blocklist on training pairs (inlined in notebook)
- Blocked-word logits processor during decoder generation

## Setup

```powershell
cd C:\Users\joe\Desktop\Title-Generator
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with `HF_TOKEN=your_huggingface_token` for dataset downloads.

`bert-base` needs **~6 GB+ VRAM** (batch size 1 + gradient accumulation). On a 4 GB GPU, set `cfg.ENCODER_MODEL` / `cfg.DECODER_MODEL` to `prajjwal1/bert-tiny` in the notebook Imports cell.

## Usage

Open `titleGenerator.ipynb` in Jupyter or VS Code/Cursor and select **Run All**.

The notebook is self-contained: configuration, dataset loaders, preprocessing, training, evaluation, and safety filtering are all inlined. No separate Python scripts are required.

Checkpoints are saved to `checkpoints/bert2bert-titles/best/`.

## Project layout

| File | Purpose |
|------|---------|
| `titleGenerator.ipynb` | Full pipeline + report (Run All) |
| `requirements.txt` | Python dependencies |
| `data/` | Hybrid train/val JSONL (generated; gitignored) |
| `checkpoints/` | Fine-tuned model weights (gitignored) |
