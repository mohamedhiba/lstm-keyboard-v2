# LSTM Keyboard v2 (Next-Word Prediction)

A from-scratch (self-written) **word-level** LSTM language model that predicts the next word given a text prefix.

What you can do right now:
- Train an LSTM LM on WikiText-2 text files
- Evaluate **loss / perplexity / top-1 / top-5**
- Run a CLI next-word predictor (`infer.py`) + multi-token generation

> Note: this is intentionally built without copy-pasting other implementations.

---

## Project structure

```
.
├── src/
│   ├── data.py      # vocab + encoding + contiguous LM batching
│   ├── model.py     # Embedding -> LSTM -> Linear
│   ├── train.py     # training loop + checkpointing
│   ├── eval.py      # eval loss/ppl/topk
│   └── infer.py     # top-k next word + generation
├── tests/
│   └── test_shapes.py
├── data/
│   └── raw/         # train.txt valid.txt test.txt (not committed)
├── checkpoints/     # best.pt (not committed)
└── requirements.txt
```

---

## Milestone checklist

### Milestone 0 — Repo setup
- [DONE] Create repo + venv
- [DONE] Add folder structure: `src/`, `tests/`
- [DONE] Add `requirements.txt`
- [DONE] First commit + push

---

### Milestone 1 — Data pipeline (`src/data.py`)
- [DONE] Load dataset text (train/valid/test)
- [DONE] Tokenize (word-level)
- [DONE] Build vocab from **train only**
- [DONE] Encode/decode
- [DONE] Batch builder producing `(x, y)` with shapes `(B, T)`
- [DONE] Sanity prints: vocab size, sample decode, batch shapes, shift check

Run:
```bash
python -m src.data
```

---

### Milestone 2 — Model forward pass (`src/model.py`)
- [DONE] Implement `Embedding -> LSTM -> Linear`
- [DONE] Output logits shape `(B, T, V)`
- [DONE] Add `tests/test_shapes.py` with assertions

Run:
```bash
pytest -q
```

---

### Milestone 3 — Training loop (`src/train.py`)
- [DONE] Cross-entropy loss over `(B*T, V)`
- [DONE] Optimizer (Adam), gradient clipping, checkpoint saving
- [DONE] Logging loss every N steps
- [DONE] Overfit-1-batch test (prove loss drops a lot)

Overfit-one-batch sanity:
```bash
python src/train.py --overfit-one-batch --steps 300 --lr 1e-2
```

Full training:
```bash
python src/train.py --epochs 10 --lr 1e-3 --log-every 100
```

---

### Milestone 4 — Evaluation + inference (`src/eval.py`, `src/infer.py`)
- [DONE] Eval loss / perplexity
- [DONE] Eval top-1 and top-5
- [DONE] CLI next-word suggestions (top-k)
- [DONE] CLI generation with sampling + quality controls

Eval:
```bash
python src/eval.py --ckpt checkpoints/best.pt --split valid
python src/eval.py --ckpt checkpoints/best.pt --split test
```

Next-word (top-k):
```bash
python src/infer.py --ckpt checkpoints/best.pt --text "I want to" --k 10
```

Generation (recommended settings):
```bash
python src/infer.py --ckpt checkpoints/best.pt --text "The people that" \
  --generate 60 --strategy sample --temperature 0.9 --topk 50 --seed 7 \
  --ban-unk --ban-tokens '\",@-@' --repeat-penalty 1.2 --repeat-window 80 --stop-eos
```

---

### Milestone 5 — Export + API demo (`src/export.py`, `api/`)
- [ ] Export model (TorchScript or `torch.save` bundle)
- [ ] FastAPI endpoint: `POST /predict` with `{ "text": "...", "k": 5 }`

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

This repo expects **plain text files** at:

```
data/raw/train.txt
data/raw/valid.txt
data/raw/test.txt
```

WikiText-2 raw is the default target. You can put any text there as long as the splits exist.

> Important: `data/` should NOT be committed to GitHub.

---

## Current results (WikiText-2, word-level)

Checkpoint: `checkpoints/best.pt`

| Split | Loss  | PPL   | Top-1 | Top-5 |
|------:|:-----:|:-----:|:-----:|:-----:|
| valid | 5.5729 | 263.20 | 20.09% | 38.78% |
| test  | 5.5241 | 250.66 | 20.24% | 39.37% |

---

## GitHub / what not to push

Make sure your `.gitignore` includes at least:

- `data/`
- `checkpoints/`
- `.venv/`

So you only push code, tests, and docs.

---

## Notes

- This is a **baseline** LSTM LM. Output quality improves a lot with subword tokenization (BPE) and/or stronger models.
- Next planned upgrade: **Milestone 5 (incremental decoding with hidden state)** for faster, keyboard-like suggestions.

