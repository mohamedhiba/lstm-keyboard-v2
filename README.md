# LSTM Keyboard v2 (Next-Word Prediction)

A from-scratch (self-written) **word-level** LSTM language model that predicts the next word given a text prefix.

✅ Built end-to-end: data pipeline → vocab/encoding → LSTM LM → training/eval → CLI inference → export → **FastAPI demo**.

---

## Live demo (Cloud Run)

- **Service:** https://lstm-keyboard-demo-743198811832.us-central1.run.app/
- **Swagger docs:** https://lstm-keyboard-demo-743198811832.us-central1.run.app/docs
- **Health:** https://lstm-keyboard-demo-743198811832.us-central1.run.app/health

**Model export release (used by the demo):**
- https://github.com/mohamedhiba/lstm-keyboard-v2/releases/tag/deployment

Quick check:
```bash
curl -s https://lstm-keyboard-demo-743198811832.us-central1.run.app/health | python -m json.tool
```

---

## Quick links

- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Download WikiText-2](#download-wikitext-2)
- [Train](#train)
- [Evaluate](#evaluate)
- [CLI inference](#cli-inference)
- [Export](#export)
- [API demo (FastAPI)](#api-demo-fastapi)
- [Deploy the demo (Cloud Run)](#deploy-the-demo-cloud-run)
- [Current results](#current-results-wikitext-2-word-level)
- [Milestone checklist](#milestone-checklist)

> This repo is intentionally built without copy-pasting other implementations.

---

## Project structure

```text
.
├── api/
│   └── main.py          # FastAPI demo (predict + generate)
├── src/
│   ├── __init__.py
│   ├── data.py          # vocab + encoding + contiguous LM batching
│   ├── model.py         # Embedding -> LSTM -> Linear (+ cached-state decoding)
│   ├── train.py         # training loop + checkpointing
│   ├── eval.py          # eval loss/ppl/topk
│   ├── infer.py         # top-k next word + generation (cached-state)
│   └── export.py        # export bundle for API/demo
├── tests/
│   ├── test_shapes.py
│   └── test_ms5_cached_decode.py
├── data/
│   └── raw/             # train.txt valid.txt test.txt (NOT committed)
├── checkpoints/         # best.pt (NOT committed)
├── exports/             # lstm_lm_export.pt (NOT committed)
└── requirements.txt
```

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Download WikiText-2

This project expects plain text files at:

```text
data/raw/train.txt
data/raw/valid.txt
data/raw/test.txt
```

Recommended way (HuggingFace datasets → write text files yourself):

```bash
python -c "from datasets import load_dataset; ds=load_dataset('wikitext','wikitext-2-raw-v1'); print(ds)"
```

Then write the splits to `data/raw/*.txt` using your own small script (kept out of the repo) or your existing pipeline.

Sanity check:
```bash
python -m src.data
```

---

## Train

Overfit-one-batch sanity (must PASS):
```bash
python src/train.py --overfit-one-batch --steps 300 --lr 1e-2
```

Full training:
```bash
python src/train.py --epochs 10 --lr 1e-3 --log-every 100
```

---

## Evaluate

```bash
python src/eval.py --ckpt checkpoints/best.pt --split valid
python src/eval.py --ckpt checkpoints/best.pt --split test
```

Metrics:
- **Loss**: average cross-entropy per token
- **PPL (perplexity)**: `exp(loss)` (lower is better)
- **Top-1**: % of time the correct next word was the model’s #1 guess
- **Top-5**: % of time the correct next word was within the model’s top 5 guesses

---

## CLI inference

Top-k next-word suggestions:
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

## Export

Export a portable bundle (weights + vocab + config) for the API/demo:

```bash
python src/export.py --ckpt checkpoints/best.pt --out exports/lstm_lm_export.pt
```

---

## API demo (FastAPI)

### Run locally

1) Export first (required):
```bash
python src/export.py --ckpt checkpoints/best.pt --out exports/lstm_lm_export.pt
```

2) Start the server:
```bash
uvicorn api.main:app --reload
```

Open:
- `http://127.0.0.1:8000/docs` (interactive Swagger UI)
- `http://127.0.0.1:8000/health`

### Endpoints

- `GET /health` → confirms model loaded
- `POST /predict` → top-k next-word suggestions
- `POST /generate` → multi-token generation

### Example: predict

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I want to","k":10}' | python -m json.tool
```

### Example: generate

```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"I want to","generate":40,"strategy":"sample","temperature":0.95,"topk":40,"seed":1,"ban_unk":true,"repeat_penalty":1.15,"repeat_window":80,"stop_eos":true}' \
| python -m json.tool
```

---

## Deploy the demo (Cloud Run)

This repo is set up so the Cloud Run service can **download the exported model at startup** (instead of committing `exports/` to git).

### Prereqs

- Google Cloud project with billing enabled
- APIs enabled: Cloud Run, Cloud Build, Artifact Registry

### Deploy

From the repo root:

```bash
gcloud run deploy lstm-keyboard-demo \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --set-env-vars "EXPORT_URL=https://github.com/mohamedhiba/lstm-keyboard-v2/releases/download/deployment/lstm_lm_export.pt,EXPORT_PATH=/tmp/lstm_lm_export.pt"
```

### Verify

```bash
curl -s https://lstm-keyboard-demo-743198811832.us-central1.run.app/health | python -m json.tool

curl -s -X POST https://lstm-keyboard-demo-743198811832.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I want to","k":10}' | python -m json.tool
```

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
- `exports/`
- `.venv/`

So you only push code, tests, and docs.

---

## Milestone checklist

### Milestone 0 — Repo setup
- [x] Create repo + venv
- [x] Add folder structure: `src/`, `tests/`
- [x] Add `requirements.txt`
- [x] First commit + push

### Milestone 1 — Data pipeline (`src/data.py`)
- [x] Load dataset text (train/valid/test)
- [x] Tokenize (word-level)
- [x] Build vocab from **train only**
- [x] Encode/decode
- [x] Batch builder producing `(x, y)` with shapes `(B, T)`
- [x] Sanity prints: vocab size, sample decode, batch shapes, shift check

### Milestone 2 — Model forward pass (`src/model.py`)
- [x] Implement `Embedding -> LSTM -> Linear`
- [x] Output logits shape `(B, T, V)`
- [x] Add tests with shape assertions

### Milestone 3 — Training loop (`src/train.py`)
- [x] Cross-entropy loss over `(B*T, V)`
- [x] Optimizer (Adam), gradient clipping, checkpoint saving
- [x] Logging loss every N steps
- [x] Overfit-1-batch test (prove loss drops a lot)

### Milestone 4 — Evaluation + inference (`src/eval.py`, `src/infer.py`)
- [x] Eval loss / perplexity
- [x] Eval top-1 and top-5
- [x] CLI next-word suggestions (top-k)
- [x] CLI generation with sampling + quality controls

### Milestone 5 — Export + API demo (`src/export.py`, `api/`)
- [x] Export model bundle (`torch.save`: weights + vocab + config)
- [x] Cached-state incremental decoding (faster generation)
- [x] FastAPI demo:
  - [x] `GET /health`
  - [x] `POST /predict` with `{ "text": "...", "k": 5 }`
  - [x] `POST /generate` for multi-token generation
  - [x] Swagger docs at `/docs`

---

## Notes

- This is a baseline LSTM LM. Output quality improves a lot with subword tokenization (BPE) and/or stronger models.
- This project intentionally focuses on understanding and building the full pipeline end-to-end.