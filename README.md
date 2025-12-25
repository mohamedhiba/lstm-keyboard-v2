
# LSTM Keyboard v2 (Next-Word Prediction)

A from-scratch (self-written) LSTM language model that predicts the next word given a text prefix.  
Includes training, evaluation (top-k accuracy / perplexity), export, and a small FastAPI demo.

## Milestone Checklist

### Milestone 0 — Repo setup
- [DONE] Create repo + venv
- [DONE] Add folder structure: `src/`, `api/`, `tests/`
- [DONE] Add `requirements.txt`
- [ ] First commit + push

**Done when:** imports work and repo runs basic commands.

---

### Milestone 1 — Data pipeline (`src/data.py`)
- [ ] Download/load dataset (default: WikiText-2)
- [ ] Build word-level vocabulary + UNK token
- [ ] Encode/decode functions
- [ ] Batch builder producing `(x, y)` with shapes `(B, T)`
- [ ] Sanity prints: vocab size, sample decode, batch shapes

**Done when:** `python -m src.data` prints correct shapes and readable decoded samples.

---

### Milestone 2 — Model forward pass (`src/model.py`)
- [ ] Implement `Embedding -> LSTM -> Linear`
- [ ] Output logits shape `(B, T, V)`
- [ ] Add `tests/test_shapes.py` with assertions

**Done when:** `pytest` passes and forward pass runs.

---

### Milestone 3 — Training loop (`src/train.py`)
- [ ] Cross-entropy loss over `(B*T, V)`
- [ ] Optimizer (Adam), gradient clipping, checkpoint saving
- [ ] Logging loss every N steps
- [ ] Overfit-1-batch test (prove loss drops a lot)

**Done when:** you can train an epoch and produce a checkpoint; overfit test succeeds.

---

### Milestone 4 — Evaluation + inference (`src/eval.py`, `src/infer.py`)
- [ ] Evaluate top-1 accuracy
- [ ] Evaluate top-5 accuracy
- [ ] Perplexity (recommended)
- [ ] `predict_next(prefix, k)` returns top-k suggestions

**Done when:** evaluation prints metrics and inference returns sensible suggestions.

---

### Milestone 5 — Export + API demo (`src/export.py`, `api/main.py`)
- [ ] Export model (TorchScript preferred)
- [ ] FastAPI endpoints:
  - [ ] `GET /health`
  - [ ] `POST /predict` with `{ "text": "...", "k": 5 }`
- [ ] README: clean run instructions + example outputs

**Done when:** API runs via `uvicorn` and `curl` returns suggestions.

---

## Quickstart

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Data sanity check

```bash
python -m src.data
```

### 3) Train

```bash
python -m src.train --epochs 1 --batch-size 64 --seq-len 30
```

### 4) Evaluate

```bash
python -m src.eval --ckpt checkpoints/model.pt
```

### 5) Inference (CLI)

```bash
python -m src.infer --text "i want to" --k 5 --ckpt checkpoints/model.pt
```

### 6) Run API

```bash
uvicorn api.main:app --reload
```

Test it:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"i want to","k":5}'
```

---

## Configuration Defaults (recommended)

* Dataset: WikiText-2
* Vocab: word-level (+ `<UNK>`)
* `seq_len=30`, `batch_size=64`
* Embedding dim: 256
* LSTM hidden: 512
* LSTM layers: 2
* Dropout: 0.2
* Optimizer: Adam, lr = 1e-3
* Grad clip: 1.0

---

## Results (fill in after you run)

| Model      | Params | Top-1 | Top-5 | Perplexity |
| ---------- | ------ | ----- | ----- | ---------- |
| LSTM-2x512 | TBD    | TBD   | TBD   | TBD        |

---

## Notes

This project is intentionally built without copying external code. Only official docs are referenced.
