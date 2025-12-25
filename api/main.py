from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LSTMLanguageModel


# ---------- helpers ----------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_prefix(text: str, word2idx: Dict[str, int]) -> List[int]:
    toks = [t for t in text.strip().split() if t]
    unk = word2idx.get("<unk>", 0)
    return [word2idx.get(t, unk) for t in toks]


def decode_tokens(ids: List[int], idx2word: List[str]) -> List[str]:
    out: List[str] = []
    for tid in ids:
        if 0 <= tid < len(idx2word):
            tok = idx2word[tid]
            if tok != "<eos>":
                out.append(tok)
        else:
            out.append("<bad_id>")
    return out


def join_tokens(tokens: List[str]) -> str:
    if not tokens:
        return ""
    no_space_before = {",", ".", ":", ";", "!", "?", ")", "]", "}", "'s", "'"}
    no_space_after = {"(", "[", "{"}

    s = ""
    quote_open = True
    for tok in tokens:
        if tok == '"':
            if quote_open:
                if s and s[-1] not in no_space_after and s[-1] != " ":
                    s += " "
                s += '"'
                quote_open = False
            else:
                s += '"'
                quote_open = True
            continue

        if not s:
            s = tok
        elif tok in no_space_before:
            s += tok
        elif s[-1] in no_space_after:
            s += tok
        elif s[-1] == '"' and not quote_open:
            s += tok
        else:
            s += " " + tok
    return s


# ---------- export loading ----------
def load_export(export_path: Path) -> Dict[str, Any]:
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")
    obj = torch.load(export_path, map_location="cpu")
    if obj.get("format") != "lstm-keyboard-v2-export-v1":
        raise ValueError("Not a recognized export format.")
    return obj


# ---------- FastAPI ----------
app = FastAPI(title="LSTM Keyboard v2 Demo", version="1.0")


# ---------- Root and favicon endpoints ----------
@app.get("/")
def root() -> RedirectResponse:
    # Send users to interactive docs by default
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico")
def favicon() -> Dict[str, Any]:
    # Avoid noisy 404s in the terminal
    return {"ok": True}

DEVICE = pick_device()
EXPORT_PATH = Path("exports/lstm_lm_export.pt")

MODEL: Optional[LSTMLanguageModel] = None
WORD2IDX: Optional[Dict[str, int]] = None
IDX2WORD: Optional[List[str]] = None


class PredictRequest(BaseModel):
    text: str
    k: int = 10


class PredictResponse(BaseModel):
    text: str
    k: int
    suggestions: List[Dict[str, Any]]


class GenerateRequest(BaseModel):
    text: str
    generate: int = 40
    strategy: str = "sample"  # "greedy" or "sample"
    temperature: float = 0.95
    topk: int = 40
    seed: int = 0
    ban_unk: bool = True
    repeat_penalty: float = 1.15
    repeat_window: int = 80
    stop_eos: bool = True


class GenerateResponse(BaseModel):
    text: str
    completion: str


@app.on_event("startup")
def _startup() -> None:
    global MODEL, WORD2IDX, IDX2WORD

    exp = load_export(EXPORT_PATH)
    WORD2IDX = exp["word2idx"]
    IDX2WORD = exp["idx2word"]
    V = len(IDX2WORD)

    cfg = exp.get("config", {})
    embed_dim = int(cfg.get("embed_dim", 256))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.2))

    m = LSTMLanguageModel(
        vocab_size=V,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    m.load_state_dict(exp["model_state_dict"], strict=True)
    m.eval()

    MODEL = m


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "device": str(DEVICE),
        "export_path": str(EXPORT_PATH),
        "loaded": MODEL is not None,
        "vocab": None if IDX2WORD is None else len(IDX2WORD),
    }


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest) -> PredictResponse:
    assert MODEL is not None and WORD2IDX is not None and IDX2WORD is not None

    ids = encode_prefix(req.text, WORD2IDX)
    if not ids:
        return PredictResponse(text=req.text, k=req.k, suggestions=[])

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)
    logits, _ = MODEL.forward_with_state(x)
    last = logits[0, -1, :]
    probs = torch.softmax(last, dim=-1)

    k = max(1, min(int(req.k), probs.numel()))
    top = torch.topk(probs, k=k)

    suggestions: List[Dict[str, Any]] = []
    for tid, p in zip(top.indices.tolist(), top.values.tolist()):
        tok = IDX2WORD[tid]
        suggestions.append({"token": tok, "prob": float(p), "logp": float(math.log(max(p, 1e-12)))})
    return PredictResponse(text=req.text, k=k, suggestions=suggestions)


@app.post("/generate", response_model=GenerateResponse)
@torch.no_grad()
def generate(req: GenerateRequest) -> GenerateResponse:
    assert MODEL is not None and WORD2IDX is not None and IDX2WORD is not None

    if req.seed:
        torch.manual_seed(int(req.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(req.seed))

    prefix_ids = encode_prefix(req.text, WORD2IDX)
    if not prefix_ids:
        return GenerateResponse(text=req.text, completion=req.text)

    eos_id = WORD2IDX.get("<eos>")
    unk_id = WORD2IDX.get("<unk>")

    # run prefix once
    prefix_x = torch.tensor(prefix_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logits, state = MODEL.forward_with_state(prefix_x, state=None)
    last = logits[0, -1, :]

    ids = list(prefix_ids)
    gen_ids: List[int] = []

    for _ in range(int(req.generate)):
        if req.strategy == "greedy":
            last_mod = last.clone()
            if req.ban_unk and unk_id is not None:
                last_mod[int(unk_id)] = -1e9
            # repeat penalty
            if req.repeat_penalty > 1.0:
                recent = ids[-int(req.repeat_window):]
                for t in set(int(x) for x in recent):
                    last_mod[t] = last_mod[t] - math.log(float(req.repeat_penalty))
            next_id = int(torch.argmax(last_mod).item())
        else:
            temp = max(float(req.temperature), 1e-6)
            scaled = last / temp
            probs = torch.softmax(scaled, dim=-1)

            if req.ban_unk and unk_id is not None:
                probs[int(unk_id)] = 0.0

            if req.repeat_penalty > 1.0:
                recent = ids[-int(req.repeat_window):]
                for t in set(int(x) for x in recent):
                    probs[t] = probs[t] / float(req.repeat_penalty)

            probs = probs / probs.sum()
            if req.topk and 0 < int(req.topk) < probs.numel():
                vals, idxs = torch.topk(probs, k=int(req.topk))
                vals = vals / vals.sum()
                choice = torch.multinomial(vals, 1).item()
                next_id = int(idxs[choice].item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())

        if req.stop_eos and eos_id is not None and next_id == int(eos_id):
            break

        gen_ids.append(next_id)
        ids.append(next_id)

        step_x = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
        step_logits, state = MODEL.step(step_x, state)
        last = step_logits[0, -1, :]

    full_tokens = req.text.strip().split()
    full_tokens += decode_tokens(gen_ids, IDX2WORD)
    completion = join_tokens(full_tokens)

    return GenerateResponse(text=req.text, completion=completion)