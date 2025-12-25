
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import batchify, encode_file, iter_batches
from src.model import LSTMLanguageModel

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_for_ce(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """CrossEntropyLoss expects (N, C) logits and (N,) targets."""
    B, T, V = logits.shape
    return logits.reshape(B * T, V), targets.reshape(B * T)


def compute_topk_acc(logits_flat: torch.Tensor, y_flat: torch.Tensor, k: int) -> float:
    """Top-k accuracy for flattened tensors.

    logits_flat: (N, V)
    y_flat: (N,)
    """
    # (N, k)
    topk = torch.topk(logits_flat, k=k, dim=-1).indices
    # (N, 1)
    y = y_flat.unsqueeze(1)
    correct = (topk == y).any(dim=1).float().sum().item()
    return float(correct) / max(int(y_flat.numel()), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0

    total_top1_correct = 0.0
    total_top5_correct = 0.0

    for batch_i, (x, y) in enumerate(iter_batches(data, seq_len=seq_len)):
        if max_batches is not None and batch_i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B, T, V)
        logits_flat, y_flat = flatten_for_ce(logits, y)

        loss = criterion(logits_flat, y_flat)

        # weight by token count
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens

        # top-1
        preds = torch.argmax(logits_flat, dim=-1)
        total_top1_correct += float((preds == y_flat).float().sum().item())

        # top-5
        total_top5_correct += float(
            (torch.topk(logits_flat, k=5, dim=-1).indices == y_flat.unsqueeze(1)).any(dim=1).float().sum().item()
        )

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    top1 = total_top1_correct / max(total_tokens, 1)
    top5 = total_top5_correct / max(total_tokens, 1)

    return {
        "loss": float(avg_loss),
        "ppl": float(ppl),
        "top1": float(top1),
        "top5": float(top5),
        "tokens": float(total_tokens),
    }


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location=device)

    # Be robust to different key names in case you tweaked train.py
    if "model_state_dict" not in payload:
        if "model_state" in payload:
            payload["model_state_dict"] = payload["model_state"]
        elif "state_dict" in payload:
            payload["model_state_dict"] = payload["state_dict"]

    if "idx2word" not in payload and "itos" in payload:
        payload["idx2word"] = payload["itos"]
    if "word2idx" not in payload and "stoi" in payload:
        payload["word2idx"] = payload["stoi"]

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained LSTM LM checkpoint (Milestone 4)")

    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--split", type=str, choices=["train", "valid", "test"], default="valid")

    p.add_argument("--batch-size", type=int, default=64, help="Must match training batchify batch size")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--max-batches", type=int, default=0, help="0 = no limit (evaluate full split)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = pick_device()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint(ckpt_path, device)

    # Pull vocab
    if "word2idx" not in payload or "idx2word" not in payload:
        print("Checkpoint keys:", list(payload.keys()))
        raise KeyError("Checkpoint must contain word2idx and idx2word to run eval.")

    word2idx: Dict[str, int] = payload["word2idx"]
    idx2word: List[str] = payload["idx2word"]
    V = len(idx2word)

    # Pull model hyperparams
    cfg = payload.get("config", {})
    embed_dim = int(cfg.get("embed_dim", 256))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.2))

    model = LSTMLanguageModel(
        vocab_size=V,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if "model_state_dict" not in payload:
        print("Checkpoint keys:", list(payload.keys()))
        raise KeyError("Checkpoint is missing model_state_dict (or equivalent).")

    model.load_state_dict(payload["model_state_dict"], strict=True)

    # Load + encode split
    data_dir = Path(args.data_dir)
    fname = {
        "train": "train.txt",
        "valid": "valid.txt",
        "test": "test.txt",
    }[args.split]

    split_path = data_dir / fname
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ids = encode_file(split_path, word2idx)
    data = batchify(ids, args.batch_size)

    max_batches = None if args.max_batches == 0 else int(args.max_batches)
    out = evaluate(model=model, data=data, seq_len=args.seq_len, device=device, max_batches=max_batches)

    print("\n=== EVAL ===")
    print(f"device: {device}")
    print(f"ckpt: {ckpt_path}")
    print(f"split: {args.split} ({split_path})")
    print(f"vocab: {V}")
    print(f"batch_size: {args.batch_size} | seq_len: {args.seq_len}")
    if max_batches is not None:
        print(f"max_batches: {max_batches}")

    print(f"loss:  {out['loss']:.4f}")
    print(f"ppl:   {out['ppl']:.2f}")
    print(f"top1:  {out['top1']*100:.2f}%")
    print(f"top5:  {out['top5']*100:.2f}%")
    print(f"tokens:{int(out['tokens'])}")


if __name__ == "__main__":
    main()