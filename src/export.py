from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained checkpoint into a portable artifact.")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="Input training checkpoint")
    p.add_argument("--out", type=str, default="exports/lstm_lm_export.pt", help="Output export path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(ckpt_path, map_location="cpu")

    # Be robust to naming differences
    if "model_state_dict" not in payload:
        if "model_state" in payload:
            payload["model_state_dict"] = payload["model_state"]
        elif "state_dict" in payload:
            payload["model_state_dict"] = payload["state_dict"]
        else:
            raise KeyError("Checkpoint missing model weights (model_state_dict/state_dict).")

    if "idx2word" not in payload and "itos" in payload:
        payload["idx2word"] = payload["itos"]
    if "word2idx" not in payload and "stoi" in payload:
        payload["word2idx"] = payload["stoi"]

    if "idx2word" not in payload or "word2idx" not in payload:
        raise KeyError("Checkpoint must contain idx2word and word2idx to export.")

    export: Dict[str, Any] = {
        "format": "lstm-keyboard-v2-export-v1",
        "model_state_dict": payload["model_state_dict"],
        "config": payload.get("config", {}),
        "idx2word": payload["idx2word"],
        "word2idx": payload["word2idx"],
    }

    torch.save(export, out_path)
    print("\n=== EXPORT ===")
    print(f"in:   {ckpt_path}")
    print(f"out:  {out_path}")
    print(f"vocab:{len(export['idx2word'])}")
    print(f"keys: {list(export.keys())}")


if __name__ == "__main__":
    main()