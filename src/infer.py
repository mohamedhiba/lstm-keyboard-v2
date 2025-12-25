from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LSTMLanguageModel

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location=device)

    # Be robust to naming differences
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


def encode_prefix(prefix: str, word2idx: Dict[str, int]) -> List[int]:
    # No <eos> here; we want to predict the next word after the prefix.
    toks = [t for t in prefix.strip().split() if t]
    unk = word2idx.get("<unk>", 0)
    return [word2idx.get(t, unk) for t in toks]


@torch.no_grad()
def predict_next(
    model: LSTMLanguageModel,
    ids: List[int],
    k: int,
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    if len(ids) == 0:
        raise ValueError("Prefix is empty after tokenization. Provide at least one word.")

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    logits = model(x)  # (1, T, V)

    # Use last time step
    last = logits[0, -1, :]  # (V,)

    # Convert to probabilities
    probs = torch.softmax(last, dim=-1)

    topk = torch.topk(probs, k=min(k, probs.numel()), dim=-1)
    top_ids = topk.indices.tolist()
    top_probs = topk.values.tolist()

    return top_ids, top_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-word inference for LSTM LM (Milestone 4)")

    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--text", type=str, required=True, help='Prefix text, e.g. "i want to"')
    p.add_argument("--k", type=int, default=5)

    p.add_argument("--generate", type=int, default=0, help="Number of tokens to generate after the prefix (0 = just next-word top-k)")
    p.add_argument("--strategy", type=str, choices=["greedy", "sample"], default="greedy", help="Decoding strategy")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (only for --strategy sample)")
    p.add_argument("--topk", type=int, default=0, help="If >0, sample only from top-k tokens (only for --strategy sample)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling (0 = don't set)")
    p.add_argument("--stop-eos", action="store_true", help="Stop generation when <eos> is produced")
    p.add_argument("--ban-unk", action="store_true", help="Prevent the model from generating <unk>")
    p.add_argument("--repeat-penalty", type=float, default=1.0, help=">1.0 discourages repeating recent tokens (1.0 = off)")
    p.add_argument("--repeat-window", type=int, default=50, help="How many recent tokens to consider for repeat penalty")
    p.add_argument(
        "--ban-tokens",
        type=str,
        default="",
        help="Comma-separated list of literal tokens to ban during generation (e.g. '\"' or '@-@').",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = pick_device()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint(ckpt_path, device)

    if "word2idx" not in payload or "idx2word" not in payload:
        print("Checkpoint keys:", list(payload.keys()))
        raise KeyError("Checkpoint must contain word2idx and idx2word.")

    word2idx: Dict[str, int] = payload["word2idx"]
    idx2word: List[str] = payload["idx2word"]
    V = len(idx2word)

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
    model.eval()

    if args.seed and int(args.seed) != 0:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    ids = encode_prefix(args.text, word2idx)
    top_ids, top_probs = predict_next(model, ids=ids, k=args.k, device=device)

    print("\n=== INFER ===")
    print(f"device: {device}")
    print(f"ckpt: {ckpt_path}")
    print(f"prefix: {args.text!r}")
    print(f"tokens_in_prefix: {len(ids)}")

    print("\nTop-k next words:")
    for rank, (tid, p) in enumerate(zip(top_ids, top_probs), start=1):
        word = idx2word[tid] if 0 <= tid < len(idx2word) else "<bad_id>"
        print(f"{rank:>2}. {word:<20}  p={p:.4f}  (logp={math.log(max(p, 1e-12)):.4f})")

    if int(args.generate) > 0:
        eos_id = word2idx.get("<eos>")
        unk_id = word2idx.get("<unk>")

        ban_token_ids: List[int] = []
        if args.ban_tokens:
            for raw in args.ban_tokens.split(","):
                tok = raw.strip()
                if not tok:
                    continue
                if tok in word2idx:
                    ban_token_ids.append(int(word2idx[tok]))

        gen_ids = generate_tokens(
            model=model,
            prefix_ids=ids,
            num_generate=int(args.generate),
            device=device,
            strategy=str(args.strategy),
            temperature=float(args.temperature),
            topk=int(args.topk),
            stop_eos=bool(args.stop_eos),
            eos_id=eos_id,
            ban_unk=bool(args.ban_unk),
            unk_id=unk_id,
            repeat_penalty=float(args.repeat_penalty),
            repeat_window=int(args.repeat_window),
            ban_token_ids=ban_token_ids,
        )
        prefix_tokens = [t for t in args.text.strip().split() if t]
        gen_tokens = decode_tokens(gen_ids, idx2word)
        full = join_tokens(prefix_tokens + gen_tokens)

        print("\nGenerated completion:")
        print(full)


def _sample_next_id(probs: torch.Tensor, topk: int = 0) -> int:
    """Sample one token id from a probability distribution."""
    if topk is not None and topk > 0 and topk < probs.numel():
        vals, idxs = torch.topk(probs, k=topk)
        vals = vals / vals.sum()
        choice = torch.multinomial(vals, num_samples=1).item()
        return int(idxs[choice].item())
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_tokens(
    model: LSTMLanguageModel,
    prefix_ids: List[int],
    num_generate: int,
    device: torch.device,
    strategy: str = "greedy",
    temperature: float = 1.0,
    topk: int = 0,
    stop_eos: bool = False,
    eos_id: int | None = None,
    ban_unk: bool = False,
    unk_id: int | None = None,
    repeat_penalty: float = 1.0,
    repeat_window: int = 50,
    ban_token_ids: List[int] | None = None,
) -> List[int]:
    """Generate `num_generate` token ids after the prefix."""
    if len(prefix_ids) == 0:
        raise ValueError("Prefix is empty after tokenization. Provide at least one word.")
    if num_generate <= 0:
        return []

    generated: List[int] = []
    ids = list(prefix_ids)

    def apply_penalties_to_probs(probs: torch.Tensor) -> torch.Tensor:
        # Ban <unk>
        if ban_unk and unk_id is not None and 0 <= int(unk_id) < probs.numel():
            probs[int(unk_id)] = 0.0

        # Ban arbitrary tokens
        if ban_token_ids:
            for tid in ban_token_ids:
                if 0 <= int(tid) < probs.numel():
                    probs[int(tid)] = 0.0

        # Repetition penalty on recent tokens
        rp = float(repeat_penalty)
        if rp > 1.0:
            window = max(int(repeat_window), 0)
            recent = ids[-window:] if window > 0 else ids
            if recent:
                # Downweight tokens that appeared recently
                unique = set(int(t) for t in recent)
                for t in unique:
                    if 0 <= t < probs.numel():
                        probs[t] = probs[t] / rp

        # Renormalize safely
        s = probs.sum()
        if s.item() <= 0:
            # Fallback: uniform over all tokens (or all except unk if banned)
            probs = torch.ones_like(probs)
            if ban_unk and unk_id is not None and 0 <= int(unk_id) < probs.numel():
                probs[int(unk_id)] = 0.0
            if ban_token_ids:
                for tid in ban_token_ids:
                    if 0 <= int(tid) < probs.numel():
                        probs[int(tid)] = 0.0
            probs = probs / probs.sum()
        else:
            probs = probs / s
        return probs

    for _ in range(num_generate):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        logits = model(x)  # (1, T, V)
        last = logits[0, -1, :]  # (V,)

        if strategy == "greedy":
            # Greedy on logits, but respect bans and repetition penalty by masking logits.
            last = last.clone()

            # Ban <unk>
            if ban_unk and unk_id is not None and 0 <= int(unk_id) < last.numel():
                last[int(unk_id)] = -1e9

            # Ban arbitrary tokens
            if ban_token_ids:
                for tid in ban_token_ids:
                    if 0 <= int(tid) < last.numel():
                        last[int(tid)] = -1e9

            # Repetition penalty (greedy): downweight recently seen tokens
            rp = float(repeat_penalty)
            if rp > 1.0:
                window = max(int(repeat_window), 0)
                recent = ids[-window:] if window > 0 else ids
                if recent:
                    unique = set(int(t) for t in recent)
                    for t in unique:
                        if 0 <= t < last.numel():
                            last[t] = last[t] - math.log(rp)

            next_id = int(torch.argmax(last).item())
        else:
            # sampling
            temp = max(float(temperature), 1e-6)
            scaled = last / temp
            probs = torch.softmax(scaled, dim=-1)
            probs = apply_penalties_to_probs(probs)
            next_id = _sample_next_id(probs, topk=topk)

        # Stop on <eos> (optional)
        if stop_eos and eos_id is not None and int(next_id) == int(eos_id):
            break

        generated.append(next_id)
        ids.append(next_id)

    return generated


def decode_tokens(ids: List[int], idx2word: List[str]) -> List[str]:
    out: List[str] = []
    for tid in ids:
        if 0 <= tid < len(idx2word):
            tok = idx2word[tid]
            if tok == "<eos>":
                continue
            out.append(tok)
        else:
            out.append("<bad_id>")
    return out


def join_tokens(tokens: List[str]) -> str:
    """Simple detokenizer for word-level tokens.

    Handles common punctuation spacing and treats the token '"' as an opening/closing quote.
    """
    if not tokens:
        return ""

    no_space_before = {",", ".", ":", ";", "!", "?", ")", "]", "}", "'s", "'"}
    no_space_after = {"(", "[", "{"}

    s = ""
    quote_open = True

    for i, tok in enumerate(tokens):
        if tok == '"':
            if quote_open:
                # opening quote: add space unless we're at start or after an opening bracket
                if s and s[-1] not in no_space_after and s[-1] != " ":
                    s += " "
                s += '"'
                quote_open = False
            else:
                # closing quote: attach to previous token
                s += '"'
                quote_open = True
            continue

        if not s:
            s = tok
            continue

        if tok in no_space_before:
            s += tok
        elif s and s[-1] in no_space_after:
            s += tok
        elif s and s[-1] == '"' and not quote_open:
            # after an opening quote, no space
            s += tok
        else:
            s += " " + tok

    return s

if __name__ == "__main__":
    main()