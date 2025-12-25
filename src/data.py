
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from collections import Counter

# Milestone 1 goals:
# - Build vocab from TRAIN ONLY
# - Encode train/valid/test into 1D token-id tensors
# - Build batches (x, y) with shapes (B, T)


def tokenize_line(line: str) -> List[str]:
    """Word-level tokenization using whitespace split.

    Adds <eos> at the end of each non-empty line.
    """
    line = line.strip()
    if not line:
        return []
    return line.split() + ["<eos>"]


def build_vocab(train_path: Path, min_freq: int = 2) -> Tuple[Dict[str, int], List[str]]:
    """Build vocabulary from TRAIN ONLY using a frequency counter.

    Tokens with frequency < min_freq are mapped to <unk>.
    """
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1")

    # 1) Count token frequencies on TRAIN
    counter: Counter[str] = Counter()
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            counter.update(tokenize_line(line))

    # 2) Create vocab with special tokens first
    word2idx: Dict[str, int] = {"<unk>": 0}
    idx2word: List[str] = ["<unk>"]

    # Ensure <eos> exists even if someone changes tokenization later
    if "<eos>" not in word2idx:
        word2idx["<eos>"] = len(idx2word)
        idx2word.append("<eos>")

    # 3) Add tokens that meet min_freq (skip specials already added)
    for tok, freq in counter.items():
        if tok in word2idx:
            continue
        if freq >= min_freq:
            word2idx[tok] = len(idx2word)
            idx2word.append(tok)

    return word2idx, idx2word


def encode_file(path: Path, word2idx: Dict[str, int]) -> torch.Tensor:
    """Encode a text file into a 1D tensor of token ids.

    Unknown tokens map to <unk>.
    """
    unk_id = word2idx["<unk>"]
    ids: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            for tok in tokenize_line(line):
                ids.append(word2idx.get(tok, unk_id))

    return torch.tensor(ids, dtype=torch.int64)


def decode_ids(ids: List[int], idx2word: List[str]) -> List[str]:
    """Decode a list of ids back to tokens (for sanity checks)."""
    out: List[str] = []
    for i in ids:
        if 0 <= i < len(idx2word):
            out.append(idx2word[i])
        else:
            out.append("<bad_id>")
    return out


def batchify(token_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshape a long 1D token-id tensor into (B, N) for contiguous LM batching."""
    if token_ids.dim() != 1:
        raise ValueError(f"batchify expects 1D token_ids, got shape {tuple(token_ids.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n_batches = token_ids.size(0) // batch_size
    trimmed_len = n_batches * batch_size
    token_ids = token_ids[:trimmed_len]
    return token_ids.view(batch_size, -1)


def get_batch(source: torch.Tensor, start_idx: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get one (x, y) pair from batchified data.

    source: (B, N)
    x: (B, T)
    y: (B, T) == x shifted by 1
    """
    if source.dim() != 2:
        raise ValueError(f"get_batch expects source with shape (B, N), got {tuple(source.shape)}")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    max_t = source.size(1) - 1  # need one extra token for y
    if start_idx < 0 or start_idx >= max_t:
        raise ValueError(f"start_idx out of range: {start_idx} (valid: 0..{max_t-1})")

    T = min(seq_len, max_t - start_idx)
    x = source[:, start_idx : start_idx + T]
    y = source[:, start_idx + 1 : start_idx + 1 + T]
    return x, y


def iter_batches(source: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Yield successive (x, y) batches across time for contiguous LM training."""
    if source.dim() != 2:
        raise ValueError(f"iter_batches expects source with shape (B, N), got {tuple(source.shape)}")
    # last usable start index is (N-2) because y needs one extra token
    for t in range(0, source.size(1) - 1, seq_len):
        yield get_batch(source, start_idx=t, seq_len=seq_len)


def main() -> None:
    # Use raw data for now. Switch to Path("data/clean") later if you create cleaned files.
    data_dir = Path("data/raw")
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Could not find {train_path}. Expected WikiText-2 saved to data/raw/train.txt"
        )

    # 1) Build vocab from train only
    min_freq = 2
    word2idx, idx2word = build_vocab(train_path, min_freq=min_freq)

    # 2) Encode splits
    train_ids = encode_file(train_path, word2idx)
    print("vocab size:", len(idx2word))
    print("train_ids length:", int(train_ids.numel()))

    if valid_path.exists():
        valid_ids = encode_file(valid_path, word2idx)
        unk = word2idx["<unk>"]
        print("valid_ids length:", int(valid_ids.numel()))
        print("valid <unk> count:", int((valid_ids == unk).sum().item()))

    if test_path.exists():
        test_ids = encode_file(test_path, word2idx)
        unk = word2idx["<unk>"]
        print("test_ids length:", int(test_ids.numel()))
        print("test <unk> count:", int((test_ids == unk).sum().item()))

    # 3) Batch builder sanity check (x, y) shapes
    batch_size = 64
    seq_len = 30

    train_data = batchify(train_ids, batch_size)  # (B, N)
    x, y = get_batch(train_data, start_idx=0, seq_len=seq_len)  # (B, T)

    print("train_data shape (B,N):", tuple(train_data.shape))
    print("x shape (B,T):", tuple(x.shape))
    print("y shape (B,T):", tuple(y.shape))

    # Demonstrate iterating across time (first 2 batches)
    for batch_i, (bx, by) in enumerate(iter_batches(train_data, seq_len=seq_len)):
        print(f"iter batch {batch_i} shapes:", tuple(bx.shape), tuple(by.shape))
        if batch_i >= 1:
            break

    # Show y is x shifted by one (decode first row)
    x0 = x[0, :10].tolist()
    y0 = y[0, :10].tolist()
    print("x[0][:10] tokens:", decode_ids(x0, idx2word))
    print("y[0][:10] tokens:", decode_ids(y0, idx2word))

    # Extra sanity: first 30 items from the flat stream
    first_30_ids = train_ids[:30].tolist()
    print("first 30 tokens:", decode_ids(first_30_ids, idx2word))
    print("first 30 ids:", first_30_ids)


if __name__ == "__main__":
    main()
