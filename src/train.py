from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from data import batchify, build_vocab, encode_file, iter_batches
from model import LSTMLanguageModel


@dataclass
class TrainConfig:
    """Configuration for training."""
    data_dir: str
    min_freq: int
    batch_size: int
    seq_len: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    grad_clip: float
    epochs: int
    log_every: int
    seed: int


def pick_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_for_ce(
    logits: torch.Tensor, 
    targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten logits and targets for CrossEntropyLoss.
    
    CrossEntropyLoss expects:
    - logits: (N, C) where N=batch, C=classes
    - targets: (N,)
    
    Args:
        logits: (B, T, V)
        targets: (B, T)
    
    Returns:
        logits_flat: (B*T, V)
        targets_flat: (B*T,)
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    return logits_flat, targets_flat


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> float:
    """Evaluate model on validation/test data.
    
    Returns average loss per token.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in iter_batches(data, seq_len=seq_len):
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits = model(x)  # (B, T, V)
        logits_flat, y_flat = flatten_for_ce(logits, y)
        
        # Calculate loss
        loss = criterion(logits_flat, y_flat)
        
        # Accumulate weighted by number of tokens
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    
    return total_loss / max(total_tokens, 1)


def train_one_epoch(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_every: int,
) -> float:
    """Train for one epoch.
    
    Returns average loss per token for the epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_tokens = 0
    
    for step, (x, y) in enumerate(iter_batches(data, seq_len=seq_len), start=1):
        x = x.to(device)
        y = y.to(device)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        logits = model(x)  # (B, T, V)
        logits_flat, y_flat = flatten_for_ce(logits, y)
        loss = criterion(logits_flat, y_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        
        # Log progress
        if log_every > 0 and step % log_every == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
            print(f"  step {step:>6} | avg_loss {avg_loss:.4f} | ppl {ppl:.2f}")
    
    return total_loss / max(total_tokens, 1)


def overfit_one_batch(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    steps: int,
) -> None:
    """Overfit on a single batch to verify model can learn.
    
    This is a critical sanity check. If loss doesn't drop significantly,
    something is wrong with the model or training setup.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get exactly one batch
    first_batch = next(iter(iter_batches(data, seq_len=seq_len)))
    x, y = first_batch
    x = x.to(device)
    y = y.to(device)
    
    print(f"\n{'='*60}")
    print("OVERFIT-ONE-BATCH TEST")
    print(f"{'='*60}")
    print(f"Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"Training for {steps} steps on this single batch...")
    print(f"Expected: Loss should drop from ~4.6 to < 0.5\n")
    
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        
        # Forward + backward
        logits = model(x)
        logits_flat, y_flat = flatten_for_ce(logits, y)
        loss = criterion(logits_flat, y_flat)
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Log progress
        if step == 1 or step % 25 == 0 or step == steps:
            print(f"step {step:>4}/{steps} | loss {loss.item():.4f}")
    
    # Final check
    final_loss = loss.item()
    print(f"\n{'='*60}")
    if final_loss < 0.5:
        print("✅ PASS: Model can overfit! Loss dropped significantly.")
    elif final_loss < 2.0:
        print("⚠️  WARNING: Loss dropped but not enough. Check learning rate.")
    else:
        print("❌ FAIL: Loss didn't drop. Something is wrong!")
    print(f"{'='*60}\n")


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    config: TrainConfig,
    word2idx: Dict[str, int],
    idx2word: List[str],
    val_loss: float,
) -> None:
    """Save model checkpoint with all necessary info for inference."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "val_loss": float(val_loss),
    }
    
    torch.save(payload, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Train LSTM language model (Milestone 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    p.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Directory containing train.txt, valid.txt, test.txt"
    )
    p.add_argument(
        "--min-freq", 
        type=int, 
        default=2,
        help="Minimum token frequency to include in vocab"
    )
    
    # Batching
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=30)
    
    # Model architecture
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    
    # Optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    
    # Checkpointing
    p.add_argument(
        "--ckpt-dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    
    # Debug mode
    p.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Debug: train on single batch to verify learning"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of steps for overfit-one-batch mode"
    )
    
    return p.parse_args()


def main() -> None:
    """Main training pipeline."""
    args = parse_args()
    
    # Create config
    cfg = TrainConfig(
        data_dir=args.data_dir,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        log_every=args.log_every,
        seed=args.seed,
    )
    
    # Setup
    set_seed(cfg.seed)
    device = pick_device()
    print(f"🖥️  Device: {device}")
    
    # Load data
    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. "
            f"Expected WikiText-2 data in {data_dir}/"
        )
    
    # Build vocab from train only
    print("\n📚 Building vocabulary...")
    word2idx, idx2word = build_vocab(train_path, min_freq=cfg.min_freq)
    V = len(idx2word)
    print(f"Vocab size: {V:,}")
    
    # Encode data
    print("\n📊 Encoding data...")
    train_ids = encode_file(train_path, word2idx)
    train_data = batchify(train_ids, cfg.batch_size)
    print(f"Train data: {tuple(train_data.shape)} (B, N)")
    
    if not valid_path.exists():
        raise FileNotFoundError(
            f"Missing {valid_path}. "
            f"Expected validation split at {valid_path}"
        )
    
    valid_ids = encode_file(valid_path, word2idx)
    valid_data = batchify(valid_ids, cfg.batch_size)
    print(f"Valid data: {tuple(valid_data.shape)} (B, N)")
    
    # Create model
    print("\n🧠 Creating model...")
    model = LSTMLanguageModel(
        vocab_size=V,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    best_path = ckpt_dir / "best.pt"
    
    # GATE: Overfit one batch
    if args.overfit_one_batch:
        overfit_one_batch(
            model=model,
            data=train_data,
            seq_len=cfg.seq_len,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.grad_clip,
            steps=args.steps,
        )
        return
    
    # Full training loop
    print(f"\n🚀 Starting training for {cfg.epochs} epochs...\n")
    print(f"{'='*60}")
    
    best_val_loss = float("inf")
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{cfg.epochs}")
        print(f"{'-'*60}")
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            data=train_data,
            seq_len=cfg.seq_len,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.grad_clip,
            log_every=cfg.log_every,
        )
        train_ppl = math.exp(train_loss) if train_loss < 50 else float("inf")
        
        # Validate
        val_loss = evaluate(
            model=model,
            data=valid_data,
            seq_len=cfg.seq_len,
            device=device,
        )
        val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
        
        # Log epoch summary
        print(f"{'-'*60}")
        print(
            f"📊 Epoch {epoch} Summary: "
            f"train_loss={train_loss:.4f} (ppl={train_ppl:.2f}) | "
            f"val_loss={val_loss:.4f} (ppl={val_ppl:.2f})"
        )
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_path=best_path,
                model=model,
                config=cfg,
                word2idx=word2idx,
                idx2word=idx2word,
                val_loss=val_loss,
            )
            print(f"✨ New best validation loss: {best_val_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {best_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()