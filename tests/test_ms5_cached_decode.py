import pytest
import torch
import sys
from pathlib import Path

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


@pytest.mark.parametrize("num_layers", [1, 2])
def test_ms5_cached_decode_step_shapes(num_layers: int) -> None:
    """Milestone 5: ensure cached-state decoding works and shapes are correct.

    We verify:
    - forward_with_state(prefix) returns (B,T,V) logits and (h,c)
    - step(token, state) returns (B,1,V) logits and updated (h,c)
    - state shapes match (num_layers, B, hidden_dim)
    """

    device = pick_device()

    vocab_size = 100
    embed_dim = 16
    hidden_dim = 32

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    ).to(device)

    assert hasattr(model, "forward_with_state"), "Model missing forward_with_state() (MS5)"
    assert hasattr(model, "step"), "Model missing step() (MS5)"

    B = 1
    T = 5

    prefix = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)

    logits, state = model.forward_with_state(prefix)

    assert logits.shape == (B, T, vocab_size)
    assert isinstance(state, tuple) and len(state) == 2

    h, c = state
    assert h.shape == (num_layers, B, hidden_dim)
    assert c.shape == (num_layers, B, hidden_dim)

    # step through a few tokens
    token = prefix[:, -1]  # (B,)
    for _ in range(10):
        step_logits, state = model.step(token, state)
        assert step_logits.shape == (B, 1, vocab_size)
        h, c = state
        assert h.shape == (num_layers, B, hidden_dim)
        assert c.shape == (num_layers, B, hidden_dim)

        # next token = argmax (deterministic)
        token = torch.argmax(step_logits[:, -1, :], dim=-1)


def test_ms5_step_accepts_shape_B() -> None:
    """step() should accept token_ids shaped (B,) as well as (B,1)."""

    device = pick_device()

    vocab_size = 50
    embed_dim = 8
    hidden_dim = 16

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
    ).to(device)

    B = 4
    tokens_B = torch.randint(0, vocab_size, (B,), device=device, dtype=torch.long)
    logits1, state1 = model.step(tokens_B, state=None)
    assert logits1.shape == (B, 1, vocab_size)

    tokens_B1 = tokens_B.unsqueeze(1)
    logits2, state2 = model.step(tokens_B1, state=None)
    assert logits2.shape == (B, 1, vocab_size)

    # States should have same shapes
    assert state1[0].shape == state2[0].shape
    assert state1[1].shape == state2[1].shape