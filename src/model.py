from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    """Word-level LSTM language model.
    
    Input:
      x: LongTensor of shape (B, T) containing token ids
    Output:
      logits: FloatTensor of shape (B, T, V)
    Optionally can return the LSTM state (h, c) for stateful decoding.
    """
    
    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2,) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if dropout < 0:
            raise ValueError("dropout must be >= 0")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # NOTE: PyTorch's LSTM only applies dropout when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        # x should be (B, T) of integer token ids
        if x.dim() != 2:
            raise ValueError(f"Expected x with shape (B, T), got {tuple(x.shape)}")
        if x.dtype != torch.long:
            raise TypeError(f"Embedding expects x dtype torch.long, got {x.dtype}")
        
        # Step 1: Convert token ids -> dense vectors with dropout
        # (B, T) -> (B, T, E)
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Step 2: LSTM over time
        # (B, T, E) -> (B, T, H)
        lstm_out, new_state = self.lstm(embedded, state)
        
        # Step 3: Apply dropout and project to vocab logits
        # (B, T, H) -> (B, T, V)
        lstm_out = self.output_dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        if return_state:
            return logits, new_state
        return logits