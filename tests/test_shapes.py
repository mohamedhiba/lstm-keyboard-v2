"""
Milestone 2 Tests: Shape validation and training mechanics

Tests verify:
1. Forward pass produces correct output shape
2. Loss computation works without errors
3. Backward pass successfully computes gradients
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

# Ensure the project root (repo folder) is on sys.path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LSTMLanguageModel


class TestLSTMShapes:
    """Test suite for LSTM model shape correctness and basic training mechanics."""
    
    @pytest.fixture
    def model_config(self):
        """Standard model configuration for testing."""
        return {
            "vocab_size": 100,
            "embed_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
        }
    
    @pytest.fixture
    def batch_config(self):
        """Standard batch configuration for testing."""
        return {
            "batch_size": 8,
            "seq_len": 20,
        }
    
    @pytest.fixture
    def model(self, model_config):
        """Create a fresh model instance."""
        return LSTMLanguageModel(**model_config)
    
    def test_forward_shape_correct(self, model, model_config, batch_config):
        """Test 1: Forward pass produces correct output shape (B, T, V)."""
        B = batch_config["batch_size"]
        T = batch_config["seq_len"]
        V = model_config["vocab_size"]
        
        # Create random input token ids in valid range [0, V-1]
        x = torch.randint(0, V, size=(B, T), dtype=torch.long)
        
        # Run forward pass
        logits = model(x)
        
        # Assert correct shape
        assert logits.shape == (B, T, V), (
            f"Expected logits shape (B={B}, T={T}, V={V}), "
            f"but got {tuple(logits.shape)}"
        )
        
        # Additional sanity checks
        assert logits.dtype == torch.float32, "Logits should be float32"
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"
    
    def test_loss_computation(self, model, model_config, batch_config):
        """Test 2: Loss can be computed without errors."""
        B = batch_config["batch_size"]
        T = batch_config["seq_len"]
        V = model_config["vocab_size"]
        
        # Create random input and target sequences
        x = torch.randint(0, V, size=(B, T), dtype=torch.long)
        y = torch.randint(0, V, size=(B, T), dtype=torch.long)
        
        # Forward pass
        logits = model(x)
        
        # Flatten for cross-entropy loss
        # Cross-entropy expects: (N, C) for input and (N,) for targets
        # where N = batch size and C = number of classes
        logits_flat = logits.reshape(B * T, V)  # (B*T, V)
        y_flat = y.reshape(B * T)  # (B*T,)
        
        # Compute loss (should not crash)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, y_flat)
        
        # Verify loss properties
        assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        
        # Loss should be reasonable (roughly -log(1/V) for random init)
        # For V=100, we expect loss around 4.6
        assert 0 < loss.item() < 20, f"Loss {loss.item()} seems unreasonable"
    
    def test_backward_pass(self, model, model_config, batch_config):
        """Test 3: Backward pass successfully computes gradients."""
        B = batch_config["batch_size"]
        T = batch_config["seq_len"]
        V = model_config["vocab_size"]
        
        # Create random input and target sequences
        x = torch.randint(0, V, size=(B, T), dtype=torch.long)
        y = torch.randint(0, V, size=(B, T), dtype=torch.long)
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        logits_flat = logits.reshape(B * T, V)
        y_flat = y.reshape(B * T)
        loss = nn.CrossEntropyLoss()(logits_flat, y_flat)
        
        # Backward pass (should not crash)
        loss.backward()
        
        # Verify gradients exist and are valid for key parameters
        params_to_check = [
            ("embedding.weight", model.embedding.weight),
            ("fc.weight", model.fc.weight),
            ("fc.bias", model.fc.bias),
        ]
        
        for name, param in params_to_check:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf"
            assert param.grad.shape == param.shape, f"Gradient shape mismatch for {name}"
    
    def test_different_batch_sizes(self, model, model_config):
        """Test that model works with various batch sizes."""
        V = model_config["vocab_size"]
        
        batch_sizes = [1, 4, 16, 32]
        seq_len = 10
        
        for B in batch_sizes:
            x = torch.randint(0, V, size=(B, seq_len), dtype=torch.long)
            logits = model(x)
            assert logits.shape == (B, seq_len, V), f"Failed for batch_size={B}"
    
    def test_different_sequence_lengths(self, model, model_config):
        """Test that model works with various sequence lengths."""
        V = model_config["vocab_size"]
        
        batch_size = 4
        seq_lengths = [1, 5, 10, 50, 100]
        
        for T in seq_lengths:
            x = torch.randint(0, V, size=(batch_size, T), dtype=torch.long)
            logits = model(x)
            assert logits.shape == (batch_size, T, V), f"Failed for seq_len={T}"
    
    def test_stateful_forward(self, model, model_config, batch_config):
        """Test that stateful forward pass works correctly."""
        B = batch_config["batch_size"]
        T = batch_config["seq_len"]
        V = model_config["vocab_size"]
        
        x = torch.randint(0, V, size=(B, T), dtype=torch.long)
        
        # Forward with state
        logits, state = model(x, return_state=True)
        
        # Check logits shape
        assert logits.shape == (B, T, V)
        
        # Check state structure
        assert isinstance(state, tuple), "State should be a tuple"
        assert len(state) == 2, "State should contain (h, c)"
        
        h, c = state
        num_layers = model_config["num_layers"]
        hidden_dim = model_config["hidden_dim"]
        
        assert h.shape == (num_layers, B, hidden_dim), f"Hidden state shape incorrect"
        assert c.shape == (num_layers, B, hidden_dim), f"Cell state shape incorrect"
    
    def test_gradient_flow_through_all_layers(self, model):
        """Test that gradients flow through all model components."""
        B, T, V = 4, 10, 100
        
        x = torch.randint(0, V, size=(B, T), dtype=torch.long)
        y = torch.randint(0, V, size=(B, T), dtype=torch.long)
        
        # Forward + backward
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.reshape(B * T, V), y.reshape(B * T))
        loss.backward()
        
        # Check all named parameters have gradients
        params_with_no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_with_no_grad.append(name)
        
        assert len(params_with_no_grad) == 0, (
            f"These parameters have no gradients: {params_with_no_grad}"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_model_rejects_invalid_input_shape(self):
        """Test that model raises error for wrong input dimensions."""
        model = LSTMLanguageModel(
            vocab_size=100,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )
        
        # 1D input (should be 2D)
        x_1d = torch.randint(0, 100, size=(10,), dtype=torch.long)
        with pytest.raises(ValueError, match="Expected x with shape"):
            model(x_1d)
        
        # 3D input (should be 2D)
        x_3d = torch.randint(0, 100, size=(2, 10, 5), dtype=torch.long)
        with pytest.raises(ValueError, match="Expected x with shape"):
            model(x_3d)
    
    def test_model_rejects_wrong_dtype(self):
        """Test that model raises error for non-long tensor."""
        model = LSTMLanguageModel(
            vocab_size=100,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Float input (should be long)
        x_float = torch.randn(4, 10)
        with pytest.raises(TypeError, match="expects x dtype torch.long"):
            model(x_float)
    
    def test_model_parameter_validation(self):
        """Test that model constructor validates parameters."""
        with pytest.raises(ValueError, match="vocab_size must be > 0"):
            LSTMLanguageModel(vocab_size=0, embed_dim=64, hidden_dim=128)
        
        with pytest.raises(ValueError, match="embed_dim must be > 0"):
            LSTMLanguageModel(vocab_size=100, embed_dim=0, hidden_dim=128)
        
        with pytest.raises(ValueError, match="hidden_dim must be > 0"):
            LSTMLanguageModel(vocab_size=100, embed_dim=64, hidden_dim=0)
        
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            LSTMLanguageModel(vocab_size=100, embed_dim=64, hidden_dim=128, num_layers=0)
        
        with pytest.raises(ValueError, match="dropout must be >= 0"):
            LSTMLanguageModel(vocab_size=100, embed_dim=64, hidden_dim=128, dropout=-0.1)


if __name__ == "__main__":
    # Allow running tests directly with: python test_shapes.py
    pytest.main([__file__, "-v"])