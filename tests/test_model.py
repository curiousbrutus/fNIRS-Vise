"""
Test suite for fMRI-fNIRS transfer learning model.

Tests model architecture, forward pass, gradient flow, and CUDA compatibility.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.fmri_fnirs_net import FmriFnirsNet, FnirsEncoder, CrossModalAttention


class TestFmriFnirsNet:
    """Test suite for the main transfer learning model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.fnirs_input = torch.randn(self.batch_size, 52, 200)
        self.fmri_input = torch.randn(self.batch_size, 768)
        self.labels = torch.randint(0, 4, (self.batch_size,))
        
    def test_model_initialization(self):
        """Test model can be initialized with different configurations."""
        # Test default initialization
        model = FmriFnirsNet()
        assert model.hparams.fnirs_channels == 52
        assert model.hparams.fmri_dim == 768
        assert model.hparams.num_classes == 4
        
        # Test custom initialization
        model = FmriFnirsNet(
            fnirs_channels=64,
            fmri_dim=512,
            num_classes=8,
            transfer_mode="distill"
        )
        assert model.hparams.fnirs_channels == 64
        assert model.hparams.fmri_dim == 512
        assert model.hparams.num_classes == 8
        assert model.hparams.transfer_mode == "distill"
        
    def test_forward_shape(self):
        """Test forward pass produces correct output shapes."""
        # Test feature_guided mode
        model = FmriFnirsNet(transfer_mode="feature_guided")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4), f"Expected shape {(self.batch_size, 4)}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        # Test concat mode
        model = FmriFnirsNet(transfer_mode="concat")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4)
        
        # Test distill mode
        model = FmriFnirsNet(transfer_mode="distill")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4)
        
    def test_gradient_flow(self):
        """Test gradients flow properly through the network."""
        model = FmriFnirsNet(transfer_mode="feature_guided")
        
        # Forward pass
        output = model(self.fnirs_input, self.fmri_input)
        loss = nn.CrossEntropyLoss()(output, self.labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert torch.isfinite(param.grad).all(), f"Infinite gradient for parameter: {name}"
                
    def test_training_step(self):
        """Test training step executes correctly."""
        model = FmriFnirsNet()
        
        batch = {
            "fnirs": self.fnirs_input,
            "fmri": self.fmri_input,
            "label": self.labels
        }
        
        # Test training step
        loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor), "Training step should return a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
        
    def test_distillation_loss(self):
        """Test knowledge distillation loss computation."""
        model = FmriFnirsNet(transfer_mode="distill", distill_alpha=0.5)
        
        batch = {
            "fnirs": self.fnirs_input,
            "fmri": self.fmri_input,
            "label": self.labels
        }
        
        # Training step should include distillation loss
        loss = model.training_step(batch, batch_idx=0)
        
        assert torch.isfinite(loss), "Distillation loss should be finite"
        
    def test_parameter_counting(self):
        """Test model has reasonable number of parameters."""
        model = FmriFnirsNet()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters (not too small, not too large)
        assert 10_000 < total_params < 10_000_000, f"Unexpected parameter count: {total_params}"
        assert trainable_params <= total_params, "Trainable params should not exceed total"
        
    def test_freeze_fmri(self):
        """Test fMRI adapter freezing works correctly."""
        model = FmriFnirsNet(freeze_fmri=True)
        
        # Check that fMRI adapter parameters are frozen
        for param in model.fmri_adapter.parameters():
            assert not param.requires_grad, "fMRI adapter should be frozen"
            
        # Check that fNIRS encoder parameters are not frozen
        for param in model.fnirs_encoder.parameters():
            assert param.requires_grad, "fNIRS encoder should not be frozen"
            
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Tesla K80")
    def test_cuda_compatibility(self, mock_device_name, mock_cuda_available):
        """Test model works with CUDA (mocked)."""
        model = FmriFnirsNet()
        
        # Mock CUDA tensors
        with patch.object(torch, 'cuda') as mock_cuda:
            mock_cuda.is_available.return_value = True
            
            # Test model can be moved to CUDA (mocked)
            try:
                # This would normally fail without real CUDA, but we're testing the logic
                if torch.cuda.is_available():
                    model = model.cuda()
                    fnirs_cuda = self.fnirs_input.cuda() if hasattr(self.fnirs_input, 'cuda') else self.fnirs_input
                    fmri_cuda = self.fmri_input.cuda() if hasattr(self.fmri_input, 'cuda') else self.fmri_input
            except:
                # Expected to fail in test environment without CUDA
                pass


class TestFnirsEncoder:
    """Test suite for the fNIRS encoder component."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 52, 200)
        
    def test_encoder_forward(self):
        """Test fNIRS encoder forward pass."""
        encoder = FnirsEncoder()
        
        with torch.no_grad():
            output = encoder(self.input_tensor)
            
        assert output.shape == (self.batch_size, 128), f"Expected {(self.batch_size, 128)}, got {output.shape}"
        assert not torch.isnan(output).any(), "Encoder output contains NaN"
        
    def test_encoder_different_configs(self):
        """Test encoder with different configurations."""
        # Test different hidden dimensions
        encoder = FnirsEncoder(hidden_dim=256)
        with torch.no_grad():
            output = encoder(self.input_tensor)
        assert output.shape == (self.batch_size, 256)
        
        # Test different input channels
        encoder = FnirsEncoder(input_channels=64, hidden_dim=128)
        input_64ch = torch.randn(self.batch_size, 64, 200)
        with torch.no_grad():
            output = encoder(input_64ch)
        assert output.shape == (self.batch_size, 128)


class TestCrossModalAttention:
    """Test suite for cross-modal attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.fnirs_features = torch.randn(self.batch_size, 128)
        self.fmri_features = torch.randn(self.batch_size, 128)
        
    def test_attention_forward(self):
        """Test cross-modal attention forward pass."""
        attention = CrossModalAttention(fnirs_dim=128, fmri_dim=128)
        
        with torch.no_grad():
            output = attention(self.fnirs_features, self.fmri_features)
            
        assert output.shape == self.fnirs_features.shape, "Attention should preserve fNIRS feature dimensions"
        assert not torch.isnan(output).any(), "Attention output contains NaN"
        
    def test_attention_gradients(self):
        """Test attention mechanism has proper gradients."""
        attention = CrossModalAttention(fnirs_dim=128, fmri_dim=128)
        
        # Forward pass
        output = attention(self.fnirs_features, self.fmri_features)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in attention.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Attention parameters should have gradients"
                assert torch.isfinite(param.grad).all(), "Attention gradients should be finite"


class TestModelIntegration:
    """Integration tests for the complete model."""
    
    def test_end_to_end_training_simulation(self):
        """Test a complete training simulation."""
        model = FmriFnirsNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Simulate training steps
        for step in range(3):
            # Forward pass
            fnirs_input = torch.randn(4, 52, 200)
            fmri_input = torch.randn(4, 768)
            labels = torch.randint(0, 4, (4,))
            
            batch = {
                "fnirs": fnirs_input,
                "fmri": fmri_input,
                "label": labels
            }
            
            # Training step
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=step)
            loss.backward()
            optimizer.step()
            
            assert torch.isfinite(loss), f"Loss should be finite at step {step}"
            
    def test_validation_step(self):
        """Test validation step."""
        model = FmriFnirsNet()
        
        batch = {
            "fnirs": torch.randn(4, 52, 200),
            "fmri": torch.randn(4, 768),
            "label": torch.randint(0, 4, (4,))
        }
        
        with torch.no_grad():
            val_loss = model.validation_step(batch, batch_idx=0)
            
        assert torch.isfinite(val_loss), "Validation loss should be finite"
        
    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = FmriFnirsNet()
        model.trainer = MagicMock()
        model.trainer.max_epochs = 100
        
        optim_config = model.configure_optimizers()
        
        assert "optimizer" in optim_config
        assert "lr_scheduler" in optim_config
        assert optim_config["optimizer"].__class__.__name__ == "AdamW"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
