"""
Test suite for fMRI-fNIRS transfer learning model.

Tests model architecture, forward pass, gradient flow, and CUDA compatibility.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.fmri_fnirs_net import Model, FmriFnirsNet


class TestModel:
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
        model = Model()
        assert model.hparams.fnirs_channels == 52
        assert model.hparams.fmri_dim == 768
        assert model.hparams.num_classes == 4
        
        # Test custom initialization
        model = Model(
            fnirs_channels=64,
            fmri_dim=512,
            num_classes=8,
            transfer_mode="knowledge_distill"
        )
        assert model.hparams.fnirs_channels == 64
        assert model.hparams.fmri_dim == 512
        assert model.hparams.num_classes == 8
        assert model.hparams.transfer_mode == "knowledge_distill"
        
    def test_forward_shape(self):
        """Test forward pass produces correct output shapes."""
        # Test feature_guided mode
        model = Model(transfer_mode="feature_guided")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4), f"Expected shape {(self.batch_size, 4)}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        # Test weight_init mode
        model = Model(transfer_mode="weight_init")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4)
        
        # Test knowledge_distill mode
        model = Model(transfer_mode="knowledge_distill")
        with torch.no_grad():
            output = model(self.fnirs_input, self.fmri_input)
            
        assert output.shape == (self.batch_size, 4)
        
    def test_gradient_flow(self):
        """Test gradients flow properly through the network."""
        model = Model(transfer_mode="feature_guided")
        
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
        model = Model()
        
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
        model = Model(transfer_mode="knowledge_distill", distill_alpha=0.5)
        
        batch = {
            "fnirs": self.fnirs_input,
            "fmri": self.fmri_input,
            "label": self.labels,
            "teacher_logits": torch.randn(self.batch_size, 4)  # Mock teacher predictions
        }
        
        # Training step should include distillation loss
        loss = model.training_step(batch, batch_idx=0)
        
        assert torch.isfinite(loss), "Distillation loss should be finite"
        
    def test_parameter_counting(self):
        """Test model has reasonable number of parameters."""
        model = Model()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters (not too small, not too large)
        assert 10_000 < total_params < 10_000_000, f"Unexpected parameter count: {total_params}"
        assert trainable_params <= total_params, "Trainable params should not exceed total"
        
    def test_backbone_selection(self):
        """Test different external backbone selection."""
        backbones = ["fNIRS-T", "fNIRSNet", "fNIRS2MW", "custom"]
        
        for backbone in backbones:
            model = Model(backbone=backbone)
            
            with torch.no_grad():
                output = model(self.fnirs_input, self.fmri_input)
                
            assert output.shape == (self.batch_size, 4), f"Backbone {backbone} failed"
            
    def test_freeze_backbone(self):
        """Test backbone freezing works correctly."""
        model = Model(freeze_backbone=True)
        
        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad, "Backbone should be frozen"
            
        # Check that classifier parameters are not frozen
        for param in model.classifier.parameters():
            assert param.requires_grad, "Classifier should not be frozen"
    
    def test_validation_step(self):
        """Test validation step."""
        model = Model()
        
        batch = {
            "fnirs": self.fnirs_input,
            "fmri": self.fmri_input,
            "label": self.labels
        }
        
        with torch.no_grad():
            val_loss = model.validation_step(batch, batch_idx=0)
            
        assert torch.isfinite(val_loss), "Validation loss should be finite"
        
    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = Model()
        config = model.configure_optimizers()
        
        assert "optimizer" in config, "Should return optimizer"
        assert "lr_scheduler" in config, "Should return learning rate scheduler"


# Test backward compatibility alias
def test_backward_compatibility():
    """Test that FmriFnirsNet alias still works."""
    model1 = Model()
    model2 = FmriFnirsNet()
    
    # Should be the same class
    assert type(model1) == type(model2)
    assert isinstance(model2, Model)
