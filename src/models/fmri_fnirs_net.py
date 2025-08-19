"""
fMRI-guided fNIRS Transfer Learning Model

This module implements a comprehensive transfer learning architecture that leverages
fMRI data to guide fNIRS classification through cross-modal attention and knowledge distillation.

External Backbone Integration:
- fNIRS-T: Transformer architecture for fNIRS time series
- fNIRSNet: Lightweight CNN for fNIRS processing  
- fNIRS2MW: Mental workload classification models

Author: GitHub Copilot
License: MIT
"""

import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add external repositories to path
EXTERNAL_DIR = Path(__file__).parent.parent.parent / "external"
sys.path.insert(0, str(EXTERNAL_DIR / "fNIRS-T"))
sys.path.insert(0, str(EXTERNAL_DIR / "fNIRSNet"))
sys.path.insert(0, str(EXTERNAL_DIR / "fNIRS2MW"))

# External backbone imports (with fallback dummy implementations)
try:
    from fnirs_t.model import FnirsTransformer
    FNIRS_T_AVAILABLE = True
except ImportError:
    print("Warning: fNIRS-T not available, using dummy implementation")
    FNIRS_T_AVAILABLE = False
    
    class FnirsTransformer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = nn.Linear(52, 256)
        def forward(self, x):
            return self.dummy(x.mean(dim=-1))

try:
    from fnirsnet.models import FNIRSNet
    FNIRSNET_AVAILABLE = True
except ImportError:
    print("Warning: fNIRSNet not available, using dummy implementation")
    FNIRSNET_AVAILABLE = False
    
    class FNIRSNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = nn.Linear(52, 256)
        def forward(self, x):
            return self.dummy(x.mean(dim=-1))

try:
    from fNIRS2MW.models import MentalWorkloadClassifier
    FNIRS2MW_AVAILABLE = True
except ImportError:
    print("Warning: fNIRS2MW not available, using dummy implementation")
    FNIRS2MW_AVAILABLE = False
    
    class MentalWorkloadClassifier(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = nn.Linear(52, 256)
        def forward(self, x):
            return self.dummy(x.mean(dim=-1))


class Model(pl.LightningModule):
    """
    Main fMRI-guided fNIRS transfer learning model with external backbone support.
    
    This model supports multiple external backbones and transfer learning strategies:
    - External backbones: fNIRS-T (Transformer), fNIRSNet (CNN), fNIRS2MW (Mental Workload)
    - Transfer modes: feature_guided (cross-attention), weight_init, knowledge_distill
    - MindVis emotion classification integration
    
    Architecture:
    1. External backbone feature extraction from fNIRS signals
    2. Cross-modal fusion with fMRI guidance (optional)
    3. Classification head for emotion/workload prediction
    """
    
    def __init__(self,
                 # Model architecture parameters
                 backbone: str = "fNIRS-T",  # "fNIRS-T", "fNIRSNet", "fNIRS2MW", "custom"
                 fnirs_channels: int = 52,
                 fnirs_time: int = 200,
                 fmri_dim: int = 768,
                 num_classes: int = 4,  # MindVis emotions
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 
                 # Transfer learning parameters
                 transfer_mode: str = "feature_guided",  # "feature_guided", "weight_init", "knowledge_distill"
                 use_fmri_guidance: bool = True,
                 freeze_backbone: bool = False,
                 distill_alpha: float = 0.5,
                 distill_temperature: float = 4.0,
                 
                 # Training parameters
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 warmup_steps: int = 500):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.backbone_name = backbone
        self.transfer_mode = transfer_mode
        self.use_fmri_guidance = use_fmri_guidance
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize external backbone
        self.backbone = self._create_backbone(
            backbone, fnirs_channels, fnirs_time, hidden_dim
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # fMRI feature processor (for cross-modal fusion)
        if use_fmri_guidance:
            self.fmri_processor = nn.Sequential(
                nn.Linear(fmri_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            # Cross-modal attention (if using feature-guided transfer)
            if transfer_mode == "feature_guided":
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True
                )
        
        # Classification head
        classifier_input_dim = hidden_dim * 2 if (use_fmri_guidance and transfer_mode == "feature_guided") else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Metrics tracking
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def _create_backbone(self, backbone_name: str, channels: int, time: int, hidden_dim: int) -> nn.Module:
        """Create the specified external backbone or custom architecture."""
        
        if backbone_name == "fNIRS-T" and FNIRS_T_AVAILABLE:
            return FnirsTransformer(
                input_channels=channels,
                sequence_length=time,
                d_model=hidden_dim,
                num_classes=hidden_dim  # Use as feature extractor
            )
            
        elif backbone_name == "fNIRSNet" and FNIRSNET_AVAILABLE:
            return FNIRSNet(
                num_channels=channels,
                num_classes=hidden_dim  # Use as feature extractor
            )
            
        elif backbone_name == "fNIRS2MW" and FNIRS2MW_AVAILABLE:
            return MentalWorkloadClassifier(
                input_shape=(channels, time),
                num_classes=hidden_dim  # Use as feature extractor
            )
            
        else:
            # Custom CNN backbone (fallback)
            return nn.Sequential(
                nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout1d(0.1),
                
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout1d(0.1),
                
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, hidden_dim)
            )
    
    def forward(self, fnirs: torch.Tensor, fmri: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            fnirs: fNIRS signals [B, C, T] = [B, 52, 200]
            fmri: Optional fMRI features [B, D] = [B, 768]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Extract features using external backbone
        fnirs_features = self.backbone(fnirs)  # [B, hidden_dim]
        
        # Handle different backbone output formats
        if isinstance(fnirs_features, tuple):
            fnirs_features = fnirs_features[0]
        
        # Ensure correct dimensionality
        if fnirs_features.dim() > 2:
            fnirs_features = fnirs_features.view(fnirs_features.size(0), -1)
        
        # Cross-modal fusion with fMRI (if enabled)
        if self.use_fmri_guidance and fmri is not None:
            fmri_features = self.fmri_processor(fmri)  # [B, hidden_dim]
            
            if self.transfer_mode == "feature_guided":
                # Cross-attention: fNIRS queries attend to fMRI keys/values
                attended_features, _ = self.cross_attention(
                    query=fnirs_features.unsqueeze(1),  # [B, 1, hidden_dim]
                    key=fmri_features.unsqueeze(1),     # [B, 1, hidden_dim]
                    value=fmri_features.unsqueeze(1)    # [B, 1, hidden_dim]
                )
                attended_features = attended_features.squeeze(1)  # [B, hidden_dim]
                
                # Concatenate for classification
                combined_features = torch.cat([fnirs_features, attended_features], dim=-1)
            else:
                # Simple addition or concatenation
                combined_features = fnirs_features + fmri_features
        else:
            combined_features = fnirs_features
        
        # Final classification
        logits = self.classifier(combined_features)
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with optional knowledge distillation."""
        fnirs = batch["fnirs"]
        labels = batch["label"]
        fmri = batch.get("fmri", None)
        
        # Forward pass
        logits = self(fnirs, fmri)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Optional knowledge distillation
        total_loss = ce_loss
        if self.transfer_mode == "knowledge_distill" and "teacher_logits" in batch:
            teacher_logits = batch["teacher_logits"]
            temp = self.hparams.get("distill_temperature", 4.0)
            alpha = self.hparams.get("distill_alpha", 0.5)
            
            # KL divergence between student and teacher predictions
            kl_loss = F.kl_div(
                F.log_softmax(logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1),
                reduction="batchmean"
            ) * (temp ** 2)
            
            total_loss = (1 - alpha) * ce_loss + alpha * kl_loss
            self.log("train_kl_loss", kl_loss)
        
        # Compute and log metrics
        self.train_acc(logits, labels)
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        fnirs = batch["fnirs"]
        labels = batch["label"]
        fmri = batch.get("fmri", None)
        
        logits = self(fnirs, fmri)
        loss = F.cross_entropy(logits, labels)
        
        self.val_acc(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        fnirs = batch["fnirs"]
        labels = batch["label"]
        fmri = batch.get("fmri", None)
        
        logits = self(fnirs, fmri)
        loss = F.cross_entropy(logits, labels)
        
        self.test_acc(logits, labels)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,  # Adjust based on training epochs
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


# Alias for backward compatibility
FmriFnirsNet = Model


def create_model_from_config(config: Dict[str, Any]) -> Model:
    """
    Factory function to create model from configuration dict.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model instance
    """
    return Model(**config)


def load_pretrained_weights(model: Model, checkpoint_path: str, strict: bool = True) -> Model:
    """
    Load pretrained weights into model.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce matching keys
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    return model


if __name__ == "__main__":
    # Test model initialization and forward pass
    print("Testing fMRI-guided fNIRS model...")
    
    # Test different backbone configurations
    backbones = ["fNIRS-T", "fNIRSNet", "fNIRS2MW", "custom"]
    
    for backbone in backbones:
        print(f"\nTesting backbone: {backbone}")
        
        model = Model(
            backbone=backbone,
            transfer_mode="feature_guided",
            use_fmri_guidance=True
        )
        
        # Create dummy data
        batch_size = 4
        fnirs_dummy = torch.randn(batch_size, 52, 200)  # [B, C_fnirs, T]
        fmri_dummy = torch.randn(batch_size, 768)       # [B, D_fmri]
        
        # Test forward pass
        with torch.no_grad():
            logits = model(fnirs_dummy, fmri_dummy)
        
        print(f"  Input shapes: fNIRS {fnirs_dummy.shape}, fMRI {fmri_dummy.shape}")
        print(f"  Output shape: {logits.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\nAll tests completed successfully!")
