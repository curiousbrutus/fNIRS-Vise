"""
Complete fMRI-guided fNIRS Transfer Learning Model

Production-ready PyTorch Lightning module for brain signal transfer learning.
"""

from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import math


class FnirsEncoder(nn.Module):
    """
    Lightweight CNN1D encoder for fNIRS signals.
    
    Architecture: 3×Conv1D(k=7,5,3) → GAP → 128-D
    Input: [B, C_fnirs, T] = [B, 52, 200]  
    Output: [B, 128]
    """
    
    def __init__(self, 
                 input_channels: int = 52,
                 input_time: int = 200,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.hidden_dim = hidden_dim
        
        # 3-block CNN1D with progressive channel expansion
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 32, kernel_size=7, stride=2),
            self._make_conv_block(32, 64, kernel_size=5, stride=2), 
            self._make_conv_block(64, 128, kernel_size=3, stride=2)
        ])
        
        # Global average pooling + projection to hidden_dim
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                        kernel_size: int, stride: int) -> nn.Module:
        """Create convolutional block with BatchNorm, ReLU, and Dropout."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: [B, 52, 200] → [B, 128]
        """
        # Apply conv blocks sequentially
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        # Global average pooling: [B, 128, T'] → [B, 128]
        x = self.gap(x).squeeze(-1)
        
        # Project to hidden dimension
        x = self.projection(x)
        
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-attention mechanism for fusing fNIRS and fMRI features.
    fNIRS features attend to fMRI features for guidance.
    """
    
    def __init__(self, fnirs_dim: int, fmri_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fnirs_dim = fnirs_dim
        self.fmri_dim = fmri_dim
        self.hidden_dim = hidden_dim
        
        # Query from fNIRS, Key/Value from fMRI
        self.q_proj = nn.Linear(fnirs_dim, hidden_dim)
        self.k_proj = nn.Linear(fmri_dim, hidden_dim)
        self.v_proj = nn.Linear(fmri_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, fnirs_dim)
        self.norm = nn.LayerNorm(fnirs_dim)
        
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, fnirs_feat: torch.Tensor, fmri_feat: torch.Tensor) -> torch.Tensor:
        """Cross-modal attention fusion."""
        # Compute attention
        q = self.q_proj(fnirs_feat)  # [B, hidden_dim]
        k = self.k_proj(fmri_feat)   # [B, hidden_dim]  
        v = self.v_proj(fmri_feat)   # [B, hidden_dim]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [B, hidden_dim]
        attended = self.out_proj(attended)        # [B, fnirs_dim]
        
        # Residual connection and normalization
        output = self.norm(fnirs_feat + attended)
        
        return output


class FmriFnirsNet(L.LightningModule):
    """
    Complete fMRI-guided fNIRS transfer learning model.
    
    Args:
        fnirs_channels: Number of fNIRS channels (default: 52)
        fmri_dim: fMRI feature dimension (default: 768)
        num_classes: Number of classification classes (default: 4)
        transfer_mode: Transfer learning strategy ("feature_guided", "distill", "concat")
        freeze_fmri: Whether to freeze fMRI adapter during training
        distill_alpha: Weight for distillation loss (0.0-1.0)
        learning_rate: Initial learning rate (default: 3e-4)
        weight_decay: L2 regularization strength (default: 1e-4)
    """
    
    def __init__(self,
                 fnirs_channels: int = 52,
                 fmri_dim: int = 768,
                 num_classes: int = 4,
                 transfer_mode: str = "feature_guided",
                 freeze_fmri: bool = True,
                 distill_alpha: float = 0.5,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Core architecture
        self.fnirs_encoder = FnirsEncoder(
            input_channels=fnirs_channels,
            input_time=200,  # Fixed for this implementation
            hidden_dim=128
        )
        
        # fMRI adapter: 768-D → 128-D
        self.fmri_adapter = nn.Sequential(
            nn.Linear(fmri_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        
        # Fusion mechanism based on transfer mode
        if transfer_mode == "feature_guided":
            self.fusion = CrossModalAttention(128, 128, 128)
            fusion_dim = 128  # Attention preserves dimension
        else:
            # Concatenation for other modes
            self.fusion = lambda x, y: torch.cat([x, y], dim=-1)
            fusion_dim = 256  # 128 + 128
            
        # Classification head: fusion_dim → 64 → num_classes
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Teacher model for distillation (simple MLP)
        if transfer_mode == "distill":
            self.teacher = nn.Sequential(
                nn.Linear(fmri_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
            # Freeze teacher parameters
            for param in self.teacher.parameters():
                param.requires_grad = False
                
        # Freeze fMRI adapter if requested
        if freeze_fmri:
            for param in self.fmri_adapter.parameters():
                param.requires_grad = False
                
        # Metrics for tracking
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, fnirs: torch.Tensor, fmri: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            fnirs: fNIRS data [B, C_fnirs, T]
            fmri: fMRI features [B, D_fmri]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Encode both modalities
        h_fnirs = self.fnirs_encoder(fnirs)   # [B, 128]
        h_fmri = self.fmri_adapter(fmri)      # [B, 128]
        
        # Fuse features based on transfer mode
        if self.hparams.transfer_mode == "feature_guided":
            fused = self.fusion(h_fnirs, h_fmri)  # Cross-attention
        else:
            fused = self.fusion(h_fnirs, h_fmri)  # Concatenation
            
        # Classification
        logits = self.head(fused)
        
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with optional knowledge distillation."""
        fnirs, fmri, labels = batch["fnirs"], batch["fmri"], batch["label"]
        
        # Forward pass
        logits = self(fnirs, fmri)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Knowledge distillation (if enabled)
        if self.hparams.transfer_mode == "distill":
            with torch.no_grad():
                teacher_logits = self.teacher(fmri)
            
            # Soft targets from teacher with temperature scaling
            temperature = 3.0
            distill_loss = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Combined loss
            alpha = self.hparams.distill_alpha
            loss = (1 - alpha) * ce_loss + alpha * distill_loss
            
            self.log("train_distill_loss", distill_loss, prog_bar=True)
        else:
            loss = ce_loss
            
        # Update metrics
        self.train_acc(logits, labels)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce_loss", ce_loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        fnirs, fmri, labels = batch["fnirs"], batch["fmri"], batch["label"]
        
        logits = self(fnirs, fmri)
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics
        self.val_acc(logits, labels)
        
        # Logging
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        fnirs, fmri, labels = batch["fnirs"], batch["fmri"], batch["label"]
        
        logits = self(fnirs, fmri)
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics
        self.test_acc(logits, labels)
        
        # Logging
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        
        return loss
    
    def configure_optimizers(self):
        """Configure AdamW optimizer with CosineAnnealingLR scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


if __name__ == "__main__":
    # Test model initialization and forward pass
    model = FmriFnirsNet(
        fnirs_channels=52,
        fmri_dim=768,
        num_classes=4,
        transfer_mode="feature_guided"
    )
    
    # Create dummy batch
    batch_size = 4
    fnirs_dummy = torch.randn(batch_size, 52, 200)  # [B, C_fnirs, T]
    fmri_dummy = torch.randn(batch_size, 768)       # [B, D_fmri]
    
    # Forward pass
    with torch.no_grad():
        logits = model(fnirs_dummy, fmri_dummy)
        
    print(f"Model Test Results:")
    print(f"  fNIRS input: {fnirs_dummy.shape}")
    print(f"  fMRI input: {fmri_dummy.shape}")
    print(f"  Output logits: {logits.shape}")  # Should be [4, 4]
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print("✓ Model test successful!")
