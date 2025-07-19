"""
Main training script for fMRI-teacher → fNIRS-student transfer learning.

Usage:
    python train.py --transfer_mode feature_guided --batch_size 8
    python train.py --transfer_mode distill --distill_alpha 0.5 --freeze_fmri
"""

import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor, 
    DeviceStatsMonitor, ModelPruning
)
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data_module import FmriFnirsDataModule
from src.model import FmriGuidedFnirsNet, load_pretrained_weights

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def create_trainer(args) -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks and logger."""
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_acc",
            patience=10,
            mode="max",
            verbose=True
        ),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename=f"{args.transfer_mode}-{{epoch:02d}}-{{val_acc:.3f}}",
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Add GPU monitoring if available
    if torch.cuda.is_available():
        callbacks.append(DeviceStatsMonitor())
    
    # W&B Logger
    logger = WandbLogger(
        project="fmri-fnirs-transfer",
        name=f"{args.transfer_mode}_alpha{args.distill_alpha}_freeze{args.freeze_fmri}",
        save_dir="./logs",
        log_model=True
    )
    
    # Trainer configuration for Colab K80 12GB
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision for memory
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,  # Gradient accumulation
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        deterministic=True,  # For reproducibility
        benchmark=False,     # Disable for deterministic behavior
    )
    
    return trainer


def main():
    """Main training function."""
    parser = ArgumentParser(description="fMRI-fNIRS Transfer Learning")
    
    # Model arguments
    parser.add_argument("--transfer_mode", type=str, default="feature_guided",
                       choices=["feature_guided", "weight_init", "distill"],
                       help="Transfer learning strategy")
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                       help="Weight for distillation loss (0.0-1.0)")
    parser.add_argument("--freeze_fmri", action="store_true", default=False,
                       help="Freeze fMRI adapter during training")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for encoders")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="L2 regularization strength")
    
    # Data arguments
    parser.add_argument("--fnirs_data_path", type=str, default="./data/fNIRS/nemo_pre",
                       help="Path to fNIRS preprocessed data")
    parser.add_argument("--fmri_data_path", type=str, default="./data/mindvis_features", 
                       help="Path to fMRI MinD-Vis features")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (keep ≤8 for K80 12GB)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    parser.add_argument("--test_subject", type=str, default=None,
                       help="Subject ID for LOSO test split")
    
    # Training arguments  
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum training epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Transfer learning arguments
    parser.add_argument("--pretrained_fmri_path", type=str, default=None,
                       help="Path to pre-trained fMRI model for weight init")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Custom experiment name for logging")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                       help="Tags for W&B experiment")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Validate GPU memory constraints
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 8.0 and args.batch_size > 4:
            print(f"Warning: Large batch size ({args.batch_size}) for limited GPU memory")
            
    # Initialize data module
    print("Setting up data...")
    dm = FmriFnirsDataModule(
        fnirs_data_path=args.fnirs_data_path,
        fmri_data_path=args.fmri_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_subject=args.test_subject
    )
    
    # Initialize model
    print(f"Creating model with transfer mode: {args.transfer_mode}")
    model = FmriGuidedFnirsNet(
        transfer_mode=args.transfer_mode,
        distill_alpha=args.distill_alpha,
        freeze_fmri=args.freeze_fmri,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load pre-trained weights if specified
    if args.pretrained_fmri_path and args.transfer_mode == "weight_init":
        print(f"Loading pre-trained weights from {args.pretrained_fmri_path}")
        model = load_pretrained_weights(model, args.pretrained_fmri_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} / {total_params:,} trainable")
    
    # Create trainer
    trainer = create_trainer(args)
    
    # Log hyperparameters to W&B
    if trainer.logger:
        trainer.logger.log_hyperparams(vars(args))
        if args.tags:
            wandb.config.update({"tags": args.tags})
    
    # Training
    print("Starting training...")
    try:
        trainer.fit(model, dm)
        
        # Test on best checkpoint
        if trainer.checkpoint_callback.best_model_path:
            print(f"Testing best model: {trainer.checkpoint_callback.best_model_path}")
            trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path, datamodule=dm)
        else:
            print("Testing current model...")
            trainer.test(model, dm)
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Cleanup
    if trainer.logger:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
