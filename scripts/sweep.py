#!/usr/bin/env python3
"""
Hydra Sweep Script for fMRI-fNIRS Transfer Learning

Launches hyperparameter sweeps using Hydra's multirun functionality.
Automatically logs to WandB project "fmri-fnirs-codespace".
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import wandb
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.fmri_fnirs_net import Model
from data.datamodule import FmriFnirsDataModule


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks and logger"""
    
    # WandB logger
    wandb_logger = WandbLogger(
        project=cfg.experiment.project,
        name=f"{cfg.experiment.name}_{cfg.model.backbone}_{cfg.transfer.mode}",
        tags=cfg.experiment.tags,
        save_dir="./sweep_logs",
        offline=False
    )
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        filename="epoch{epoch:02d}-val_acc{val_acc:.3f}",
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.get("patience", 10),
        mode="min",
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        deterministic=False  # Set to True for full reproducibility (slower)
    )
    
    return trainer


def create_model(cfg: DictConfig) -> Model:
    """Create model from config"""
    model = Model(
        backbone=cfg.model.backbone,
        transfer_mode=cfg.transfer.mode,
        num_classes=cfg.model.get("num_classes", 4),
        fnirs_channels=cfg.model.get("fnirs_channels", 52),
        fnirs_time=cfg.model.get("fnirs_time", 200),
        fmri_dim=cfg.model.get("fmri_features", 768),
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 1e-5),
        distill_alpha=cfg.transfer.get("distill_alpha", 0.5),
        distill_temperature=cfg.transfer.get("distill_temperature", 4.0),
        freeze_backbone=cfg.transfer.get("freeze_backbone", False)
    )
    return model


def create_datamodule(cfg: DictConfig) -> FmriFnirsDataModule:
    """Create data module from config"""
    datamodule = FmriFnirsDataModule(
        fnirs_data_path=cfg.data.get("fnirs_data_path", "./data/fnirs"),
        fmri_data_path=cfg.data.get("fmri_data_path", "./data/fmri"),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 2),
        cv_method=cfg.data.get("cv_method", "loso"),
        test_size=cfg.data.get("test_size", 0.2),
        use_fnirs2mw=cfg.data.get("use_fnirs2mw", True)
    )
    return datamodule


@hydra.main(version_base=None, config_path="../configs", config_name="sweep")
def run_experiment(cfg: DictConfig) -> float:
    """
    Run single experiment with given configuration.
    
    Returns validation accuracy for optimization.
    """
    
    print("=" * 80)
    print("STARTING EXPERIMENT")
    print("=" * 80)
    print(f"Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    
    # Create components
    model = create_model(cfg)
    datamodule = create_datamodule(cfg)
    trainer = create_trainer(cfg)
    
    # Setup data
    datamodule.setup()
    
    # Log model summary
    print(f"\nModel: {cfg.model.backbone} with {cfg.transfer.mode} transfer")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    try:
        # Train model
        trainer.fit(model, datamodule)
        
        # Test model
        test_results = trainer.test(model, datamodule, ckpt_path="best")
        
        # Extract validation accuracy for optimization
        val_acc = trainer.callback_metrics.get("val_acc", 0.0)
        test_acc = test_results[0].get("test_acc", 0.0) if test_results else 0.0
        
        print("=" * 80)
        print("EXPERIMENT COMPLETED")
        print(f"Best Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("=" * 80)
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Close wandb run
        wandb.finish()
        
        return float(val_acc)
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        wandb.finish()
        return 0.0


def main():
    """Main sweep launcher"""
    print("🚀 Launching fMRI-fNIRS Transfer Learning Sweep")
    print(f"📊 Project: fmri-fnirs-codespace")
    print(f"💾 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💿 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run the experiment
    run_experiment()


if __name__ == "__main__":
    main()
    
    print("=== fMRI-fNIRS Transfer Learning Sweep ===")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Extract sweep parameters
    transfer_modes = cfg.sweep.transfer_mode
    distill_alphas = cfg.sweep.distill_alpha  
    freeze_fmri_options = cfg.sweep.freeze_fmri
    learning_rates = cfg.sweep.learning_rate
    batch_sizes = cfg.sweep.batch_size
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        transfer_modes, distill_alphas, freeze_fmri_options, 
        learning_rates, batch_sizes
    ))
    
    print(f"Total experiments: {len(param_combinations)}")
    
    results = []
    
    for i, (transfer_mode, alpha, freeze_fmri, lr, bs) in enumerate(param_combinations, 1):
        
        # Skip invalid combinations
        if transfer_mode != "distill" and alpha != 0.5:
            continue  # Only vary alpha for distillation
            
        print(f"\n--- Experiment {i}/{len(param_combinations)} ---")
        print(f"Mode: {transfer_mode}, Alpha: {alpha}, Freeze: {freeze_fmri}, LR: {lr}, BS: {bs}")
        
        # Build command
        cmd = [
            sys.executable, "src/train.py",
            "--transfer_mode", str(transfer_mode),
            "--distill_alpha", str(alpha),
            "--learning_rate", str(lr),
            "--batch_size", str(bs),
            "--max_epochs", str(cfg.training.max_epochs),
            "--seed", str(cfg.training.seed),
            "--experiment_name", f"sweep_{transfer_mode}_a{alpha}_f{freeze_fmri}_lr{lr}_bs{bs}",
            "--tags", "sweep", transfer_mode, f"alpha_{alpha}"
        ]
        
        if freeze_fmri:
            cmd.append("--freeze_fmri")
            
        # Add other fixed parameters
        if cfg.data.fnirs_data_path:
            cmd.extend(["--fnirs_data_path", cfg.data.fnirs_data_path])
        if cfg.data.fmri_data_path:
            cmd.extend(["--fmri_data_path", cfg.data.fmri_data_path])
            
        try:
            # Run experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2h timeout
            
            if result.returncode == 0:
                print(f"✓ Experiment completed successfully")
                results.append({
                    "transfer_mode": transfer_mode,
                    "distill_alpha": alpha,
                    "freeze_fmri": freeze_fmri,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "status": "success"
                })
            else:
                print(f"✗ Experiment failed with code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                results.append({
                    "transfer_mode": transfer_mode,
                    "distill_alpha": alpha,
                    "freeze_fmri": freeze_fmri,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "status": "failed",
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"✗ Experiment timed out")
            results.append({
                "transfer_mode": transfer_mode,
                "distill_alpha": alpha,
                "freeze_fmri": freeze_fmri,
                "learning_rate": lr,
                "batch_size": bs,
                "status": "timeout"
            })
            
        except Exception as e:
            print(f"✗ Experiment error: {e}")
            results.append({
                "transfer_mode": transfer_mode,
                "distill_alpha": alpha,
                "freeze_fmri": freeze_fmri,
                "learning_rate": lr,
                "batch_size": bs,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print(f"\n=== Sweep Summary ===")
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] != "success"])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    results_path = Path("sweep_results.csv")
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


def manual_sweep():
    """Manual sweep without Hydra (fallback option)."""
    
    # Define sweep parameters
    configs = {
        "transfer_mode": ["feature_guided", "distill", "weight_init"],
        "distill_alpha": [0.0, 0.3, 0.5, 0.7],
        "freeze_fmri": [True, False],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [4, 8]
    }
    
    # Generate combinations (filtered)
    experiments = []
    
    for mode in configs["transfer_mode"]:
        for freeze in configs["freeze_fmri"]:
            for lr in configs["learning_rate"]:
                for bs in configs["batch_size"]:
                    
                    if mode == "distill":
                        # Test different alpha values for distillation
                        for alpha in configs["distill_alpha"]:
                            experiments.append({
                                "transfer_mode": mode,
                                "distill_alpha": alpha,
                                "freeze_fmri": freeze,
                                "learning_rate": lr,
                                "batch_size": bs
                            })
                    else:
                        # Fixed alpha for non-distillation modes
                        experiments.append({
                            "transfer_mode": mode,
                            "distill_alpha": 0.5,  # Unused
                            "freeze_fmri": freeze,
                            "learning_rate": lr,
                            "batch_size": bs
                        })
    
    print(f"Manual sweep: {len(experiments)} experiments")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n--- Manual Experiment {i}/{len(experiments)} ---")
        
        cmd = [
            sys.executable, "src/train.py",
            "--transfer_mode", exp["transfer_mode"],
            "--distill_alpha", str(exp["distill_alpha"]),
            "--learning_rate", str(exp["learning_rate"]),
            "--batch_size", str(exp["batch_size"]),
            "--max_epochs", "50",  # Reduced for sweep
            "--experiment_name", f"manual_sweep_{i}",
            "--tags", "manual_sweep", exp["transfer_mode"]
        ]
        
        if exp["freeze_fmri"]:
            cmd.append("--freeze_fmri")
            
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, timeout=3600)  # 1h timeout
            print(f"✓ Experiment {i} completed")
        except Exception as e:
            print(f"✗ Experiment {i} failed: {e}")


if __name__ == "__main__":
    # Try Hydra first, fallback to manual if config missing
    try:
        sweep_main()
    except Exception as e:
        print(f"Hydra sweep failed: {e}")
        print("Falling back to manual sweep...")
        manual_sweep()
