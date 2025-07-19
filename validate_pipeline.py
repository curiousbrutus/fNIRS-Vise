#!/usr/bin/env python3
"""
Transfer Learning Pipeline Validation Script
============================================

Quick validation script to ensure all components of the fMRI-teacher → fNIRS-student 
transfer learning pipeline are working correctly.

Usage:
    python validate_pipeline.py [--quick] [--verbose]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import pytorch_lightning as pl


def validate_imports():
    """Validate all critical imports work."""
    print("🔍 Validating imports...")
    
    try:
        from src.models.fmri_fnirs_net import Model
        print("  ✅ Model import successful")
    except Exception as e:
        print(f"  ❌ Model import failed: {e}")
        return False
    
    try:
        from src.data.datamodule import FmriFnirsDataModule
        print("  ✅ DataModule import successful")
    except Exception as e:
        print(f"  ❌ DataModule import failed: {e}")
        return False
    
    return True


def validate_model_architectures():
    """Test all backbone variants."""
    print("\n🏗️ Validating model architectures...")
    
    backbones = ["fNIRS-T", "fNIRSNet", "fNIRS2MW", "custom"]
    transfer_modes = ["feature_guided", "weight_init", "knowledge_distill"]
    
    from src.models.fmri_fnirs_net import Model
    
    for backbone in backbones:
        for transfer_mode in transfer_modes:
            try:
                model = Model(
                    backbone=backbone,
                    transfer_mode=transfer_mode,
                    use_fmri_guidance=True
                )
                
                # Test forward pass
                batch_size = 2
                fnirs = torch.randn(batch_size, 52, 200)
                fmri = torch.randn(batch_size, 768)
                
                with torch.no_grad():
                    logits = model(fnirs, fmri)
                    assert logits.shape == (batch_size, 4), f"Wrong output shape: {logits.shape}"
                
                print(f"  ✅ {backbone} + {transfer_mode}")
                
            except Exception as e:
                print(f"  ❌ {backbone} + {transfer_mode} failed: {e}")
                return False
    
    return True


def validate_data_pipeline():
    """Test data loading and batch creation."""
    print("\n📊 Validating data pipeline...")
    
    try:
        from src.data.datamodule import FmriFnirsDataModule, FmriFnirsDataset
        import tempfile
        
        # Create temporary directory for dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create datamodule
            dm = FmriFnirsDataModule(
                data_root=temp_path,
                batch_size=4
            )
            
            # Prepare dummy data  
            dm.prepare_data()
            
            # Create dataset directly with tensor format (bypass fNIRS2MW)
            dataset = FmriFnirsDataset(
                fnirs_data_path=dm.fnirs_path,
                fmri_data_path=dm.fmri_path,
                use_fnirs2mw=False  # Force tensor format
            )
            
            if len(dataset) == 0:
                print("  ⚠️  No saved data found, using fallback dummy generation")
                # The dataset will auto-generate dummy data in _load_tensor_data
            
            print(f"  ✅ Dataset length: {len(dataset)}")
            
            # Test single sample
            sample = dataset[0]
            assert "fnirs" in sample, "Missing fNIRS data in sample"
            assert "fmri" in sample, "Missing fMRI data in sample"
            assert "labels" in sample, "Missing labels in sample"
            
            print(f"  ✅ Sample shapes: fNIRS {sample['fnirs'].shape}, fMRI {sample['fmri'].shape}")
            
            # Test dataloader creation
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            
            assert "fnirs" in batch, "Missing fNIRS data in batch"
            assert "fmri" in batch, "Missing fMRI data in batch"
            assert "labels" in batch, "Missing labels in batch"
            
            print(f"  ✅ Batch shapes: fNIRS {batch['fnirs'].shape}, fMRI {batch['fmri'].shape}, Labels {batch['labels'].shape}")
        
    except Exception as e:
        print(f"  ❌ Data pipeline failed: {e}")
        return False
    
    return True


def validate_training_loop():
    """Test training step execution."""
    print("\n🏋️ Validating training loop...")
    
    try:
        from src.models.fmri_fnirs_net import Model
        from src.data.datamodule import FmriFnirsDataset
        import tempfile
        
        # Create model
        model = Model(backbone="custom", transfer_mode="feature_guided")
        
        # Create simple dataset with dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset = FmriFnirsDataset(
                fnirs_data_path=temp_path / "fnirs",
                fmri_data_path=temp_path / "fmri",
                use_fnirs2mw=False  # Use dummy data generation
            )
            
            # Create dataloader
            from torch.utils.data import DataLoader
            train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
            
            # Test training step
            model.train()
            batch = next(iter(train_loader))
            loss = model.training_step(batch, 0)
            
            assert isinstance(loss, torch.Tensor), "Training step should return tensor"
            assert loss.numel() == 1, "Loss should be scalar"
            assert not torch.isnan(loss), "Loss should not be NaN"
            
            print(f"  ✅ Training step successful, loss: {loss.item():.4f}")
            
            # Test validation step
            model.eval()
            val_loss = model.validation_step(batch, 0)
            
            print(f"  ✅ Validation step successful, loss: {val_loss.item():.4f}")
        
    except Exception as e:
        print(f"  ❌ Training loop failed: {e}")
        return False
    
    return True


def validate_configuration():
    """Test Hydra configuration loading."""
    print("\n⚙️ Validating configuration...")
    
    try:
        import hydra
        from hydra import compose, initialize
        from omegaconf import DictConfig
        
        # Test base config
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="config")
            print(f"  ✅ Base config loaded: {list(cfg.keys())}")
        
        # Test sweep config  
        with initialize(version_base=None, config_path="configs"):
            sweep_cfg = compose(config_name="sweep")
            print(f"  ✅ Sweep config loaded: {list(sweep_cfg.keys())}")
        
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False
    
    return True


def run_full_validation(quick=False, verbose=False):
    """Run complete pipeline validation."""
    print("🚀 Starting Transfer Learning Pipeline Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run validation steps
    steps = [
        ("Imports", validate_imports),
        ("Model Architectures", validate_model_architectures),
        ("Data Pipeline", validate_data_pipeline),
        ("Configuration", validate_configuration),
    ]
    
    if not quick:
        steps.append(("Training Loop", validate_training_loop))
    
    results = {}
    for step_name, step_func in steps:
        try:
            success = step_func()
            results[step_name] = success
            if not success:
                print(f"\n❌ {step_name} validation failed!")
                break
        except Exception as e:
            print(f"\n💥 {step_name} validation crashed: {e}")
            results[step_name] = False
            if verbose:
                import traceback
                traceback.print_exc()
            break
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = all(results.values())
    
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{step:20s}: {status}")
    
    if all_passed:
        print(f"\n🎉 ALL VALIDATIONS PASSED! ({elapsed:.2f}s)")
        print("\n📋 Your fMRI→fNIRS transfer learning pipeline is ready!")
        print("\nNext steps:")
        print("  1. Run full tests: pytest tests/ -v")
        print("  2. Start training: python scripts/sweep.py")
        print("  3. Check roadmap: cat TRANSFER_LEARNING_ROADMAP.md")
    else:
        print(f"\n⚠️ SOME VALIDATIONS FAILED! ({elapsed:.2f}s)")
        print("\nTroubleshooting:")
        print("  1. Check error messages above")
        print("  2. Ensure pip install -e . completed successfully")
        print("  3. Review TRANSFER_LEARNING_ROADMAP.md")
        return False
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate transfer learning pipeline")
    parser.add_argument("--quick", action="store_true", 
                       help="Skip time-intensive validations")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed error messages")
    
    args = parser.parse_args()
    
    success = run_full_validation(quick=args.quick, verbose=args.verbose)
    sys.exit(0 if success else 1)
