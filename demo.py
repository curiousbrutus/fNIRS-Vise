#!/usr/bin/env python3
"""
Quick demo script to test the transfer learning pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.models.fmri_fnirs_net import Model as FmriGuidedFnirsNet
from src.data.datamodule import FmriFnirsDataModule
from src.utils import memory_usage_analysis

def create_dummy_data():
    """Create dummy data for testing."""
    print("Creating dummy data...")
    
    # Create data directory
    data_dir = Path("./data/dummy")
    fnirs_dir = data_dir / "fNIRS" / "nemo_pre"
    fmri_dir = data_dir / "mindvis_features"
    
    fnirs_dir.mkdir(parents=True, exist_ok=True)
    fmri_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy samples
    subjects = ["sub01", "sub02", "sub03"]
    emotions = ["emotion0", "emotion1", "emotion2", "emotion3"]
    
    sample_count = 0
    for subject in subjects:
        for emotion in emotions:
            for trial in range(3):  # 3 trials per condition
                
                # Generate dummy fNIRS data [52, 200]
                fnirs_data = torch.randn(52, 200) * 0.1  # Small values like real fNIRS
                
                # Generate dummy fMRI features [768]
                fmri_data = torch.randn(768) * 0.5
                
                # Save files
                base_name = f"{subject}_{emotion}_{trial:02d}"
                
                torch.save(fnirs_data, fnirs_dir / f"{base_name}.pt")
                torch.save(fmri_data, fmri_dir / f"{base_name}.pt")
                
                sample_count += 1
    
    print(f"Created {sample_count} dummy samples in {data_dir}")
    return str(fnirs_dir), str(fmri_dir)

def test_model_architectures():
    """Test different model configurations."""
    print("\n=== Testing Model Architectures ===")
    
    configs = [
        {"transfer_mode": "feature_guided", "name": "Cross-Attention"},
        {"transfer_mode": "knowledge_distill", "distill_alpha": 0.5, "name": "Knowledge Distillation"},  
        {"transfer_mode": "weight_init", "name": "Weight Initialization"}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        model = FmriGuidedFnirsNet(**{k: v for k, v in config.items() if k != 'name'})
        
        # Test forward pass
        batch_size = 4
        fnirs_dummy = torch.randn(batch_size, 52, 200)
        fmri_dummy = torch.randn(batch_size, 768)
        
        with torch.no_grad():
            logits = model(fnirs_dummy, fmri_dummy)
            
        print(f"  ✓ Output shape: {logits.shape}")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

def test_data_loading():
    """Test data loading pipeline."""
    print("\n=== Testing Data Loading ===")
    
    # Create dummy data
    fnirs_path, fmri_path = create_dummy_data()
    
    # Test DataModule
    dm = FmriFnirsDataModule(
        data_root=str(Path("./data/dummy").parent),
        batch_size=4,
        test_subject="sub01"
    )
    
    dm.setup()
    
    print(f"Train samples: {len(dm.train_dataset)}")
    print(f"Val samples: {len(dm.val_dataset)}")  
    print(f"Test samples: {len(dm.test_dataset)}")
    
    # Test data loading
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  fNIRS: {batch['fnirs'].shape}")
    print(f"  fMRI: {batch['fmri'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  ✓ Data loading successful")

def test_training_step():
    """Test a single training step."""
    print("\n=== Testing Training Step ===")
    
    # Create model
    model = FmriGuidedFnirsNet(transfer_mode="feature_guided")
    
    # Create dummy batch
    batch = {
        "fnirs": torch.randn(4, 52, 200),
        "fmri": torch.randn(4, 768),
        "label": torch.randint(0, 4, (4,))
    }
    
    # Test training step
    model.train()
    loss = model.training_step(batch, 0)
    
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training step successful")

def test_memory_usage():
    """Test GPU memory usage."""
    print("\n=== Testing Memory Usage ===")
    
    if torch.cuda.is_available():
        model = FmriGuidedFnirsNet().cuda()
        
        memory_stats = memory_usage_analysis(model, {
            "fnirs": (8, 52, 200),  # Max batch size
            "fmri": (8, 768)
        })
        
        print("✓ Memory analysis completed")
    else:
        print("CUDA not available, skipping memory test")

def main():
    """Run all demo tests."""
    print("🧠 fMRI-fNIRS Transfer Learning Demo")
    print("=" * 40)
    
    try:
        test_model_architectures()
        test_data_loading() 
        test_training_step()
        test_memory_usage()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("1. Prepare your real fNIRS and fMRI data")
        print("2. Run: python src/train.py --transfer_mode feature_guided")
        print("3. Monitor training with W&B")
        print("4. Evaluate with: python scripts/evaluate.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
