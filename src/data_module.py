"""
fNIRS-fMRI Transfer Learning Dataset Module

This module provides the data loading utilities for the fMRI-teacher → fNIRS-student 
transfer learning pipeline. Handles loading fMRI latent features and fNIRS signals
with proper train/test splits using Leave-One-Subject-Out (LOSO) validation.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


class FmriFnirsDataset(Dataset):
    """
    Dataset for loading paired fMRI and fNIRS data with emotion labels.
    
    Returns dictionary with:
    - "fnirs": Tensor[C_fnirs, T] where C_fnirs=52 channels, T=200 timepoints
    - "fmri": Tensor[D_fmri] where D_fmri=768 (MinD-Vis pooled features)  
    - "label": Tensor scalar for emotion class 0-3
    - "subject": subject ID for LOSO splits
    """
    
    def __init__(self, 
                 fnirs_data_path: Union[str, Path],
                 fmri_data_path: Union[str, Path],
                 subject_ids: Optional[List[str]] = None,
                 transform: Optional[callable] = None):
        """
        Initialize dataset with paths to fNIRS and fMRI data.
        
        Args:
            fnirs_data_path: Path to fNIRS tensor files (.pt or .npy)
            fmri_data_path: Path to fMRI latent features (.pt or .npy) 
            subject_ids: Optional list of subject IDs to filter
            transform: Optional transform to apply to fNIRS data
        """
        self.fnirs_data_path = Path(fnirs_data_path)
        self.fmri_data_path = Path(fmri_data_path)
        self.transform = transform
        
        # Load data indices and metadata
        self.samples = self._load_sample_metadata()
        
        if subject_ids is not None:
            self.samples = [s for s in self.samples if s['subject'] in subject_ids]
            
        self.label_encoder = LabelEncoder()
        labels = [s['label'] for s in self.samples]
        self.label_encoder.fit(labels)
        
    def _load_sample_metadata(self) -> List[Dict]:
        """Load metadata for all available samples."""
        samples = []
        
        # TODO: Implement actual data loading based on your file structure
        # This is a template - adjust paths based on your actual data organization
        for fnirs_file in self.fnirs_data_path.glob("*.pt"):
            # Extract metadata from filename (adjust parsing logic as needed)
            parts = fnirs_file.stem.split('_')
            subject = parts[0]  # e.g., "sub01" 
            condition = parts[1]  # e.g., "emotion1"
            trial = parts[2] if len(parts) > 2 else "0"
            
            # Map condition to emotion label (adjust mapping as needed)
            emotion_map = {"emotion0": 0, "emotion1": 1, "emotion2": 2, "emotion3": 3}
            label = emotion_map.get(condition, 0)
            
            # Find corresponding fMRI file
            fmri_file = self.fmri_data_path / f"{fnirs_file.stem}.pt"
            if fmri_file.exists():
                samples.append({
                    'fnirs_path': fnirs_file,
                    'fmri_path': fmri_file,
                    'subject': subject,
                    'label': label,
                    'condition': condition,
                    'trial': trial
                })
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load fNIRS data [C_fnirs, T] = [52, 200]
        fnirs_data = torch.load(sample['fnirs_path'])
        if fnirs_data.dim() == 1:
            # Reshape if flattened: [52*200] -> [52, 200]
            fnirs_data = fnirs_data.view(52, 200)
        
        # Load fMRI latent features [D_fmri] = [768]
        fmri_data = torch.load(sample['fmri_path'])
        if fmri_data.dim() > 1:
            fmri_data = fmri_data.flatten()  # Ensure 1D
            
        # Apply transforms
        if self.transform:
            fnirs_data = self.transform(fnirs_data)
            
        return {
            "fnirs": fnirs_data.float(),
            "fmri": fmri_data.float(), 
            "label": torch.tensor(sample['label'], dtype=torch.long),
            "subject": sample['subject']
        }


class FmriFnirsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for fMRI-fNIRS transfer learning.
    
    Implements Leave-One-Subject-Out (LOSO) cross-validation splits
    with proper data loading for GPU training.
    """
    
    def __init__(self,
                 fnirs_data_path: str = "./data/fNIRS/nemo_pre",
                 fmri_data_path: str = "./data/mindvis_features", 
                 batch_size: int = 8,
                 num_workers: int = 2,
                 pin_memory: bool = True,
                 test_subject: Optional[str] = None,
                 val_split: float = 0.2):
        """
        Initialize DataModule.
        
        Args:
            fnirs_data_path: Path to fNIRS preprocessed tensors
            fmri_data_path: Path to fMRI MinD-Vis features  
            batch_size: Batch size for training (keep ≤8 for K80 12GB)
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory for GPU transfer
            test_subject: Subject ID to hold out for testing (LOSO)
            val_split: Fraction of remaining subjects for validation
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.fnirs_data_path = fnirs_data_path
        self.fmri_data_path = fmri_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_subject = test_subject
        self.val_split = val_split
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None 
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        if self.dataset is None:
            self.dataset = FmriFnirsDataset(
                fnirs_data_path=self.fnirs_data_path,
                fmri_data_path=self.fmri_data_path
            )
            
        # Get all unique subjects
        subjects = list(set(sample['subject'] for sample in self.dataset.samples))
        
        if self.test_subject is not None:
            # LOSO: hold out specified test subject
            test_subjects = [self.test_subject]
            train_val_subjects = [s for s in subjects if s != self.test_subject]
        else:
            # Use first subject as test by default
            test_subjects = [subjects[0]]
            train_val_subjects = subjects[1:]
            
        # Split remaining subjects into train/val
        if len(train_val_subjects) > 1:
            n_val = max(1, int(len(train_val_subjects) * self.val_split))
            val_subjects = train_val_subjects[:n_val]
            train_subjects = train_val_subjects[n_val:]
        else:
            # Only one subject left - use for both train and val
            train_subjects = train_val_subjects
            val_subjects = train_val_subjects
            
        # Create subject-specific datasets
        self.train_dataset = FmriFnirsDataset(
            self.fnirs_data_path, self.fmri_data_path, subject_ids=train_subjects
        )
        self.val_dataset = FmriFnirsDataset(
            self.fnirs_data_path, self.fmri_data_path, subject_ids=val_subjects
        )
        self.test_dataset = FmriFnirsDataset(
            self.fnirs_data_path, self.fmri_data_path, subject_ids=test_subjects
        )
        
        print(f"LOSO Split - Train: {len(train_subjects)} subjects ({len(self.train_dataset)} samples)")
        print(f"             Val: {len(val_subjects)} subjects ({len(self.val_dataset)} samples)")  
        print(f"             Test: {len(test_subjects)} subjects ({len(self.test_dataset)} samples)")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # For consistent batch sizes
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def create_loso_splits(dataset: FmriFnirsDataset, n_splits: int = 5) -> List[Tuple[List[str], List[str]]]:
    """
    Create Leave-One-Subject-Out splits for cross-validation.
    
    Args:
        dataset: The dataset to split
        n_splits: Number of CV folds (for GroupKFold)
        
    Returns:
        List of (train_subjects, test_subjects) tuples
    """
    subjects = list(set(sample['subject'] for sample in dataset.samples))
    
    splits = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        test_subjects = [test_subject]
        splits.append((train_subjects, test_subjects))
        
    return splits


if __name__ == "__main__":
    # Test the DataModule
    dm = FmriFnirsDataModule(batch_size=4)
    dm.setup()
    
    # Test data loading
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print("Batch shapes:")
    print(f"  fNIRS: {batch['fnirs'].shape}")  # Should be [4, 52, 200]
    print(f"  fMRI: {batch['fmri'].shape}")    # Should be [4, 768]  
    print(f"  Labels: {batch['label'].shape}") # Should be [4]
    print(f"  Subjects: {batch['subject']}")
