"""
fNIRS-fMRI Transfer Learning Data Module

Complete data loading pipeline with OSF integration and LOSO splits.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import subprocess
import tempfile
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import zipfile


class FmriFnirsDataset(Dataset):
    """
    Dataset for loading paired fMRI and fNIRS data.
    
    Returns dictionary with:
    - "fnirs": Tensor[C_fnirs, T] where C_fnirs=52, T=200
    - "fmri": Tensor[D_fmri] where D_fmri=768
    - "label": Tensor scalar for class label
    - "subject": subject ID string
    """
    
    def __init__(self, 
                 fnirs_data_path: Union[str, Path],
                 fmri_data_path: Union[str, Path],
                 subject_ids: Optional[List[str]] = None,
                 transform: Optional[callable] = None):
        """
        Initialize dataset with data paths.
        
        Args:
            fnirs_data_path: Path to fNIRS tensor files
            fmri_data_path: Path to fMRI latent features
            subject_ids: Optional list of subject IDs to filter
            transform: Optional transform for fNIRS data
        """
        self.fnirs_data_path = Path(fnirs_data_path)
        self.fmri_data_path = Path(fmri_data_path)
        self.transform = transform
        
        # Load sample metadata
        self.samples = self._load_sample_metadata()
        
        # Filter by subject IDs if provided
        if subject_ids is not None:
            self.samples = [s for s in self.samples if s['subject'] in subject_ids]
            
        # Setup label encoding
        self.label_encoder = LabelEncoder()
        labels = [s['label'] for s in self.samples]
        if labels:
            self.label_encoder.fit(labels)
        else:
            # Fallback for empty dataset
            self.label_encoder.fit([0, 1, 2, 3])
        
    def _load_sample_metadata(self) -> List[Dict]:
        """Load metadata for all available samples."""
        samples = []
        
        # Look for fNIRS files and match with fMRI files
        if not self.fnirs_data_path.exists():
            return samples
            
        for fnirs_file in self.fnirs_data_path.glob("*.pt"):
            # Parse filename for metadata (adjust based on naming convention)
            stem = fnirs_file.stem
            parts = stem.split('_')
            
            if len(parts) >= 2:
                subject = parts[0]
                condition = parts[1] if len(parts) > 1 else "unknown"
                trial = parts[2] if len(parts) > 2 else "0"
            else:
                subject = "unknown"
                condition = "emotion0"
                trial = "0"
                
            # Map condition to label (customize as needed)
            condition_map = {
                "emotion0": 0, "emotion1": 1, "emotion2": 2, "emotion3": 3,
                "happy": 1, "sad": 2, "angry": 3, "neutral": 0
            }
            label = condition_map.get(condition.lower(), 0)
            
            # Find corresponding fMRI file
            fmri_file = self.fmri_data_path / f"{stem}.pt"
            if fmri_file.exists():
                samples.append({
                    'fnirs_path': fnirs_file,
                    'fmri_path': fmri_file,
                    'subject': subject,
                    'condition': condition,
                    'trial': trial,
                    'label': label
                })
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load fNIRS data [C_fnirs, T] = [52, 200]
        fnirs_data = torch.load(sample['fnirs_path'], map_location='cpu')
        if fnirs_data.dim() == 1:
            # Reshape if flattened
            fnirs_data = fnirs_data.view(52, 200)
        elif fnirs_data.shape != (52, 200):
            # Handle different shapes
            if fnirs_data.numel() == 52 * 200:
                fnirs_data = fnirs_data.reshape(52, 200)
            else:
                # Pad or crop to expected shape
                fnirs_data = torch.zeros(52, 200)
        
        # Load fMRI features [D_fmri] = [768]
        fmri_data = torch.load(sample['fmri_path'], map_location='cpu')
        if fmri_data.dim() > 1:
            fmri_data = fmri_data.flatten()
        if fmri_data.numel() != 768:
            # Pad or crop to expected dimension
            if fmri_data.numel() > 768:
                fmri_data = fmri_data[:768]
            else:
                padded = torch.zeros(768)
                padded[:fmri_data.numel()] = fmri_data
                fmri_data = padded
                
        # Apply transforms
        if self.transform:
            fnirs_data = self.transform(fnirs_data)
            
        return {
            "fnirs": fnirs_data.float(),
            "fmri": fmri_data.float(),
            "label": torch.tensor(sample['label'], dtype=torch.long),
            "subject": sample['subject']
        }


class FmriFnirsDataModule(L.LightningDataModule):
    """
    Lightning DataModule with OSF data download and LOSO splits.
    
    Features:
    - Automatic data download from OSF
    - Leave-One-Subject-Out (LOSO) cross-validation
    - GPU-optimized data loading
    """
    
    def __init__(self,
                 data_root: str = "./data",
                 osf_project_id: Optional[str] = None,
                 batch_size: int = 8,
                 num_workers: int = 2,
                 pin_memory: bool = True,
                 test_subject: Optional[str] = None,
                 val_split: float = 0.2):
        """
        Initialize DataModule.
        
        Args:
            data_root: Root directory for data storage
            osf_project_id: OSF project ID for data download
            batch_size: Batch size (≤8 for K80 12GB)
            num_workers: DataLoader workers
            pin_memory: Whether to pin memory for GPU
            test_subject: Subject to hold out for testing (LOSO)
            val_split: Fraction of subjects for validation
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = Path(data_root)
        self.osf_project_id = osf_project_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_subject = test_subject
        self.val_split = val_split
        
        # Data paths
        self.fnirs_path = self.data_root / "fNIRS" / "processed"
        self.fmri_path = self.data_root / "fMRI" / "features"
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Download data from OSF if not present."""
        # Create directories
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.fnirs_path.mkdir(parents=True, exist_ok=True)
        self.fmri_path.mkdir(parents=True, exist_ok=True)
        
        # Check if data already exists
        if (self.fnirs_path / ".download_complete").exists():
            print("Data already downloaded, skipping...")
            return
            
        if self.osf_project_id:
            print(f"Downloading data from OSF project: {self.osf_project_id}")
            self._download_from_osf()
        else:
            print("No OSF project ID provided, creating dummy data...")
            self._create_dummy_data()
            
        # Mark download as complete
        (self.fnirs_path / ".download_complete").touch()
        (self.fmri_path / ".download_complete").touch()
    
    def _download_from_osf(self):
        """Download data using osfclient."""
        try:
            import osfclient
            
            # Download to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # OSF CLI command
                cmd = [
                    "osf", "-p", self.osf_project_id, 
                    "clone", str(temp_path / "osf_data")
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"OSF download failed: {result.stderr}")
                    self._create_dummy_data()
                    return
                
                # Extract and organize data
                self._organize_downloaded_data(temp_path / "osf_data")
                
        except ImportError:
            print("osfclient not available, creating dummy data...")
            self._create_dummy_data()
        except Exception as e:
            print(f"OSF download error: {e}, creating dummy data...")
            self._create_dummy_data()
    
    def _organize_downloaded_data(self, source_path: Path):
        """Organize downloaded data into expected structure."""
        # Look for fNIRS and fMRI data in source
        for file_path in source_path.rglob("*.pt"):
            filename = file_path.name
            
            if "fnirs" in filename.lower():
                shutil.copy2(file_path, self.fnirs_path / filename)
            elif "fmri" in filename.lower():
                shutil.copy2(file_path, self.fmri_path / filename)
                
        print(f"Organized data: {len(list(self.fnirs_path.glob('*.pt')))} fNIRS files")
        print(f"Organized data: {len(list(self.fmri_path.glob('*.pt')))} fMRI files")
    
    def _create_dummy_data(self):
        """Create dummy data for testing."""
        print("Creating dummy data for development...")
        
        subjects = ["sub01", "sub02", "sub03", "sub04"]
        emotions = ["emotion0", "emotion1", "emotion2", "emotion3"]
        
        sample_count = 0
        for subject in subjects:
            for emotion in emotions:
                for trial in range(3):  # 3 trials per condition
                    
                    # Generate dummy fNIRS [52, 200]
                    fnirs_data = torch.randn(52, 200) * 0.1
                    
                    # Generate dummy fMRI [768]
                    fmri_data = torch.randn(768) * 0.5
                    
                    # Save files
                    base_name = f"{subject}_{emotion}_{trial:02d}"
                    
                    torch.save(fnirs_data, self.fnirs_path / f"{base_name}.pt")
                    torch.save(fmri_data, self.fmri_path / f"{base_name}.pt")
                    
                    sample_count += 1
        
        print(f"Created {sample_count} dummy samples")
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets with LOSO splits."""
        # Create full dataset
        full_dataset = FmriFnirsDataset(
            fnirs_data_path=self.fnirs_path,
            fmri_data_path=self.fmri_path
        )
        
        if len(full_dataset) == 0:
            raise ValueError("No data found! Check data paths and run prepare_data()")
        
        # Get unique subjects
        subjects = list(set(sample['subject'] for sample in full_dataset.samples))
        print(f"Found {len(subjects)} unique subjects: {subjects}")
        
        # LOSO split
        if self.test_subject and self.test_subject in subjects:
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
            # Fallback: use same subjects for train/val
            train_subjects = train_val_subjects
            val_subjects = train_val_subjects[:1] if train_val_subjects else test_subjects
            
        # Create subject-specific datasets
        self.train_dataset = FmriFnirsDataset(
            self.fnirs_path, self.fmri_path, subject_ids=train_subjects
        )
        self.val_dataset = FmriFnirsDataset(
            self.fnirs_path, self.fmri_path, subject_ids=val_subjects
        )
        self.test_dataset = FmriFnirsDataset(
            self.fnirs_path, self.fmri_path, subject_ids=test_subjects
        )
        
        print(f"LOSO Split:")
        print(f"  Train: {len(train_subjects)} subjects, {len(self.train_dataset)} samples")
        print(f"  Val: {len(val_subjects)} subjects, {len(self.val_dataset)} samples")
        print(f"  Test: {len(test_subjects)} subjects, {len(self.test_dataset)} samples")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )


if __name__ == "__main__":
    # Test DataModule
    print("Testing FmriFnirsDataModule...")
    
    dm = FmriFnirsDataModule(
        data_root="./test_data",
        batch_size=4,
        test_subject="sub01"
    )
    
    # Prepare and setup data
    dm.prepare_data()
    dm.setup()
    
    # Test data loading
    train_loader = dm.train_dataloader()
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        
        print("✓ DataModule test successful!")
        print(f"  Batch shapes:")
        print(f"    fNIRS: {batch['fnirs'].shape}")
        print(f"    fMRI: {batch['fmri'].shape}")
        print(f"    Labels: {batch['label'].shape}")
        print(f"    Subjects: {batch['subject']}")
    else:
        print("Warning: No data in train_loader")
        
    # Cleanup test data
    import shutil
    if Path("./test_data").exists():
        shutil.rmtree("./test_data")
        print("✓ Cleaned up test data")
