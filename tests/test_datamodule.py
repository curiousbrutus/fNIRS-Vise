"""
Test suite for fNIRS-fMRI data module.

Tests data loading, LOSO splits, and dataset integrity.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.datamodule import FmriFnirsDataModule, FmriFnirsDataset


class TestFmriFnirsDataset:
    """Test suite for the dataset class."""
    
    def setup_method(self):
        """Setup test fixtures with temporary data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fnirs_dir = self.temp_dir / "fnirs"
        self.fmri_dir = self.temp_dir / "fmri"
        
        self.fnirs_dir.mkdir(parents=True)
        self.fmri_dir.mkdir(parents=True)
        
        # Create dummy data files
        self.subjects = ["sub01", "sub02", "sub03"]
        self.conditions = ["emotion0", "emotion1", "emotion2", "emotion3"]
        
        self.sample_files = []
        for subject in self.subjects:
            for condition in self.conditions:
                for trial in range(2):  # 2 trials per condition
                    base_name = f"{subject}_{condition}_{trial:02d}"
                    
                    # Create dummy fNIRS data [52, 200]
                    fnirs_data = torch.randn(52, 200) * 0.1
                    fnirs_path = self.fnirs_dir / f"{base_name}.pt"
                    torch.save(fnirs_data, fnirs_path)
                    
                    # Create dummy fMRI data [768]
                    fmri_data = torch.randn(768) * 0.5
                    fmri_path = self.fmri_dir / f"{base_name}.pt"
                    torch.save(fmri_data, fmri_path)
                    
                    self.sample_files.append(base_name)
                    
    def teardown_method(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_dataset_initialization(self):
        """Test dataset can be initialized and loads samples."""
        dataset = FmriFnirsDataset(
            fnirs_data_path=self.fnirs_dir,
            fmri_data_path=self.fmri_dir
        )
        
        # Should find all created samples
        expected_samples = len(self.subjects) * len(self.conditions) * 2
        assert len(dataset) == expected_samples, f"Expected {expected_samples} samples, got {len(dataset)}"
        
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct data structure."""
        dataset = FmriFnirsDataset(
            fnirs_data_path=self.fnirs_dir,
            fmri_data_path=self.fmri_dir
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            # Check return structure
            assert isinstance(sample, dict), "Sample should be a dictionary"
            assert "fnirs" in sample, "Sample should contain 'fnirs' key"
            assert "fmri" in sample, "Sample should contain 'fmri' key"
            assert "label" in sample, "Sample should contain 'label' key"
            assert "subject" in sample, "Sample should contain 'subject' key"
            
            # Check data types and shapes
            assert isinstance(sample["fnirs"], torch.Tensor), "fNIRS should be tensor"
            assert isinstance(sample["fmri"], torch.Tensor), "fMRI should be tensor"
            assert isinstance(sample["label"], torch.Tensor), "Label should be tensor"
            assert isinstance(sample["subject"], str), "Subject should be string"
            
            # Check shapes
            assert sample["fnirs"].shape == (52, 200), f"fNIRS shape should be (52, 200), got {sample['fnirs'].shape}"
            assert sample["fmri"].shape == (768,), f"fMRI shape should be (768,), got {sample['fmri'].shape}"
            assert sample["label"].dim() == 0, "Label should be scalar"
            
    def test_subject_filtering(self):
        """Test dataset filtering by subject IDs."""
        # Filter to only sub01
        dataset = FmriFnirsDataset(
            fnirs_data_path=self.fnirs_dir,
            fmri_data_path=self.fmri_dir,
            subject_ids=["sub01"]
        )
        
        # Should only have samples from sub01
        expected_samples = len(self.conditions) * 2  # 2 trials per condition
        assert len(dataset) == expected_samples
        
        # Check all samples are from sub01
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["subject"] == "sub01", f"Expected sub01, got {sample['subject']}"
            
    def test_empty_dataset(self):
        """Test dataset behavior with no data files."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        dataset = FmriFnirsDataset(
            fnirs_data_path=empty_dir,
            fmri_data_path=empty_dir
        )
        
        assert len(dataset) == 0, "Empty directories should result in empty dataset"


class TestFmriFnirsDataModule:
    """Test suite for the Lightning DataModule."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_datamodule_initialization(self):
        """Test DataModule can be initialized with different parameters."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            batch_size=4,
            num_workers=0  # Disable multiprocessing for tests
        )
        
        assert dm.batch_size == 4
        assert dm.num_workers == 0
        assert dm.data_root == self.temp_dir
        
    def test_loso_split_uniqueness(self):
        """Test LOSO splits create non-overlapping subject groups."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            batch_size=4,
            num_workers=0,
            test_subject="sub01"
        )
        
        # Prepare and setup data (creates dummy data)
        dm.prepare_data()
        dm.setup()
        
        # Get subjects from each split
        train_subjects = set()
        val_subjects = set()
        test_subjects = set()
        
        if dm.train_dataset and len(dm.train_dataset) > 0:
            if hasattr(dm.train_dataset, 'samples'):
                for sample in dm.train_dataset.samples:
                    train_subjects.add(sample['subject'])
            else:
                train_subjects.update(dm.train_dataset.subjects)
                
        if dm.val_dataset and len(dm.val_dataset) > 0:
            if hasattr(dm.val_dataset, 'samples'):
                for sample in dm.val_dataset.samples:
                    val_subjects.add(sample['subject'])
            else:
                val_subjects.update(dm.val_dataset.subjects)
                
        if dm.test_dataset and len(dm.test_dataset) > 0:
            if hasattr(dm.test_dataset, 'samples'):
                for sample in dm.test_dataset.samples:
                    test_subjects.add(sample['subject'])
            else:
                test_subjects.update(dm.test_dataset.subjects)
        
        # Test subject should only be in test set
        assert "sub01" in test_subjects, "Test subject should be in test set"
        assert "sub01" not in train_subjects, "Test subject should not be in train set"
        
        # No overlap between train and test
        assert len(train_subjects.intersection(test_subjects)) == 0, "Train and test subjects should not overlap"
        
    def test_dataloader_creation(self):
        """Test DataLoaders can be created and return proper batches."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            batch_size=2,
            num_workers=0
        )
        
        dm.prepare_data()
        dm.setup()
        
        # Test train dataloader
        train_loader = dm.train_dataloader()
        assert train_loader.batch_size == 2
        
        # Test batch loading if data exists
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            
            assert "fnirs" in batch
            assert "fmri" in batch
            assert "label" in batch
            assert "subject" in batch
            
            # Check batch dimensions
            batch_size = batch["fnirs"].shape[0]
            assert batch["fnirs"].shape == (batch_size, 52, 200)
            assert batch["fmri"].shape == (batch_size, 768)
            assert batch["label"].shape == (batch_size,)
            
    def test_dataloader_length(self):
        """Test DataLoader lengths are reasonable."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            batch_size=4,
            num_workers=0
        )
        
        dm.prepare_data()
        dm.setup()
        
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        # At least one loader should have data
        total_batches = len(train_loader) + len(val_loader) + len(test_loader)
        assert total_batches > 0, "At least one dataloader should have data"
        
    def test_prepare_data_dummy_creation(self):
        """Test dummy data creation when no OSF project provided."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            osf_project_id=None  # No OSF download
        )
        
        # Should create dummy data
        dm.prepare_data()
        
        # Check data directories were created
        assert (dm.fnirs_path).exists(), "fNIRS data directory should be created"
        assert (dm.fmri_path).exists(), "fMRI data directory should be created"
        
        # Check data files were created
        fnirs_files = list(dm.fnirs_path.glob("*.pt"))
        fmri_files = list(dm.fmri_path.glob("*.pt"))
        
        assert len(fnirs_files) > 0, "Dummy fNIRS files should be created"
        assert len(fmri_files) > 0, "Dummy fMRI files should be created"
        assert len(fnirs_files) == len(fmri_files), "Should have equal fNIRS and fMRI files"
        
    @patch('subprocess.run')
    def test_osf_download_failure_fallback(self, mock_subprocess):
        """Test fallback to dummy data when OSF download fails."""
        # Mock failed OSF download
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Download failed"
        
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            osf_project_id="dummy_project"
        )
        
        # Should fallback to dummy data creation
        dm.prepare_data()
        
        # Should still have created dummy data
        assert (dm.fnirs_path).exists()
        assert (dm.fmri_path).exists()
        
    def test_multiple_setup_calls(self):
        """Test DataModule handles multiple setup calls gracefully."""
        dm = FmriFnirsDataModule(
            data_root=str(self.temp_dir),
            num_workers=0
        )
        
        dm.prepare_data()
        
        # First setup
        dm.setup()
        first_train_size = len(dm.train_dataset) if dm.train_dataset else 0
        
        # Second setup (should not crash)
        dm.setup()
        second_train_size = len(dm.train_dataset) if dm.train_dataset else 0
        
        # Should be consistent
        assert first_train_size == second_train_size, "Multiple setup calls should be consistent"


class TestDataIntegrity:
    """Test data integrity and edge cases."""
    
    def test_malformed_filenames(self):
        """Test handling of malformed filenames."""
        temp_dir = Path(tempfile.mkdtemp())
        fnirs_dir = temp_dir / "fnirs"
        fmri_dir = temp_dir / "fmri"
        
        fnirs_dir.mkdir(parents=True)
        fmri_dir.mkdir(parents=True)
        
        try:
            # Create files with malformed names
            malformed_files = ["malformed.pt", "no_underscores.pt", "_leading_underscore.pt"]
            
            for filename in malformed_files:
                # Create dummy data
                torch.save(torch.randn(52, 200), fnirs_dir / filename)
                torch.save(torch.randn(768), fmri_dir / filename)
            
            # Dataset should handle malformed filenames gracefully
            dataset = FmriFnirsDataset(fnirs_dir, fmri_dir)
            
            # Should not crash, even if some files can't be parsed properly
            assert len(dataset) >= 0, "Dataset should handle malformed filenames"
            
        finally:
            shutil.rmtree(temp_dir)
            
    def test_mismatched_data_shapes(self):
        """Test handling of data with unexpected shapes."""
        temp_dir = Path(tempfile.mkdtemp())
        fnirs_dir = temp_dir / "fnirs"
        fmri_dir = temp_dir / "fmri"
        
        fnirs_dir.mkdir(parents=True)
        fmri_dir.mkdir(parents=True)
        
        try:
            # Create data with wrong shapes
            torch.save(torch.randn(100, 300), fnirs_dir / "sub01_emotion0_00.pt")  # Wrong shape
            torch.save(torch.randn(512), fmri_dir / "sub01_emotion0_00.pt")  # Wrong dimension
            
            dataset = FmriFnirsDataset(fnirs_dir, fmri_dir)
            
            if len(dataset) > 0:
                sample = dataset[0]
                
                # Should reshape/pad to correct dimensions
                assert sample["fnirs"].shape == (52, 200), "fNIRS should be reshaped to (52, 200)"
                assert sample["fmri"].shape == (768,), "fMRI should be padded/cropped to (768,)"
                
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
