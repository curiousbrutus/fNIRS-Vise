from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch
import os

class FmriFnirsDataset(Dataset):
    """Dataset for streaming fMRI and fNIRS data."""
    def __init__(self, fmri_dir: str, fnirs_dir: str, subject_ids: list):
        self.fmri_dir = fmri_dir
        self.fnirs_dir = fnirs_dir
        self.subject_ids = subject_ids
        self.data = self._load_data()

    def _load_data(self):
        # In a real scenario, this would stream data from OSF or disk
        # For now, use dummy data
        data = []
        for subject_id in self.subject_ids:
            # Assuming file naming convention
            fmri_file = os.path.join(self.fmri_dir, f'fmri_{subject_id}.pt')
            fnirs_file = os.path.join(self.fnirs_dir, f'fnirs_{subject_id}.pt')
            # Load and append data (replace with actual loading logic)
            fmri_data = torch.randn(768)
            fnirs_data = torch.randn(52 * 200)
            label = torch.randn(1) # Dummy label
            data.append((fmri_data, fnirs_data, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FmriFnirsDataModule(LightningDataModule):
    """DataModule for fMRI and fNIRS data with LOSO split."""
    def __init__(self, fmri_dir: str, fnirs_dir: str, subject_ids: list, batch_size: int = 32):
        super().__init__()
        self.fmri_dir = fmri_dir
        self.fnirs_dir = fnirs_dir
        self.subject_ids = subject_ids
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Leave-One-Subject-Out split
        if stage == 'fit':
            self.train_subjects = self.subject_ids[:-1] # Use all but the last subject for training
            self.val_subjects = [self.subject_ids[-1]] # Use the last subject for validation
            self.train_dataset = FmriFnirsDataset(self.fmri_dir, self.fnirs_dir, self.train_subjects)
            self.val_dataset = FmriFnirsDataset(self.fmri_dir, self.fnirs_dir, self.val_subjects)
        elif stage == 'test':
            self.test_subjects = self.subject_ids # Use all subjects for testing (adjust as needed)
            self.test_dataset = FmriFnirsDataset(self.fmri_dir, self.fnirs_dir, self.test_subjects)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
