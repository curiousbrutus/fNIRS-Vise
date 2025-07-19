import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from sklearn.model_selection import GroupShuffleSplit
import os
import gdown

class MindVisfNIRSDataset(Dataset):
    """Dataset for streaming fMRI and fNIRS data from MindVis and NEMO."""
    def __init__(self, data_paths: list):
        self.data_paths = data_paths
        self.data = self._load_data()

    def _load_data(self):
        # In a real scenario, this would load data from specified paths
        # For now, use dummy data simulating the structure
        data = []
        for _ in self.data_paths:
            fmri_data = torch.randn(768)
            fnirs_data = torch.randn(52, 200)
            label = torch.randint(0, 4, (1,)).item() # Dummy label (assuming 4 classes)
            data.append({'fmri': fmri_data, 'fnirs': fnirs_data, 'label': label})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FmriFnirsDataModule(LightningDataModule):
    """DataModule for fMRI and fNIRS data with LOSO split."""
    def __init__(self, data_dir: str = 'data', batch_size: int = 8, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subject_ids = [f'subject_{i:03d}' for i in range(30)] # Assuming 30 subjects for NEMO

    def prepare_data(self):
        # Download data from OSF if not exists (stub)
        # In a real scenario, use osfclient or gdown to download actual data
        os.makedirs(self.data_dir, exist_ok=True)
        print("Data download (stub): Assuming data is available in {}".format(self.data_dir))

    def setup(self, stage: str):
        # Leave-One-Subject-Out split using GroupShuffleSplit
        groups = [i for i in range(len(self.subject_ids)) for _ in range(10)] # Dummy groups, assuming 10 samples per subject
        gss = GroupShuffleSplit(n_splits=len(self.subject_ids), test_size=1, random_state=42)

        all_data_paths = [os.path.join(self.data_dir, f'subject_{i:03d}_sample_{j:02d}.pt') for i in range(30) for j in range(10)] # Dummy paths

        if stage == 'fit':
            train_indices, val_indices = next(gss.split(all_data_paths, groups=groups))
            self.train_data_paths = [all_data_paths[i] for i in train_indices]
            self.val_data_paths = [all_data_paths[i] for i in val_indices]
            self.train_dataset = MindVisfNIRSDataset(self.train_data_paths)
            self.val_dataset = MindVisfNIRSDataset(self.val_data_paths)
        elif stage == 'test':
            self.test_data_paths = all_data_paths # For testing, use all data (adjust as needed)
            self.test_dataset = MindVisfNIRSDataset(self.test_data_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
