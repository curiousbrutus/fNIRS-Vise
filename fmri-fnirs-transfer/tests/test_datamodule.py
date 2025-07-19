import pytest
from src.data.datamodule import FmriFnirsDataModule

def test_loso_split():
    """Tests the Leave-One-Subject-Out split logic."""
    subject_ids = [f'subject_{i:03d}' for i in range(10)] # 10 subjects
    datamodule = FmriFnirsDataModule(
        fmri_dir='fake_fmri_dir',
        fnirs_dir='fake_fnirs_dir',
        subject_ids=subject_ids
    )

    datamodule.setup(stage='fit')

    # Check train/validation split
    assert len(datamodule.train_subjects) == 9
    assert len(datamodule.val_subjects) == 1
    assert datamodule.val_subjects[0] == subject_ids[-1]

    datamodule.setup(stage='test')
    assert len(datamodule.test_subjects) == 10
