import pytest
import torch
from unittest.mock import MagicMock

# Mock CUDA for CPU-only testing
@pytest.fixture(autouse=True)
def mock_cuda(mocker):
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.memory_allocated', return_value=0)
    mocker.patch('torch.cuda.memory_reserved', return_value=0)
    mock_properties = MagicMock()
    mock_properties.total_memory = 1024**3 * 16 # Mock 16GB GPU memory
    mocker.patch('torch.cuda.get_device_properties', return_value=mock_properties)
    mocker.patch('torch.cuda.synchronize')

from src.models.fmri_fnirs_net import FmriGuidedFnirsNet

def test_model_forward_and_backward():
    """Tests forward pass and gradient flow."""
    model = FmriGuidedFnirsNet()
    fmri_data = torch.randn(1, 768, requires_grad=True)
    fnirs_data = torch.randn(1, 52 * 200, requires_grad=True)
    labels = torch.randn(1, 52 * 200)

    outputs = model(fmri_data, fnirs_data)
    loss = model.criterion(outputs, labels)
    loss.backward()

    # Check if gradients are computed
    assert fmri_data.grad is not None
    assert fnirs_data.grad is not None
