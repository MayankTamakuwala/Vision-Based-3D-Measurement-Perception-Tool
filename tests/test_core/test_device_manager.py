"""Tests for DeviceManager."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.device_manager import DeviceManager


@pytest.mark.unit
class TestDeviceManager:
    """Test cases for DeviceManager."""

    def test_initialization(self):
        """Test DeviceManager initialization."""
        dm = DeviceManager()
        assert dm.device is not None
        assert dm.device_name in ['cuda', 'mps', 'cpu']

    def test_device_detection(self):
        """Test automatic device detection."""
        dm = DeviceManager()

        if torch.cuda.is_available():
            assert 'cuda' in dm.device_name
        elif torch.backends.mps.is_available():
            assert dm.device_name == 'mps'
        else:
            assert dm.device_name == 'cpu'

    def test_preferred_device_cpu(self):
        """Test forcing CPU device."""
        dm = DeviceManager(preferred_device='cpu')
        assert dm.device_name == 'cpu'
        assert dm.device == torch.device('cpu')

    def test_get_device_info(self):
        """Test device info retrieval."""
        dm = DeviceManager()
        info = dm.get_device_info()

        assert 'device_name' in info
        assert 'available' in info
        assert 'device_count' in info

    def test_mixed_precision_dtype(self):
        """Test mixed precision dtype."""
        dm = DeviceManager()
        dtype = dm.get_mixed_precision_dtype()

        assert dtype in [torch.float32, torch.float16, torch.bfloat16]

    def test_optimize_for_inference(self):
        """Test inference optimization settings."""
        dm = DeviceManager()

        # Create dummy model
        model = torch.nn.Linear(10, 10)

        optimized = dm.optimize_for_inference(model)

        # Check that model is in eval mode
        assert not optimized.training

    def test_clear_cache(self):
        """Test CUDA cache clearing."""
        dm = DeviceManager()

        # Should not raise error even on CPU
        dm.clear_cache()

    def test_repr(self):
        """Test string representation."""
        dm = DeviceManager()
        repr_str = repr(dm)

        assert 'DeviceManager' in repr_str
        assert dm.device_name in repr_str


@pytest.mark.gpu
class TestDeviceManagerGPU:
    """GPU-specific tests for DeviceManager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device detection."""
        dm = DeviceManager(preferred_device='cuda')
        assert 'cuda' in dm.device_name
        assert dm.get_device_info()['device_count'] > 0

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device(self):
        """Test MPS device detection."""
        dm = DeviceManager(preferred_device='mps')
        assert dm.device_name == 'mps'
