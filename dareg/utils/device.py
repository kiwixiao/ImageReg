"""
DAREG Device Management

Platform-aware GPU/CPU device selection:
- macOS: CPU only (MPS does not support grid_sampler_3d)
- Linux: CUDA GPU if available, fallback to CPU
"""

import platform
import torch
from typing import Optional
from .logging_config import get_logger

logger = get_logger("device")


def get_device(device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    Get computation device with platform-aware defaults.

    Auto mode (device=None):
    - Linux: CUDA GPU if available, else CPU
    - macOS: CPU always (MPS lacks grid_sampler_3d support)

    Args:
        device: Explicit device string ("cuda", "cpu", "mps") or None for auto
        verbose: Whether to log device selection

    Returns:
        torch.device instance
    """
    if device is not None:
        selected = torch.device(device)
    elif platform.system() == "Darwin":
        # macOS: MPS does not support grid_sampler_3d, always use CPU
        selected = torch.device("cpu")
        if verbose:
            logger.info("macOS detected — using CPU (MPS lacks grid_sampler_3d)")
    elif torch.cuda.is_available():
        selected = torch.device("cuda")
    else:
        selected = torch.device("cpu")

    if verbose:
        logger.info(f"Using device: {selected}")
        if selected.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return selected


class DeviceManager:
    """
    Context manager for device memory management

    Handles GPU memory cleanup and provides memory usage tracking.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.initial_memory = 0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.initial_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def memory_used(self) -> float:
        """Returns memory used in MB since context entry"""
        if self.device.type == "cuda":
            current = torch.cuda.memory_allocated()
            return (current - self.initial_memory) / 1e6
        return 0.0

    def log_memory(self, label: str = "Memory"):
        """Log current memory usage"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logger.debug(f"{label}: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
