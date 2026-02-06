"""
DAREG Image Pair Data Structure

Dataclass holding source and target images with metadata and grids.
Supports both original and normalized (common grid) versions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import numpy as np

# Try to import deepali Image - will be available at runtime
try:
    from deepali.data import Image
    from deepali.core import Grid
except ImportError:
    Image = None
    Grid = None


@dataclass
class ImageMetadata:
    """Metadata for a single image"""
    path: str
    shape: tuple
    spacing: tuple  # Voxel spacing in mm
    origin: tuple   # World origin
    direction: tuple  # Direction cosines (flattened 3x3)
    dtype: str


@dataclass
class ImagePair:
    """
    Container for source and target images

    Holds both original resolution images and normalized versions
    resampled to a common grid for registration.

    Attributes:
        source_original: Source image at original resolution
        target_original: Target image at original resolution
        source_normalized: Source image resampled to common grid
        target_normalized: Target image resampled to common grid
        common_grid: Common coordinate grid for registration
        source_metadata: Source image metadata
        target_metadata: Target image metadata
        segmentation: Optional segmentation image
    """
    # Original resolution images
    source_original: Any = None  # deepali.data.Image
    target_original: Any = None  # deepali.data.Image

    # Normalized images on common grid
    source_normalized: Any = None  # deepali.data.Image
    target_normalized: Any = None  # deepali.data.Image

    # Common grid for registration
    common_grid: Any = None  # deepali.core.Grid

    # Metadata
    source_metadata: Optional[ImageMetadata] = None
    target_metadata: Optional[ImageMetadata] = None

    # Optional segmentation
    segmentation: Optional[Any] = None  # deepali.data.Image or None

    # Original SimpleITK images for header preservation
    source_sitk: Any = None
    target_sitk: Any = None
    segmentation_sitk: Any = None

    def __post_init__(self):
        """Validate image pair after initialization"""
        if self.source_normalized is not None and self.target_normalized is not None:
            # Verify shapes match on common grid
            src_shape = tuple(self.source_normalized.shape) if hasattr(self.source_normalized, 'shape') else None
            tgt_shape = tuple(self.target_normalized.shape) if hasattr(self.target_normalized, 'shape') else None

            if src_shape and tgt_shape and src_shape != tgt_shape:
                raise ValueError(
                    f"Normalized source and target shapes must match. "
                    f"Got source={src_shape}, target={tgt_shape}"
                )

    @property
    def shape(self) -> tuple:
        """Get shape of normalized images (common grid shape)"""
        if self.source_normalized is not None:
            return tuple(self.source_normalized.tensor().shape[-3:])
        elif self.common_grid is not None:
            return tuple(self.common_grid.shape)
        return None

    @property
    def spacing(self) -> tuple:
        """Get spacing of common grid in mm"""
        if self.common_grid is not None:
            spacing = self.common_grid.spacing()
            if isinstance(spacing, torch.Tensor):
                return tuple(spacing.tolist())
            return tuple(spacing)
        return None

    def get_source_tensor(self, normalized: bool = True) -> torch.Tensor:
        """
        Get source image as tensor

        Args:
            normalized: If True, return normalized version on common grid

        Returns:
            Image tensor [C, D, H, W] or [D, H, W]
        """
        img = self.source_normalized if normalized else self.source_original
        if img is not None:
            return img.tensor()
        return None

    def get_target_tensor(self, normalized: bool = True) -> torch.Tensor:
        """
        Get target image as tensor

        Args:
            normalized: If True, return normalized version on common grid

        Returns:
            Image tensor [C, D, H, W] or [D, H, W]
        """
        img = self.target_normalized if normalized else self.target_original
        if img is not None:
            return img.tensor()
        return None

    def to_device(self, device: torch.device) -> "ImagePair":
        """
        Move all tensors to specified device

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        if self.source_normalized is not None:
            self.source_normalized = self.source_normalized.to(device)
        if self.target_normalized is not None:
            self.target_normalized = self.target_normalized.to(device)
        if self.segmentation is not None:
            self.segmentation = self.segmentation.to(device)
        return self

    def summary(self) -> Dict[str, Any]:
        """Get summary of image pair"""
        return {
            "source": {
                "path": self.source_metadata.path if self.source_metadata else None,
                "shape": self.source_metadata.shape if self.source_metadata else None,
                "spacing": self.source_metadata.spacing if self.source_metadata else None,
            },
            "target": {
                "path": self.target_metadata.path if self.target_metadata else None,
                "shape": self.target_metadata.shape if self.target_metadata else None,
                "spacing": self.target_metadata.spacing if self.target_metadata else None,
            },
            "common_grid": {
                "shape": self.shape,
                "spacing": self.spacing,
            },
            "has_segmentation": self.segmentation is not None,
        }
