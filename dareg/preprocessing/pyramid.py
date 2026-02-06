"""
DAREG Multi-Resolution Pyramid

Create and manage image pyramids for coarse-to-fine registration.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from deepali.data import Image
from deepali.core import Grid

from .grid_manager import get_anisotropic_pyramid_dims
from ..utils.logging_config import get_logger

logger = get_logger("pyramid")


@dataclass
class PyramidLevel:
    """Single level of image pyramid"""
    level: int
    image: Image
    grid: Grid
    scale_factor: Tuple[float, ...]


class ImagePyramid:
    """
    Multi-resolution image pyramid for registration

    Supports MIRTK-style anisotropic downsampling where coarse dimensions
    (e.g., thick slices in Z) are not further downsampled.
    """

    def __init__(
        self,
        image: Image,
        num_levels: int = 4,
        anisotropy_threshold: float = 2.0,
    ):
        """
        Create image pyramid

        Args:
            image: Source image
            num_levels: Number of pyramid levels (1 = finest/original)
            anisotropy_threshold: Threshold for anisotropic handling
        """
        self.source_image = image
        self.num_levels = num_levels
        self.anisotropy_threshold = anisotropy_threshold
        self.levels: Dict[int, PyramidLevel] = {}

        # Determine which dimensions to downsample
        self.downsample_dims = get_anisotropic_pyramid_dims(
            image.grid(), anisotropy_threshold
        )

        # Build pyramid
        self._build_pyramid()

    def _build_pyramid(self):
        """Build all pyramid levels"""
        logger.info(f"Creating {self.num_levels}-level pyramid")

        if self.downsample_dims:
            logger.info(f"  Anisotropic mode: downsampling dims {self.downsample_dims}")

        # Use deepali's pyramid() method
        pyramid_dict = self.source_image.pyramid(
            levels=self.num_levels,
            start=0,  # Finest level index
            end=self.num_levels - 1,  # Coarsest level index
            dims=self.downsample_dims,
        )

        # Store levels
        for level in range(self.num_levels):
            level_image = pyramid_dict[level]
            scale = self._compute_scale_factor(level_image.grid())

            self.levels[level] = PyramidLevel(
                level=level,
                image=level_image,
                grid=level_image.grid(),
                scale_factor=scale,
            )

            logger.debug(
                f"  Level {level}: shape={tuple(level_image.grid().shape)}, "
                f"scale={[f'{s:.2f}' for s in scale]}"
            )

    def _compute_scale_factor(self, level_grid: Grid) -> Tuple[float, ...]:
        """Compute scale factor relative to original"""
        orig_shape = self.source_image.grid().shape
        level_shape = level_grid.shape

        scales = []
        for orig_s, level_s in zip(orig_shape, level_shape):
            scales.append(float(level_s) / float(orig_s))

        return tuple(scales)

    def __getitem__(self, level: int) -> PyramidLevel:
        """Get pyramid level by index"""
        if level not in self.levels:
            raise KeyError(f"Invalid pyramid level: {level}")
        return self.levels[level]

    def __len__(self) -> int:
        """Number of pyramid levels"""
        return self.num_levels

    def coarse_to_fine(self) -> List[PyramidLevel]:
        """Iterate levels from coarsest to finest"""
        for level in range(self.num_levels - 1, -1, -1):
            yield self.levels[level]

    def fine_to_coarse(self) -> List[PyramidLevel]:
        """Iterate levels from finest to coarsest"""
        for level in range(self.num_levels):
            yield self.levels[level]


def create_pyramid(
    image: Image,
    num_levels: int = 4,
    anisotropy_threshold: float = 2.0,
) -> ImagePyramid:
    """
    Create image pyramid

    Args:
        image: Source image
        num_levels: Number of pyramid levels
        anisotropy_threshold: Threshold for anisotropic handling

    Returns:
        ImagePyramid object
    """
    return ImagePyramid(image, num_levels, anisotropy_threshold)


def create_paired_pyramids(
    source: Image,
    target: Image,
    num_levels: int = 4,
    anisotropy_threshold: float = 2.0,
) -> Tuple[ImagePyramid, ImagePyramid]:
    """
    Create matched pyramids for source and target

    Args:
        source: Source image
        target: Target image
        num_levels: Number of pyramid levels
        anisotropy_threshold: Threshold for anisotropic handling

    Returns:
        Tuple of (source_pyramid, target_pyramid)
    """
    source_pyramid = create_pyramid(source, num_levels, anisotropy_threshold)
    target_pyramid = create_pyramid(target, num_levels, anisotropy_threshold)

    # Verify levels match
    for level in range(num_levels):
        src_shape = source_pyramid[level].grid.shape
        tgt_shape = target_pyramid[level].grid.shape

        if src_shape != tgt_shape:
            logger.warning(
                f"Pyramid level {level} shape mismatch: "
                f"source={src_shape}, target={tgt_shape}"
            )

    return source_pyramid, target_pyramid
