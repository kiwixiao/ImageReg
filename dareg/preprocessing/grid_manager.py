"""
DAREG Grid Management

Utilities for creating and managing coordinate grids.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from deepali.core import Grid

from ..utils.logging_config import get_logger

logger = get_logger("grid")


def create_common_grid(
    source_grid: Grid,
    target_grid: Grid,
    method: str = "target",
    target_spacing: Optional[Tuple[float, float, float]] = None,
) -> Grid:
    """
    Create a common grid for registration

    Args:
        source_grid: Source image grid
        target_grid: Target image grid
        method: Grid creation method
            - "target": Use target grid as-is
            - "union": Create grid covering union of both
            - "intersection": Create grid covering intersection
            - "resample": Create grid with specified spacing
        target_spacing: Target spacing for "resample" method

    Returns:
        Common Grid for registration
    """
    if method == "target":
        # Use target grid directly
        common_grid = target_grid
        logger.debug(f"Using target grid: shape={tuple(common_grid.shape)}")

    elif method == "union":
        # Compute bounding box of both grids
        src_bbox = compute_bounding_box(source_grid)
        tgt_bbox = compute_bounding_box(target_grid)

        # Union of bounding boxes
        min_corner = torch.minimum(src_bbox[0], tgt_bbox[0])
        max_corner = torch.maximum(src_bbox[1], tgt_bbox[1])

        # Use finer spacing
        spacing = torch.minimum(
            torch.tensor(source_grid.spacing()),
            torch.tensor(target_grid.spacing())
        )

        # Compute shape
        extent = max_corner - min_corner
        shape = torch.ceil(extent / spacing).int().tolist()

        # Preserve direction from target grid (reference standard in medical imaging)
        direction = target_grid.direction()
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()

        common_grid = Grid(
            shape=tuple(shape),
            spacing=tuple(spacing.tolist()),
            origin=tuple(min_corner.tolist()),
            direction=direction,  # Preserve orientation from target
        )
        logger.debug(f"Union grid: shape={tuple(common_grid.shape)}, spacing={tuple(spacing.tolist())}")

    elif method == "intersection":
        # Compute bounding box of both grids
        src_bbox = compute_bounding_box(source_grid)
        tgt_bbox = compute_bounding_box(target_grid)

        # Intersection of bounding boxes
        min_corner = torch.maximum(src_bbox[0], tgt_bbox[0])
        max_corner = torch.minimum(src_bbox[1], tgt_bbox[1])

        # Use finer spacing
        spacing = torch.minimum(
            torch.tensor(source_grid.spacing()),
            torch.tensor(target_grid.spacing())
        )

        # Compute shape
        extent = max_corner - min_corner
        if (extent <= 0).any():
            raise ValueError("Source and target grids do not overlap")

        shape = torch.ceil(extent / spacing).int().tolist()

        # Preserve direction from target grid (reference standard in medical imaging)
        direction = target_grid.direction()
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()

        common_grid = Grid(
            shape=tuple(shape),
            spacing=tuple(spacing.tolist()),
            origin=tuple(min_corner.tolist()),
            direction=direction,  # Preserve orientation from target
        )
        logger.debug(f"Intersection grid: shape={tuple(common_grid.shape)}")

    elif method == "resample":
        if target_spacing is None:
            raise ValueError("target_spacing required for resample method")

        # Use target bounding box with new spacing
        bbox = compute_bounding_box(target_grid)
        extent = bbox[1] - bbox[0]
        spacing = torch.tensor(target_spacing)
        shape = torch.ceil(extent / spacing).int().tolist()

        # Preserve direction from target grid (reference standard in medical imaging)
        direction = target_grid.direction()
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()

        common_grid = Grid(
            shape=tuple(shape),
            spacing=target_spacing,
            origin=tuple(bbox[0].tolist()),
            direction=direction,  # Preserve orientation from target
        )
        logger.debug(f"Resampled grid: shape={tuple(common_grid.shape)}, spacing={target_spacing}")

    else:
        raise ValueError(f"Unknown grid method: {method}")

    return common_grid


def compute_bounding_box(grid: Grid) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute world-coordinate bounding box of a grid

    Args:
        grid: Grid to compute bounding box for

    Returns:
        Tuple of (min_corner, max_corner) tensors
    """
    shape = torch.tensor(grid.shape)
    spacing = torch.tensor(grid.spacing())
    origin = torch.tensor(grid.origin()) if hasattr(grid, 'origin') else torch.zeros(3)

    # Compute extent
    extent = shape.float() * spacing

    min_corner = origin
    max_corner = origin + extent

    return min_corner, max_corner


def compute_anisotropy_ratio(grid: Grid) -> float:
    """
    Compute anisotropy ratio of a grid

    Args:
        grid: Grid to analyze

    Returns:
        Ratio of max/min spacing (1.0 = isotropic)
    """
    spacing = grid.spacing()
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.cpu().numpy()

    min_spacing = float(np.min(spacing))
    max_spacing = float(np.max(spacing))

    if min_spacing > 0:
        ratio = max_spacing / min_spacing
    else:
        ratio = 1.0

    return ratio


def get_anisotropic_pyramid_dims(
    grid: Grid,
    anisotropy_threshold: float = 2.0,
) -> Optional[Tuple[int, ...]]:
    """
    Determine which dimensions to downsample for anisotropic images

    MIRTK-style behavior: For highly anisotropic images (e.g., Z=3mm vs XY=1.25mm),
    only downsample the fine dimensions (XY), not the coarse dimension (Z).

    Args:
        grid: Image grid
        anisotropy_threshold: Ratio threshold for "coarse" dimension

    Returns:
        Tuple of dimension indices to downsample, or None for isotropic
    """
    spacing = grid.spacing()
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.cpu().numpy()
    else:
        spacing = np.array(spacing)

    min_spacing = float(spacing.min())
    max_spacing = float(spacing.max())

    # Check if anisotropic
    if max_spacing / min_spacing < anisotropy_threshold:
        return None  # Isotropic - downsample all dims

    # Find fine dimensions (close to min spacing)
    fine_dims = []
    for i, s in enumerate(spacing):
        if float(s) / min_spacing < anisotropy_threshold:
            fine_dims.append(i)

    logger.debug(f"Anisotropic grid: spacing={spacing}, fine_dims={fine_dims}")
    return tuple(fine_dims) if fine_dims else None
