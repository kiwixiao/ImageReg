"""
DAREG Segmentation Transformation

Transform segmentation labels using nearest neighbor interpolation
to preserve discrete label values.

Includes Newton-Raphson approximation for FFD inverse (MIRTK-style).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Any
from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial

from ..utils.logging_config import get_logger

logger = get_logger("segmentation")


class ApproximateInverseTransform:
    """
    Approximate inverse of FFD using fixed-point iteration (MIRTK-style).

    For a forward transform T: x → y = x + u(x), the inverse finds x given y.

    Fixed-point iteration algorithm:
        x₀ = y  (initial guess)
        x_{n+1} = y - u(x_n)  (update)
        Repeat until convergence

    This is equivalent to MIRTK's Newton-Raphson approximation for FFD inverse.
    Reference: MIRTK FreeFormTransformation3D::Inverse()

    Args:
        forward_transform: The forward FFD transform
        max_iterations: Maximum number of iterations (default: 10)
        tolerance: Convergence tolerance in normalized coordinates (default: 1e-5)
    """

    def __init__(self, forward_transform, max_iterations: int = 10, tolerance: float = 1e-5):
        self.forward_transform = forward_transform
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Get grid from forward transform
        if hasattr(forward_transform, 'grid'):
            self._grid = forward_transform.grid()
        else:
            self._grid = None

    def grid(self):
        """Return the grid associated with this transform"""
        return self._grid

    def __call__(self, points: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply approximate inverse transform to points.

        Given points y in target space, find corresponding points x in source space
        such that forward_transform(x) ≈ y.

        Args:
            points: Input points in target space [N, D, H, W, 3] or [D, H, W, 3]

        Returns:
            Points in source space (approximate inverse)
        """
        # Initialize with identity (x₀ = y)
        x = points.clone()

        # Fixed-point iteration: x_{n+1} = y - u(x_n)
        for i in range(self.max_iterations):
            # Compute forward transform at current guess
            # forward_transform(x) = x + u(x)
            y_approx = self.forward_transform(x, **kwargs)

            # Displacement at current guess: u(x) = forward(x) - x
            displacement = y_approx - x

            # Update: x_new = y - u(x_current)
            x_new = points - displacement

            # Check convergence
            delta = (x_new - x).abs().max().item()
            x = x_new

            if delta < self.tolerance:
                logger.debug(f"FFD inverse converged in {i+1} iterations (delta={delta:.2e})")
                break
        else:
            logger.debug(f"FFD inverse reached max iterations ({self.max_iterations}), delta={delta:.2e}")

        return x


def approximate_ffd_inverse(transform, max_iterations: int = 10, tolerance: float = 1e-5):
    """
    Create an approximate inverse of an FFD transform using Newton-Raphson iteration.

    This is the MIRTK-style approach for computing FFD inverse, which doesn't have
    an analytical inverse like SVFFD.

    For segmentation propagation, we need the inverse transform:
    - Forward: frame_i → frame_0 (what we have from registration)
    - Inverse: frame_0 → frame_i (what we need for segmentation)

    Args:
        transform: Forward FFD transform
        max_iterations: Max iterations for convergence (default: 10)
        tolerance: Convergence tolerance (default: 1e-5)

    Returns:
        ApproximateInverseTransform that can be used like a regular transform
    """
    return ApproximateInverseTransform(transform, max_iterations, tolerance)


def transform_segmentation(
    segmentation: Image,
    transform: Any,
    output_grid: Optional[Grid] = None,
) -> Image:
    """
    Transform segmentation using nearest neighbor interpolation

    CRITICAL: Segmentations MUST use nearest neighbor interpolation
    to preserve discrete label values. Linear interpolation would
    create invalid intermediate values.

    Args:
        segmentation: Segmentation image with discrete labels
        transform: Transform to apply (SpatialTransform or ApproximateInverseTransform)
        output_grid: Optional output grid

    Returns:
        Transformed segmentation
    """
    if output_grid is None:
        output_grid = segmentation.grid()

    # Prepare input tensor
    tensor = segmentation.tensor()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    # Handle ApproximateInverseTransform separately (not a SpatialTransform)
    if isinstance(transform, ApproximateInverseTransform):
        logger.debug("Using manual grid_sample for ApproximateInverseTransform")

        # Get the segmentation's grid for coordinate conversion
        seg_grid = segmentation.grid()

        # CRITICAL FIX: When output_grid differs from segmentation grid, we need to:
        # 1. Get world coords from output_grid (where we want the result)
        # 2. Apply inverse transform in the transform's cube space
        # 3. Convert result to SEGMENTATION's cube space for grid_sample
        #
        # The key insight: grid_sample expects coords in the INPUT tensor's space,
        # which is the segmentation's grid, NOT the output_grid!

        # CRITICAL BUG FIX: output_grid.coords() returns CUBE coords (normalized [-1,1])
        # NOT world coords! Must convert to actual world coords first.
        output_cube_coords = output_grid.coords(device=tensor.device, dtype=torch.float32)
        if output_cube_coords.dim() == 4:
            output_cube_coords = output_cube_coords.unsqueeze(0)  # [1, D, H, W, 3]

        # Convert cube coords to ACTUAL world coordinates (in mm)
        output_world_coords = output_grid.cube_to_world(output_cube_coords)

        # Convert to transform's cube space for the inverse transform operation
        # Use the transform's grid if available, otherwise use output_grid
        transform_grid = transform.grid() if transform.grid() is not None else output_grid
        cube_coords_for_transform = transform_grid.world_to_cube(output_world_coords)

        logger.debug(f"Output grid shape: {output_grid.shape}, Seg grid shape: {seg_grid.shape}")
        logger.debug(f"World coords range: [{output_world_coords.min():.2f}, {output_world_coords.max():.2f}]")
        logger.debug(f"Cube coords for transform: [{cube_coords_for_transform.min():.2f}, {cube_coords_for_transform.max():.2f}]")

        # Apply inverse transform in cube space
        # inverse_transform maps target cube coords → source cube coords
        with torch.no_grad():
            source_cube_coords = transform(cube_coords_for_transform)

        logger.debug(f"Source cube coords (transform space): [{source_cube_coords.min():.2f}, {source_cube_coords.max():.2f}]")

        # Convert source cube coords back to world coords, then to SEGMENTATION's cube space
        # This is the CRITICAL step: grid_sample expects coords in the INPUT tensor's space!
        source_world_coords = transform_grid.cube_to_world(source_cube_coords)
        source_seg_cube_coords = seg_grid.world_to_cube(source_world_coords)

        logger.debug(f"Source world coords: [{source_world_coords.min():.2f}, {source_world_coords.max():.2f}]")
        logger.debug(f"Source seg cube coords (for grid_sample): [{source_seg_cube_coords.min():.2f}, {source_seg_cube_coords.max():.2f}]")

        # Check if coords are within valid range for grid_sample
        in_bounds_ratio = ((source_seg_cube_coords >= -1) & (source_seg_cube_coords <= 1)).float().mean()
        logger.debug(f"Fraction of coords in [-1, 1] range: {in_bounds_ratio:.2%}")

        # Apply grid_sample with nearest neighbor
        # CRITICAL: Use source_seg_cube_coords which is in segmentation's space
        warped = F.grid_sample(
            tensor.float(),  # grid_sample requires float
            source_seg_cube_coords,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )

        # Round to ensure integer labels
        warped = warped.round()

        # Remove batch dim
        if warped.dim() == 5:
            warped = warped.squeeze(0)

        # Log unique labels for debugging
        unique_labels = torch.unique(warped)
        logger.debug(f"Segmentation transformed (manual): tensor shape {warped.shape}, "
                    f"grid shape {output_grid.shape}, labels: {unique_labels.tolist()}")

        return Image(data=warped, grid=output_grid)

    # Standard path: use ImageTransformer for SpatialTransform
    # NOTE: Do NOT call transform.grid_() for FFD/SVFFD transforms
    # FFD/SVFFD transforms can only change grid size following 2n-1 B-spline rule
    # Instead, the ImageTransformer will handle resampling to output_grid via target parameter

    # Create transformer with nearest neighbor interpolation
    # CRITICAL: Use correct deepali API parameter names
    # - target (not target_grid) for output sampling grid
    # - sampling (not mode) for interpolation method
    # - padding takes PaddingMode enum or scalar value
    transformer = spatial.ImageTransformer(
        transform,
        target=output_grid,
        sampling="nearest",  # CRITICAL: nearest neighbor for segmentations
        padding=0,  # Zero padding outside domain
    )

    # Apply transform
    with torch.no_grad():
        warped = transformer(tensor)

    # Remove batch dim only (keep channel dim for deepali Image)
    # warped shape: [N, C, D, H, W] -> [C, D, H, W]
    if warped.dim() == 5:
        warped = warped.squeeze(0)  # Remove batch dim

    # Ensure integer labels
    warped = warped.round()

    # Get the actual output grid from the transformer to match the warped tensor shape
    # The transformer's target grid is what the warped tensor is sampled on
    actual_output_grid = transformer.target_grid()

    logger.debug(f"Segmentation transformed: tensor shape {warped.shape}, grid shape {actual_output_grid.shape}")

    return Image(data=warped, grid=actual_output_grid)


def compute_label_overlap(
    segmentation1: Image,
    segmentation2: Image,
    labels: Optional[list] = None,
) -> dict:
    """
    Compute overlap metrics between two segmentations

    Args:
        segmentation1: First segmentation
        segmentation2: Second segmentation
        labels: Optional list of labels to evaluate (default: all non-zero)

    Returns:
        Dictionary with per-label Dice scores
    """
    seg1 = segmentation1.tensor()
    seg2 = segmentation2.tensor()

    # Get unique labels
    if labels is None:
        all_labels = torch.unique(torch.cat([seg1.flatten(), seg2.flatten()]))
        labels = [int(l) for l in all_labels if l != 0]

    results = {}
    for label in labels:
        mask1 = (seg1 == label).float()
        mask2 = (seg2 == label).float()

        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum()

        if union > 0:
            dice = float(2 * intersection / union)
        else:
            dice = 1.0 if intersection == 0 else 0.0

        results[f"dice_label_{label}"] = dice

    # Mean Dice
    if results:
        results["mean_dice"] = sum(results.values()) / len(results)

    return results
