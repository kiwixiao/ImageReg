"""
DAREG Transform Application

Apply transforms to images at various resolutions.
"""

import torch
from typing import Optional, Any
from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial

from ..utils.logging_config import get_logger

logger = get_logger("transformer")


def apply_transform(
    image: Image,
    transform: Any,
    output_grid: Optional[Grid] = None,
    sampling: str = "linear",
    padding: str = "zeros",
) -> Image:
    """
    Apply transform to image

    Args:
        image: Input image to transform
        transform: Transform to apply (Rigid, Affine, or FFD)
        output_grid: Optional output grid (defaults to image grid)
        sampling: Interpolation mode ("linear" or "nearest")
        padding: Padding mode ("zeros" or "border")

    Returns:
        Transformed image
    """
    if output_grid is None:
        output_grid = image.grid()

    # Set transform output grid
    if hasattr(transform, 'grid_'):
        transform.grid_(output_grid)

    # Create transformer using correct deepali API
    transformer = spatial.ImageTransformer(
        transform,
        target=output_grid,
        sampling=sampling,
        padding=padding,
    )

    # Prepare input tensor
    tensor = image.tensor()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    # Apply transform
    with torch.no_grad():
        warped = transformer(tensor)

    # Remove batch/channel dims
    warped = warped.squeeze(0)
    if warped.dim() == 4 and warped.shape[0] == 1:
        warped = warped.squeeze(0)

    return Image(data=warped, grid=output_grid)


def warp_image(
    image: Image,
    flow_field: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Image:
    """
    Warp image using displacement/flow field

    Args:
        image: Input image
        flow_field: Displacement field [D, H, W, 3] or [3, D, H, W]
        mode: Interpolation mode
        padding_mode: Padding mode

    Returns:
        Warped image
    """
    tensor = image.tensor()
    grid = image.grid()

    # Ensure proper dimensions
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    # Create sampling grid from flow field
    if flow_field.dim() == 4 and flow_field.shape[0] == 3:
        # [3, D, H, W] -> [D, H, W, 3]
        flow_field = flow_field.permute(1, 2, 3, 0)

    # Get base coordinates
    coords = grid.coords(device=tensor.device)  # [D, H, W, 3]

    # Add displacement
    warped_coords = coords + flow_field

    # Convert to normalized coordinates for grid_sample
    warped_normalized = grid.world_to_cube(warped_coords.unsqueeze(0))

    # Sample
    with torch.no_grad():
        warped = torch.nn.functional.grid_sample(
            tensor,
            warped_normalized,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )

    # Remove batch/channel dims
    warped = warped.squeeze(0)
    if warped.dim() == 4 and warped.shape[0] == 1:
        warped = warped.squeeze(0)

    return Image(data=warped, grid=grid)


def apply_transform_preserve_resolution(
    source_image: Image,
    transform: Any,
    target_grid: Grid,
    sampling: str = "linear",
    padding: str = "border",
) -> Image:
    """
    Apply transform to source image while preserving source's original resolution.

    This is the MIRTK convention: the output image has the source's spacing/resolution,
    but is moved to align with the target coordinate space.

    Equivalent to: mirtk transform-image source.nii output.nii -dofin T.dof
    (without -target option, which would resample to target grid)

    CRITICAL: The transform was learned on a resampled common_grid. We must:
    1. Create output positions at source's resolution, covering target's extent
    2. Convert output positions to common_grid's cube space (where transform operates)
    3. Apply transform to get source positions in common_grid's cube space
    4. Convert source positions to source_image's cube space
    5. Sample from source_image using grid_sample

    Args:
        source_image: Source image to transform (will keep its resolution)
        transform: Transform to apply (operates in common_grid cube space)
        target_grid: Target grid (the common_grid where transform was learned)
        sampling: Image interpolation mode ("linear" or "nearest")
        padding: Image extrapolation mode ("zeros" or "border")

    Returns:
        Transformed image with source's original resolution, aligned to target space
    """
    import torch.nn.functional as F

    source_grid = source_image.grid()

    # PRESERVE ORIGINAL RESOLUTION: Output grid is IDENTICAL to source grid
    # Same shape, spacing, direction - only the CONTENT is transformed
    # This ensures the output looks exactly like the original in ITK-SNAP
    output_grid = source_grid

    logger.debug(f"apply_transform_preserve_resolution:")
    logger.debug(f"  Source grid: shape={source_grid.shape}, spacing={source_grid.spacing().tolist()}")
    logger.debug(f"  Target grid (common_grid): shape={target_grid.shape}, spacing={target_grid.spacing().tolist()}")
    logger.debug(f"  Output grid: shape={output_grid.shape}, spacing={output_grid.spacing().tolist()} (same as source)")

    # Prepare input tensor
    tensor = source_image.tensor()
    original_dtype = tensor.dtype
    is_integer = tensor.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)

    # grid_sample requires float tensors
    if is_integer:
        tensor = tensor.float()

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    # Manual coordinate transformation through common_grid (target_grid)
    # The transform was learned on target_grid (common_grid), so it expects
    # cube coordinates in that space.
    with torch.no_grad():
        # Step 1: Get output positions in cube coords (normalized [-1,1])
        output_cube_coords = output_grid.coords(device=tensor.device, dtype=torch.float32)
        if output_cube_coords.dim() == 4:
            output_cube_coords = output_cube_coords.unsqueeze(0)  # [1, D, H, W, 3]

        # Step 2: Convert output cube coords to WORLD coords (mm)
        output_world_coords = output_grid.cube_to_world(output_cube_coords)

        # Step 3: Convert world coords to target_grid (common_grid) cube space
        # This is where the transform operates!
        target_cube_coords = target_grid.world_to_cube(output_world_coords)

        logger.debug(f"  Output world range: [{output_world_coords.min():.2f}, {output_world_coords.max():.2f}]")
        logger.debug(f"  Target cube range: [{target_cube_coords.min():.2f}, {target_cube_coords.max():.2f}]")

        # Step 4: Apply transform in target_grid's cube space
        # Transform maps target cube coords -> source cube coords (in common_grid space)
        source_cube_in_target_space = transform(target_cube_coords)

        logger.debug(f"  Source cube (target space) range: [{source_cube_in_target_space.min():.2f}, {source_cube_in_target_space.max():.2f}]")

        # Step 5: Convert source cube coords from target_grid space to WORLD coords
        source_world_coords = target_grid.cube_to_world(source_cube_in_target_space)

        # Step 6: Convert world coords to SOURCE_IMAGE's cube space for grid_sample
        source_cube_coords = source_grid.world_to_cube(source_world_coords)

        logger.debug(f"  Source world range: [{source_world_coords.min():.2f}, {source_world_coords.max():.2f}]")
        logger.debug(f"  Source cube range: [{source_cube_coords.min():.2f}, {source_cube_coords.max():.2f}]")

        # Check valid range
        in_bounds = ((source_cube_coords >= -1) & (source_cube_coords <= 1)).float().mean()
        logger.debug(f"  Fraction in valid range: {in_bounds:.2%}")

        # Step 7: Sample from source image using grid_sample
        # Determine interpolation mode
        if sampling == "nearest":
            mode = "nearest"
        else:
            mode = "bilinear"

        # Determine padding mode
        if padding == "zeros":
            padding_mode = "zeros"
        else:
            padding_mode = "border"

        warped = F.grid_sample(
            tensor,
            source_cube_coords,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )

    # Remove batch dim only, keep channel dim [N,C,D,H,W] -> [C,D,H,W]
    warped = warped.squeeze(0)

    # Convert back to original dtype if it was integer (e.g., segmentation labels)
    if is_integer:
        warped = warped.round().to(original_dtype)

    return Image(data=warped, grid=output_grid)


def compose_transforms(
    transform1: Any,
    transform2: Any,
) -> spatial.SequentialTransform:
    """
    Compose two transforms (T2 o T1)

    Args:
        transform1: First transform to apply
        transform2: Second transform to apply

    Returns:
        Composed SequentialTransform
    """
    return spatial.SequentialTransform(transform1, transform2)
