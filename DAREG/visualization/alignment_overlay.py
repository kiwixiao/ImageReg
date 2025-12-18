"""
Alignment Overlay Visualization

Creates overlay visualizations showing target (gray) + moved source (green)
at 50% transparency for each registration stage:
- Pure rigid
- Rigid + affine
- Rigid + affine + FFD/SVFFD
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deepali" / "src"))

from deepali.data import Image
from deepali.core import functional as U
import deepali.spatial as spatial

from ..utils.logging_config import get_logger

logger = get_logger("alignment_overlay")


def create_warped_grid_image(grid, transform, stride: int = 8, target_size: int = 256):
    """
    Create warped grid image using deepali's built-in grid_image function

    Reference: PairwiseImageRegistration-deepali_demo.pdf

    Args:
        grid: The image grid (deepali Grid object)
        transform: The transform to apply (can be SequentialTransform, FFD, etc.)
        stride: Grid line spacing (default 8)
        target_size: Resolution for high-res grid (default 256)

    Returns:
        warped_grid_np: Warped grid as numpy array [D, H, W]
        original_grid_np: Original grid as numpy array [D, H, W]
    """
    try:
        # Create high-resolution grid for smoother visualization
        max_dim = max(grid.shape)
        target_size = min(target_size, max_dim * 2)  # Don't oversample too much
        grid_highres = grid.resize(target_size)

        # Create grid image using deepali's built-in function
        # inverted=True gives white lines on black background
        grid_image = U.grid_image(grid_highres, num=1, stride=stride, inverted=True)

        # Apply the transform to warp the grid image
        grid_transformer = spatial.ImageTransformer(transform, grid_highres, padding="zeros")
        warped_grid = grid_transformer(grid_image)

        # Convert to numpy
        original_grid_np = grid_image.squeeze().detach().cpu().numpy()
        warped_grid_np = warped_grid.squeeze().detach().cpu().numpy()

        return warped_grid_np, original_grid_np

    except Exception as e:
        logger.warning(f"Could not create warped grid: {e}")
        return None, None


def create_green_colormap():
    """Create green-to-black colormap for source overlay"""
    colors = [
        (0.0, 0.0, 0.0),  # Black (low intensity)
        (0.0, 0.5, 0.0),  # Dark green
        (0.0, 1.0, 0.0),  # Bright green (high intensity)
    ]
    return LinearSegmentedColormap.from_list('green_black', colors, N=256)


def create_alignment_overlay_figure(
    target_image: Image,
    source_images: List[Tuple[str, Image]],  # List of (stage_name, warped_source)
    output_path: Path,
    slice_axis: str = "axial",  # "axial", "coronal", "sagittal"
    slice_index: Optional[int] = None,  # None = middle slice
    alpha: float = 0.5,
) -> Path:
    """
    Create overlay visualization of alignment stages

    Args:
        target_image: Target/reference image (shown in gray)
        source_images: List of (stage_name, warped_source_image) tuples
        output_path: Output path for saved figure
        slice_axis: Which axis to slice ("axial", "coronal", "sagittal")
        slice_index: Slice index (None = middle)
        alpha: Transparency (0.5 = 50%)

    Returns:
        Path to saved figure
    """
    # Extract target tensor
    target_np = target_image.tensor().squeeze().cpu().numpy()
    if target_np.ndim == 4:
        target_np = target_np[0]  # Remove channel dim

    # Determine slice index
    if slice_axis == "axial":
        dim = 0  # Z
        if slice_index is None:
            slice_index = target_np.shape[0] // 2
        target_slice = target_np[slice_index, :, :]
    elif slice_axis == "coronal":
        dim = 1  # Y
        if slice_index is None:
            slice_index = target_np.shape[1] // 2
        target_slice = target_np[:, slice_index, :]
    else:  # sagittal
        dim = 2  # X
        if slice_index is None:
            slice_index = target_np.shape[2] // 2
        target_slice = target_np[:, :, slice_index]

    # Normalize target to [0, 1]
    target_slice = (target_slice - target_slice.min()) / (target_slice.max() - target_slice.min() + 1e-8)

    # Create figure with subplots for each stage
    n_stages = len(source_images)
    fig, axes = plt.subplots(1, n_stages + 1, figsize=(5 * (n_stages + 1), 5))

    if n_stages == 0:
        axes = [axes]

    # Green colormap for source
    green_cmap = create_green_colormap()

    # Plot target only (first subplot)
    axes[0].imshow(target_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Target (Reference)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot each alignment stage
    for i, (stage_name, warped_source) in enumerate(source_images):
        ax = axes[i + 1]

        # Extract source tensor
        source_np = warped_source.tensor().squeeze().cpu().numpy()
        if source_np.ndim == 4:
            source_np = source_np[0]

        # Get corresponding slice
        if slice_axis == "axial":
            source_slice = source_np[slice_index, :, :] if slice_index < source_np.shape[0] else source_np[source_np.shape[0]//2, :, :]
        elif slice_axis == "coronal":
            source_slice = source_np[:, slice_index, :] if slice_index < source_np.shape[1] else source_np[:, source_np.shape[1]//2, :]
        else:
            source_slice = source_np[:, :, slice_index] if slice_index < source_np.shape[2] else source_np[:, :, source_np.shape[2]//2]

        # Normalize source to [0, 1]
        source_slice = (source_slice - source_slice.min()) / (source_slice.max() - source_slice.min() + 1e-8)

        # Plot target in gray (background)
        ax.imshow(target_slice, cmap='gray', vmin=0, vmax=1, alpha=alpha)

        # Plot source in green (overlay)
        ax.imshow(source_slice, cmap=green_cmap, vmin=0, vmax=1, alpha=alpha)

        ax.set_title(f'{stage_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Add legend
    fig.text(0.5, 0.02, 'Gray = Target (Frame 0)  |  Green = Moved Source (Static)',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved alignment overlay: {output_path}")
    return output_path


def create_alignment_overlay_3views(
    target_image: Image,
    source_images: List[Tuple[str, Image]],
    output_path: Path,
    alpha: float = 0.5,
    transforms: Optional[List[Tuple[str, any]]] = None,  # List of (stage_name, transform) for grid overlay
    show_grid: bool = True,
    grid_stride: int = 12,
    grid_alpha: float = 0.4,
) -> Path:
    """
    Create 3-view (axial, coronal, sagittal) overlay visualization with optional grid deformation

    Args:
        target_image: Target/reference image
        source_images: List of (stage_name, warped_source_image) tuples
        output_path: Output path
        alpha: Transparency for image overlay
        transforms: Optional list of (stage_name, transform) for grid deformation overlay
        show_grid: Whether to show grid deformation (default True if transforms provided)
        grid_stride: Grid line spacing (default 12)
        grid_alpha: Transparency for grid overlay (default 0.4)

    Returns:
        Path to saved figure
    """
    # Extract target tensor
    target_np = target_image.tensor().squeeze().cpu().numpy()
    if target_np.ndim == 4:
        target_np = target_np[0]

    # Get middle slices
    d_mid = target_np.shape[0] // 2
    h_mid = target_np.shape[1] // 2
    w_mid = target_np.shape[2] // 2

    # Normalize target
    target_norm = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)

    # Create figure: rows = stages, cols = views
    n_stages = len(source_images)
    fig, axes = plt.subplots(n_stages, 3, figsize=(15, 5 * n_stages))

    if n_stages == 1:
        axes = axes.reshape(1, -1)

    green_cmap = create_green_colormap()
    view_names = ['Axial', 'Coronal', 'Sagittal']

    # Prepare warped grids if transforms are provided
    warped_grids = {}
    if transforms is not None and show_grid:
        grid = target_image.grid()
        for stage_name, transform in transforms:
            if transform is not None:
                warped_grid_np, original_grid_np = create_warped_grid_image(
                    grid, transform, stride=grid_stride
                )
                if warped_grid_np is not None:
                    warped_grids[stage_name] = warped_grid_np
                    logger.debug(f"Created warped grid for stage: {stage_name}")

    for row_idx, (stage_name, warped_source) in enumerate(source_images):
        # Extract and normalize source
        source_np = warped_source.tensor().squeeze().cpu().numpy()
        if source_np.ndim == 4:
            source_np = source_np[0]
        source_norm = (source_np - source_np.min()) / (source_np.max() - source_np.min() + 1e-8)

        # Get slices for each view
        target_slices = [
            target_norm[d_mid, :, :],  # Axial
            target_norm[:, h_mid, :],  # Coronal
            target_norm[:, :, w_mid],  # Sagittal
        ]

        # Handle potential shape mismatch
        sd_mid = min(d_mid, source_norm.shape[0] - 1)
        sh_mid = min(h_mid, source_norm.shape[1] - 1)
        sw_mid = min(w_mid, source_norm.shape[2] - 1)

        source_slices = [
            source_norm[sd_mid, :, :],
            source_norm[:, sh_mid, :],
            source_norm[:, :, sw_mid],
        ]

        # Get warped grid for this stage if available
        stage_grid = warped_grids.get(stage_name, None)
        if stage_grid is not None:
            # Calculate grid slice indices (grid may be different resolution)
            gd_mid = stage_grid.shape[0] // 2
            gh_mid = stage_grid.shape[1] // 2
            gw_mid = stage_grid.shape[2] // 2
            grid_slices = [
                stage_grid[gd_mid, :, :],  # Axial
                stage_grid[:, gh_mid, :],  # Coronal
                stage_grid[:, :, gw_mid],  # Sagittal
            ]
        else:
            grid_slices = [None, None, None]

        for col_idx, (tgt_slice, src_slice, view_name) in enumerate(zip(target_slices, source_slices, view_names)):
            ax = axes[row_idx, col_idx]

            # Plot target (gray) and source (green) with transparency
            ax.imshow(tgt_slice, cmap='gray', vmin=0, vmax=1, alpha=alpha)
            ax.imshow(src_slice, cmap=green_cmap, vmin=0, vmax=1, alpha=alpha)

            # Overlay warped grid if available (in orange/red for visibility)
            grid_slice = grid_slices[col_idx]
            if grid_slice is not None:
                # Resize grid slice to match image slice if needed
                from scipy.ndimage import zoom
                if grid_slice.shape != tgt_slice.shape:
                    zoom_factors = (tgt_slice.shape[0] / grid_slice.shape[0],
                                   tgt_slice.shape[1] / grid_slice.shape[1])
                    grid_slice = zoom(grid_slice, zoom_factors, order=1)

                # Create orange colormap for grid overlay
                grid_normalized = (grid_slice - grid_slice.min()) / (grid_slice.max() - grid_slice.min() + 1e-8)
                # Only show where grid lines exist (values > threshold)
                grid_mask = grid_normalized > 0.1
                grid_rgba = np.zeros((*grid_slice.shape, 4))
                grid_rgba[grid_mask, 0] = 1.0   # Red channel
                grid_rgba[grid_mask, 1] = 0.4   # Green channel (orange tint)
                grid_rgba[grid_mask, 2] = 0.0   # Blue channel
                grid_rgba[grid_mask, 3] = grid_alpha  # Alpha
                ax.imshow(grid_rgba)

            # Column titles (view names) - only on first row
            if row_idx == 0:
                ax.set_title(view_name, fontsize=14, fontweight='bold')

            # Row labels (stage names) - add prominent text label on left side
            if col_idx == 0:
                # Use text annotation for more prominent row label
                ax.text(-0.15, 0.5, stage_name, transform=ax.transAxes,
                       fontsize=13, fontweight='bold', va='center', ha='right',
                       rotation=90, color='navy')
            ax.axis('off')

    # Add legend
    legend_text = 'Gray = Target (Frame 0)  |  Green = Moved Source (Static)'
    if warped_grids:
        legend_text += '  |  Orange = Deformation Grid'
    fig.text(0.5, 0.01, legend_text, ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.04, left=0.12)  # Extra left margin for row labels

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved 3-view alignment overlay: {output_path}")
    return output_path


def save_alignment_progression(
    target_image: Image,
    rigid_result: Optional[Image],
    affine_result: Optional[Image],
    ffd_result: Optional[Image],
    output_dir: Path,
    prefix: str = "alignment",
    rigid_transform: Optional[any] = None,
    affine_transform: Optional[any] = None,
    ffd_transform: Optional[any] = None,
    show_grid: bool = True,
) -> List[Path]:
    """
    Save alignment progression visualizations

    Creates:
    1. Single-view overlay (axial middle slice)
    2. 3-view overlay (axial, coronal, sagittal) with optional grid deformation

    Args:
        target_image: Target (frame 0)
        rigid_result: Source after rigid (or None)
        affine_result: Source after rigid+affine (or None)
        ffd_result: Source after rigid+affine+ffd (or None)
        output_dir: Output directory
        prefix: Filename prefix
        rigid_transform: Rigid transform for grid visualization (or None)
        affine_transform: Affine transform for grid visualization (or None)
        ffd_transform: FFD transform for grid visualization (or None)
        show_grid: Whether to show grid deformation overlay (default True)

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of stages
    stages = []
    transforms = []

    if rigid_result is not None:
        stages.append(("After Rigid", rigid_result))
        if rigid_transform is not None:
            transforms.append(("After Rigid", rigid_transform))
    if affine_result is not None:
        stages.append(("After Rigid+Affine", affine_result))
        if affine_transform is not None:
            transforms.append(("After Rigid+Affine", affine_transform))
    if ffd_result is not None:
        stages.append(("After Rigid+Affine+FFD", ffd_result))
        if ffd_transform is not None:
            transforms.append(("After Rigid+Affine+FFD", ffd_transform))

    if not stages:
        logger.warning("No alignment results provided for visualization")
        return []

    saved_paths = []

    # Create single-view overlay
    single_path = output_dir / f"{prefix}_overlay_axial.png"
    create_alignment_overlay_figure(
        target_image, stages, single_path, slice_axis="axial"
    )
    saved_paths.append(single_path)

    # Create 3-view overlay with optional grid deformation
    three_view_path = output_dir / f"{prefix}_overlay_3views.png"
    create_alignment_overlay_3views(
        target_image, stages, three_view_path,
        transforms=transforms if show_grid and transforms else None,
        show_grid=show_grid
    )
    saved_paths.append(three_view_path)

    return saved_paths
