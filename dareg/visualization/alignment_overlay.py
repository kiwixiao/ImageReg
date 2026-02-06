"""
Alignment Overlay Visualization

Creates overlay visualizations showing target (gray) + moved source (green)
at 50% transparency for each registration stage:
- Pure rigid
- Rigid + affine
- Rigid + affine + FFD/SVFFD
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch

from deepali.data import Image
from deepali.core import functional as U
import deepali.spatial as spatial

from ..utils.logging_config import get_logger

logger = get_logger("alignment_overlay")


def create_warped_grid_image(grid, transform, stride: int = 8, target_size: int = 256):
    """
    Create warped grid image using deepali's built-in grid_image function

    Reference: PairwiseImageRegistration-deepali_demo.pdf (page 8-9)

    The correct pattern from deepali demo:
        highres_grid = transform.grid().resize(512)
        grid_image = U.grid_image(highres_grid, ...)
        grid_transformer = spatial.ImageTransformer(transform, highres_grid, ...)
        warped_grid = grid_transformer(grid_image)

    KEY: Use transform.grid().resize() - start from transform's own grid,
    not an external grid. ImageTransformer handles coordinate conversion internally.

    Args:
        grid: The image grid (deepali Grid object) - used as fallback
        transform: The transform to apply (can be SequentialTransform, FFD, etc.)
        stride: Grid line spacing (default 8)
        target_size: Resolution for high-res grid (default 256)

    Returns:
        warped_grid_np: Warped grid as numpy array [D, H, W]
        original_grid_np: Original grid as numpy array [D, H, W]
    """
    try:
        # CRITICAL: Use transform's own grid, then resize for visualization
        # This is the pattern from deepali demo PDF (page 8-9)
        if hasattr(transform, 'grid') and callable(transform.grid):
            transform_grid = transform.grid()
        else:
            # Fallback to provided grid if transform doesn't have grid()
            transform_grid = grid
            logger.debug("Transform has no grid() method, using provided grid")

        # For anisotropic grids, use the original grid directly (don't resize)
        # resize() creates an isotropic cubic grid which breaks coordinate system
        # for transforms learned on anisotropic data
        # Instead, just use the transform's grid directly
        grid_highres = transform_grid

        logger.debug(f"Grid visualization: shape={grid_highres.shape}, spacing={[f'{s:.4f}' for s in grid_highres.spacing().tolist()]}")

        # Create grid image using deepali's built-in function
        # inverted=False: grid lines = 1 (white), cell interiors = 0 (black)
        # This allows us to add green color at grid line positions
        grid_image = U.grid_image(grid_highres, num=1, stride=stride, inverted=False)

        # Apply the transform to warp the grid image
        # ImageTransformer(transform, target_grid) handles coordinate conversion internally
        # padding="border" preserves grid lines at edges
        grid_transformer = spatial.ImageTransformer(transform, grid_highres, padding="border")
        warped_grid = grid_transformer(grid_image)

        # Convert to numpy
        original_grid_np = grid_image.squeeze().detach().cpu().numpy()
        warped_grid_np = warped_grid.squeeze().detach().cpu().numpy()

        return warped_grid_np, original_grid_np

    except Exception as e:
        logger.warning(f"Could not create warped grid: {e}")
        import traceback
        traceback.print_exc()
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

            # Overlay warped deformation grid (solid black lines for visibility)
            grid_slice = grid_slices[col_idx]
            if grid_slice is not None:
                # Resize grid slice to match image slice if needed
                from scipy.ndimage import zoom
                if grid_slice.shape != tgt_slice.shape:
                    zoom_factors = (tgt_slice.shape[0] / grid_slice.shape[0],
                                   tgt_slice.shape[1] / grid_slice.shape[1])
                    grid_slice = zoom(grid_slice, zoom_factors, order=1)

                # Create SOLID BLACK lines for deformation grid (high visibility)
                grid_normalized = (grid_slice - grid_slice.min()) / (grid_slice.max() - grid_slice.min() + 1e-8)
                # Grid lines are white (high values) in inverted=True mode
                # Threshold to detect grid lines
                grid_mask = grid_normalized > 0.15
                grid_rgba = np.zeros((*grid_slice.shape, 4))
                # BLACK color with high opacity for solid visible lines
                grid_rgba[grid_mask, 0] = 0.0   # Red channel (black)
                grid_rgba[grid_mask, 1] = 0.0   # Green channel (black)
                grid_rgba[grid_mask, 2] = 0.0   # Blue channel (black)
                grid_rgba[grid_mask, 3] = 0.85  # High alpha for solid appearance
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
        legend_text += '  |  Black Grid = Deformation'
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


def create_standalone_grid_3views(
    grid,
    transforms: List[Tuple[str, any]],
    output_path: Path,
    grid_stride: int = 12,
) -> Path:
    """
    Create standalone 3-view (axial, coronal, sagittal) deformed grid visualization

    Shows deformed grid with DISPLACEMENT VECTORS (arrows) indicating movement direction and magnitude.

    Args:
        grid: The image grid (deepali Grid object)
        transforms: List of (stage_name, transform) tuples
        output_path: Output path
        grid_stride: Grid line spacing (default 12)

    Returns:
        Path to saved figure
    """
    from scipy.ndimage import zoom

    # Filter out None transforms
    valid_transforms = [(name, t) for name, t in transforms if t is not None]

    if not valid_transforms:
        logger.warning("No valid transforms provided for grid visualization")
        return None

    # Create warped grids and compute displacement fields for each stage
    warped_grids = {}
    displacement_fields = {}
    original_grid_np = None

    for stage_name, transform in valid_transforms:
        warped_grid_np, orig_grid_np = create_warped_grid_image(
            grid, transform, stride=grid_stride
        )
        if warped_grid_np is not None:
            warped_grids[stage_name] = warped_grid_np
            if original_grid_np is None:
                original_grid_np = orig_grid_np

            # Compute displacement field for vector visualization
            disp_field = _compute_displacement_field(grid, transform)
            if disp_field is not None:
                displacement_fields[stage_name] = disp_field

    if not warped_grids or original_grid_np is None:
        logger.warning("Could not create any warped grids")
        return None

    # Create figure: rows = stages, cols = views (axial, coronal, sagittal)
    n_stages = len(warped_grids)
    fig, axes = plt.subplots(n_stages, 3, figsize=(15, 5 * n_stages), facecolor='black')

    if n_stages == 1:
        axes = axes.reshape(1, -1)

    view_names = ['Axial', 'Coronal', 'Sagittal']

    for row_idx, (stage_name, warped_grid) in enumerate(warped_grids.items()):
        # Get middle slices
        d_mid = warped_grid.shape[0] // 2
        h_mid = warped_grid.shape[1] // 2
        w_mid = warped_grid.shape[2] // 2

        # Warped grid slices
        warped_slices = [
            warped_grid[d_mid, :, :],  # Axial
            warped_grid[:, h_mid, :],  # Coronal
            warped_grid[:, :, w_mid],  # Sagittal
        ]

        # Get displacement field for this stage
        disp = displacement_fields.get(stage_name)

        for col_idx, (warped_slice, view_name) in enumerate(zip(warped_slices, view_names)):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')

            # Display warped grid (white lines on black background)
            ax.imshow(warped_slice, cmap='gray', vmin=0, vmax=1)

            # Add displacement vectors if available
            if disp is not None:
                _add_displacement_vectors(ax, disp, col_idx, d_mid, h_mid, w_mid, warped_slice.shape)

            # Column titles (view names) - only on first row
            if row_idx == 0:
                ax.set_title(view_name, fontsize=14, fontweight='bold', color='white')

            # Row labels (stage names)
            if col_idx == 0:
                ax.text(-0.15, 0.5, stage_name, transform=ax.transAxes,
                       fontsize=13, fontweight='bold', va='center', ha='right',
                       rotation=90, color='white')
            ax.axis('off')

    # Add title
    fig.suptitle('Deformation Grid with Displacement Vectors (red arrows)', fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.02, left=0.12, top=0.94)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

    logger.info(f"Saved standalone grid 3-view with vectors: {output_path}")
    return output_path


def _compute_displacement_field(grid, transform) -> Optional[np.ndarray]:
    """
    Compute displacement field from transform using deepali's flow() API

    Returns:
        Displacement field as numpy array [3, D, H, W] in normalized coords
    """
    try:
        import torch

        with torch.no_grad():
            # Use transform's flow() method if available (returns FlowFields)
            if hasattr(transform, 'flow'):
                flow = transform.flow(grid)
                # FlowFields tensor is [N, D, H, W, 3] or [N, 3, D, H, W]
                flow_tensor = flow.tensor()
                if flow_tensor.dim() == 5:
                    if flow_tensor.shape[-1] == 3:  # [N, D, H, W, 3]
                        flow_tensor = flow_tensor.permute(0, 4, 1, 2, 3)  # -> [N, 3, D, H, W]
                    flow_tensor = flow_tensor.squeeze(0)  # [3, D, H, W]
                disp_np = flow_tensor.cpu().numpy()
                return disp_np

            # Fallback: Use ImageTransformer to compute displacement
            # Create identity grid and transform it
            transformer = spatial.ImageTransformer(transform, grid)

            # Get the displacement by comparing identity to transformed coords
            # ImageTransformer internally computes the sampling grid
            # We can access it via the flow field
            if hasattr(transformer, 'flow'):
                flow = transformer.flow(grid)
                flow_tensor = flow.tensor()
                if flow_tensor.dim() == 5:
                    if flow_tensor.shape[-1] == 3:
                        flow_tensor = flow_tensor.permute(0, 4, 1, 2, 3)
                    flow_tensor = flow_tensor.squeeze(0)
                disp_np = flow_tensor.cpu().numpy()
                return disp_np

            # Last fallback: compute manually using grid points
            # Get normalized grid coordinates
            coords = grid.coords(channels_last=True)  # [D, H, W, 3]
            coords = coords.unsqueeze(0)  # [1, D, H, W, 3]

            # For SequentialTransform, we need to use it differently
            # Get warped coordinates by applying transform
            if hasattr(transform, 'disp'):
                disp = transform.disp(grid)
                disp_np = disp.squeeze(0).cpu().numpy()
                if disp_np.shape[-1] == 3:  # [D, H, W, 3]
                    disp_np = np.transpose(disp_np, (3, 0, 1, 2))  # -> [3, D, H, W]
                return disp_np

        return None

    except Exception as e:
        logger.warning(f"Could not compute displacement field: {e}")
        import traceback
        traceback.print_exc()
        return None


def _add_displacement_vectors(ax, disp: np.ndarray, view_idx: int,
                              d_mid: int, h_mid: int, w_mid: int,
                              slice_shape: tuple):
    """
    Add displacement vector arrows to plot

    Args:
        ax: Matplotlib axis
        disp: Displacement field [3, D, H, W]
        view_idx: 0=axial, 1=coronal, 2=sagittal
        d_mid, h_mid, w_mid: Middle slice indices
        slice_shape: Shape of the 2D slice being displayed
    """
    # Subsample for cleaner visualization (every 8-12 points)
    step = max(8, min(slice_shape) // 15)

    if view_idx == 0:  # Axial (D slice) - show X,Y displacement
        # disp shape: [3, D, H, W] where 3 = (x, y, z)
        dx = disp[0, d_mid, ::step, ::step]  # X displacement
        dy = disp[1, d_mid, ::step, ::step]  # Y displacement
        y_coords, x_coords = np.mgrid[0:dx.shape[0], 0:dx.shape[1]]
    elif view_idx == 1:  # Coronal (H slice) - show X,Z displacement
        dx = disp[0, ::step, h_mid, ::step]  # X displacement
        dz = disp[2, ::step, h_mid, ::step]  # Z displacement
        dy = dz  # Use Z as Y for display
        y_coords, x_coords = np.mgrid[0:dx.shape[0], 0:dx.shape[1]]
    else:  # Sagittal (W slice) - show Y,Z displacement
        dy_actual = disp[1, ::step, ::step, w_mid]  # Y displacement
        dz = disp[2, ::step, ::step, w_mid]  # Z displacement
        dx = dy_actual
        dy = dz
        y_coords, x_coords = np.mgrid[0:dx.shape[0], 0:dx.shape[1]]

    # Scale vectors for visibility
    # The displacement is in normalized coords [-1, 1], scale to voxels
    scale_x = slice_shape[1] / 2.0
    scale_y = slice_shape[0] / 2.0

    dx_scaled = dx * scale_x
    dy_scaled = dy * scale_y

    # Compute magnitude for color coding
    magnitude = np.sqrt(dx_scaled**2 + dy_scaled**2)
    max_mag = magnitude.max() if magnitude.max() > 0 else 1.0

    # Only draw arrows where there's significant displacement
    threshold = max_mag * 0.05  # 5% of max displacement
    mask = magnitude > threshold

    if mask.any():
        # Draw arrows with quiver
        ax.quiver(
            x_coords[mask] * step,
            y_coords[mask] * step,
            dx_scaled[mask],
            dy_scaled[mask],
            color='red',
            scale=1,
            scale_units='xy',
            angles='xy',
            width=0.005,
            headwidth=4,
            headlength=5,
            alpha=0.8,
        )


def create_grid_comparison_figure(
    grid,
    transform,
    output_path: Path,
    grid_stride: int = 12,
) -> Path:
    """
    Create comparison figure: Direct transform vs World coordinate conversion

    This validates that both methods produce identical results (they should theoretically).

    Method A: ImageTransformer(transform, grid) - using deepali's built-in transformer
    Method B: Manual grid_sample using transform's flow field

    Args:
        grid: The image grid (deepali Grid object)
        transform: Transform to apply
        output_path: Output path
        grid_stride: Grid line spacing

    Returns:
        Path to saved figure
    """
    import torch

    try:
        # Create grid image
        grid_image = U.grid_image(grid, num=1, stride=grid_stride, inverted=True)

        # Method A: Direct ImageTransformer (standard deepali method)
        transformer_a = spatial.ImageTransformer(transform, grid, padding="border")
        warped_a = transformer_a(grid_image)
        warped_a_np = warped_a.squeeze().detach().cpu().numpy()

        # Method B: Manual grid_sample using flow field
        with torch.no_grad():
            # Get flow field from transform (displacement in normalized coords)
            if hasattr(transform, 'flow'):
                flow = transform.flow(grid)
                flow_tensor = flow.tensor()  # [N, D, H, W, 3] typically
            else:
                # Fallback - just use Method A result for comparison
                warped_b_np = warped_a_np
                max_diff = 0.0
                logger.info("Transform has no flow() method, skipping Method B")

            if hasattr(transform, 'flow'):
                # Get identity grid coordinates
                identity_coords = grid.coords(channels_last=True)  # [D, H, W, 3]
                identity_coords = identity_coords.unsqueeze(0)  # [1, D, H, W, 3]

                # Add flow to get warped coordinates
                # Flow is displacement, so warped = identity + flow
                if flow_tensor.shape[-1] == 3:  # [N, D, H, W, 3]
                    warped_coords = identity_coords + flow_tensor
                else:  # [N, 3, D, H, W]
                    flow_tensor = flow_tensor.permute(0, 2, 3, 4, 1)  # -> [N, D, H, W, 3]
                    warped_coords = identity_coords + flow_tensor

                # Prepare grid image for sampling
                grid_img_5d = grid_image
                if grid_img_5d.dim() == 3:
                    grid_img_5d = grid_img_5d.unsqueeze(0).unsqueeze(0)
                elif grid_img_5d.dim() == 4:
                    grid_img_5d = grid_img_5d.unsqueeze(0)

                # Sample using grid_sample
                warped_b = torch.nn.functional.grid_sample(
                    grid_img_5d,
                    warped_coords,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True,
                )

                warped_b_np = warped_b.squeeze().detach().cpu().numpy()

                # Compute difference
                diff = np.abs(warped_a_np - warped_b_np)
                max_diff = diff.max()

        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')

        d_mid = warped_a_np.shape[0] // 2

        # Row 1: Method A (ImageTransformer)
        axes[0, 0].imshow(warped_a_np[d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Method A: ImageTransformer\n(Axial)', fontsize=12, color='white')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(warped_a_np[:, warped_a_np.shape[1]//2, :], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Method A: ImageTransformer\n(Coronal)', fontsize=12, color='white')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(warped_a_np[:, :, warped_a_np.shape[2]//2], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Method A: ImageTransformer\n(Sagittal)', fontsize=12, color='white')
        axes[0, 2].axis('off')

        # Row 2: Method B (manual grid_sample)
        axes[1, 0].imshow(warped_b_np[d_mid, :, :], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Method B: grid_sample+flow\n(Axial)', fontsize=12, color='white')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(warped_b_np[:, warped_b_np.shape[1]//2, :], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Method B: grid_sample+flow\n(Coronal)', fontsize=12, color='white')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(warped_b_np[:, :, warped_b_np.shape[2]//2], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('Method B: grid_sample+flow\n(Sagittal)', fontsize=12, color='white')
        axes[1, 2].axis('off')

        # Add comparison info
        fig.suptitle(f'Grid Warp Comparison: ImageTransformer vs grid_sample+flow\nMax difference: {max_diff:.6f} (should be ~0)',
                    fontsize=14, fontweight='bold', color='white', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

        logger.info(f"Saved grid comparison figure: {output_path} (max_diff={max_diff:.6f})")
        return output_path

    except Exception as e:
        logger.error(f"Could not create grid comparison figure: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_displacement_magnitude_figure(
    grid,
    transforms: List[Tuple[str, any]],
    output_path: Path,
) -> Path:
    """
    Create displacement magnitude visualization showing WHERE motion is strongest.

    Shows:
    - Row per transform stage
    - Heatmap of displacement magnitude (color = how much motion)
    - Multi-slice view (multiple Z slices to see variation)
    - Statistics (max, mean displacement in mm)

    Args:
        grid: The image grid (deepali Grid object)
        transforms: List of (stage_name, transform) tuples
        output_path: Output path

    Returns:
        Path to saved figure
    """
    import torch

    # Filter out None transforms
    valid_transforms = [(name, t) for name, t in transforms if t is not None]

    if not valid_transforms:
        logger.warning("No valid transforms provided")
        return None

    # Get grid spacing for converting to mm
    spacing = grid.spacing()
    if hasattr(spacing, 'tolist'):
        spacing = spacing.tolist()
    spacing = np.array(spacing)  # [z, y, x] in mm

    # Compute displacement fields for each transform
    displacement_data = {}
    for stage_name, transform in valid_transforms:
        disp = _compute_displacement_field(grid, transform)
        if disp is not None:
            displacement_data[stage_name] = disp

    if not displacement_data:
        logger.warning("Could not compute any displacement fields")
        return None

    n_stages = len(displacement_data)

    # Create figure: rows = stages, cols = [Axial slices (3), Magnitude stats]
    fig = plt.figure(figsize=(20, 5 * n_stages), facecolor='black')

    for row_idx, (stage_name, disp) in enumerate(displacement_data.items()):
        # disp shape: [3, D, H, W] in normalized coords
        # Convert to mm: multiply by half the grid extent
        # Physical extent = shape * spacing (in mm)
        import torch
        shape_tensor = torch.tensor(list(grid.shape), dtype=torch.float32)
        spacing_tensor = grid.spacing()
        extent = (shape_tensor * spacing_tensor).numpy()  # [z, y, x] in mm
        scale = extent / 2.0  # normalized [-1,1] -> mm

        # Scale displacement to mm
        disp_mm = np.zeros_like(disp)
        for i in range(3):
            disp_mm[i] = disp[i] * scale[i]

        # Compute displacement magnitude in mm
        disp_magnitude = np.sqrt(disp_mm[0]**2 + disp_mm[1]**2 + disp_mm[2]**2)

        # Statistics
        max_disp = disp_magnitude.max()
        mean_disp = disp_magnitude.mean()
        std_disp = disp_magnitude.std()

        # Get multiple Z slices (show 5 slices across the volume)
        D = disp_magnitude.shape[0]
        slice_indices = [int(D * f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
        slice_indices = [min(s, D-1) for s in slice_indices]

        # Create subplots for this row
        for col_idx, slice_idx in enumerate(slice_indices):
            ax = fig.add_subplot(n_stages, 6, row_idx * 6 + col_idx + 1)
            ax.set_facecolor('black')

            # Displacement magnitude heatmap for this slice
            magnitude_slice = disp_magnitude[slice_idx, :, :]

            # Use 'hot' colormap - black=0, red/yellow=high
            im = ax.imshow(magnitude_slice, cmap='hot', vmin=0, vmax=max(max_disp, 0.1))
            ax.set_title(f'Z={slice_idx}/{D}\n({slice_idx * spacing[0]:.1f}mm)',
                        fontsize=10, color='white')
            ax.axis('off')

            # Add colorbar to last slice
            if col_idx == 4:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.ax.set_ylabel('mm', color='white', fontsize=10)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Add statistics panel
        ax_stats = fig.add_subplot(n_stages, 6, row_idx * 6 + 6)
        ax_stats.set_facecolor('black')
        ax_stats.axis('off')

        stats_text = f"""
{stage_name}

Displacement Stats:
  Max:  {max_disp:.2f} mm
  Mean: {mean_disp:.2f} mm
  Std:  {std_disp:.2f} mm

Grid: {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}
Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm
"""
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, color='white', family='monospace',
                     verticalalignment='center')

    fig.suptitle('Displacement Magnitude Across Z-Slices (mm)',
                fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

    logger.info(f"Saved displacement magnitude figure: {output_path}")
    return output_path


def create_3d_motion_summary(
    grid,
    transform,
    output_path: Path,
    stage_name: str = "Transform",
) -> Path:
    """
    Create comprehensive 3D motion summary visualization.

    Shows:
    - XY plane (Axial): displacement vectors
    - XZ plane (Coronal): displacement vectors
    - YZ plane (Sagittal): displacement vectors
    - 3D displacement magnitude with all components visible
    - Per-axis displacement breakdown

    Args:
        grid: The image grid
        transform: The transform to visualize
        output_path: Output path
        stage_name: Name of the transform stage

    Returns:
        Path to saved figure
    """
    import torch

    disp = _compute_displacement_field(grid, transform)
    if disp is None:
        logger.warning("Could not compute displacement field")
        return None

    # Get grid spacing and extent for mm conversion
    import torch
    grid_spacing = grid.spacing()
    if hasattr(grid_spacing, 'tolist'):
        spacing = np.array(grid_spacing.tolist())
    elif hasattr(grid_spacing, 'cpu'):
        spacing = grid_spacing.cpu().numpy()
    else:
        spacing = np.array(list(grid_spacing))

    # Physical extent = shape * spacing (in mm)
    shape_tensor = torch.tensor(list(grid.shape), dtype=torch.float32)
    spacing_tensor = grid.spacing()
    extent = (shape_tensor * spacing_tensor).numpy()  # [z, y, x] in mm
    scale = extent / 2.0

    # Convert to mm
    disp_mm = np.zeros_like(disp)
    for i in range(3):
        disp_mm[i] = disp[i] * scale[i]

    # Compute total magnitude
    total_mag = np.sqrt(disp_mm[0]**2 + disp_mm[1]**2 + disp_mm[2]**2)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='black')

    D, H, W = disp_mm.shape[1:]
    d_mid, h_mid, w_mid = D // 2, H // 2, W // 2

    # Row 1: Per-axis displacement (X, Y, Z components separately)
    axis_names = ['X (Left-Right)', 'Y (Ant-Post)', 'Z (Sup-Inf)', 'Total Magnitude']
    axis_data = [disp_mm[0], disp_mm[1], disp_mm[2], total_mag]
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'hot']  # Diverging for signed, hot for magnitude

    for col, (name, data, cmap) in enumerate(zip(axis_names, axis_data, cmaps)):
        ax = axes[0, col]
        ax.set_facecolor('black')

        # Show axial slice (middle Z)
        slice_data = data[d_mid, :, :]

        if cmap == 'RdBu_r':
            # Symmetric colormap for signed displacement
            vmax = max(abs(slice_data.min()), abs(slice_data.max()), 0.1)
            im = ax.imshow(slice_data, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=max(slice_data.max(), 0.1))

        ax.set_title(f'{name}\nMax: {data.max():.2f}mm, Min: {data.min():.2f}mm',
                    fontsize=11, color='white')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Row 2: Vector field views with arrows (subsampled)
    step = max(8, min(H, W) // 12)

    # Axial view (XY plane at middle Z)
    ax = axes[1, 0]
    ax.set_facecolor('black')
    ax.imshow(total_mag[d_mid, :, :], cmap='gray', alpha=0.5)
    y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
    dx = disp_mm[0, d_mid, ::step, ::step]  # X displacement
    dy = disp_mm[1, d_mid, ::step, ::step]  # Y displacement
    mag = np.sqrt(dx**2 + dy**2)
    mask = mag > mag.max() * 0.05
    if mask.any():
        ax.quiver(x_coords[mask], y_coords[mask], dx[mask], dy[mask],
                 color='red', scale=50, width=0.004, headwidth=4)
    ax.set_title(f'Axial (Z={d_mid}): X-Y motion', fontsize=11, color='white')
    ax.axis('off')

    # Coronal view (XZ plane at middle Y)
    ax = axes[1, 1]
    ax.set_facecolor('black')
    ax.imshow(total_mag[:, h_mid, :], cmap='gray', alpha=0.5)
    z_coords, x_coords = np.mgrid[0:D:max(1,step//3), 0:W:step]
    dx = disp_mm[0, ::max(1,step//3), h_mid, ::step]  # X displacement
    dz = disp_mm[2, ::max(1,step//3), h_mid, ::step]  # Z displacement
    mag = np.sqrt(dx**2 + dz**2)
    mask = mag > mag.max() * 0.05
    if mask.any():
        ax.quiver(x_coords[mask], z_coords[mask], dx[mask], dz[mask],
                 color='cyan', scale=50, width=0.004, headwidth=4)
    ax.set_title(f'Coronal (Y={h_mid}): X-Z motion', fontsize=11, color='white')
    ax.axis('off')

    # Sagittal view (YZ plane at middle X)
    ax = axes[1, 2]
    ax.set_facecolor('black')
    ax.imshow(total_mag[:, :, w_mid], cmap='gray', alpha=0.5)
    z_coords, y_coords = np.mgrid[0:D:max(1,step//3), 0:H:step]
    dy = disp_mm[1, ::max(1,step//3), ::step, w_mid]  # Y displacement
    dz = disp_mm[2, ::max(1,step//3), ::step, w_mid]  # Z displacement
    mag = np.sqrt(dy**2 + dz**2)
    mask = mag > mag.max() * 0.05
    if mask.any():
        ax.quiver(y_coords[mask], z_coords[mask], dy[mask], dz[mask],
                 color='lime', scale=50, width=0.004, headwidth=4)
    ax.set_title(f'Sagittal (X={w_mid}): Y-Z motion', fontsize=11, color='white')
    ax.axis('off')

    # Statistics summary
    ax = axes[1, 3]
    ax.set_facecolor('black')
    ax.axis('off')

    stats_text = f"""
{stage_name}

DISPLACEMENT STATISTICS (mm):

Total Magnitude:
  Max:  {total_mag.max():.3f}
  Mean: {total_mag.mean():.3f}
  Std:  {total_mag.std():.3f}

Per-Axis Max Displacement:
  X (L-R): {abs(disp_mm[0]).max():.3f} mm
  Y (A-P): {abs(disp_mm[1]).max():.3f} mm
  Z (S-I): {abs(disp_mm[2]).max():.3f} mm

Grid: {D}×{H}×{W} voxels
Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm
Extent: {extent[0]:.1f}×{extent[1]:.1f}×{extent[2]:.1f} mm
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=11, color='white', family='monospace',
           verticalalignment='top')

    fig.suptitle(f'3D Motion Summary: {stage_name}',
                fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

    logger.info(f"Saved 3D motion summary: {output_path}")
    return output_path


def create_multi_slice_displacement_view(
    grid,
    transform,
    output_path: Path,
    target_slices: int = 20,
    stage_name: str = "Transform",
    separate_figures: bool = True,
) -> List[Path]:
    """
    Create multi-slice displacement visualization for all 3 anatomical directions.

    Shows slices expanding from middle equally in both directions, every other slice.
    Auto-handles when a direction doesn't have enough slices.

    Args:
        grid: The image grid
        transform: The transform to visualize
        output_path: Output path (base name, will append direction suffix)
        target_slices: Target number of slices to show per direction (default 20)
        stage_name: Name of the transform stage
        separate_figures: If True, create 3 separate figures (one per direction)

    Returns:
        List of paths to saved figures
    """
    import torch

    disp = _compute_displacement_field(grid, transform)
    if disp is None:
        logger.warning("Could not compute displacement field")
        return None

    # Get grid spacing and compute physical extent
    grid_spacing = grid.spacing()
    if hasattr(grid_spacing, 'tolist'):
        spacing = np.array(grid_spacing.tolist())
    elif hasattr(grid_spacing, 'cpu'):
        spacing = grid_spacing.cpu().numpy()
    else:
        spacing = np.array(list(grid_spacing))

    # Physical extent = shape * spacing (in mm)
    shape_tensor = torch.tensor(list(grid.shape), dtype=torch.float32)
    spacing_tensor = grid.spacing()
    extent = (shape_tensor * spacing_tensor).numpy()
    scale = extent / 2.0

    # Convert displacement to mm
    disp_mm = np.zeros_like(disp)
    for i in range(3):
        disp_mm[i] = disp[i] * scale[i]

    # Compute total magnitude
    total_mag = np.sqrt(disp_mm[0]**2 + disp_mm[1]**2 + disp_mm[2]**2)

    # Get dimensions: disp shape is [3, D, H, W] = [3, Z, Y, X]
    D, H, W = disp_mm.shape[1:]  # Z, Y, X dimensions

    def get_slice_indices(dim_size: int, target_count: int, step: int = 2) -> list:
        """Get slice indices expanding from middle, every 'step' slices."""
        mid = dim_size // 2
        indices = [mid]

        # Expand in both directions
        offset = step
        while len(indices) < target_count and offset < dim_size:
            # Add slice below middle
            if mid - offset >= 0:
                indices.insert(0, mid - offset)
            # Add slice above middle
            if mid + offset < dim_size and len(indices) < target_count:
                indices.append(mid + offset)
            offset += step

        # If still not enough (step too large), fill with step=1
        if len(indices) < target_count:
            offset = 1
            while len(indices) < target_count:
                if mid - offset >= 0 and (mid - offset) not in indices:
                    indices.insert(0, mid - offset)
                if mid + offset < dim_size and (mid + offset) not in indices and len(indices) < target_count:
                    indices.append(mid + offset)
                offset += 1
                if offset >= dim_size:
                    break

        return sorted(indices)

    # Get slice indices for each direction
    z_indices = get_slice_indices(D, target_slices, step=2)
    y_indices = get_slice_indices(H, target_slices, step=2)
    x_indices = get_slice_indices(W, target_slices, step=2)

    # Global colormap range
    vmax = total_mag.max()
    if vmax < 0.1:
        vmax = 0.1

    direction_info = [
        ('Axial', 'Z', z_indices, D, spacing[0], lambda idx: total_mag[idx, :, :]),
        ('Coronal', 'Y', y_indices, H, spacing[1], lambda idx: total_mag[:, idx, :]),
        ('Sagittal', 'X', x_indices, W, spacing[2], lambda idx: total_mag[:, :, idx]),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    if separate_figures:
        # Create separate figure for each direction
        for dir_name, axis_name, indices, dim_size, dim_spacing, slice_func in direction_info:
            n_slices = len(indices)

            # Calculate grid layout: aim for ~5 rows
            n_cols = min(n_slices, 5)
            n_rows = (n_slices + n_cols - 1) // n_cols

            # Figure size: each slice ~2.5 inches wide, ~2 inches tall
            fig_width = n_cols * 2.5 + 1.5  # extra for colorbar
            fig_height = n_rows * 2.2 + 1.0  # extra for title

            fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(fig_width, fig_height),
                                    facecolor='black',
                                    gridspec_kw={'width_ratios': [1]*n_cols + [0.3]})

            if n_rows == 1:
                axes = axes.reshape(1, -1)

            # Plot each slice
            for i, slice_idx in enumerate(indices):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                ax.set_facecolor('black')

                slice_data = slice_func(slice_idx)
                im = ax.imshow(slice_data, cmap='hot', vmin=0, vmax=vmax, aspect='equal')

                phys_pos = slice_idx * dim_spacing
                ax.set_title(f'{axis_name}={slice_idx} ({phys_pos:.1f}mm)',
                           fontsize=9, color='white', fontweight='bold')
                ax.axis('off')

            # Hide unused subplot cells
            for i in range(n_slices, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)

            # Add colorbar in last column (merge all rows)
            # Hide individual colorbar cells
            for row in range(n_rows):
                axes[row, n_cols].set_visible(False)

            # Create a single colorbar axis spanning all rows
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
            cbar.ax.set_ylabel('Displacement (mm)', color='white', fontsize=11)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Title with stats
            fig.suptitle(f'{dir_name} View ({axis_name}-axis): {stage_name}\n'
                        f'{n_slices}/{dim_size} slices | Spacing: {dim_spacing:.2f}mm | '
                        f'Max: {vmax:.2f}mm, Mean: {total_mag.mean():.2f}mm',
                        fontsize=12, fontweight='bold', color='white', y=0.98)

            plt.tight_layout()
            plt.subplots_adjust(top=0.88, right=0.90, wspace=0.05, hspace=0.25)

            # Save with direction suffix
            base_name = output_path.stem
            dir_path = output_path.parent / f"{base_name}_{dir_name.lower()}.png"
            plt.savefig(dir_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()

            saved_paths.append(dir_path)
            logger.info(f"Saved {dir_name} view: {dir_path}")
            logger.info(f"  {n_slices} slices from {indices[0]} to {indices[-1]}")

    else:
        # Original combined figure (kept for backward compatibility)
        n_z, n_y, n_x = len(z_indices), len(y_indices), len(x_indices)
        max_cols = max(n_z, n_y, n_x)

        fig_height = 4 * 3
        fig_width = 1.2 * max_cols + 2
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')

        for row_idx, (dir_name, axis_name, indices, dim_size, dim_spacing, slice_func) in enumerate(direction_info):
            n_slices = len(indices)

            for col_idx, slice_idx in enumerate(indices):
                ax = fig.add_subplot(3, max_cols + 1, row_idx * (max_cols + 1) + col_idx + 1)
                ax.set_facecolor('black')

                slice_data = slice_func(slice_idx)
                im = ax.imshow(slice_data, cmap='hot', vmin=0, vmax=vmax, aspect='auto')

                phys_pos = slice_idx * dim_spacing
                ax.set_title(f'{slice_idx}\n({phys_pos:.1f}mm)', fontsize=7, color='white')
                ax.axis('off')

            ax_info = fig.add_subplot(3, max_cols + 1, row_idx * (max_cols + 1) + max_cols + 1)
            ax_info.set_facecolor('black')
            ax_info.axis('off')

            cbar = plt.colorbar(im, ax=ax_info, fraction=0.8, pad=0.05, orientation='vertical')
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
            cbar.ax.set_ylabel('mm', color='white', fontsize=9)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            info_text = f"{dir_name} ({axis_name})\n{n_slices}/{dim_size} slices\nSpacing: {dim_spacing:.2f}mm"
            ax_info.text(0.5, 0.3, info_text, transform=ax_info.transAxes,
                        fontsize=9, color='white', ha='center', va='center',
                        family='monospace')

        fig.suptitle(f'Multi-Slice Displacement: {stage_name}\n'
                    f'Max: {total_mag.max():.2f}mm, Mean: {total_mag.mean():.2f}mm',
                    fontsize=12, fontweight='bold', color='white', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.05)

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

        saved_paths.append(output_path)
        logger.info(f"Saved multi-slice displacement view: {output_path}")

    return saved_paths


def create_multi_slice_grid_with_arrows(
    grid,
    transform,
    anatomical_image: np.ndarray,
    output_path: Path,
    target_slices: int = 20,
    stage_name: str = "Transform",
    arrow_step: int = 8,
    grid_stride: int = 12,
    grid_alpha: float = 0.6,
) -> List[Path]:
    """
    Create multi-slice visualization with anatomical image, warped grid overlay, and displacement arrows.

    For each slice shows:
    - Grayscale anatomical image as background
    - Transparent warped grid overlay (green)
    - Red displacement arrows showing movement direction/magnitude

    Args:
        grid: The image grid (deepali Grid object)
        transform: The transform to visualize
        anatomical_image: 3D anatomical image as numpy array [D, H, W]
        output_path: Output path (base name, will append direction suffix)
        target_slices: Target number of slices to show per direction (default 20)
        stage_name: Name of the transform stage
        arrow_step: Spacing between arrows (default 8)
        grid_stride: Grid line spacing (default 12)
        grid_alpha: Grid overlay transparency (default 0.6)

    Returns:
        List of paths to saved figures
    """
    import torch
    from scipy.ndimage import zoom

    # Compute displacement field
    disp = _compute_displacement_field(grid, transform)
    if disp is None:
        logger.warning("Could not compute displacement field")
        return []

    # Get grid spacing and compute physical extent
    grid_spacing = grid.spacing()
    if hasattr(grid_spacing, 'tolist'):
        spacing = np.array(grid_spacing.tolist())
    elif hasattr(grid_spacing, 'cpu'):
        spacing = grid_spacing.cpu().numpy()
    else:
        spacing = np.array(list(grid_spacing))

    # Physical extent = shape * spacing (in mm)
    shape_tensor = torch.tensor(list(grid.shape), dtype=torch.float32)
    spacing_tensor = grid.spacing()
    extent = (shape_tensor * spacing_tensor).numpy()
    scale = extent / 2.0

    # Convert displacement to mm
    disp_mm = np.zeros_like(disp)
    for i in range(3):
        disp_mm[i] = disp[i] * scale[i]

    # Compute total magnitude
    total_mag = np.sqrt(disp_mm[0]**2 + disp_mm[1]**2 + disp_mm[2]**2)

    # Get dimensions
    D, H, W = disp_mm.shape[1:]  # Z, Y, X dimensions

    # Create warped grid image
    warped_grid_np, original_grid_np = create_warped_grid_image(grid, transform, stride=grid_stride)

    if warped_grid_np is None:
        logger.warning("Could not create warped grid, using displacement magnitude only")
        warped_grid_np = np.zeros((D, H, W))

    def get_slice_indices(dim_size: int, target_count: int, step: int = 2) -> list:
        """Get slice indices expanding from middle, every 'step' slices."""
        mid = dim_size // 2
        indices = [mid]

        offset = step
        while len(indices) < target_count and offset < dim_size:
            if mid - offset >= 0:
                indices.insert(0, mid - offset)
            if mid + offset < dim_size and len(indices) < target_count:
                indices.append(mid + offset)
            offset += step

        if len(indices) < target_count:
            offset = 1
            while len(indices) < target_count:
                if mid - offset >= 0 and (mid - offset) not in indices:
                    indices.insert(0, mid - offset)
                if mid + offset < dim_size and (mid + offset) not in indices and len(indices) < target_count:
                    indices.append(mid + offset)
                offset += 1
                if offset >= dim_size:
                    break

        return sorted(indices)

    # Get slice indices for each direction
    z_indices = get_slice_indices(D, target_slices, step=2)
    y_indices = get_slice_indices(H, target_slices, step=2)
    x_indices = get_slice_indices(W, target_slices, step=2)

    # Global max for arrow scaling
    vmax_original = total_mag.max()  # Actual max displacement (may be 0)
    vmax = max(vmax_original, 0.1)  # Floor for scaling (avoid divide-by-zero)
    has_meaningful_displacement = vmax_original > 0.01  # Flag for arrow drawing

    # Debug: Log input shapes and value ranges
    logger.debug(f"Grid+arrows visualization input:")
    logger.debug(f"  anatomical_image shape: {anatomical_image.shape}, dtype: {anatomical_image.dtype}")
    logger.debug(f"  anatomical_image range: [{anatomical_image.min():.4f}, {anatomical_image.max():.4f}]")
    logger.debug(f"  disp_mm shape: {disp_mm.shape}")
    logger.debug(f"  Expected dimensions: D={D}, H={H}, W={W}")

    # Ensure anatomical_image matches expected dimensions
    anat = anatomical_image
    if anat.shape != (D, H, W):
        # Resize anatomical image to match grid
        zoom_factors = (D / anat.shape[0], H / anat.shape[1], W / anat.shape[2])
        anat = zoom(anat, zoom_factors, order=1)

    # Resize warped grid if needed
    if warped_grid_np.shape != (D, H, W):
        grid_zoom = (D / warped_grid_np.shape[0], H / warped_grid_np.shape[1], W / warped_grid_np.shape[2])
        warped_grid_np = zoom(warped_grid_np, grid_zoom, order=1)

    # Normalize anatomical image
    anat_min, anat_max = anat.min(), anat.max()
    if anat_max > anat_min:
        anat_norm = (anat - anat_min) / (anat_max - anat_min)
    else:
        anat_norm = anat * 0

    # Normalize grid
    grid_min, grid_max = warped_grid_np.min(), warped_grid_np.max()
    if grid_max > grid_min:
        grid_norm = (warped_grid_np - grid_min) / (grid_max - grid_min)
    else:
        grid_norm = warped_grid_np * 0

    # Debug: Verify normalized values
    logger.debug(f"  anat_norm shape: {anat_norm.shape}, range: [{anat_norm.min():.4f}, {anat_norm.max():.4f}]")
    logger.debug(f"  grid_norm shape: {grid_norm.shape}, range: [{grid_norm.min():.4f}, {grid_norm.max():.4f}]")

    # Direction info: (name, axis_label, indices, dim_size, spacing,
    #                  slice_funcs for: anat, grid, disp_y, disp_x)
    direction_info = [
        ('Axial', 'Z', z_indices, D, spacing[0],
         lambda idx: anat_norm[idx, :, :],
         lambda idx: grid_norm[idx, :, :],
         lambda idx: disp_mm[1, idx, :, :],  # Y displacement
         lambda idx: disp_mm[2, idx, :, :]), # X displacement
        ('Coronal', 'Y', y_indices, H, spacing[1],
         lambda idx: anat_norm[:, idx, :],
         lambda idx: grid_norm[:, idx, :],
         lambda idx: disp_mm[0, :, idx, :],  # Z displacement
         lambda idx: disp_mm[2, :, idx, :]), # X displacement
        ('Sagittal', 'X', x_indices, W, spacing[2],
         lambda idx: anat_norm[:, :, idx],
         lambda idx: grid_norm[:, :, idx],
         lambda idx: disp_mm[0, :, :, idx],  # Z displacement
         lambda idx: disp_mm[1, :, :, idx]), # Y displacement
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Create separate figure for each direction
    for dir_name, axis_name, indices, dim_size, dim_spacing, anat_func, grid_func, disp_v_func, disp_h_func in direction_info:
        n_slices = len(indices)

        # Calculate grid layout
        n_cols = min(n_slices, 5)
        n_rows = (n_slices + n_cols - 1) // n_cols

        # Figure size
        fig_width = n_cols * 3.0 + 1.0
        fig_height = n_rows * 2.8 + 1.2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                                facecolor='black')

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each slice
        for i, slice_idx in enumerate(indices):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.set_facecolor('black')

            # Get slice data with validation
            try:
                anat_slice = anat_func(slice_idx)
                grid_slice = grid_func(slice_idx)
                disp_v = disp_v_func(slice_idx)
                disp_h = disp_h_func(slice_idx)

                # Validate slice shapes and values
                if anat_slice is None or anat_slice.size == 0:
                    logger.warning(f"Empty anatomical slice at index {slice_idx}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           color='white', transform=ax.transAxes)
                    continue

                # Ensure 2D and float dtype
                anat_slice = np.asarray(anat_slice, dtype=np.float32).squeeze()
                grid_slice = np.asarray(grid_slice, dtype=np.float32).squeeze()

                if anat_slice.ndim != 2:
                    logger.warning(f"Unexpected anat_slice shape: {anat_slice.shape}")
                    anat_slice = anat_slice.reshape(anat_slice.shape[-2], anat_slice.shape[-1])

                if grid_slice.ndim != 2:
                    grid_slice = np.zeros_like(anat_slice)

                # Ensure shapes match
                if anat_slice.shape != grid_slice.shape:
                    from scipy.ndimage import zoom as scipy_zoom
                    grid_slice = scipy_zoom(grid_slice,
                                           (anat_slice.shape[0] / max(grid_slice.shape[0], 1),
                                            anat_slice.shape[1] / max(grid_slice.shape[1], 1)),
                                           order=1)

                # Normalize slices to [0, 1] if needed
                anat_min_s, anat_max_s = anat_slice.min(), anat_slice.max()
                if anat_max_s > anat_min_s:
                    anat_slice = (anat_slice - anat_min_s) / (anat_max_s - anat_min_s)
                else:
                    anat_slice = np.zeros_like(anat_slice)

                grid_min_s, grid_max_s = grid_slice.min(), grid_slice.max()
                if grid_max_s > grid_min_s:
                    grid_slice = (grid_slice - grid_min_s) / (grid_max_s - grid_min_s)
                else:
                    grid_slice = np.zeros_like(grid_slice)

            except Exception as e:
                logger.warning(f"Error getting slice data at {slice_idx}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       color='red', transform=ax.transAxes)
                continue

            # Create RGB overlay: anatomy (gray) + grid (green)
            r_channel = anat_slice.copy()
            g_channel = anat_slice + grid_alpha * grid_slice
            b_channel = anat_slice.copy()

            rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
            rgb = np.clip(rgb, 0, 1).astype(np.float32)

            # Validate RGB array
            if not np.isfinite(rgb).all():
                logger.warning(f"Non-finite values in RGB at slice {slice_idx}, replacing with zeros")
                rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

            # Debug: Print RGB channel statistics for first slice
            if i == 0:
                logger.debug(f"  RGB DEBUG for {dir_name} slice {slice_idx}:")
                logger.debug(f"    anat_slice: shape={anat_slice.shape}, min={anat_slice.min():.4f}, max={anat_slice.max():.4f}, mean={anat_slice.mean():.4f}")
                logger.debug(f"    grid_slice: shape={grid_slice.shape}, min={grid_slice.min():.4f}, max={grid_slice.max():.4f}, mean={grid_slice.mean():.4f}")
                logger.debug(f"    r_channel: min={r_channel.min():.4f}, max={r_channel.max():.4f}")
                logger.debug(f"    g_channel: min={g_channel.min():.4f}, max={g_channel.max():.4f}")
                logger.debug(f"    b_channel: min={b_channel.min():.4f}, max={b_channel.max():.4f}")
                logger.debug(f"    rgb: shape={rgb.shape}, dtype={rgb.dtype}")
                logger.debug(f"    rgb[:,:,0] (R): min={rgb[:,:,0].min():.4f}, max={rgb[:,:,0].max():.4f}")
                logger.debug(f"    rgb[:,:,1] (G): min={rgb[:,:,1].min():.4f}, max={rgb[:,:,1].max():.4f}")
                logger.debug(f"    rgb[:,:,2] (B): min={rgb[:,:,2].min():.4f}, max={rgb[:,:,2].max():.4f}")
                # Sample center pixel
                cy, cx = rgb.shape[0]//2, rgb.shape[1]//2
                logger.debug(f"    Center pixel rgb[{cy},{cx}]: R={rgb[cy,cx,0]:.4f}, G={rgb[cy,cx,1]:.4f}, B={rgb[cy,cx,2]:.4f}")
                # Save debug image to verify RGB array
                debug_rgb_path = output_path.parent / f"DEBUG_rgb_{dir_name.lower()}_slice{slice_idx}.png"
                try:
                    from PIL import Image
                    img = Image.fromarray((rgb * 255).astype(np.uint8), mode='RGB')
                    img.save(str(debug_rgb_path))
                    logger.debug(f"    Saved debug RGB: {debug_rgb_path}")
                except ImportError:
                    logger.debug(f"    PIL not available, skipping debug image save")

            # FIX: Use explicit extent with aspect='equal' to prevent matplotlib rendering bug
            # The bug causes RGB corruption when aspect='auto' is combined with quiver overlay
            # on square arrays (axial 192x192) but not rectangular arrays (coronal/sagittal 12x192)
            h_size, w_size = rgb.shape[:2]
            ax.imshow(rgb, aspect='equal', extent=[0, w_size, h_size, 0])

            # Add displacement arrows AT GRID INTERSECTION POINTS
            # Use grid_stride to align arrows with grid lines

            # Arrows at grid intersections (every grid_stride pixels)
            # Use arrow_step as multiplier for sparser arrows (e.g., every 2nd or 3rd grid point)
            arrow_spacing = grid_stride * max(1, arrow_step // grid_stride)

            y_coords = np.arange(arrow_spacing // 2, h_size, arrow_spacing)
            x_coords = np.arange(arrow_spacing // 2, w_size, arrow_spacing)
            X, Y = np.meshgrid(x_coords, y_coords)

            # Sample displacement at arrow positions (in world coords / mm)
            U = disp_h[::arrow_spacing, ::arrow_spacing][:Y.shape[0], :X.shape[1]]
            V = disp_v[::arrow_spacing, ::arrow_spacing][:Y.shape[0], :X.shape[1]]

            # Scale arrows for visibility:
            # arrow length in pixels = displacement_mm * scale_factor
            # Target: max arrow ~20 pixels for visibility without overwhelming
            if has_meaningful_displacement:  # Only draw if there's actual displacement
                arrow_scale = 20.0 / vmax  # Max arrow = 20 pixels

                ax.quiver(X, Y, U * arrow_scale, V * arrow_scale,
                         color='red', alpha=0.7, scale=1, scale_units='xy',
                         width=0.002 * max(h_size, w_size), headwidth=3, headlength=3)

            phys_pos = slice_idx * dim_spacing
            ax.set_title(f'{axis_name}={slice_idx} ({phys_pos:.1f}mm)',
                        fontsize=9, color='white', fontweight='bold')
            ax.axis('off')

        # Hide unused subplot cells
        for i in range(n_slices, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        # Title with stats (show original vmax, not floored value)
        arrow_info = "Red=Arrows" if has_meaningful_displacement else "No arrows (identity)"
        fig.suptitle(f'{dir_name} View ({axis_name}-axis): {stage_name}\n'
                    f'{n_slices}/{dim_size} slices | Spacing: {dim_spacing:.2f}mm | '
                    f'Max Disp: {vmax_original:.2f}mm | Green=Grid, {arrow_info}',
                    fontsize=11, fontweight='bold', color='white', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, wspace=0.05, hspace=0.25)

        # Save with direction suffix
        base_name = output_path.stem
        dir_path = output_path.parent / f"{base_name}_{dir_name.lower()}.png"
        plt.savefig(dir_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

        saved_paths.append(dir_path)
        logger.info(f"Saved {dir_name} grid+arrows view: {dir_path}")

    return saved_paths


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
    ffd_label: str = "FFD",
    show_grid: bool = False,
) -> List[Path]:
    """
    Save alignment progression visualizations

    Creates:
    1. Single-view overlay (axial middle slice)
    2. 3-view overlay (axial, coronal, sagittal)
    3. Standalone deformed grid 3-view PNG (if transforms provided)

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
        show_grid: Whether to show grid deformation overlay on alignment image (default False)

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
        stages.append((f"After Rigid+Affine+{ffd_label}", ffd_result))
        if ffd_transform is not None:
            transforms.append((f"After Rigid+Affine+{ffd_label}", ffd_transform))

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

    # Create standalone deformed grid visualization (if transforms provided)
    if transforms:
        grid_path = output_dir / f"{prefix}_deformed_grid_3views.png"
        result = create_standalone_grid_3views(
            target_image.grid(),
            transforms,
            grid_path
        )
        if result is not None:
            saved_paths.append(grid_path)

    return saved_paths
