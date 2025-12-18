"""
DAREG Deformation Visualization

Visualize deformation fields, grid warping, and flow fields.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepali.data import Image
from deepali.core import Grid
from deepali.core import functional as U
import deepali.spatial as spatial

from ..utils.logging_config import get_logger

logger = get_logger("deformation")


def plot_grid_deformation(
    transform,
    grid: Grid,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Grid Deformation",
    slice_idx: Optional[int] = None,
    grid_spacing: int = 16,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot grid deformation showing original and warped grids

    Args:
        transform: Deformation transform
        grid: Image grid
        output_path: Optional path to save figure
        title: Figure title
        slice_idx: Slice index (default: middle)
        grid_spacing: Spacing between grid lines
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    # Create grid image using deepali's utility
    try:
        grid_image = U.grid_image(grid, num=1, stride=grid_spacing, inverted=True)
    except Exception:
        # Fallback: create simple grid manually
        shape = tuple(grid.shape)
        grid_image = _create_grid_image(shape, grid_spacing)
        grid_image = grid_image.unsqueeze(0)  # Add batch dim

    # Warp grid
    transformer = spatial.ImageTransformer(transform, grid, padding="border")
    warped_grid = transformer(grid_image)

    # Get middle slice
    if slice_idx is None:
        slice_idx = grid.shape[0] // 2

    orig_slice = grid_image.squeeze().cpu().numpy()[slice_idx]
    warp_slice = warped_grid.squeeze().cpu().detach().numpy()[slice_idx]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title("Original Grid")
    axes[0].axis('off')

    axes[1].imshow(warp_slice, cmap='gray')
    axes[1].set_title("Warped Grid")
    axes[1].axis('off')

    # Difference
    diff = np.abs(warp_slice - orig_slice)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Deformation Magnitude")
    axes[2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def plot_displacement_field(
    displacement: torch.Tensor,
    background: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Displacement Field",
    slice_idx: Optional[int] = None,
    subsample: int = 4,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot displacement/flow field as quiver plot

    Args:
        displacement: Displacement field [3, D, H, W] or [D, H, W, 3]
        background: Optional background image for overlay
        output_path: Optional path to save figure
        title: Figure title
        slice_idx: Slice index (default: middle)
        subsample: Subsample factor for arrows
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    # Convert to numpy [D, H, W, 3]
    disp_np = _to_numpy_flow(displacement)

    if slice_idx is None:
        slice_idx = disp_np.shape[0] // 2

    # Get slice components
    dx = disp_np[slice_idx, :, :, 0]
    dy = disp_np[slice_idx, :, :, 1]
    magnitude = np.sqrt(dx**2 + dy**2)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Magnitude heatmap
    im0 = axes[0].imshow(magnitude, cmap='hot')
    axes[0].set_title("Displacement Magnitude")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Quiver plot
    h, w = dx.shape
    y, x = np.mgrid[0:h:subsample, 0:w:subsample]
    dx_sub = dx[::subsample, ::subsample]
    dy_sub = dy[::subsample, ::subsample]

    if background is not None:
        bg_slice = background[slice_idx] if background.ndim == 3 else background
        axes[1].imshow(bg_slice, cmap='gray', alpha=0.7)

    axes[1].quiver(x, y, dx_sub, dy_sub, color='blue', scale=20)
    axes[1].set_title("Flow Vectors")
    axes[1].axis('off')

    # HSV representation
    hsv = _flow_to_hsv(dx, dy)
    axes[2].imshow(hsv)
    axes[2].set_title("Flow Direction (HSV)")
    axes[2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def plot_jacobian_map(
    displacement: torch.Tensor,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Jacobian Determinant",
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot Jacobian determinant map showing local volume changes

    Values < 0: Folding (topology violation)
    Values < 1: Compression
    Values > 1: Expansion

    Args:
        displacement: Displacement field
        output_path: Optional path to save figure
        title: Figure title
        slice_idx: Slice index
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    from deepali.losses import functional as L

    # Compute Jacobian determinant
    disp = _ensure_5d_flow(displacement)
    jac_det = L.jacobian_det(disp, add_identity=True)
    jac_np = jac_det.squeeze().cpu().detach().numpy()

    if slice_idx is None:
        slice_idx = jac_np.shape[0] // 2

    jac_slice = jac_np[slice_idx]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use diverging colormap centered at 1.0
    vmin = min(0, jac_slice.min())
    vmax = max(2, jac_slice.max())

    im = ax.imshow(jac_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Jacobian Determinant")

    # Add statistics
    folding_pct = (jac_np < 0).mean() * 100
    ax.text(0.02, 0.98, f"Folding: {folding_pct:.2f}%\nMin: {jac_np.min():.3f}\nMax: {jac_np.max():.3f}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def _create_grid_image(shape: tuple, spacing: int) -> torch.Tensor:
    """Create simple grid image"""
    D, H, W = shape
    grid = torch.zeros(D, H, W)

    # Add grid lines
    for d in range(0, D, spacing):
        grid[d, :, :] = 1.0
    for h in range(0, H, spacing):
        grid[:, h, :] = 1.0
    for w in range(0, W, spacing):
        grid[:, :, w] = 1.0

    return grid


def _to_numpy_flow(flow: torch.Tensor) -> np.ndarray:
    """Convert flow tensor to numpy [D, H, W, 3]"""
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    # Handle dimensions
    if flow.ndim == 5:
        flow = flow.squeeze(0)  # Remove batch
    if flow.shape[0] == 3:
        # [3, D, H, W] -> [D, H, W, 3]
        flow = np.transpose(flow, (1, 2, 3, 0))

    return flow


def _ensure_5d_flow(flow: torch.Tensor) -> torch.Tensor:
    """Ensure flow is 5D [N, C, D, H, W]"""
    if flow.dim() == 4:
        if flow.shape[-1] == 3:
            # [D, H, W, 3] -> [3, D, H, W]
            flow = flow.permute(3, 0, 1, 2)
        flow = flow.unsqueeze(0)
    return flow


def _flow_to_hsv(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Convert flow to HSV color representation"""
    import colorsys

    h, w = dx.shape
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Normalize
    mag_norm = magnitude / (magnitude.max() + 1e-8)
    angle_norm = (angle + np.pi) / (2 * np.pi)  # [0, 1]

    # Create HSV image
    hsv = np.zeros((h, w, 3))
    hsv[:, :, 0] = angle_norm  # Hue = direction
    hsv[:, :, 1] = 1.0  # Saturation = 1
    hsv[:, :, 2] = mag_norm  # Value = magnitude

    # Convert to RGB
    rgb = np.zeros_like(hsv)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])

    return rgb
