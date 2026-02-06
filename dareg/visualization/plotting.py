"""
DAREG Basic Plotting

Side-by-side and overlay visualizations for registration results.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepali.data import Image

from ..utils.logging_config import get_logger

logger = get_logger("plotting")


def plot_side_by_side(
    source: Image,
    target: Image,
    warped_source: Optional[Image] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Registration Comparison",
    slice_idx: Optional[int] = None,
    view: str = "axial",
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Create side-by-side comparison of source, target, and warped source

    Args:
        source: Source image
        target: Target image
        warped_source: Optional warped source image
        output_path: Optional path to save figure
        title: Figure title
        slice_idx: Slice index (default: middle)
        view: View orientation ("axial", "sagittal", "coronal")
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    # Get tensors
    src_np = _to_numpy(source.tensor())
    tgt_np = _to_numpy(target.tensor())
    wrp_np = _to_numpy(warped_source.tensor()) if warped_source else None

    # Get slice
    src_slice = _get_slice(src_np, slice_idx, view)
    tgt_slice = _get_slice(tgt_np, slice_idx, view)
    wrp_slice = _get_slice(wrp_np, slice_idx, view) if wrp_np is not None else None

    # Create figure
    n_cols = 3 if wrp_slice is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    axes[0].imshow(src_slice, cmap='gray')
    axes[0].set_title("Source")
    axes[0].axis('off')

    axes[1].imshow(tgt_slice, cmap='gray')
    axes[1].set_title("Target")
    axes[1].axis('off')

    if wrp_slice is not None:
        axes[2].imshow(wrp_slice, cmap='gray')
        axes[2].set_title("Warped Source")
        axes[2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def plot_overlay(
    image1: Image,
    image2: Image,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Overlay",
    slice_idx: Optional[int] = None,
    view: str = "axial",
    colors: Tuple[str, str] = ("red", "green"),
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Create color overlay of two images

    Args:
        image1: First image (shown in first color)
        image2: Second image (shown in second color)
        output_path: Optional path to save figure
        title: Figure title
        slice_idx: Slice index (default: middle)
        view: View orientation
        colors: Tuple of color names for each image
        alpha: Transparency
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    # Get slices
    img1_np = _to_numpy(image1.tensor())
    img2_np = _to_numpy(image2.tensor())

    img1_slice = _get_slice(img1_np, slice_idx, view)
    img2_slice = _get_slice(img2_np, slice_idx, view)

    # Normalize to [0, 1]
    img1_norm = _normalize(img1_slice)
    img2_norm = _normalize(img2_slice)

    # Create RGB overlay
    h, w = img1_norm.shape
    overlay = np.zeros((h, w, 3))

    # Map colors
    color_map = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "magenta": (1, 0, 1),
        "cyan": (0, 1, 1),
        "yellow": (1, 1, 0),
    }

    c1 = color_map.get(colors[0], (1, 0, 0))
    c2 = color_map.get(colors[1], (0, 1, 0))

    for i in range(3):
        overlay[:, :, i] = img1_norm * c1[i] * alpha + img2_norm * c2[i] * alpha

    overlay = np.clip(overlay, 0, 1)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def save_visualization(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 150,
):
    """Save matplotlib figure"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)

    # Remove batch/channel dimensions
    while arr.ndim > 3:
        arr = arr.squeeze(0)

    return arr


def _get_slice(
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    view: str = "axial",
) -> np.ndarray:
    """Get 2D slice from 3D volume"""
    if slice_idx is None:
        # Use middle slice
        if view == "axial":
            slice_idx = volume.shape[1] // 2
        elif view == "sagittal":
            slice_idx = volume.shape[0] // 2
        else:  # coronal
            slice_idx = volume.shape[2] // 2

    if view == "axial":
        return volume[:, slice_idx, :]
    elif view == "sagittal":
        return volume[slice_idx, :, :]
    else:  # coronal
        return volume[:, :, slice_idx]


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]"""
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val > 0:
        return (arr - min_val) / (max_val - min_val)
    return arr - min_val
