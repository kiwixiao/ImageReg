#!/usr/bin/env python3
"""
Segmentation Progression Viewer

Multi-slice visualization for segmentation propagation through frames.
Creates separate visualizations for sagittal, axial, and coronal views.

Each view shows:
- 8 slices per row (spread from mid-slice to both sides)
- One row per frame
- Clear visualization of segmentation changes over time
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import torch


def get_spread_slices(dim_size: int, num_slices: int = 8) -> List[int]:
    """
    Get slice indices spread from mid-slice to both sides.

    Args:
        dim_size: Size of the dimension
        num_slices: Number of slices to extract (default 8)

    Returns:
        List of slice indices, centered around mid-slice
    """
    if dim_size < num_slices:
        return list(range(dim_size))

    mid = dim_size // 2

    # Calculate step size to spread slices evenly
    # We want slices from approximately 10% to 90% of the volume
    margin = max(1, int(dim_size * 0.1))
    valid_range = dim_size - 2 * margin
    step = max(1, valid_range // (num_slices - 1))

    # Generate symmetric offsets from center
    half_slices = num_slices // 2
    slices = []

    for i in range(-half_slices, num_slices - half_slices):
        idx = mid + i * step
        idx = max(0, min(dim_size - 1, idx))
        slices.append(idx)

    # Remove duplicates while preserving order
    seen = set()
    unique_slices = []
    for s in slices:
        if s not in seen:
            seen.add(s)
            unique_slices.append(s)

    return unique_slices


def create_segmentation_progression_view(
    segmentation_sequence: List,
    output_dir: Path,
    view: str = "sagittal",
    num_slices: int = 8,
    colormap: str = "viridis",
    title_prefix: str = "Segmentation Progression",
    spacing: Optional[Tuple[float, float, float]] = None
) -> Path:
    """
    Create multi-slice segmentation progression for a single view.

    Args:
        segmentation_sequence: List of segmentation images (deepali Image or tensors)
        output_dir: Output directory for the PNG
        view: One of "sagittal", "axial", "coronal"
        num_slices: Number of slices per row (default 8)
        colormap: Matplotlib colormap (default "viridis")
        title_prefix: Title prefix for the figure
        spacing: Optional (D, H, W) spacing in mm for correct aspect ratio.
                 If None, assumes isotropic (aspect=1).

    Returns:
        Path to saved PNG file
    """
    if not segmentation_sequence:
        return None

    num_frames = len(segmentation_sequence)

    # Determine axis for each view
    # For 3D array [D, H, W]: D=depth/slice, H=height, W=width
    # Sagittal: slice along W (width) -> shows D x H
    # Axial: slice along D (depth) -> shows H x W
    # Coronal: slice along H (height) -> shows D x W
    view_axis = {"sagittal": 2, "axial": 0, "coronal": 1}
    axis = view_axis.get(view.lower(), 0)

    # Calculate aspect ratio for correct display of anisotropic images
    # For array [D, H, W] with spacing [sp_D, sp_H, sp_W]:
    #   Sagittal (axis=2): shows [D, H] -> aspect = sp_D / sp_H
    #   Axial (axis=0): shows [H, W] -> aspect = sp_H / sp_W
    #   Coronal (axis=1): shows [D, W] -> aspect = sp_D / sp_W
    if spacing is not None:
        sp_D, sp_H, sp_W = spacing
        if axis == 2:  # Sagittal
            aspect_ratio = sp_D / sp_H
        elif axis == 0:  # Axial
            aspect_ratio = sp_H / sp_W
        else:  # Coronal
            aspect_ratio = sp_D / sp_W
    else:
        aspect_ratio = 1.0  # Isotropic assumption

    # Get first segmentation to determine dimensions
    first_seg = segmentation_sequence[0]
    if hasattr(first_seg, 'tensor'):
        seg_np = first_seg.tensor().squeeze().cpu().numpy()
    elif isinstance(first_seg, torch.Tensor):
        seg_np = first_seg.squeeze().cpu().numpy()
    else:
        seg_np = np.array(first_seg).squeeze()

    dim_size = seg_np.shape[axis]
    slice_indices = get_spread_slices(dim_size, num_slices)
    actual_num_slices = len(slice_indices)

    # Create figure with one row per frame
    fig_width = 2 * actual_num_slices
    fig_height = 2 * num_frames
    fig, axes = plt.subplots(num_frames, actual_num_slices,
                             figsize=(fig_width, fig_height),
                             squeeze=False)

    # Process each frame
    for frame_idx, seg in enumerate(segmentation_sequence):
        if hasattr(seg, 'tensor'):
            seg_np = seg.tensor().squeeze().cpu().numpy()
        elif isinstance(seg, torch.Tensor):
            seg_np = seg.squeeze().cpu().numpy()
        else:
            seg_np = np.array(seg).squeeze()

        for slice_idx, slice_pos in enumerate(slice_indices):
            ax = axes[frame_idx, slice_idx]

            # Extract slice based on view
            if axis == 0:  # Axial
                slice_2d = seg_np[slice_pos, :, :]
            elif axis == 1:  # Coronal
                slice_2d = seg_np[:, slice_pos, :]
            else:  # Sagittal (axis == 2)
                slice_2d = seg_np[:, :, slice_pos]

            ax.imshow(slice_2d, cmap=colormap, interpolation='nearest', aspect=aspect_ratio)
            ax.axis('off')

            # Add slice number label on first row
            if frame_idx == 0:
                ax.set_title(f's={slice_pos}', fontsize=8)

        # Add frame label on first column
        axes[frame_idx, 0].set_ylabel(f'Frame {frame_idx}', fontsize=10, rotation=0,
                                       ha='right', va='center')

    # Set overall title
    view_names = {"sagittal": "Sagittal", "axial": "Axial", "coronal": "Coronal"}
    view_name = view_names.get(view.lower(), view.capitalize())
    fig.suptitle(f'{title_prefix} - {view_name} View', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"segmentation_progression_{view.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_all_segmentation_progression_views(
    segmentation_sequence: List,
    output_dir: Path,
    num_slices: int = 8,
    colormap: str = "viridis",
    title_prefix: str = "Segmentation Progression",
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Create segmentation progression visualizations for all three views.

    Args:
        segmentation_sequence: List of segmentation images
        output_dir: Output directory for the PNGs
        num_slices: Number of slices per row (default 8)
        colormap: Matplotlib colormap (default "viridis")
        title_prefix: Title prefix for the figures
        spacing: Optional (D, H, W) spacing in mm for correct aspect ratio

    Returns:
        Tuple of (sagittal_path, axial_path, coronal_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for view in ["sagittal", "axial", "coronal"]:
        try:
            path = create_segmentation_progression_view(
                segmentation_sequence,
                output_dir,
                view=view,
                num_slices=num_slices,
                colormap=colormap,
                title_prefix=title_prefix,
                spacing=spacing
            )
            paths.append(path)
            if path:
                print(f"   Created {view} progression: {path.name}")
        except Exception as e:
            print(f"   Warning: Failed to create {view} progression: {e}")
            paths.append(None)

    return tuple(paths)


def create_segmentation_overlay_progression(
    image_sequence: List,
    segmentation_sequence: List,
    output_dir: Path,
    view: str = "axial",
    num_slices: int = 8,
    seg_alpha: float = 0.3,
    title_prefix: str = "Segmentation Overlay",
    spacing: Optional[Tuple[float, float, float]] = None
) -> Optional[Path]:
    """
    Create segmentation overlay on anatomical images for progression visualization.

    This shows the segmentation overlaid on the anatomical image for each frame,
    making it easier to assess registration quality.

    Args:
        image_sequence: List of anatomical images
        segmentation_sequence: List of segmentation masks
        output_dir: Output directory
        view: View orientation
        num_slices: Number of slices per row
        seg_alpha: Transparency for segmentation overlay
        title_prefix: Title prefix
        spacing: Optional (D, H, W) spacing in mm for correct aspect ratio

    Returns:
        Path to saved PNG
    """
    if not image_sequence or not segmentation_sequence:
        return None

    if len(image_sequence) != len(segmentation_sequence):
        print(f"   Warning: Image count ({len(image_sequence)}) != Seg count ({len(segmentation_sequence)})")
        return None

    num_frames = len(image_sequence)

    # Determine axis
    view_axis = {"sagittal": 2, "axial": 0, "coronal": 1}
    axis = view_axis.get(view.lower(), 0)

    # Calculate aspect ratio for correct display of anisotropic images
    if spacing is not None:
        sp_D, sp_H, sp_W = spacing
        if axis == 2:  # Sagittal
            aspect_ratio = sp_D / sp_H
        elif axis == 0:  # Axial
            aspect_ratio = sp_H / sp_W
        else:  # Coronal
            aspect_ratio = sp_D / sp_W
    else:
        aspect_ratio = 1.0  # Isotropic assumption

    # Get dimensions from first image
    first_img = image_sequence[0]
    if hasattr(first_img, 'tensor'):
        img_np = first_img.tensor().squeeze().cpu().numpy()
    elif isinstance(first_img, torch.Tensor):
        img_np = first_img.squeeze().cpu().numpy()
    else:
        img_np = np.array(first_img).squeeze()

    dim_size = img_np.shape[axis]
    slice_indices = get_spread_slices(dim_size, num_slices)
    actual_num_slices = len(slice_indices)

    # Create figure
    fig_width = 2 * actual_num_slices
    fig_height = 2 * num_frames
    fig, axes = plt.subplots(num_frames, actual_num_slices,
                             figsize=(fig_width, fig_height),
                             squeeze=False)

    # Process each frame
    for frame_idx in range(num_frames):
        # Get image
        img = image_sequence[frame_idx]
        if hasattr(img, 'tensor'):
            img_np = img.tensor().squeeze().cpu().numpy()
        elif isinstance(img, torch.Tensor):
            img_np = img.squeeze().cpu().numpy()
        else:
            img_np = np.array(img).squeeze()

        # Get segmentation
        seg = segmentation_sequence[frame_idx]
        if hasattr(seg, 'tensor'):
            seg_np = seg.tensor().squeeze().cpu().numpy()
        elif isinstance(seg, torch.Tensor):
            seg_np = seg.squeeze().cpu().numpy()
        else:
            seg_np = np.array(seg).squeeze()

        for slice_idx, slice_pos in enumerate(slice_indices):
            ax = axes[frame_idx, slice_idx]

            # Extract slices
            if axis == 0:
                img_slice = img_np[slice_pos, :, :]
                seg_slice = seg_np[slice_pos, :, :]
            elif axis == 1:
                img_slice = img_np[:, slice_pos, :]
                seg_slice = seg_np[:, slice_pos, :]
            else:
                img_slice = img_np[:, :, slice_pos]
                seg_slice = seg_np[:, :, slice_pos]

            # Normalize image for display
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            # Create RGB overlay
            rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

            # Add segmentation overlay in color (e.g., green)
            seg_mask = seg_slice > 0.5
            rgb[seg_mask, 1] = np.clip(rgb[seg_mask, 1] + seg_alpha, 0, 1)

            ax.imshow(rgb, interpolation='nearest', aspect=aspect_ratio)
            ax.axis('off')

            if frame_idx == 0:
                ax.set_title(f's={slice_pos}', fontsize=8)

        axes[frame_idx, 0].set_ylabel(f'Frame {frame_idx}', fontsize=10, rotation=0,
                                       ha='right', va='center')

    view_names = {"sagittal": "Sagittal", "axial": "Axial", "coronal": "Coronal"}
    view_name = view_names.get(view.lower(), view.capitalize())
    fig.suptitle(f'{title_prefix} - {view_name} View', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f"segmentation_overlay_{view.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path
