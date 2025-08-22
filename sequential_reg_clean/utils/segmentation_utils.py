#!/usr/bin/env python3
"""
Segmentation Transformation Utilities
Handles transformation of binary segmentation masks using registration transforms
"""

import sys
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')

import torch
import numpy as np
from pathlib import Path
from typing import Optional

import deepali.spatial as spatial
from deepali.data import Image

from .image_loader import ImagePair, save_image_with_header_preservation


def transform_segmentation_mask(segmentation: Image, transform, 
                               target_grid: Optional[object] = None, 
                               transform_name: str = "transform") -> Image:
    """
    Transform segmentation mask using registration transform with nearest neighbor interpolation
    
    Args:
        segmentation: Source segmentation mask (binary/label image)
        transform: Registration transform (rigid, affine, etc.)
        target_grid: Target grid to transform to (if None, uses transform's grid)
        transform_name: Name of transform for logging
        
    Returns:
        Transformed segmentation mask with discrete labels preserved
    """
    
    print(f"🎯 Transforming segmentation mask using {transform_name}...")
    
    # Create image transformer for segmentation (will use nearest neighbor manually)
    if target_grid is not None:
        transformer = spatial.ImageTransformer(transform, target_grid)
    else:
        transformer = spatial.ImageTransformer(transform)
    
    # Apply transformation using spatial transformer
    with torch.no_grad():
        # Apply transformation to the segmentation
        seg_tensor = segmentation.batch().tensor()
        transformed_tensor = transformer(seg_tensor)
        
        # Remove batch dimension if it was added
        if transformed_tensor.dim() == 5:  # [N, C, D, H, W]
            transformed_tensor = transformed_tensor.squeeze(0)  # [C, D, H, W]
        if transformed_tensor.dim() == 4 and transformed_tensor.shape[0] == 1:  # [1, D, H, W]
            transformed_tensor = transformed_tensor.squeeze(0)  # [D, H, W]
    
    # Get the correct grid for the transformed segmentation
    if target_grid is not None:
        result_grid = target_grid
    else:
        result_grid = segmentation.grid()
    
    # Round to nearest integer to ensure discrete labels (important for segmentation)
    rounded_tensor = torch.round(transformed_tensor)
    
    # For segmentation transformation, we'll return the tensor directly
    # and handle the grid/image creation in the calling function
    # Create a simple wrapper to maintain the interface
    class TransformedSegmentation:
        def __init__(self, tensor, reference_image):
            self._tensor = tensor
            self._reference = reference_image
        
        def tensor(self):
            return self._tensor
            
        def device(self):
            return self._reference.device
    
    transformed_seg = TransformedSegmentation(rounded_tensor, segmentation)
    
    # Log transformation statistics
    original_labels = torch.unique(segmentation.tensor()).cpu().numpy()
    transformed_labels = torch.unique(transformed_seg.tensor()).cpu().numpy()
    
    print(f"   📊 Original labels: {original_labels.tolist()}")
    print(f"   📊 Transformed labels: {transformed_labels.tolist()}")
    print(f"   📐 Shape: {segmentation.tensor().shape} → {transformed_seg.tensor().shape}")
    
    return transformed_seg


def save_transformed_segmentations(image_pair: ImagePair, rigid_transform,
                                 affine_transform: Optional[object],
                                 output_dir: Path, stage_limit: str = "affine"):
    """
    Save transformed segmentation masks for all registration stages
    
    Args:
        image_pair: Image pair containing source segmentation
        rigid_transform: Rigid registration transform
        affine_transform: Affine registration transform (optional)
        output_dir: Output directory for results
        stage_limit: Registration stage limit
    """
    
    if image_pair.source_seg_original is None:
        print("ℹ️  No source segmentation to transform")
        return
    
    print("\\n🎯 TRANSFORMING SEGMENTATION MASKS")
    print("=" * 50)
    
    # Create segmentation results directory
    seg_dir = output_dir / "segmentation_results"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original segmentation (reference)
    print("💾 Saving original segmentation references...")
    save_image_with_header_preservation(
        image_pair.source_seg_original.tensor(),
        image_pair.source_seg_original,
        seg_dir / "source_seg_original.nii.gz",
        "Original source segmentation"
    )
    
    save_image_with_header_preservation(
        image_pair.source_seg_common.tensor(),
        image_pair.source_seg_common,
        seg_dir / "source_seg_common.nii.gz", 
        "Source segmentation in common coordinates"
    )
    
    # Transform with rigid registration
    print("🔧 Applying rigid transformation to segmentation...")
    
    # Transform to target space (FINAL RESULT - keep original source resolution)
    print("   📏 Moving segmentation to target space (keep source resolution)...")
    seg_rigid_target = transform_segmentation_mask(
        image_pair.source_seg_original, 
        rigid_transform, 
        image_pair.source_original_grid,  # Use source grid to preserve source resolution!
        "rigid_to_target"
    )
    
    save_image_with_header_preservation(
        seg_rigid_target.tensor(),
        image_pair.source_seg_original,  # Use source segmentation as reference for header/resolution
        seg_dir / "source_seg_moved_to_target_rigid.nii.gz",
        "Source segmentation moved to target space (rigid, source resolution)"
    )
    
    # Transform to source space (bidirectional)
    inverse_rigid = rigid_transform.inverse()
    seg_rigid_source = transform_segmentation_mask(
        image_pair.source_seg_common,  # Use common space version
        inverse_rigid,
        image_pair.source_original_grid,
        "inverse_rigid"
    )
    
    save_image_with_header_preservation(
        seg_rigid_source.tensor(),
        image_pair.source_original,
        seg_dir / "source_seg_rigid_in_source_space.nii.gz", 
        "Source segmentation after rigid (back to source space)"
    )
    
    # Transform with affine registration (if available)
    if stage_limit in ["affine", "svffd"] and affine_transform is not None:
        print("🔧 Applying affine transformation to segmentation...")
        
        # Transform to target space (FINAL RESULT - keep original source resolution)
        print("   📏 Moving segmentation to target space (keep source resolution)...")
        seg_affine_target = transform_segmentation_mask(
            image_pair.source_seg_original,
            affine_transform,
            image_pair.source_original_grid,  # Use source grid to preserve source resolution!
            "affine_to_target"
        )
        
        save_image_with_header_preservation(
            seg_affine_target.tensor(),
            image_pair.source_seg_original,  # Use source segmentation as reference for header/resolution
            seg_dir / "source_seg_moved_to_target_affine.nii.gz",
            "Source segmentation moved to target space (affine, source resolution)"
        )
        
        # Transform to source space (bidirectional)
        inverse_affine = affine_transform.inverse()
        seg_affine_source = transform_segmentation_mask(
            image_pair.source_seg_common,
            inverse_affine,
            image_pair.source_original_grid,
            "inverse_affine"
        )
        
        save_image_with_header_preservation(
            seg_affine_source.tensor(),
            image_pair.source_original,
            seg_dir / "source_seg_affine_in_source_space.nii.gz",
            "Source segmentation after affine (back to source space)"
        )
    
    # Also save final results in main results directory (like image registration)
    final_results_dir = output_dir / "final_results"
    final_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Copying final segmentation results to main results directory...")
    
    # Copy rigid result
    if (seg_dir / "source_seg_moved_to_target_rigid.nii.gz").exists():
        import shutil
        shutil.copy2(
            seg_dir / "source_seg_moved_to_target_rigid.nii.gz",
            final_results_dir / "source_seg_moved_to_target_rigid.nii.gz"
        )
        print("   ✅ source_seg_moved_to_target_rigid.nii.gz")
    
    # Copy affine result (if available)
    if stage_limit in ["affine", "svffd"] and affine_transform is not None:
        if (seg_dir / "source_seg_moved_to_target_affine.nii.gz").exists():
            shutil.copy2(
                seg_dir / "source_seg_moved_to_target_affine.nii.gz", 
                final_results_dir / "source_seg_moved_to_target_affine.nii.gz"
            )
            print("   ✅ source_seg_moved_to_target_affine.nii.gz")
    
    print(f"✅ SEGMENTATION TRANSFORMATION COMPLETE")
    print(f"   📂 Detailed results: {seg_dir}")
    print(f"   📂 Final results: {final_results_dir}")


def create_segmentation_overlay_visualization(image_pair: ImagePair, 
                                           rigid_transform,
                                           affine_transform: Optional[object],
                                           output_dir: Path, stage_limit: str = "affine"):
    """
    Create visualization showing segmentation overlays before and after registration
    
    Args:
        image_pair: Image pair with segmentation
        rigid_transform: Rigid transformation
        affine_transform: Affine transformation (optional)
        output_dir: Output directory
        stage_limit: Registration stage limit
    """
    
    if image_pair.source_seg_original is None:
        print("ℹ️  No segmentation for overlay visualization")
        return
    
    print("🎨 Creating segmentation overlay visualizations...")
    
    import matplotlib.pyplot as plt
    
    # Get anatomical images
    source_common = image_pair.source_normalized.tensor().squeeze().detach().cpu().numpy()
    target_common = image_pair.target_normalized.tensor().squeeze().detach().cpu().numpy()
    seg_common = image_pair.source_seg_common.tensor().squeeze().detach().cpu().numpy()
    
    # Transform segmentation with rigid
    seg_rigid = transform_segmentation_mask(
        image_pair.source_seg_common, rigid_transform, transform_name="rigid_for_viz"
    )
    seg_rigid_np = seg_rigid.tensor().squeeze().detach().cpu().numpy()
    
    # Transform segmentation with affine (if available)
    seg_affine_np = None
    if stage_limit in ["affine", "svffd"] and affine_transform is not None:
        seg_affine = transform_segmentation_mask(
            image_pair.source_seg_common, affine_transform, transform_name="affine_for_viz"
        )
        seg_affine_np = seg_affine.tensor().squeeze().detach().cpu().numpy()
    
    # Create visualization
    num_cols = 4 if seg_affine_np is not None else 3
    fig, axes = plt.subplots(3, num_cols, figsize=(6*num_cols, 18))
    fig.suptitle('Segmentation Overlay Visualization - Registration Progress', fontsize=16, fontweight='bold')
    
    # Get middle slices
    d_mid = source_common.shape[0] // 2
    h_mid = source_common.shape[1] // 2  
    w_mid = source_common.shape[2] // 2
    
    view_names = ['Sagittal', 'Axial', 'Coronal']
    slices = [
        (lambda x: x[d_mid, :, :], lambda x: x[d_mid, :, :]),  # Sagittal
        (lambda x: x[:, h_mid, :], lambda x: x[:, h_mid, :]),  # Axial
        (lambda x: x[:, :, w_mid], lambda x: x[:, :, w_mid])   # Coronal
    ]
    
    for row, (view_name, (img_slice, seg_slice)) in enumerate(zip(view_names, slices)):
        
        # Column 1: Original source + segmentation
        axes[row, 0].imshow(img_slice(source_common), cmap='gray', alpha=0.8)
        axes[row, 0].contour(seg_slice(seg_common), levels=[0.5], colors='red', linewidths=2, alpha=0.8)
        axes[row, 0].set_title(f'Original Source + Segmentation ({view_name})')
        axes[row, 0].axis('off')
        
        # Column 2: Target (reference)
        axes[row, 1].imshow(img_slice(target_common), cmap='gray', alpha=0.8)
        axes[row, 1].set_title(f'Target Reference ({view_name})')
        axes[row, 1].axis('off')
        
        # Column 3: After rigid registration
        axes[row, 2].imshow(img_slice(target_common), cmap='gray', alpha=0.8)
        axes[row, 2].contour(seg_slice(seg_rigid_np), levels=[0.5], colors='green', linewidths=2, alpha=0.8)
        axes[row, 2].set_title(f'Rigid: Segmentation on Target ({view_name})')
        axes[row, 2].axis('off')
        
        # Column 4: After affine registration (if available)
        if seg_affine_np is not None:
            axes[row, 3].imshow(img_slice(target_common), cmap='gray', alpha=0.8)
            axes[row, 3].contour(seg_slice(seg_affine_np), levels=[0.5], colors='blue', linewidths=2, alpha=0.8)
            axes[row, 3].set_title(f'Affine: Segmentation on Target ({view_name})')
            axes[row, 3].axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Original Segmentation'),
        Line2D([0], [0], color='green', lw=2, label='After Rigid Registration'),
    ]
    if seg_affine_np is not None:
        legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='After Affine Registration'))
    
    fig.legend(legend_elements, [el.get_label() for el in legend_elements], 
              loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save visualization
    overlay_path = output_dir / "debug_analysis" / "segmentation_overlay_progression.png"
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Segmentation overlay visualization: {overlay_path.name}")