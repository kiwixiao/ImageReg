#!/usr/bin/env python3
"""
Image Loading and Common Coordinate System Utilities
Handles loading images and setting up common world coordinate space
"""

import sys
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')

import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass

# Fix version check for deepali
import importlib.metadata
original_version = importlib.metadata.version
def mock_version(package_name):
    if package_name == "hf-deepali":
        return "0.1.0"
    return original_version(package_name)
importlib.metadata.version = mock_version

from deepali.data import Image
from deepali.core import Grid


@dataclass
class ImagePair:
    """Container for source and target images in different coordinate systems"""
    # Original images (in their native coordinate systems)
    source_original: Image
    target_original: Image
    
    # Common coordinate space (for registration)
    source_common: Image
    target_common: Image
    
    # Normalized for registration optimization
    source_normalized: Image
    target_normalized: Image
    
    # Metadata
    common_grid: Grid
    source_original_grid: Grid
    target_original_grid: Grid
    
    # Segmentation masks (optional) - must come after required fields in dataclass
    source_seg_original: Image = None
    source_seg_common: Image = None


def load_image_pair(source_path: str, target_path: str, device: str = "cpu") -> ImagePair:
    """
    Load source and target images and create common coordinate space
    Also automatically loads source segmentation mask if available
    
    Args:
        source_path: Path to source/moving image
        target_path: Path to target/fixed image  
        device: Device for computation
        
    Returns:
        ImagePair with all coordinate systems set up, including segmentation if found
    """
    
    print("🏥 LOADING IMAGE PAIR FOR SEQUENTIAL REGISTRATION")
    print("=" * 60)
    
    device = torch.device(device)
    
    # Load original images in their native coordinate systems
    print(f"📂 Loading source: {Path(source_path).name}")
    print(f"📂 Loading target: {Path(target_path).name}")
    
    source_original = Image.read(source_path, device=device)
    target_original = Image.read(target_path, device=device)
    
    print(f"✅ Source shape: {source_original.shape}, spacing: {source_original.spacing().tolist()}")
    print(f"✅ Target shape: {target_original.shape}, spacing: {target_original.spacing().tolist()}")
    
    # Store original grids for later coordinate system preservation
    source_original_grid = source_original.grid()
    target_original_grid = target_original.grid()
    
    # Create common world coordinate space (use target as reference - medical convention)
    print(f"🌍 Setting up common world coordinate space...")
    common_grid = target_original.grid()
    print(f"✅ Common grid shape: {common_grid.shape}, spacing: {common_grid.spacing().tolist()}")
    
    # Resample source to common coordinate space
    print(f"🔄 Resampling source to common coordinate space...")
    source_common = source_original.sample(common_grid)
    target_common = target_original  # Already in common space
    
    # Normalize images for registration optimization (zero mean, unit variance)
    print(f"🔄 Normalizing images for registration...")
    source_normalized = normalize_medical_image(source_common)
    target_normalized = normalize_medical_image(target_common)
    
    print(f"✅ Normalization complete:")
    print(f"   Source range: [{source_normalized.tensor().min():.3f}, {source_normalized.tensor().max():.3f}]")
    print(f"   Target range: [{target_normalized.tensor().min():.3f}, {target_normalized.tensor().max():.3f}]")
    
    # =========================================================================
    # SEGMENTATION MASK LOADING (OPTIONAL)
    # =========================================================================
    source_seg_original = None
    source_seg_common = None
    
    # Try to find source segmentation mask
    source_dir = Path(source_path).parent
    source_name = Path(source_path).stem.replace('.nii', '')  # Remove .nii extension
    
    # Common segmentation naming patterns
    seg_patterns = [
        f"{source_name}_seg_mask.nii.gz",
        f"{source_name}_seg.nii.gz", 
        f"{source_name}_mask.nii.gz",
        "source_seg_mask.nii.gz",
        "source_seg.nii.gz"
    ]
    
    for pattern in seg_patterns:
        seg_path = source_dir / pattern
        if seg_path.exists():
            print(f"🎯 Loading source segmentation: {seg_path.name}")
            
            # Load segmentation mask
            seg_sitk = sitk.ReadImage(str(seg_path))
            
            # Convert to deepali Image (original resolution)
            seg_data = sitk.GetArrayFromImage(seg_sitk)  # Keep original ITK array format (Z, Y, X)
            seg_tensor = torch.from_numpy(seg_data.astype(np.float32)).to(device)
            
            if seg_tensor.dim() == 3:
                seg_tensor = seg_tensor.unsqueeze(0)  # Add channel dimension: [C, Z, Y, X]
            
            # Debug: Check tensor and grid shapes
            print(f"   Segmentation tensor shape: {seg_tensor.shape}")
            print(f"   Source original grid shape: {source_original_grid.shape}")
            
            # Create segmentation image with proper grid
            source_seg_original = Image(seg_tensor, source_original_grid, device=device)
            
            # Resample segmentation to common coordinate space using NEAREST NEIGHBOR
            print(f"🔄 Resampling segmentation to common coordinate space...")
            source_seg_common = source_seg_original.sample(common_grid, mode='nearest')
            
            print(f"✅ Segmentation loaded and resampled")
            print(f"   Original shape: {source_seg_original.tensor().shape}")
            print(f"   Common shape: {source_seg_common.tensor().shape}")
            print(f"   Unique labels: {torch.unique(source_seg_common.tensor()).tolist()}")
            break
    
    if source_seg_original is None:
        print(f"ℹ️  No source segmentation found (searched: {seg_patterns})")
    
    # Create ImagePair container
    image_pair = ImagePair(
        source_original=source_original,
        target_original=target_original,
        source_common=source_common,
        target_common=target_common,
        source_normalized=source_normalized,
        target_normalized=target_normalized,
        source_seg_original=source_seg_original,
        source_seg_common=source_seg_common,
        common_grid=common_grid,
        source_original_grid=source_original_grid,
        target_original_grid=target_original_grid
    )
    
    print(f"✅ IMAGE PAIR SETUP COMPLETE")
    print(f"   📐 Common coordinate space established")
    print(f"   📐 Original coordinate systems preserved")
    print(f"   📐 Images normalized for registration")
    
    return image_pair


def normalize_medical_image(image: Image) -> Image:
    """
    Normalize medical image to zero mean, unit variance
    
    Args:
        image: Input medical image
        
    Returns:
        Normalized image
    """
    tensor = image.tensor()
    
    # Compute statistics
    mean_val = tensor.mean()
    std_val = tensor.std()
    
    # Normalize (avoid division by zero)
    if std_val > 1e-6:
        normalized_tensor = (tensor - mean_val) / std_val
    else:
        normalized_tensor = tensor - mean_val
    
    # Create new image with normalized data
    normalized_image = Image(normalized_tensor, image.grid())
    
    return normalized_image


def save_image_with_header_preservation(tensor: torch.Tensor, reference_image: Image, 
                                      output_path: Path, description: str = ""):
    """
    Save tensor as medical image with full header preservation using SimpleITK
    
    Args:
        tensor: Image tensor to save
        reference_image: Reference image for header information
        output_path: Output file path (.nii.gz)
        description: Description for logging
    """
    
    print(f"💾 Saving: {description} -> {output_path.name}")
    
    try:
        # Ensure tensor is on CPU and properly shaped
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Remove batch dimensions
        while tensor.dim() > 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D: {tensor.shape}")
        
        # Convert to numpy
        numpy_array = tensor.detach().numpy()
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(numpy_array)
        
        # Set spacing
        spacing = reference_image.spacing()
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy().tolist()
        sitk_image.SetSpacing(spacing)
        
        # Set origin
        try:
            origin = reference_image.grid().origin()
            if isinstance(origin, torch.Tensor):
                origin = origin.cpu().numpy().tolist()
            sitk_image.SetOrigin(origin)
        except:
            sitk_image.SetOrigin([0.0, 0.0, 0.0])
        
        # Set direction matrix
        try:
            direction = reference_image.grid().direction()
            if isinstance(direction, torch.Tensor):
                direction = direction.cpu().numpy().flatten().tolist()
            sitk_image.SetDirection(direction)
        except:
            sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write image
        sitk.WriteImage(sitk_image, str(output_path))
        
        print(f"✅ Saved: {output_path.name} - Size: {sitk_image.GetSize()}")
        
    except Exception as e:
        print(f"❌ Failed to save {description}: {e}")
        raise


def create_side_by_side_comparison(source_common: Image, target_common: Image, 
                                 output_dir: Path, stage_name: str = ""):
    """
    Create side-by-side visualization of images in common coordinate space
    
    Args:
        source_common: Source image in common coordinates
        target_common: Target image in common coordinates  
        output_dir: Output directory
        stage_name: Stage name for filename
    """
    
    print(f"🎨 Creating side-by-side comparison for {stage_name}...")
    
    import matplotlib.pyplot as plt
    
    # Get middle slices for visualization
    source_tensor = source_common.tensor().detach().cpu().numpy()
    target_tensor = target_common.tensor().detach().cpu().numpy()
    
    print(f"   Source tensor shape: {source_tensor.shape}")
    print(f"   Target tensor shape: {target_tensor.shape}")
    
    # Handle different tensor dimensions
    if source_tensor.ndim == 4 and source_tensor.shape[0] == 1:
        source_tensor = source_tensor.squeeze(0)  # Remove batch dimension
    if target_tensor.ndim == 4 and target_tensor.shape[0] == 1:
        target_tensor = target_tensor.squeeze(0)  # Remove batch dimension
    
    # Get middle slice indices
    d_mid = source_tensor.shape[0] // 2
    h_mid = source_tensor.shape[1] // 2
    w_mid = source_tensor.shape[2] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Common Coordinate Space Comparison - {stage_name}', fontsize=16)
    
    # Source images
    axes[0, 0].imshow(source_tensor[d_mid, :, :], cmap='gray')
    axes[0, 0].set_title('Source - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(source_tensor[:, h_mid, :], cmap='gray')
    axes[0, 1].set_title('Source - Coronal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(source_tensor[:, :, w_mid], cmap='gray')
    axes[0, 2].set_title('Source - Sagittal')
    axes[0, 2].axis('off')
    
    # Target images
    axes[1, 0].imshow(target_tensor[d_mid, :, :], cmap='gray')
    axes[1, 0].set_title('Target - Axial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(target_tensor[:, h_mid, :], cmap='gray')
    axes[1, 1].set_title('Target - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(target_tensor[:, :, w_mid], cmap='gray')
    axes[1, 2].set_title('Target - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = output_dir / f"side_by_side_common_coords_{stage_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Side-by-side comparison saved: {output_path.name}")