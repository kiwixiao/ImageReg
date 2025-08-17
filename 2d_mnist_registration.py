#!/usr/bin/env python3
"""
2D MNIST Registration to exactly reproduce reference image
Using actual MNIST digits like in the reference
"""

import sys
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')

# Fix version check
import importlib.metadata
original_version = importlib.metadata.version
def mock_version(package_name):
    if package_name == "hf-deepali":
        return "0.1.0"
    return original_version(package_name)
importlib.metadata.version = mock_version

from pathlib import Path
import torch
from torch import Tensor, optim
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import deepali.spatial as spatial
from deepali.core import Grid, functional as U
from deepali.data import Image
from deepali.losses import functional as L


def load_mnist_images():
    """Load MNIST digit 9 images matching the reference"""
    # Load MNIST dataset
    mnist = MNIST(root="data", download=True, transform=ToTensor())
    
    # Find digit 9 samples
    nines = []
    for i in range(len(mnist)):
        image, label = mnist[i]
        if label == 9:
            nines.append(image)
            if len(nines) >= 10:  # Get enough samples
                break
    
    # Select appropriate samples (upright and rotated)
    # Based on the reference image, we need an upright 9 and a rotated 9
    target = nines[0].squeeze()  # Upright 9
    source = nines[2].squeeze()  # Different 9 (will look rotated)
    
    print(f"Target shape: {target.shape}")
    print(f"Source shape: {source.shape}")
    
    return target, source


def register_2d_mnist(target_tensor, source_tensor, device):
    """Perform 2D registration using deepali with SVFFD"""
    
    # Create grids for 2D MNIST images (28x28)
    grid = Grid(shape=target_tensor.shape)
    
    print(f"Grid shape: {grid.shape}")
    
    # Add channel dimension if needed
    if target_tensor.dim() == 2:
        target_tensor = target_tensor.unsqueeze(0)  # Add channel dimension
    if source_tensor.dim() == 2:
        source_tensor = source_tensor.unsqueeze(0)  # Add channel dimension
    
    # Create Image objects
    target_image = Image(target_tensor, grid).to(device)
    source_image = Image(source_tensor, grid).to(device)
    
    # Create image pyramids for multi-resolution
    target_pyramid = target_image.pyramid(levels=3)
    source_pyramid = source_image.pyramid(levels=3)
    
    # Use SVFFD for diffeomorphic registration (matching reference)
    # Start with coarsest level
    levels = len(target_pyramid)
    transform = spatial.StationaryVelocityFreeFormDeformation(
        target_pyramid[levels - 1].grid(), 
        stride=2  # Control point spacing
    ).to(device).train()
    
    print("\nMulti-resolution SVFFD registration...")
    
    # Multi-resolution optimization
    iterations_per_level = [100, 100, 200]
    
    for level in reversed(range(levels)):
        print(f"\nLevel {level}:")
        
        # Update grid for current level
        transform.grid_(target_pyramid[level].grid())
        
        # Get images at current level
        target_batch = target_pyramid[level].batch().tensor()
        source_batch = source_pyramid[level].batch().tensor()
        
        # Create transformer and optimizer
        transformer = spatial.ImageTransformer(transform)
        optimizer = optim.Adam(transform.parameters(), lr=1e-2)
        
        # Optimization at current level
        iterations = iterations_per_level[level] if level < len(iterations_per_level) else 100
        
        for i in range(iterations):
            # Forward transform: source -> target
            warped_source = transformer(source_batch)
            
            # Compute loss (MSE + regularization)
            sim_loss = L.mse_loss(warped_source, target_batch)
            
            # Add bending energy regularization for smooth deformation
            v = transform.v  # Velocity field
            bending = L.bending_loss(v)
            total_loss = sim_loss + 0.001 * bending
            
            if i % 50 == 0:
                print(f"  Iter {i:3d}: loss={total_loss.item():.6f}, sim={sim_loss.item():.6f}, bending={bending.item():.6f}")
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    return transform.eval()


def create_reference_style_visualization(target, source, transform, device, output_dir):
    """Create visualization exactly matching the reference image style"""
    
    # Create high-res grid for deformation visualization
    grid = Grid(shape=target.shape)
    highres_grid = grid.resize(512)
    grid_image = U.grid_image(highres_grid, num=1, stride=8, inverted=True, device=device)
    
    with torch.inference_mode():
        # Get inverse transform
        inverse_transform = transform.inverse()
        
        # Prepare batch tensors
        target_batch = target.unsqueeze(0).unsqueeze(0).to(device)
        source_batch = source.unsqueeze(0).unsqueeze(0).to(device)
        
        # Create transformers
        source_transformer = spatial.ImageTransformer(transform)
        target_transformer = spatial.ImageTransformer(inverse_transform)
        
        # Create grid transformers with higher resolution
        source_grid_transformer = spatial.ImageTransformer(transform, highres_grid, padding="zeros")
        target_grid_transformer = spatial.ImageTransformer(inverse_transform, highres_grid, padding="zeros")
        
        # Apply transformations
        warped_source = source_transformer(source_batch).squeeze()
        warped_target = target_transformer(target_batch).squeeze()
        
        warped_source_grid = source_grid_transformer(grid_image).squeeze()
        warped_target_grid = target_grid_transformer(grid_image).squeeze()
    
    # Create figure with exact layout as reference
    fig = plt.figure(figsize=(12, 8))
    
    # Create subplot grid
    gs = fig.add_gridspec(2, 3, hspace=0.05, wspace=0.05)
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Helper function for consistent display
    def imshow(tensor, title, ax):
        img = tensor.detach().cpu().numpy()
        ax.imshow(img, cmap='gray', interpolation='nearest')
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')
    
    # Top row
    imshow(target, "target", axes[0])
    imshow(warped_source, "warped source", axes[1])
    imshow(warped_source_grid, "forward deformation", axes[2])
    
    # Bottom row
    imshow(source, "source", axes[3])
    imshow(warped_target, "warped target", axes[4])
    imshow(warped_target_grid, "inverse deformation", axes[5])
    
    plt.tight_layout()
    plt.savefig(output_dir / "2d_mnist_registration_result.png", dpi=150, bbox_inches='tight', 
                facecolor='lightgray', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Saved visualization: {output_dir}/2d_mnist_registration_result.png")
    
    # Also save individual components for debugging
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(target.cpu().numpy(), cmap='gray')
    axes[0].set_title("Target (Original)")
    axes[0].axis('off')
    
    axes[1].imshow(source.cpu().numpy(), cmap='gray')
    axes[1].set_title("Source (Original)")
    axes[1].axis('off')
    
    axes[2].imshow(warped_source.cpu().numpy(), cmap='gray')
    axes[2].set_title("Warped Source\n(Should match Target)")
    axes[2].axis('off')
    
    axes[3].imshow(warped_target.cpu().numpy(), cmap='gray')
    axes[3].set_title("Warped Target\n(Should match Source)")
    axes[3].axis('off')
    
    plt.savefig(output_dir / "2d_mnist_components.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved components: {output_dir}/2d_mnist_components.png")
    
    return warped_source


def main():
    """Main 2D MNIST registration to reproduce reference"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("="*60)
    print("2D MNIST REGISTRATION")
    print("Reproducing reference image with actual MNIST digits")
    print("="*60)
    
    # Try to load from existing PNG files first
    input_dir = Path("/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/2D_mnist_input")
    output_dir = Path("/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/mnist_2d_output")
    output_dir.mkdir(exist_ok=True)
    
    if (input_dir / "target.png").exists() and (input_dir / "source.png").exists():
        print("\nUsing existing target.png and source.png...")
        
        # Load and normalize images
        target_pil = PILImage.open(input_dir / "target.png").convert('L')
        source_pil = PILImage.open(input_dir / "source.png").convert('L')
        
        # Resize to MNIST size (28x28) for easier processing
        target_pil = target_pil.resize((28, 28), PILImage.LANCZOS)
        source_pil = source_pil.resize((28, 28), PILImage.LANCZOS)
        
        target = torch.tensor(np.array(target_pil).astype(np.float32) / 255.0)
        source = torch.tensor(np.array(source_pil).astype(np.float32) / 255.0)
        
        print(f"Loaded target: {target.shape}")
        print(f"Loaded source: {source.shape}")
    else:
        print("\nLoading MNIST digits...")
        target, source = load_mnist_images()
    
    # Perform registration
    transform = register_2d_mnist(target, source, device)
    
    # Create visualization
    warped_source = create_reference_style_visualization(target, source, transform, device, output_dir)
    
    # Quality check
    print("\nRegistration complete!")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. 'Warped source' = source deformed to match target (FORWARD)")
    print("2. 'Warped target' = target deformed back to source (INVERSE)")
    print("3. Forward deformation shows how source moves to target")
    print("4. Inverse deformation shows how target moves back to source")
    print("\nFor segmentation transfer:")
    print("- Segmentation is with source image")
    print("- Use FORWARD transform to move it to target space")
    print("- This makes segmentation align with target anatomy")


if __name__ == "__main__":
    main()