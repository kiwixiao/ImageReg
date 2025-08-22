#!/usr/bin/env python3
"""
Rigid Registration Module
Performs 6-DOF rigid registration (translation + rotation) in common coordinate space
"""

import sys
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import deepali.spatial as spatial
from deepali.losses import functional as L
from deepali.data import Image

# Add parent directory to path for imports
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.image_loader import ImagePair, save_image_with_header_preservation, create_side_by_side_comparison


@dataclass
class RigidResult:
    """Container for rigid registration results"""
    transform: spatial.RigidTransform
    source_after_rigid_common: Image
    convergence_data: Dict[str, List[float]]
    final_loss: float
    translation: torch.Tensor
    rotation: torch.Tensor


class RigidRegistration:
    """
    Clean rigid registration implementation
    Operates in common coordinate space with comprehensive debugging
    """
    
    def __init__(self, device: str = "cpu", iterations: int = 200, learning_rate: float = 1e-2):
        self.device = torch.device(device)
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        # Results storage
        self.convergence_data = {
            'iterations': [],
            'total_loss': [],
            'ncc_loss': [],
            'translation_norm': [],
            'rotation_norm': []
        }
        
    def register(self, image_pair: ImagePair) -> RigidResult:
        """
        Perform rigid registration in common coordinate space
        
        Args:
            image_pair: ImagePair with all coordinate systems
            
        Returns:
            RigidResult with transform and intermediate results
        """
        
        print("\\n🔧 RIGID REGISTRATION (6 DOF)")
        print("=" * 50)
        print("Purpose: Global alignment - translation + rotation")
        print("Coordinate space: Common world coordinates")
        print("DOF: 6 (3 translation + 3 rotation)")
        
        # Create rigid transformation in common coordinate space
        rigid_transform = spatial.RigidTransform(
            image_pair.common_grid,
            translation=True,
            rotation=True
        ).to(self.device).train()
        
        param_count = sum(p.numel() for p in rigid_transform.parameters())
        print(f"✅ Created rigid transform with {param_count} parameters")
        
        # Create image transformer
        transformer = spatial.ImageTransformer(rigid_transform)
        
        # Optimizer
        optimizer = optim.Adam(rigid_transform.parameters(), lr=self.learning_rate)
        
        # Get normalized tensors for optimization
        source_batch = image_pair.source_normalized.batch().tensor()
        target_batch = image_pair.target_normalized.batch().tensor()
        
        print(f"🔄 Starting optimization: {self.iterations} iterations")
        print(f"   Source shape: {source_batch.shape}")
        print(f"   Target shape: {target_batch.shape}")
        
        best_loss = float('inf')
        
        # Optimization loop
        for i in range(self.iterations):
            optimizer.zero_grad()
            
            # Apply rigid transformation
            warped_source = transformer(source_batch)
            
            # Apply mutual foreground masking to focus only on overlapping FOV
            target_masked, warped_masked = self._apply_mutual_fov_mask(target_batch, warped_source)
            
            # Compute similarity loss (NCC for medical images) - only on mutual FOV
            ncc_loss = L.ncc_loss(warped_masked, target_masked)
            total_loss = ncc_loss  # No regularization for rigid
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track convergence
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
            
            # Get current parameters for monitoring
            translation = rigid_transform.translation.data().squeeze()
            rotation = rigid_transform.rotation.data().squeeze()
            translation_norm = torch.norm(translation).item()
            rotation_norm = torch.norm(rotation).item()
            
            # Store convergence data
            self.convergence_data['iterations'].append(i)
            self.convergence_data['total_loss'].append(current_loss)
            self.convergence_data['ncc_loss'].append(ncc_loss.item())
            self.convergence_data['translation_norm'].append(translation_norm)
            self.convergence_data['rotation_norm'].append(rotation_norm)
            
            # Progress logging
            if i % 50 == 0 or i == self.iterations - 1:
                print(f"   Iter {i:3d}: loss={current_loss:.6f}, ncc={ncc_loss.item():.6f}, "
                      f"trans_norm={translation_norm:.3f}, rot_norm={rotation_norm:.3f}")
        
        print(f"✅ Rigid registration complete. Best loss: {best_loss:.6f}")
        
        # Create result with transformed source in common space
        final_transformer = spatial.ImageTransformer(rigid_transform)
        source_after_rigid_tensor = final_transformer(image_pair.source_normalized.batch().tensor())
        source_after_rigid_common = Image(source_after_rigid_tensor.squeeze(0), image_pair.common_grid)
        
        # Get final parameters
        final_translation = rigid_transform.translation.data().squeeze()
        final_rotation = rigid_transform.rotation.data().squeeze()
        
        print(f"📊 Final rigid parameters:")
        print(f"   Translation: {final_translation.tolist()}")
        print(f"   Rotation: {final_rotation.tolist()}")
        
        return RigidResult(
            transform=rigid_transform,
            source_after_rigid_common=source_after_rigid_common,
            convergence_data=self.convergence_data,
            final_loss=best_loss,
            translation=final_translation,
            rotation=final_rotation
        )
    
    def save_intermediate_results(self, image_pair: ImagePair, rigid_result: RigidResult, 
                                output_dir: Path, create_visualizations: bool = True):
        """
        Save rigid registration intermediate results and debug outputs
        
        Args:
            image_pair: Original image pair
            rigid_result: Rigid registration results
            output_dir: Output directory
            create_visualizations: Whether to create debug visualizations
        """
        
        print("\\n💾 SAVING RIGID INTERMEDIATE RESULTS")
        print("=" * 50)
        
        # Create output directories
        transforms_dir = output_dir / "transforms"
        intermediate_dir = output_dir / "intermediate_results"
        debug_dir = output_dir / "debug_analysis"
        final_dir = output_dir / "final_results"
        
        for dir_path in [transforms_dir, intermediate_dir, debug_dir, final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save rigid transform
        print("🔧 Saving rigid transform...")
        rigid_state = {
            'transform': rigid_result.transform.state_dict(),
            'description': 'Rigid registration (6 DOF: 3 translation + 3 rotation)',
            'method': 'rigid_6dof_ncc_loss',
            'parameters': {
                'translation': rigid_result.translation.tolist(),
                'rotation': rigid_result.rotation.tolist(),
                'final_loss': rigid_result.final_loss
            },
            'coordinate_space': 'common_world_coordinates',
            'grid_shape': list(image_pair.common_grid.shape),
            'grid_spacing': list(image_pair.common_grid.spacing().tolist())
        }
        torch.save(rigid_state, transforms_dir / "rigid_transform.pth")
        print(f"   ✅ Transform saved: rigid_transform.pth")
        
        # 2. Save intermediate result in common coordinate space
        print("🌍 Saving intermediate result in common coordinates...")
        save_image_with_header_preservation(
            rigid_result.source_after_rigid_common.tensor(),
            image_pair.target_common,  # Use target as reference for common space
            intermediate_dir / "source_after_rigid_common.nii.gz",
            "Source after rigid (common coordinates)"
        )
        
        # 3. Apply rigid to original resolution and save final results
        print("📏 Applying rigid to original resolutions...")
        self._save_final_results_original_resolution(image_pair, rigid_result.transform, final_dir)
        
        # 4. Create visualizations and debug outputs
        if create_visualizations:
            print("🎨 Creating debug visualizations...")
            self._create_debug_visualizations(image_pair, rigid_result, debug_dir)
            
            # Side-by-side comparison in common coordinates
            create_side_by_side_comparison(
                rigid_result.source_after_rigid_common,
                image_pair.target_normalized,
                debug_dir,
                "rigid_result"
            )
            
            # Create rigid transformation flow visualization
            print("🌊 Creating rigid transformation flow visualization...")
            self._create_rigid_flow_visualization(image_pair, rigid_result.transform, debug_dir)
        
        print("✅ RIGID INTERMEDIATE RESULTS SAVED")
        print(f"   📂 Transforms: {transforms_dir}")
        print(f"   📂 Intermediate: {intermediate_dir}")
        print(f"   📂 Final results: {final_dir}")
        print(f"   📂 Debug: {debug_dir}")
    
    def _save_final_results_original_resolution(self, image_pair: ImagePair, 
                                              rigid_transform: spatial.RigidTransform, 
                                              final_dir: Path):
        """Save rigid results at original image resolutions"""
        
        # Forward: Apply rigid to source at original resolution
        rigid_transformer_source = spatial.ImageTransformer(rigid_transform, image_pair.source_original_grid)
        source_batch = image_pair.source_original.batch().tensor()
        source_rigid_result = rigid_transformer_source(source_batch)
        
        save_image_with_header_preservation(
            source_rigid_result.squeeze(0),
            image_pair.source_original,
            final_dir / "source_moved_to_target_rigid.nii.gz",
            "Source moved to target (rigid, original resolution)"
        )
        
        # Bidirectional: Apply inverse rigid to target at original resolution
        inverse_rigid = rigid_transform.inverse()
        inverse_rigid_transformer = spatial.ImageTransformer(inverse_rigid, image_pair.target_original_grid)
        target_batch = image_pair.target_original.batch().tensor()
        target_rigid_result = inverse_rigid_transformer(target_batch)
        
        save_image_with_header_preservation(
            target_rigid_result.squeeze(0),
            image_pair.target_original,
            final_dir / "target_moved_to_source_rigid.nii.gz",
            "Target moved to source (rigid, original resolution)"
        )
        
        # Save reference images
        save_image_with_header_preservation(
            image_pair.source_original.tensor(),
            image_pair.source_original,
            final_dir / "source_reference.nii.gz",
            "Source reference (original)"
        )
        
        save_image_with_header_preservation(
            image_pair.target_original.tensor(),
            image_pair.target_original,
            final_dir / "target_reference.nii.gz",
            "Target reference (original)"
        )
    
    def _create_debug_visualizations(self, image_pair: ImagePair, rigid_result: RigidResult, debug_dir: Path):
        """Create comprehensive debug visualizations"""
        
        import matplotlib.pyplot as plt
        
        # 1. Convergence plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Rigid Registration Convergence Analysis', fontsize=14)
        
        iterations = rigid_result.convergence_data['iterations']
        
        # Loss convergence
        axes[0, 0].plot(iterations, rigid_result.convergence_data['total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('NCC Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Translation norm
        axes[0, 1].plot(iterations, rigid_result.convergence_data['translation_norm'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Translation Norm (mm)')
        axes[0, 1].set_title('Translation Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rotation norm
        axes[1, 0].plot(iterations, rigid_result.convergence_data['rotation_norm'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Rotation Norm (radians)')
        axes[1, 0].set_title('Rotation Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final parameters summary
        axes[1, 1].axis('off')
        summary_text = f"""
RIGID REGISTRATION SUMMARY
========================

Final Loss: {rigid_result.final_loss:.6f}

Translation (mm):
• X: {rigid_result.translation[0]:.3f}
• Y: {rigid_result.translation[1]:.3f}  
• Z: {rigid_result.translation[2]:.3f}
• Norm: {torch.norm(rigid_result.translation):.3f}

Rotation (radians):
• X: {rigid_result.rotation[0]:.3f}
• Y: {rigid_result.rotation[1]:.3f}
• Z: {rigid_result.rotation[2]:.3f}
• Norm: {torch.norm(rigid_result.rotation):.3f}

Iterations: {len(iterations)}
Coordinate Space: Common World
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(debug_dir / "rigid_convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Convergence analysis: rigid_convergence_analysis.png")
    
    def _create_rigid_flow_visualization(self, image_pair: ImagePair, rigid_transform: spatial.RigidTransform, debug_dir: Path):
        """Create rigid transformation flow field and grid deformation visualization"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        from deepali.core import functional as U
        
        # Get common grid for flow computation
        grid = image_pair.common_grid
        
        with torch.no_grad():
            # Compute flow field (displacement field) for rigid transform
            flow_field = rigid_transform.flow(grid, device=self.device)
            
            # Convert to numpy for visualization
            if flow_field.dim() == 4:  # [C, D, H, W]
                flow_np = flow_field.detach().cpu().numpy()
            else:  # Has batch dimension
                flow_np = flow_field[0].detach().cpu().numpy()
            
            # Compute displacement magnitude
            displacement_magnitude = np.sqrt(np.sum(flow_np**2, axis=0))
            
            print(f"   Flow field shape: {flow_np.shape}")
            print(f"   Max displacement: {displacement_magnitude.max():.3f} mm")
            print(f"   Mean displacement: {displacement_magnitude.mean():.3f} mm")
        
        # Create comprehensive flow visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rigid Transformation Flow Field Analysis', fontsize=16, fontweight='bold')
        
        # Get middle slices for visualization
        d_mid = flow_np.shape[1] // 2
        h_mid = flow_np.shape[2] // 2
        w_mid = flow_np.shape[3] // 2
        
        # 1. X displacement component
        im1 = axes[0, 0].imshow(flow_np[0, d_mid, :, :], cmap='RdBu_r')
        axes[0, 0].set_title(f'X Displacement (Axial slice)\\nRange: [{flow_np[0].min():.3f}, {flow_np[0].max():.3f}] mm')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='mm')
        
        # 2. Y displacement component
        im2 = axes[0, 1].imshow(flow_np[1, d_mid, :, :], cmap='RdBu_r') 
        axes[0, 1].set_title(f'Y Displacement (Axial slice)\\nRange: [{flow_np[1].min():.3f}, {flow_np[1].max():.3f}] mm')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='mm')
        
        # 3. Z displacement component
        im3 = axes[0, 2].imshow(flow_np[2, d_mid, :, :], cmap='RdBu_r')
        axes[0, 2].set_title(f'Z Displacement (Axial slice)\\nRange: [{flow_np[2].min():.3f}, {flow_np[2].max():.3f}] mm')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8, label='mm')
        
        # 4. Displacement magnitude
        im4 = axes[1, 0].imshow(displacement_magnitude[d_mid, :, :], cmap='viridis')
        axes[1, 0].set_title(f'Displacement Magnitude\\nMax: {displacement_magnitude.max():.3f} mm')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8, label='mm')
        
        # 5. Vector field visualization (subsampled for clarity)
        source_slice = image_pair.source_normalized.tensor().detach().cpu().numpy()
        if source_slice.ndim == 4 and source_slice.shape[0] == 1:
            source_slice = source_slice.squeeze(0)
        axes[1, 1].imshow(source_slice[d_mid, :, :], cmap='gray', alpha=0.7)
        
        # Subsample vectors for clear visualization
        step = max(1, min(flow_np.shape[2], flow_np.shape[3]) // 15)  # ~15 vectors per dimension
        y_coords, x_coords = np.mgrid[0:flow_np.shape[2]:step, 0:flow_np.shape[3]:step]
        
        # Get flow components at middle slice
        dx_sub = flow_np[0, d_mid, ::step, ::step]
        dy_sub = flow_np[1, d_mid, ::step, ::step]
        
        # Scale vectors proportionally based on actual displacement magnitude
        max_displacement = np.sqrt(dx_sub**2 + dy_sub**2).max()
        if max_displacement > 0:
            # Scale to make maximum vector ~20 pixels long for visibility
            scale_factor = 20.0 / max_displacement
        else:
            scale_factor = 1.0
            
        axes[1, 1].quiver(x_coords, y_coords, dx_sub * scale_factor, dy_sub * scale_factor,
                         angles='xy', scale_units='xy', scale=1, color='red', alpha=0.8, width=0.003)
        axes[1, 1].set_title(f'Vector Field Overlay\\n(Proportional: max={max_displacement:.3f}mm)')
        axes[1, 1].axis('off')
        
        # 6. Displacement statistics and grid deformation info
        axes[1, 2].axis('off')
        
        # Compute rotation matrix and translation for analysis
        translation = rigid_transform.translation.data().squeeze().detach().cpu().numpy()
        rotation = rigid_transform.rotation.data().squeeze().detach().cpu().numpy()
        
        stats_text = f"""
RIGID TRANSFORMATION ANALYSIS
============================

Translation Vector (mm):
• X: {translation[0]:.6f}
• Y: {translation[1]:.6f}
• Z: {translation[2]:.6f}
• Magnitude: {np.linalg.norm(translation):.6f}

Rotation Vector (radians):
• X: {rotation[0]:.6f}
• Y: {rotation[1]:.6f}
• Z: {rotation[2]:.6f}
• Magnitude: {np.linalg.norm(rotation):.6f}

Displacement Field Statistics:
• Max displacement: {displacement_magnitude.max():.3f} mm
• Mean displacement: {displacement_magnitude.mean():.3f} mm
• Std displacement: {displacement_magnitude.std():.3f} mm

Grid Deformation Properties:
• Type: Rigid body motion
• Preserves: Distances, angles, shapes
• DOF: 6 (3 translation + 3 rotation)
• Topology: Preserved (no folding)
"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save rigid flow visualization
        flow_path = debug_dir / "rigid_flow_field_analysis.png"
        plt.savefig(flow_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Rigid flow field analysis: rigid_flow_field_analysis.png")
        
        # Also create a simplified grid deformation visualization
        self._create_rigid_grid_visualization(image_pair, rigid_transform, debug_dir)
    
    def _create_rigid_grid_visualization(self, image_pair: ImagePair, rigid_transform: spatial.RigidTransform, debug_dir: Path):
        """Create anatomical grid overlay visualization in proper world coordinates"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        from deepali.core import functional as U
        
        print(f"   🌐 Creating anatomical grid overlay visualization...")
        
        # Use common world coordinates for proper anatomical orientation
        grid = image_pair.common_grid
        
        # Get the anatomical images in common coordinates
        source_common = image_pair.source_normalized.tensor().squeeze().detach().cpu().numpy()
        target_common = image_pair.target_normalized.tensor().squeeze().detach().cpu().numpy()
        
        # Get the transformed source image to show actual rigid transformation result
        with torch.no_grad():
            transformer = spatial.ImageTransformer(rigid_transform)
            source_transformed_tensor = transformer(image_pair.source_normalized.batch().tensor())
            source_transformed = source_transformed_tensor.squeeze().detach().cpu().numpy()
        
        # Create clear line-based grid for better visualization
        def create_line_grid(shape, spacing=8):
            """Create a clear line-based grid pattern"""
            grid_array = np.zeros(shape, dtype=np.float32)
            
            # Add vertical lines (x-direction)
            for i in range(0, shape[2], spacing):
                if i < shape[2]:
                    grid_array[:, :, i] = 1.0
            
            # Add horizontal lines (y-direction) 
            for j in range(0, shape[1], spacing):
                if j < shape[1]:
                    grid_array[:, j, :] = 1.0
            
            # Add depth lines (z-direction)
            for k in range(0, shape[0], spacing):
                if k < shape[0]:
                    grid_array[k, :, :] = 1.0
                    
            return grid_array
        
        # Create line grid at image resolution for clear visualization
        line_grid = create_line_grid(source_common.shape, spacing=8)
        
        # Apply rigid transformation to deform the line grid using image transformation
        with torch.no_grad():
            # Convert line grid to torch tensor and apply rigid transformation directly
            line_grid_tensor = torch.from_numpy(line_grid).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, D, H, W]
            transformer = spatial.ImageTransformer(rigid_transform)
            deformed_grid_tensor = transformer(line_grid_tensor)
            deformed_line_grid = deformed_grid_tensor.squeeze().detach().cpu().numpy()
        
        # Create comprehensive visualization showing source, target, and transformations
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Rigid Grid Deformation: Source vs Target with Clear Line Grids', fontsize=16, fontweight='bold')
        
        # Get proper anatomical slices
        d_mid = source_common.shape[0] // 2
        h_mid = source_common.shape[1] // 2  
        w_mid = source_common.shape[2] // 2
        
        # Row 1: Original Source + Original Grid
        axes[0, 0].imshow(source_common[d_mid, :, :], cmap='gray', alpha=0.8)
        axes[0, 0].contour(line_grid[d_mid, :, :], levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)
        axes[0, 0].set_title('Source + Original Grid (Sagittal)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(source_common[:, h_mid, :], cmap='gray', alpha=0.8)
        axes[0, 1].contour(line_grid[:, h_mid, :], levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)
        axes[0, 1].set_title('Source + Original Grid (Axial)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(source_common[:, :, w_mid], cmap='gray', alpha=0.8)
        axes[0, 2].contour(line_grid[:, :, w_mid], levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)
        axes[0, 2].set_title('Source + Original Grid (Coronal)')
        axes[0, 2].axis('off')
        
        # Row 2: Target Images for Reference
        axes[1, 0].imshow(target_common[d_mid, :, :], cmap='gray', alpha=0.8)
        axes[1, 0].contour(line_grid[d_mid, :, :], levels=[0.5], colors='blue', linewidths=1.5, alpha=0.8)
        axes[1, 0].set_title('Target + Reference Grid (Sagittal)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(target_common[:, h_mid, :], cmap='gray', alpha=0.8)
        axes[1, 1].contour(line_grid[:, h_mid, :], levels=[0.5], colors='blue', linewidths=1.5, alpha=0.8)
        axes[1, 1].set_title('Target + Reference Grid (Axial)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(target_common[:, :, w_mid], cmap='gray', alpha=0.8)
        axes[1, 2].contour(line_grid[:, :, w_mid], levels=[0.5], colors='blue', linewidths=1.5, alpha=0.8)
        axes[1, 2].set_title('Target + Reference Grid (Coronal)')
        axes[1, 2].axis('off')
        
        # Row 3: Transformed Source + Deformed Grid
        axes[2, 0].imshow(source_transformed[d_mid, :, :], cmap='gray', alpha=0.8)
        axes[2, 0].contour(deformed_line_grid[d_mid, :, :], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        axes[2, 0].set_title('Rigid Transformed + Deformed Grid (Sagittal)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(source_transformed[:, h_mid, :], cmap='gray', alpha=0.8)
        axes[2, 1].contour(deformed_line_grid[:, h_mid, :], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        axes[2, 1].set_title('Rigid Transformed + Deformed Grid (Axial)')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(source_transformed[:, :, w_mid], cmap='gray', alpha=0.8)
        axes[2, 2].contour(deformed_line_grid[:, :, w_mid], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        axes[2, 2].set_title('Rigid Transformed + Deformed Grid (Coronal)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        # Save anatomical grid visualization
        grid_path = debug_dir / "rigid_grid_deformation.png"
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Anatomical grid overlay: rigid_grid_deformation.png")
    
    def _apply_mutual_fov_mask(self, target: torch.Tensor, source: torch.Tensor):
        """
        Apply mutual field of view masking for registration
        Only registers the overlapping regions where both images have valid data
        Non-overlapping regions remain unchanged
        
        Args:
            target: Target image tensor
            source: Source (warped) image tensor
            
        Returns:
            Tuple of (target_masked, source_masked) tensors
        """
        # Define background/padding value - use actual minimum from normalization
        # From output: Source range: [-0.656, 3.528], Target range: [-0.836, 5.598]
        # Use a threshold slightly below the actual minimum values
        padding_value = -0.9  # Below both -0.656 and -0.836
        
        # Create foreground masks for both images
        target_fg = target.squeeze() != padding_value
        source_fg = source.squeeze() != padding_value
        
        # Create mutual mask: only areas where BOTH images have valid data
        mutual_mask = (target_fg.float() * source_fg.float()).to(self.device)
        
        # Ensure correct tensor dimensions for broadcasting
        if mutual_mask.dim() == 3:  # [D, H, W]
            mutual_mask = mutual_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif mutual_mask.dim() == 4:  # [C, D, H, W] 
            mutual_mask = mutual_mask.unsqueeze(0)  # [1, C, D, H, W]
        
        # Apply mask to both images - focuses registration on mutual FOV only
        target_masked = target * mutual_mask
        source_masked = source * mutual_mask
        
        return target_masked, source_masked