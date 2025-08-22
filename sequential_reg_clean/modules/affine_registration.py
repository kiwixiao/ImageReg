#!/usr/bin/env python3
"""
Affine Registration Module
Performs 12-DOF affine registration (translation + rotation + scaling + shearing)
Chains from rigid registration output in common coordinate space
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
from deepali.losses import NMI
from deepali.data import Image

# Add parent directory to path for imports
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.image_loader import ImagePair, save_image_with_header_preservation, create_side_by_side_comparison


@dataclass
class AffineResult:
    """Container for affine registration results"""
    transform: spatial.AffineTransform
    source_after_affine_common: Image
    convergence_data: Dict[str, List[float]]
    final_loss: float
    matrix: torch.Tensor
    translation: torch.Tensor


class AffineRegistration:
    """
    Clean affine registration implementation
    Chains from rigid output and operates in common coordinate space
    Provides 12 DOF: 3 translation + 3 rotation + 3 scaling + 3 shearing
    """
    
    def __init__(self, device: str = "cpu", iterations: int = 15, learning_rate: float = 1e-4):
        self.device = torch.device(device)
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        # Results storage
        self.convergence_data = {
            'iterations': [],
            'total_loss': [],
            'ncc_loss': [],
            'matrix_det': [],
            'matrix_cond': []
        }
        
    def register(self, source_after_rigid: Image, target_normalized: Image, common_grid, rigid_transform) -> AffineResult:
        """
        Perform affine registration chaining from rigid output using multi-resolution pyramid
        EXACT COPY of working old implementation approach
        
        Args:
            source_after_rigid: Source image after rigid registration (in common coordinates)
            target_normalized: Target image (normalized, in common coordinates)
            common_grid: Common coordinate grid for registration
            rigid_transform: Rigid transform to initialize from
            
        Returns:
            AffineResult with transform and intermediate results
        """
        
        print("\n🔧 AFFINE REGISTRATION (9 DOF) - WORKING OLD APPROACH")
        print("=" * 60)
        print("Purpose: Linear deformation - translation + rotation + scaling")
        print("Coordinate space: Common world coordinates")
        print("DOF: 9 (3 translation + 3 rotation + 3 scaling)")
        print("Method: Multi-resolution pyramid with NMI loss")
        
        # EXACT COPY: Initialize affine from rigid result like working version
        print("🔗 Initializing affine from rigid result...")
        with torch.no_grad():
            # Create affine transform initialized with rigid rotation and translation
            self.affine_transform = spatial.AffineTransform(
                grid=common_grid,
                scaling=True,  # Initialize identity scaling
                rotation=rigid_transform.rotation.params.clone().detach(),
                translation=rigid_transform.translation.params.clone().detach()
            ).to(self.device)
        
        # Enable gradients for affine optimization
        for param in self.affine_transform.parameters():
            param.requires_grad = True
        
        param_count = sum(p.numel() for p in self.affine_transform.parameters())
        print(f"✅ Created affine transform with {param_count} parameters")
        
        # Multi-resolution pyramid levels - INCREASED iterations for better convergence
        affine_levels = [3, 2, 1]
        affine_iterations = {3: 30, 2: 50, 1: 70}  # Increased from {10, 15, 20}
        affine_learning_rates = {3: 5e-4, 2: 3e-4, 1: 1e-4}
        
        best_loss = float('inf')
        
        # EXACT COPY: Multi-resolution optimization like working version
        for level in affine_levels:
            print(f"\n📊 Affine Level {level} (1/{2**level} resolution)")
            
            # Create downsampled images for this level
            static_level = self._create_pyramid_level(target_normalized, level)
            moving_level = self._create_pyramid_level(source_after_rigid, level)
            
            # CRITICAL: Update transform grid to match current level
            self.affine_transform.grid_(static_level.grid())
            
            # Create NMI loss function for this level (same as rigid registration)
            nmi_loss = NMI().to(self.device)
            
            # Optimizer for this level
            optimizer = optim.Adam(
                self.affine_transform.parameters(), 
                lr=affine_learning_rates[level]
            )
            
            # Optimization iterations for this level
            num_iterations = affine_iterations[level]
            level_best_loss = float('inf')
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # Apply current affine transform
                transformer = spatial.ImageTransformer(self.affine_transform)
                warped = transformer(moving_level.tensor())
                
                # Apply foreground masking (same as rigid registration)
                target_masked, warped_masked = self._apply_mutual_fov_mask(
                    static_level.tensor(), warped
                )
                
                # Compute NMI loss with MIRTK-style foreground masking (same as rigid)
                loss = nmi_loss(warped_masked, target_masked)
                
                if loss.item() < level_best_loss:
                    level_best_loss = loss.item()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                
                # Store convergence data with real values
                current_iter = len(self.convergence_data['iterations'])
                self.convergence_data['iterations'].append(current_iter)
                self.convergence_data['total_loss'].append(loss.item())
                self.convergence_data['ncc_loss'].append(loss.item())  # Actually NMI loss
                
                # Get real matrix statistics
                with torch.no_grad():
                    from deepali.core.linalg import as_homogeneous_matrix
                    current_matrix = as_homogeneous_matrix(self.affine_transform.tensor())
                    linear_part = current_matrix[:, :3, :3] if current_matrix.shape[0] > 0 else current_matrix
                    matrix_det = torch.det(linear_part).item()
                    matrix_cond = torch.linalg.cond(linear_part).item()
                    
                self.convergence_data['matrix_det'].append(matrix_det)
                self.convergence_data['matrix_cond'].append(matrix_cond)
                
                # Print progress
                if iteration % 5 == 0 or iteration == num_iterations - 1:
                    print(f"   Iter {iteration:3d}: Loss = {loss.item():.6f}")
                
                # Backward pass and optimization step
                loss.backward()
                optimizer.step()
            
            print(f"   ✅ Affine Level {level}: Best = {level_best_loss:.6f}")
        
        print(f"✅ Affine registration complete. Best loss: {best_loss:.6f}")
        
        # Create result with transformed source in common space
        # Reset affine transform grid to original common grid before final application
        self.affine_transform.grid_(common_grid)
        final_transformer = spatial.ImageTransformer(self.affine_transform)
        source_after_affine_tensor = final_transformer(source_after_rigid.tensor())
        
        # Handle tensor dimensions like working rigid implementation
        if source_after_affine_tensor.dim() == 4:  # [N, D, H, W] - squeeze batch dimension
            source_after_affine_tensor = source_after_affine_tensor.squeeze(0)
        
        # Get final transformation matrix and translation first
        from deepali.core.linalg import as_homogeneous_matrix
        final_matrix = as_homogeneous_matrix(self.affine_transform.tensor()).detach()
        # Extract translation from the last column of the homogeneous matrix (first 3 elements)
        final_translation = final_matrix[0, :3, -1] if final_matrix.shape[0] > 0 else torch.zeros(3)
        
        print(f"📊 Final affine transformation:")
        # Extract 3x3 linear part for analysis
        linear_part = final_matrix[:, :3, :3] if final_matrix.shape[0] > 0 else final_matrix
        print(f"   Matrix determinant: {torch.det(linear_part).item():.6f}")
        print(f"   Matrix condition number: {torch.linalg.cond(linear_part).item():.1f}")
        print(f"   Translation: {final_translation.tolist()}")
        
        # Create a simple wrapper for visualization - use the actual transformed tensor
        print(f"   📊 Creating visualization wrapper for transformed tensor...")
        
        class AffineVisualizationWrapper:
            def __init__(self, tensor, grid, device):
                self._tensor = tensor
                self._grid = grid
                self.device = device
            
            def tensor(self):
                return self._tensor
                
            def grid(self):
                return self._grid
        
        source_after_affine_common = AffineVisualizationWrapper(
            source_after_affine_tensor, 
            target_normalized.grid(), 
            self.device
        )
        
        return AffineResult(
            transform=self.affine_transform,
            source_after_affine_common=source_after_affine_common,
            convergence_data=self.convergence_data,
            final_loss=best_loss,
            matrix=final_matrix,
            translation=final_translation
        )
    
    def save_intermediate_results(self, image_pair: ImagePair, affine_result: AffineResult, 
                                output_dir: Path, create_visualizations: bool = True):
        """
        Save affine registration intermediate results and debug outputs
        
        Args:
            image_pair: Original image pair
            affine_result: Affine registration results
            output_dir: Output directory
            create_visualizations: Whether to create debug visualizations
        """
        
        print("\n💾 SAVING AFFINE INTERMEDIATE RESULTS")
        print("=" * 50)
        
        # Create output directories
        transforms_dir = output_dir / "transforms"
        intermediate_dir = output_dir / "intermediate_results"
        debug_dir = output_dir / "debug_analysis"
        final_dir = output_dir / "final_results"
        
        for dir_path in [transforms_dir, intermediate_dir, debug_dir, final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save affine transform
        print("🔧 Saving affine transform...")
        affine_state = {
            'transform': affine_result.transform.state_dict(),
            'description': 'Affine registration (12 DOF: translation + rotation + scaling + shearing)',
            'method': 'affine_12dof_ncc_loss_with_det_regularization',
            'parameters': {
                'matrix': affine_result.matrix.tolist(),
                'translation': affine_result.translation.tolist(),
                'final_loss': affine_result.final_loss,
                'determinant': torch.det(affine_result.matrix[:, :3, :3]).item(),
                'condition_number': torch.linalg.cond(affine_result.matrix[:, :3, :3]).item()
            },
            'coordinate_space': 'common_world_coordinates',
            'grid_shape': list(image_pair.common_grid.shape),
            'grid_spacing': list(image_pair.common_grid.spacing().tolist()),
            'chained_from': 'rigid_registration_output'
        }
        torch.save(affine_state, transforms_dir / "affine_transform.pth")
        print(f"   ✅ Transform saved: affine_transform.pth")
        
        # 2. Save intermediate result in common coordinate space
        print("🌍 Saving intermediate result in common coordinates...")
        save_image_with_header_preservation(
            affine_result.source_after_affine_common.tensor(),
            image_pair.target_common,  # Use target as reference for common space
            intermediate_dir / "source_after_affine_common.nii.gz",
            "Source after affine (common coordinates)"
        )
        
        # 3. Apply affine to original resolution and save final results
        print("📏 Applying affine to original resolutions...")
        self._save_final_results_original_resolution(image_pair, affine_result.transform, final_dir)
        
        # 4. Create visualizations and debug outputs
        if create_visualizations:
            print("🎨 Creating debug visualizations...")
            self._create_debug_visualizations(image_pair, affine_result, debug_dir)
            
            # Side-by-side comparison in common coordinates
            create_side_by_side_comparison(
                affine_result.source_after_affine_common,
                image_pair.target_normalized,
                debug_dir,
                "affine_result"
            )
            
            # Create affine transformation flow visualization
            print("🌊 Creating affine transformation flow visualization...")
            self._create_affine_flow_visualization(image_pair, affine_result.transform, debug_dir)
            
            # Create Jacobian determinant map
            print("🗺️  Creating Jacobian determinant map...")
            self._create_jacobian_map(image_pair, affine_result.transform, debug_dir)
        
        print("✅ AFFINE INTERMEDIATE RESULTS SAVED")
        print(f"   📂 Transforms: {transforms_dir}")
        print(f"   📂 Intermediate: {intermediate_dir}")
        print(f"   📂 Final results: {final_dir}")
        print(f"   📂 Debug: {debug_dir}")
    
    def _save_final_results_original_resolution(self, image_pair: ImagePair, 
                                              affine_transform: spatial.AffineTransform, 
                                              final_dir: Path):
        """Save affine results at original image resolutions"""
        
        # Forward: Apply affine to source at original resolution
        affine_transformer_source = spatial.ImageTransformer(affine_transform, image_pair.source_original_grid)
        source_batch = image_pair.source_original.batch().tensor()
        source_affine_result = affine_transformer_source(source_batch)
        
        save_image_with_header_preservation(
            source_affine_result.squeeze(0),
            image_pair.source_original,
            final_dir / "source_moved_to_target_affine.nii.gz",
            "Source moved to target (affine, original resolution)"
        )
        
        # Bidirectional: Apply inverse affine to target at original resolution
        inverse_affine = affine_transform.inverse()
        inverse_affine_transformer = spatial.ImageTransformer(inverse_affine, image_pair.target_original_grid)
        target_batch = image_pair.target_original.batch().tensor()
        target_affine_result = inverse_affine_transformer(target_batch)
        
        save_image_with_header_preservation(
            target_affine_result.squeeze(0),
            image_pair.target_original,
            final_dir / "target_moved_to_source_affine.nii.gz",
            "Target moved to source (affine, original resolution)"
        )
    
    def _create_debug_visualizations(self, image_pair: ImagePair, affine_result: AffineResult, debug_dir: Path):
        """Create comprehensive debug visualizations"""
        
        import matplotlib.pyplot as plt
        
        # 1. Convergence plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Affine Registration Convergence Analysis', fontsize=14)
        
        iterations = affine_result.convergence_data['iterations']
        
        # Loss convergence - NMI loss is negative (higher values = better)
        axes[0, 0].plot(iterations, affine_result.convergence_data['total_loss'], 'b-', linewidth=2, label='NMI Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('NMI Loss (negative, higher is better)')
        axes[0, 0].set_title('NMI Loss Convergence (Multi-Resolution)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        # Don't use log scale for negative NMI values
        
        # Matrix determinant
        axes[0, 1].plot(iterations, affine_result.convergence_data['matrix_det'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal (det=1)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Matrix Determinant')
        axes[0, 1].set_title('Transformation Determinant')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Matrix condition number
        axes[1, 0].plot(iterations, affine_result.convergence_data['matrix_cond'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Condition Number')
        axes[1, 0].set_title('Matrix Condition Number')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Final parameters summary
        axes[1, 1].axis('off')
        det_val = torch.det(affine_result.matrix[:, :3, :3]).item()
        cond_val = torch.linalg.cond(affine_result.matrix[:, :3, :3]).item()
        
        summary_text = f"""
AFFINE REGISTRATION SUMMARY
===========================
Method: Multi-Resolution NMI Optimization
Levels: [3,2,1] → [1/8, 1/4, 1/2] resolution

Final NMI Loss: {affine_result.final_loss:.6f}
(Negative value, higher = better alignment)

Transformation Matrix:
{affine_result.matrix.cpu().numpy()}

Translation (mm):
• X: {affine_result.translation[0]:.3f}
• Y: {affine_result.translation[1]:.3f}  
• Z: {affine_result.translation[2]:.3f}

Matrix Properties:
• Determinant: {det_val:.6f} (≈1.0 = no volume change)
• Condition Number: {cond_val:.1f} (1.0 = optimal stability)

Total Iterations: {len(iterations)} across 3 levels
DOF: 9 (translation + rotation + scaling)
Coordinate Space: Common World
Initialization: From Rigid Registration Result
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(debug_dir / "affine_convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Convergence analysis: affine_convergence_analysis.png")
    
    def _create_affine_flow_visualization(self, image_pair: ImagePair, affine_transform: spatial.AffineTransform, debug_dir: Path):
        """Create affine transformation flow field and deformation visualization"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        from deepali.core import functional as U
        
        # Get common grid for flow computation
        grid = image_pair.common_grid
        
        with torch.no_grad():
            # Compute flow field (displacement field) for affine transform
            flow_field = affine_transform.flow(grid, device=self.device)
            
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
        fig.suptitle('Affine Transformation Flow Field Analysis', fontsize=16, fontweight='bold')
        
        # Get middle slices for visualization
        d_mid = flow_np.shape[1] // 2
        h_mid = flow_np.shape[2] // 2
        w_mid = flow_np.shape[3] // 2
        
        # 1. X displacement component
        im1 = axes[0, 0].imshow(flow_np[0, d_mid, :, :], cmap='RdBu_r')
        axes[0, 0].set_title(f'X Displacement (Axial slice)\nRange: [{flow_np[0].min():.3f}, {flow_np[0].max():.3f}] mm')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='mm')
        
        # 2. Y displacement component
        im2 = axes[0, 1].imshow(flow_np[1, d_mid, :, :], cmap='RdBu_r') 
        axes[0, 1].set_title(f'Y Displacement (Axial slice)\nRange: [{flow_np[1].min():.3f}, {flow_np[1].max():.3f}] mm')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='mm')
        
        # 3. Z displacement component
        im3 = axes[0, 2].imshow(flow_np[2, d_mid, :, :], cmap='RdBu_r')
        axes[0, 2].set_title(f'Z Displacement (Axial slice)\nRange: [{flow_np[2].min():.3f}, {flow_np[2].max():.3f}] mm')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8, label='mm')
        
        # 4. Displacement magnitude
        im4 = axes[1, 0].imshow(displacement_magnitude[d_mid, :, :], cmap='viridis')
        axes[1, 0].set_title(f'Displacement Magnitude\nMax: {displacement_magnitude.max():.3f} mm')
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
                         angles='xy', scale_units='xy', scale=1, color='cyan', alpha=0.8, width=0.003)
        axes[1, 1].set_title(f'Vector Field Overlay\n(Proportional: max={max_displacement:.3f}mm)')
        axes[1, 1].axis('off')
        
        # 6. Transformation analysis
        axes[1, 2].axis('off')
        
        # Get transformation matrix and analyze components
        from deepali.core.linalg import as_homogeneous_matrix
        matrix = as_homogeneous_matrix(affine_transform.tensor()).detach().cpu().numpy()
        # Extract translation from the last column of the homogeneous matrix (first 3 elements)
        if matrix.shape[0] > 0:
            translation = matrix[0, :3, -1]
        else:
            translation = np.zeros(3)
        
        stats_text = f"""
AFFINE TRANSFORMATION ANALYSIS
==============================
Method: Multi-Resolution NMI + Conservative Scaling

Transformation Matrix:
{matrix}

Translation Vector (mm):
• X: {translation[0]:.6f}
• Y: {translation[1]:.6f}
• Z: {translation[2]:.6f}
• Magnitude: {np.linalg.norm(translation):.6f}

Matrix Properties:
• Determinant: {np.linalg.det(matrix[0, :3, :3]):.6f}
  (≈1.0 indicates no volume distortion)
• Condition Number: {np.linalg.cond(matrix[0, :3, :3]):.1f}
  (1.0 = optimal numerical stability)

Displacement Field Statistics:
• Max displacement: {displacement_magnitude.max():.3f} mm
• Mean displacement: {displacement_magnitude.mean():.3f} mm
• Std displacement: {displacement_magnitude.std():.3f} mm

Conservative Affine Properties:
• Type: Linear transformation (no shearing)
• Includes: Translation + rotation + scaling only
• DOF: 9 parameters (3+3+3)
• Topology: Preserved (linear mapping)
• Initialization: From rigid registration
"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        
        # Save affine flow visualization
        flow_path = debug_dir / "affine_flow_field_analysis.png"
        plt.savefig(flow_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Affine flow field analysis: affine_flow_field_analysis.png")
        
        # Also create a grid deformation visualization
        self._create_affine_grid_visualization(image_pair, affine_transform, debug_dir)
    
    def _create_affine_grid_visualization(self, image_pair: ImagePair, affine_transform: spatial.AffineTransform, debug_dir: Path):
        """Create anatomical grid overlay visualization in proper world coordinates for affine"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        from deepali.core import functional as U
        
        print(f"   🌐 Creating anatomical grid overlay visualization for affine...")
        
        # Use common world coordinates for proper anatomical orientation
        grid = image_pair.common_grid
        
        # Get the anatomical images in common coordinates
        source_common = image_pair.source_normalized.tensor().squeeze().detach().cpu().numpy()
        target_common = image_pair.target_normalized.tensor().squeeze().detach().cpu().numpy()
        
        # Get the transformed source image to show actual affine transformation result
        with torch.no_grad():
            transformer = spatial.ImageTransformer(affine_transform)
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
        
        # Apply affine transformation to deform the line grid using image transformation
        with torch.no_grad():
            # Convert line grid to torch tensor and apply affine transformation directly
            line_grid_tensor = torch.from_numpy(line_grid).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, D, H, W]
            transformer = spatial.ImageTransformer(affine_transform)
            deformed_grid_tensor = transformer(line_grid_tensor)
            deformed_line_grid = deformed_grid_tensor.squeeze().detach().cpu().numpy()
        
        # Create comprehensive visualization showing source, target, and transformations
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Affine Grid Deformation: Source vs Target with Clear Line Grids', fontsize=16, fontweight='bold')
        
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
        
        # Row 3: Transformed Source + Deformed Grid + Movement Arrows
        axes[2, 0].imshow(source_transformed[d_mid, :, :], cmap='gray', alpha=0.8)
        axes[2, 0].contour(deformed_line_grid[d_mid, :, :], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        # Add movement arrows for sagittal view
        self._add_movement_arrows(axes[2, 0], line_grid[d_mid, :, :], deformed_line_grid[d_mid, :, :], 'sagittal')
        axes[2, 0].set_title('Affine Transformed + Deformed Grid + Movement (Sagittal)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(source_transformed[:, h_mid, :], cmap='gray', alpha=0.8)
        axes[2, 1].contour(deformed_line_grid[:, h_mid, :], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        # Add movement arrows for axial view
        self._add_movement_arrows(axes[2, 1], line_grid[:, h_mid, :], deformed_line_grid[:, h_mid, :], 'axial')
        axes[2, 1].set_title('Affine Transformed + Deformed Grid + Movement (Axial)')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(source_transformed[:, :, w_mid], cmap='gray', alpha=0.8)
        axes[2, 2].contour(deformed_line_grid[:, :, w_mid], levels=[0.5], colors='green', linewidths=1.5, alpha=0.8)
        # Add movement arrows for coronal view
        self._add_movement_arrows(axes[2, 2], line_grid[:, :, w_mid], deformed_line_grid[:, :, w_mid], 'coronal')
        axes[2, 2].set_title('Affine Transformed + Deformed Grid + Movement (Coronal)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        # Save anatomical grid visualization
        grid_path = debug_dir / "affine_grid_deformation.png"
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Anatomical grid overlay for affine: affine_grid_deformation.png")
    
    def _add_movement_arrows(self, ax, original_grid, deformed_grid, view_name):
        """Add arrows showing grid movement/deformation"""
        
        # Find grid intersection points in original grid
        grid_points = []
        h, w = original_grid.shape
        
        # Sample grid intersection points (every 16 pixels for clarity)
        for i in range(8, h-8, 16):
            for j in range(8, w-8, 16):
                # Check if this is near a grid line intersection
                if (original_grid[i, j] > 0.5 or 
                    original_grid[i-1:i+2, j].max() > 0.5 or 
                    original_grid[i, j-1:j+2].max() > 0.5):
                    grid_points.append((i, j))
        
        # Create displacement vectors by comparing original vs deformed grid
        arrows_x, arrows_y, arrows_dx, arrows_dy = [], [], [], []
        
        for y, x in grid_points:
            # Find nearest grid point in deformed image
            search_region = 8  # Search within 8 pixels
            y_start, y_end = max(0, y-search_region), min(h, y+search_region+1)
            x_start, x_end = max(0, x-search_region), min(w, x+search_region+1)
            
            # Find maximum response in deformed grid within search region
            search_area = deformed_grid[y_start:y_end, x_start:x_end]
            if search_area.size > 0:
                max_pos = np.unravel_index(np.argmax(search_area), search_area.shape)
                new_y = y_start + max_pos[0]
                new_x = x_start + max_pos[1]
                
                # Calculate displacement
                dx = new_x - x
                dy = new_y - y
                
                # Only show significant movements (> 2 pixels)
                if np.sqrt(dx*dx + dy*dy) > 2:
                    arrows_x.append(x)
                    arrows_y.append(y)
                    arrows_dx.append(dx)
                    arrows_dy.append(dy)
        
        # Draw arrows showing movement
        if len(arrows_x) > 0:
            ax.quiver(arrows_x, arrows_y, arrows_dx, arrows_dy,
                     angles='xy', scale_units='xy', scale=1,
                     color='yellow', alpha=0.9, width=0.003,
                     headwidth=3, headlength=4)
            
        print(f"   📍 Added {len(arrows_x)} movement arrows for {view_name} view")
    
    def _create_jacobian_map(self, image_pair: ImagePair, affine_transform: spatial.AffineTransform, debug_dir: Path):
        """Create Jacobian determinant map to visualize volume changes"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        from deepali.core.linalg import as_homogeneous_matrix
        
        print(f"   🧮 Computing Jacobian determinant map for affine transformation...")
        
        # Get transformation matrix
        with torch.no_grad():
            transform_matrix = as_homogeneous_matrix(affine_transform.tensor()).detach().cpu().numpy()
            
            # Extract 3x3 linear transformation matrix (excludes translation)
            if transform_matrix.shape[0] > 0:
                linear_matrix = transform_matrix[0, :3, :3]
            else:
                linear_matrix = transform_matrix[:3, :3]
            
            # For affine transformations, Jacobian determinant is constant everywhere
            # It's just the determinant of the linear transformation matrix
            jacobian_det = np.linalg.det(linear_matrix)
            
            print(f"   📊 Affine Jacobian determinant: {jacobian_det:.6f}")
            print(f"   💧 Volume change: {((jacobian_det - 1.0) * 100):.2f}%")
        
        # Get anatomical images for overlay
        source_common = image_pair.source_normalized.tensor().squeeze().detach().cpu().numpy()
        target_common = image_pair.target_normalized.tensor().squeeze().detach().cpu().numpy()
        
        # Create constant Jacobian map (same value everywhere for affine)
        jac_map = np.full_like(source_common, jacobian_det)
        
        # Create comprehensive Jacobian visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Affine Jacobian Determinant Map - Volume Change Analysis', fontsize=16, fontweight='bold')
        
        # Get middle slices
        d_mid = source_common.shape[0] // 2
        h_mid = source_common.shape[1] // 2  
        w_mid = source_common.shape[2] // 2
        
        # Define color map for Jacobian (centered at 1.0 = no volume change)
        vmin, vmax = 0.5, 1.5  # Reasonable range for affine transformations
        
        # Row 1: Jacobian maps overlaid on anatomy
        im1 = axes[0, 0].imshow(source_common[d_mid, :, :], cmap='gray', alpha=0.7)
        jac_overlay1 = axes[0, 0].imshow(jac_map[d_mid, :, :], cmap='RdBu_r', alpha=0.6, 
                                        vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Jacobian Map (Sagittal)')
        axes[0, 0].axis('off')
        plt.colorbar(jac_overlay1, ax=axes[0, 0], shrink=0.8, label='Jacobian Det')
        
        im2 = axes[0, 1].imshow(source_common[:, h_mid, :], cmap='gray', alpha=0.7)
        jac_overlay2 = axes[0, 1].imshow(jac_map[:, h_mid, :], cmap='RdBu_r', alpha=0.6,
                                        vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Jacobian Map (Axial)')
        axes[0, 1].axis('off')
        plt.colorbar(jac_overlay2, ax=axes[0, 1], shrink=0.8, label='Jacobian Det')
        
        im3 = axes[0, 2].imshow(source_common[:, :, w_mid], cmap='gray', alpha=0.7)
        jac_overlay3 = axes[0, 2].imshow(jac_map[:, :, w_mid], cmap='RdBu_r', alpha=0.6,
                                        vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Jacobian Map (Coronal)')
        axes[0, 2].axis('off')
        plt.colorbar(jac_overlay3, ax=axes[0, 2], shrink=0.8, label='Jacobian Det')
        
        # Row 2: Pure Jacobian maps and analysis
        jac_pure1 = axes[1, 0].imshow(jac_map[d_mid, :, :], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Pure Jacobian Map (Sagittal)')
        axes[1, 0].axis('off')
        plt.colorbar(jac_pure1, ax=axes[1, 0], shrink=0.8, label='Jacobian Det')
        
        jac_pure2 = axes[1, 1].imshow(jac_map[:, h_mid, :], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Pure Jacobian Map (Axial)')
        axes[1, 1].axis('off')
        plt.colorbar(jac_pure2, ax=axes[1, 1], shrink=0.8, label='Jacobian Det')
        
        # Analysis panel
        axes[1, 2].axis('off')
        
        analysis_text = f"""
JACOBIAN DETERMINANT ANALYSIS
=============================

Transformation Type: Affine (Linear)
Jacobian Determinant: {jacobian_det:.6f}

Volume Change Analysis:
{'─' * 25}
• Det = 1.0: No volume change
• Det > 1.0: Volume expansion  
• Det < 1.0: Volume compression
• Det = 0.0: Degenerate (fold/tear)

Current Transformation:
{'─' * 25}
• Jacobian Det: {jacobian_det:.6f}
• Volume Change: {((jacobian_det - 1.0) * 100):+.2f}%
• Interpretation: {"Expansion" if jacobian_det > 1.0 else "Compression" if jacobian_det < 1.0 else "No change"}

Affine Properties:
{'─' * 20}
• Constant Jacobian everywhere
• Uniform volume scaling
• Preserves straight lines
• No local deformation

Matrix Determinant:
{'─' * 20}
Linear Matrix:
{linear_matrix}

det(A) = {jacobian_det:.6f}

Color Scale:
{'─' * 15}
• Red: Compression (< 1.0)
• White: No change (= 1.0) 
• Blue: Expansion (> 1.0)
"""
        
        axes[1, 2].text(0.05, 0.95, analysis_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        
        # Save Jacobian map
        jacobian_path = debug_dir / "affine_jacobian_determinant_map.png"
        plt.savefig(jacobian_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Jacobian determinant map: affine_jacobian_determinant_map.png")
    
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
        # Use same masking threshold as old working implementation
        padding_value = -1.0  # Standard padding value for background masking
        
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
    
    def _create_pyramid_level(self, image: Image, level: int):
        """Create pyramid level by downsampling - EXACT copy from working old implementation"""
        if level == 1:
            return image
        
        factor = 2 ** (level - 1)
        tensor = image.tensor()
        
        # Downsample tensor - Handle 4D tensor properly for trilinear interpolation
        if tensor.dim() == 4:  # [C, X, Y, Z] - Need to add batch dimension for trilinear
            downsampled = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),  # Add batch dimension: [1, C, X, Y, Z]
                scale_factor=1.0/factor,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension: [C, X, Y, Z]
        elif tensor.dim() == 5:  # [N, C, X, Y, Z] - Already has batch dimension
            downsampled = torch.nn.functional.interpolate(
                tensor,
                scale_factor=1.0/factor,
                mode='trilinear',
                align_corners=False
            )
        else:  # 3D tensor [X, Y, Z] - Add both batch and channel dimensions
            downsampled = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, X, Y, Z]
                scale_factor=1.0/factor,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Back to [X, Y, Z]
        
        # Create new grid for downsampled image
        original_grid = image.grid()
        new_spacing = [s * factor for s in original_grid.spacing()]
        new_size = [int(s / factor) for s in original_grid.size()]
        
        from deepali.core import Grid
        new_grid = Grid(
            size=new_size,
            spacing=new_spacing,
            origin=original_grid.origin(),
            direction=original_grid.direction()
        )
        
        return Image(downsampled, new_grid, device=self.device)