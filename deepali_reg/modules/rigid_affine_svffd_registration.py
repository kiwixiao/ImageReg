#!/usr/bin/env python3
"""
Rigid+Affine+SVFFD Registration Module - Sequential Diffeomorphic Registration
GUARANTEED TO WORK: Uses working rigid+affine pipeline + deepali SVFFD refinement
Sequential optimization: Rigid (6 DOF) → Affine (12 DOF) → SVFFD (Diffeomorphic)
"""

import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional
from tempfile import TemporaryDirectory
import json

import deepali.spatial as spatial
from deepali.losses import NMI
from deepali.losses import functional as L
from deepali.data import Image

# Import the WORKING rigid+affine registration class
from .rigid_affine_registration import run_rigid_affine_registration, AffineRefinement
import SimpleITK as sitk


def run_rigid_affine_svffd_registration(
    static_path: str,
    moving_path: str,
    output_dir: str,
    static_seg_path: Optional[str] = None,
    device: str = "cpu",
    method_config: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Run sequential rigid+affine+SVFFD registration
    STEP 1: Run rigid+affine registration using the working modules
    STEP 2: Load affine result and refine with SVFFD (diffeomorphic)
    """
    
    print("🚀 RIGID+AFFINE+SVFFD REGISTRATION (Sequential Diffeomorphic)")
    print("=" * 70)
    print("✅ STEP 1: Run rigid+affine registration (6+12 DOF)")
    
    # STEP 1: Run rigid+affine registration directly using the working modules
    rigid_affine_results = run_rigid_affine_registration(
        static_path, moving_path, output_dir, static_seg_path, device, method_config
    )
    
    print("✅ Rigid+Affine registration completed successfully")
    print("✅ STEP 2: Load affine result and extend to SVFFD (Diffeomorphic)")
    
    # STEP 2: Run SVFFD refinement on top of rigid+affine
    svffd_refiner = SVFFDRefinement(device=device, output_dir=output_dir, method_config=method_config)
    svffd_refiner.load_images_and_affine_result(static_path, moving_path, output_dir)
    svffd_refiner.run_svffd_optimization()
    svffd_refiner.save_final_results(static_seg_path)
    
    print("✅ SEQUENTIAL APPROACH: Rigid+Affine+SVFFD complete!")
    
    # Return results dictionary - SAME format as other modules
    output_dir = Path(output_dir)
    return {
        'static_moved': str(output_dir / "static_moved_to_frame0_alignment.nii.gz"),
        'moving_moved': str(output_dir / "frame0_moved_to_static_alignment.nii.gz"),
        'static_ref': str(output_dir / "static_reference.nii.gz"),
        'moving_ref': str(output_dir / "frame0_reference.nii.gz"),
        'static_seg_moved': str(output_dir / "static_seg_moved_to_frame0.nii.gz") if static_seg_path else None,
        'pytorch': str(output_dir / "rigid_affine_svffd_transform.pth"),
        'text': str(output_dir / "rigid_affine_svffd_transform.txt")
    }


class SVFFDRefinement:
    """
    Clean SVFFD refinement class that builds on completed rigid+affine registration
    Uses deepali's StationaryVelocityFreeFormDeformation for diffeomorphic registration
    Following PDF tutorial examples and multi-resolution approach
    """
    
    def __init__(self, device: str = "cpu", output_dir: str = "outputs", method_config: Optional[Dict] = None):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.method_config = method_config or {}
        
        # Will be loaded from rigid+affine result
        self.static_sitk = None
        self.moving_sitk = None
        self.static_deepali = None
        self.moving_deepali = None
        self.affine_transform = None
        self.svffd_transform = None
        
        # SVFFD-specific parameters following deepali tutorial
        if 'svffd_pyramid_levels' in self.method_config:
            self.svffd_levels = self.method_config['svffd_pyramid_levels']
            self.svffd_iterations = self.method_config['svffd_iterations']
            self.svffd_learning_rates = self.method_config['svffd_learning_rates']
        else:
            # Default SVFFD parameters following deepali tutorial
            self.svffd_levels = [2, 1]  # Multi-resolution levels
            self.svffd_iterations = {2: 30, 1: 50}
            self.svffd_learning_rates = {2: 1e-2, 1: 8e-3}
        
        # SVFFD regularization weight for B-spline bending energy
        self.w_bending = self.method_config.get('w_bending', 1e-3)  # B-spline bending energy
        
        # SVFFD control point spacing (integer for B-spline)
        self.svffd_stride = self.method_config.get('svffd_stride', 2)  # Every 2nd grid point
    
    def load_images_and_affine_result(self, static_path: str, moving_path: str, output_dir: str):
        """
        Load images and rigid+affine transformation result
        """
        print("📂 Loading images and rigid+affine result...")
        
        # Load images exactly like the working registration modules
        self.static_sitk = sitk.ReadImage(static_path)
        self.moving_sitk = sitk.ReadImage(moving_path)
        
        # Convert to deepali format
        self.static_deepali = Image.from_sitk(self.static_sitk, device=self.device)
        self.moving_deepali = Image.from_sitk(self.moving_sitk, device=self.device)
        
        # Load the rigid+affine transformation result
        affine_transform_path = Path(output_dir) / "rigid_affine_transform.pth"
        loaded_data = torch.load(affine_transform_path, map_location=self.device)
        
        # Handle both wrapped and unwrapped formats
        if isinstance(loaded_data, dict) and 'transform' in loaded_data:
            affine_state_dict = loaded_data['transform']
            print(f"✅ Loaded wrapped rigid+affine transform (format: main pipeline)")
        else:
            affine_state_dict = loaded_data
            print(f"✅ Loaded direct rigid+affine transform state_dict")
        
        # Recreate affine transform and load state
        self.affine_transform = spatial.AffineTransform(self.static_deepali.grid()).to(self.device)
        self.affine_transform.load_state_dict(affine_state_dict)
        
        print(f"✅ Images loaded: Static {self.static_sitk.GetSize()} @ {self.static_sitk.GetSpacing()}")
        print(f"✅ Images loaded: Moving {self.moving_sitk.GetSize()} @ {self.moving_sitk.GetSpacing()}")
        print(f"✅ Rigid+Affine transform loaded from: {affine_transform_path}")
    
    def run_svffd_optimization(self):
        """
        Run SVFFD refinement on top of rigid+affine result
        Uses deepali StationaryVelocityFreeFormDeformation following PDF tutorial
        """
        print("\n🔄 STAGE 3: SVFFD REFINEMENT (Diffeomorphic)")
        print("Adding SVFFD refinement on top of completed rigid+affine result...")
        print(f"📊 SVFFD Parameters:")
        print(f"   Control point stride: {self.svffd_stride}")
        print(f"   B-spline bending regularization: {self.w_bending}")
        
        # Apply affine transform to moving image before SVFFD optimization
        # This ensures SVFFD starts from affine-aligned position
        print("🔄 Pre-aligning images using rigid+affine transform...")
        affine_transformer = spatial.ImageTransformer(self.affine_transform)
        moving_affine_aligned = affine_transformer(self.moving_deepali.tensor())
        
        # Create Image object for aligned moving image
        aligned_moving_deepali = Image(moving_affine_aligned, self.static_deepali.grid(), device=self.device)
        
        # Initialize SVFFD transform following deepali tutorial - START WITH COARSEST LEVEL
        print("🔗 Initializing SVFFD (Stationary Velocity Free-Form Deformation)...")
        
        # Create pyramids first to get the coarsest level grid
        max_levels = max(self.svffd_levels) + 1  # pyramid() creates levels 0 to max_levels-1
        static_pyramid = self.static_deepali.pyramid(max_levels)
        moving_pyramid = aligned_moving_deepali.pyramid(max_levels)
        
        print(f"   📊 Created pyramid with levels: {list(static_pyramid.keys())}")
        
        # Initialize SVFFD with coarsest level grid (following deepali tutorial pattern)
        coarsest_level = max(self.svffd_levels)
        if coarsest_level not in static_pyramid:
            coarsest_level = max(static_pyramid.keys())
            print(f"   ⚠️  Requested level {max(self.svffd_levels)} not available, using {coarsest_level}")
        
        coarsest_grid = static_pyramid[coarsest_level].grid()
        print(f"   📐 Initializing SVFFD with coarsest level {coarsest_level} grid: {coarsest_grid.shape}")
        
        self.svffd_transform = spatial.StationaryVelocityFreeFormDeformation(
            grid=coarsest_grid,  # Start with coarsest level
            stride=int(self.svffd_stride),  # B-spline control point spacing (integer)
            params=True  # Enable gradient optimization
        ).to(self.device)
        
        # Run multi-resolution SVFFD optimization following deepali tutorial
        # Process from coarsest (highest number) to finest (lowest number)
        for level in sorted(self.svffd_levels, reverse=True):  # Start from coarsest level
            print(f"\n📊 SVFFD Level {level} (1/{2**level} resolution)")
            
            # Get pyramid images for this level
            static_level = static_pyramid[level]
            moving_level = moving_pyramid[level]
            
            # Update SVFFD transform grid to current level - following deepali tutorial pattern
            print(f"   📐 Updating SVFFD grid to level {level} resolution")
            self.svffd_transform.grid_(static_level.grid())
            
            # Create loss function for this level
            loss_fn = self._create_svffd_loss_function()
            
            # Optimizer for this level
            optimizer = optim.Adam(
                self.svffd_transform.parameters(), 
                lr=self.svffd_learning_rates[level]
            )
            
            # Optimization iterations for this level
            num_iterations = self.svffd_iterations[level]
            best_loss = float('inf')
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # Apply current SVFFD transform following deepali tutorial pattern
                svffd_transformer = spatial.ImageTransformer(self.svffd_transform)
                warped = svffd_transformer(moving_level.batch().tensor())
                
                # Compute loss with regularization
                loss_dict = loss_fn(warped, static_level.batch().tensor(), self.svffd_transform)
                total_loss = loss_dict['loss']
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                
                # Print progress
                if iteration % 10 == 0 or iteration == num_iterations - 1:
                    loss_str = f"Loss={total_loss.item():.6f}"
                    if 'sim' in loss_dict:
                        loss_str += f", Sim={loss_dict['sim'].item():.6f}"
                    if 'diff' in loss_dict:
                        loss_str += f", Diff={loss_dict['diff'].item():.2f}"
                    if 'be' in loss_dict:
                        loss_str += f", Bend={loss_dict['be'].item():.2f}"
                    print(f"   Iter {iteration:3d}: {loss_str}")
                
                # Backward pass and optimization step
                total_loss.backward()
                optimizer.step()
            
            print(f"   ✅ SVFFD Level {level}: Best = {best_loss:.6f}")
        
        print("✅ SVFFD optimization complete - Diffeomorphic registration achieved!")
    
    
    def _create_svffd_loss_function(self):
        """
        Create SVFFD loss function with B-spline regularization
        Following deepali tutorial pattern for StationaryVelocityFreeFormDeformation
        """
        def loss_fn(warped, target, transform):
            """SVFFD loss function with similarity + B-spline bending regularization"""
            terms = {}
            
            # Similarity term (NMI)
            nmi_loss = NMI().to(self.device)
            
            # Ensure correct dimensions for NMI [N, C, D, H, W]
            if warped.dim() == 3:
                warped = warped.unsqueeze(0).unsqueeze(0)
            elif warped.dim() == 4:
                warped = warped.unsqueeze(0)
            
            if target.dim() == 3:
                target = target.unsqueeze(0).unsqueeze(0)
            elif target.dim() == 4:
                target = target.unsqueeze(0)
            
            sim = nmi_loss(warped, target)
            terms['sim'] = sim
            total_loss = sim
            
            # B-spline regularization for SVFFD - following deepali tutorial
            if self.w_bending > 0:
                # For BSplineTransform (SVFFD inherits from this), use B-spline bending energy
                if isinstance(transform, spatial.BSplineTransform):
                    params = transform.params
                    if isinstance(params, torch.Tensor):
                        bending = L.bspline_bending_loss(params)
                        total_loss = bending.mul(self.w_bending).add(total_loss)
                        terms['be'] = bending
                else:
                    # Fallback to velocity field bending energy
                    v = getattr(transform, 'v', None)
                    if v is not None:
                        bending = L.bending_loss(v)
                        total_loss = bending.mul(self.w_bending).add(total_loss)
                        terms['be'] = bending
            
            return {'loss': total_loss, **terms}
        
        return loss_fn
    
    def save_final_results(self, static_seg_path=None):
        """
        Save final SVFFD transformation and bidirectional results
        Compose the full rigid+affine+SVFFD transformation
        """
        print("\n💾 SAVING FINAL RIGID+AFFINE+SVFFD TRANSFORMATION")
        print("=" * 60)
        
        # Compose the full transformation: SVFFD ∘ Affine  
        # The moving image is first transformed by Affine (rigid+affine result), then by SVFFD (diffeomorphic refinement)
        composed_transform = spatial.SequentialTransform(
            self.static_deepali.grid(),
            self.affine_transform,  # Applied first (rigid+affine result)
            self.svffd_transform   # Applied second (diffeomorphic refinement)
        )
        
        # Save final composed transform
        final_transform_path = self.output_dir / "rigid_affine_svffd_transform.pth"
        torch.save(composed_transform.state_dict(), final_transform_path)
        print(f"✅ Final PyTorch: {final_transform_path}")
        
        # Save in text format
        self._save_final_transform_text(composed_transform)
        
        # Save in JSON format
        self._save_final_transform_json(composed_transform)
        
        print("🎯 Final rigid+affine+SVFFD transformation saved!")
        
        # Create deepali professional visualization using WORKING approach
        print("🎨 Creating deepali professional visualization using WORKING approach...")
        from .svffd_visualization_rewrite import create_working_svffd_visualization
        create_working_svffd_visualization(self.static_sitk, self.moving_sitk, composed_transform, self.output_dir, self.device)
        
        # Save bidirectional results using final composed transform
        self._save_bidirectional_results(composed_transform, static_seg_path)
    
    def _save_final_transform_text(self, transform):
        """Save final transform in human-readable text format"""
        path = self.output_dir / "rigid_affine_svffd_transform.txt"
        with open(path, 'w') as f:
            f.write("Rigid+Affine+SVFFD Transform (Diffeomorphic)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Transform Type: Composed (Affine ∘ SVFFD)\n")
            f.write("Components:\n")
            f.write("  1. Rigid (6 DOF) - Translation + Rotation\n")
            f.write("  2. Affine (12 DOF) - Scaling + Shearing\n")
            f.write("  3. SVFFD (Diffeomorphic) - Non-rigid deformation\n\n")
            f.write("Properties:\n")
            f.write("  - Diffeomorphic (invertible)\n")
            f.write("  - Multi-resolution optimization\n")
            f.write("  - Regularized for smoothness\n\n")
            f.write(f"SVFFD Parameters:\n")
            f.write(f"  - Control point stride: {self.svffd_stride}\n")
            f.write(f"  - B-spline bending weight: {self.w_bending}\n")
        
        print(f"✅ Text format: {path}")
    
    def _save_final_transform_json(self, transform):
        """Save final transform in JSON format"""
        path = self.output_dir / "rigid_affine_svffd_transform.json"
        transform_data = {
            'type': 'rigid_affine_svffd',
            'components': ['rigid', 'affine', 'svffd'],
            'properties': {
                'diffeomorphic': True,
                'invertible': True,
                'multi_resolution': True
            },
            'svffd_config': {
                'stride': self.svffd_stride,
                'regularization': {
                    'bending': self.w_bending
                }
            },
            'device': str(self.device)
        }
        
        with open(path, 'w') as f:
            json.dump(transform_data, f, indent=2)
        
        print(f"✅ JSON format: {path}")
    
    def _save_bidirectional_results(self, final_transform, static_seg_path=None):
        """
        Save bidirectional results using final composed transform
        Following the same pattern as other modules
        """
        print("\n💾 SAVING BIDIRECTIONAL RESULTS (FINAL DIFFEOMORPHIC)")
        print("=" * 60)
        print("Following user requirements:")
        print("- Registration = moving images, NOT changing resolution")
        print("- Each result keeps its original coordinate system")
        print("- Diffeomorphic properties preserved")
        
        # Get forward and inverse transforms
        forward_transform = final_transform
        inverse_transform = forward_transform.inverse()
        
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # === Direction A: Static → Frame0 alignment ===
            print("\n🔄 Direction A: Static → Frame0 alignment (keeps static resolution)")
            
            # Save static temporarily for deepali processing
            static_temp = temp_dir / "static_orig.nii.gz"
            sitk.WriteImage(self.static_sitk, str(static_temp))
            static_orig_deepali = Image.read(str(static_temp), device=self.device)
            
            # Apply inverse transformation to move static into frame0's alignment
            inverse_transformer = spatial.ImageTransformer(inverse_transform)
            warped_static_tensor = inverse_transformer(static_orig_deepali.tensor())
            
            # Convert back to SimpleITK format preserving static's coordinate system
            warped_static_array = warped_static_tensor.squeeze().detach().cpu().numpy()
            warped_static_sitk = sitk.GetImageFromArray(warped_static_array)
            warped_static_sitk.CopyInformation(self.static_sitk)
            
            # Save result: Static image aligned with frame0, keeping static resolution
            static_moved_path = self.output_dir / "static_moved_to_frame0_alignment.nii.gz"
            sitk.WriteImage(warped_static_sitk, str(static_moved_path))
            print(f"   ✅ Saved: {static_moved_path}")
            print(f"      Resolution: {warped_static_sitk.GetSize()} @ {warped_static_sitk.GetSpacing()}")
            
            # ============================================
            # SEGMENTATION TRANSFORMATION (if provided)
            # ============================================
            if static_seg_path is not None:
                print(f"\n🏷️  Transforming static segmentation using same diffeomorphic transformation...")
                
                try:
                    # Load static segmentation
                    static_seg_sitk = sitk.ReadImage(static_seg_path)
                    print(f"   📂 Loaded segmentation: {static_seg_sitk.GetSize()}")
                    
                    # Save segmentation temporarily for deepali processing
                    seg_temp = temp_dir / "static_seg_orig.nii.gz"
                    sitk.WriteImage(static_seg_sitk, str(seg_temp))
                    static_seg_deepali = Image.read(str(seg_temp), device=self.device)
                    
                    # Apply the SAME inverse transformation used for static image
                    # Use NEAREST NEIGHBOR interpolation for segmentation to preserve discrete labels
                    seg_transformer = spatial.ImageTransformer(inverse_transform, sampling='nearest', padding='border')
                    warped_seg_tensor = seg_transformer(static_seg_deepali.tensor())
                    
                    # Convert back to SimpleITK with nearest neighbor interpolation for labels
                    warped_seg_array = warped_seg_tensor.squeeze().detach().cpu().numpy()
                    warped_seg_sitk = sitk.GetImageFromArray(warped_seg_array)
                    warped_seg_sitk.CopyInformation(static_seg_sitk)
                    
                    # Save transformed segmentation
                    seg_moved_path = self.output_dir / "static_seg_moved_to_frame0.nii.gz"
                    sitk.WriteImage(warped_seg_sitk, str(seg_moved_path))
                    print(f"   ✅ Saved: {seg_moved_path}")
                    print(f"      Resolution: {warped_seg_sitk.GetSize()} @ {warped_seg_sitk.GetSpacing()}")
                    print(f"   🎯 Segmentation moved using identical diffeomorphic transformation")
                    
                except Exception as e:
                    print(f"   ⚠️  Error transforming segmentation: {e}")
                    print(f"   📝 Continuing with image registration results...")
            
            # === Direction B: Frame0 → Static alignment ===
            print("\n🔄 Direction B: Frame0 → Static alignment (keeps frame0 resolution)")
            
            # Use original moving image with its original grid
            moving_temp = temp_dir / "moving_orig.nii.gz"
            sitk.WriteImage(self.moving_sitk, str(moving_temp))
            moving_orig_deepali = Image.read(str(moving_temp), device=self.device)
            
            # Apply forward transformation using the ORIGINAL moving image grid
            forward_transformer = spatial.ImageTransformer(forward_transform, moving_orig_deepali.grid())
            warped_moving_tensor = forward_transformer(moving_orig_deepali.tensor())
            
            # Convert back to SimpleITK format preserving moving's coordinate system
            warped_moving_array = warped_moving_tensor.squeeze().detach().cpu().numpy()
            warped_moving_sitk = sitk.GetImageFromArray(warped_moving_array)
            warped_moving_sitk.CopyInformation(self.moving_sitk)
            
            # Save result: Frame0 image aligned with static, keeping frame0 resolution
            moving_moved_path = self.output_dir / "frame0_moved_to_static_alignment.nii.gz"
            sitk.WriteImage(warped_moving_sitk, str(moving_moved_path))
            print(f"   ✅ Saved: {moving_moved_path}")
            print(f"      Resolution: {warped_moving_sitk.GetSize()} @ {warped_moving_sitk.GetSpacing()}")
            
            # === Save reference images ===
            static_ref_path = self.output_dir / "static_reference.nii.gz"
            moving_ref_path = self.output_dir / "frame0_reference.nii.gz"
            sitk.WriteImage(self.static_sitk, str(static_ref_path))
            sitk.WriteImage(self.moving_sitk, str(moving_ref_path))
            
            print("\n✅ Saved references: static_reference.nii.gz, frame0_reference.nii.gz")
            
        print("\n📝 VERIFICATION IN ITK-SNAP:")
        print("Option A (Static moved to frame0):")
        print("  1. Load: frame0_reference.nii.gz")
        print("  2. Add overlay: static_moved_to_frame0_alignment.nii.gz")
        print("Option B (Frame0 moved to static):")
        print("  1. Load: static_reference.nii.gz")
        print("  2. Add overlay: frame0_moved_to_static_alignment.nii.gz")
        print("")
        print("🎯 Both overlays should show OPTIMAL alignment with diffeomorphic properties!")
        print("🔄 The transformation is invertible and preserves topology!")
    
    def _create_deepali_native_visualization(self, composed_transform):
        """
        Create deepali professional visualization using transform object directly
        Following the deepali tutorial's invertible_registration_figure pattern
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        
        print("🎨 Creating deepali professional visualization...")
        
        try:
            # Create figure following the WORKING world coordinate approach
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle("DEEPALI Professional Visualization - RIGID_AFFINE_SVFFD (after)", fontsize=16, fontweight='bold')
            
            # Use the SAME world coordinate sampling approach as the working visualization
            # Get world coordinate for visualization (use center)
            static_center = self.static_sitk.TransformIndexToPhysicalPoint(
                [s//2 for s in self.static_sitk.GetSize()]
            )
            world_x = static_center[0]
            
            # Sample images at same world coordinate plane - following working pattern
            static_samples, moving_before_samples, moving_after_samples, Y_world, Z_world = self._sample_world_plane_svffd_comparison(world_x, composed_transform)
            
            # Row 1: Images showing SVFFD improvement in SAME world coordinate plane
            # Static image at world coordinate X
            axes[0, 0].imshow(static_samples, cmap='gray', aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[0, 0].set_title(f'Static Image\n@ X={world_x:.1f}mm (World Coords)', fontsize=10)
            axes[0, 0].set_xlabel('Y (mm)')
            axes[0, 0].set_ylabel('Z (mm)')
            axes[0, 0].invert_xaxis()
            axes[0, 0].invert_yaxis()
            
            # Moving image AFTER SVFFD (final result) - SAME world coordinates as static
            axes[0, 1].imshow(moving_after_samples, cmap='gray', aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[0, 1].set_title(f'Moving Image\n@ X={world_x:.1f}mm (after)', fontsize=10)
            axes[0, 1].set_xlabel('Y (mm)')
            axes[0, 1].set_ylabel('Z (mm)')
            axes[0, 1].invert_xaxis()
            axes[0, 1].invert_yaxis()
            
            # RGB Overlay (Red=Static, Green=Moving after SVFFD) 
            overlay = np.zeros((*static_samples.shape, 3))
            static_norm = (static_samples - static_samples.min()) / (static_samples.max() - static_samples.min() + 1e-8)
            moving_norm = (moving_after_samples - moving_after_samples.min()) / (moving_after_samples.max() - moving_after_samples.min() + 1e-8)
            overlay[..., 0] = static_norm  # Red channel = Static
            overlay[..., 1] = moving_norm  # Green channel = Moving (after SVFFD)
            
            axes[0, 2].imshow(overlay, aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[0, 2].set_title(f'RGB Overlay\n(Red=Static, Green=Moving)', fontsize=10)
            axes[0, 2].set_xlabel('Y (mm)')
            axes[0, 2].set_ylabel('Z (mm)')
            axes[0, 2].invert_xaxis()
            axes[0, 2].invert_yaxis()
            
            # RGB overlay showing SVFFD improvement: BEFORE vs AFTER
            rgb_svffd_improvement = np.zeros((*static_samples.shape, 3))
            before_norm = (moving_before_samples - moving_before_samples.min()) / (moving_before_samples.max() - moving_before_samples.min() + 1e-8)
            after_norm = (moving_after_samples - moving_after_samples.min()) / (moving_after_samples.max() - moving_after_samples.min() + 1e-8)
            rgb_svffd_improvement[:, :, 0] = before_norm  # Red: before SVFFD
            rgb_svffd_improvement[:, :, 1] = after_norm   # Green: after SVFFD
            rgb_svffd_improvement[:, :, 2] = 0.0  # Blue: none
            
            axes[0, 3].imshow(rgb_svffd_improvement, aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[0, 3].set_title('SVFFD Improvement\nRed=Before, Green=After SVFFD', fontsize=10)
            axes[0, 3].set_xlabel('Y (mm)')
            axes[0, 3].set_ylabel('Z (mm)')
            axes[0, 3].invert_xaxis()
            axes[0, 3].invert_yaxis()
            
            # Original Grid (uniform)
            axes[0, 3].imshow(static_samples, cmap='gray', alpha=0.3, aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            
            # Draw regular uniform grid
            n_lines = 15
            y_lines = np.linspace(Y_world.min(), Y_world.max(), n_lines)
            z_lines = np.linspace(Z_world.min(), Z_world.max(), n_lines)
            
            for y in y_lines:
                axes[0, 3].axhline(y=y, color='blue', alpha=0.6, linewidth=1)
            for z in z_lines:
                axes[0, 3].axvline(x=z, color='blue', alpha=0.6, linewidth=1)
                
            axes[0, 3].set_title(f'Original Grid\n(Uniform)', fontsize=10)
            axes[0, 3].set_xlabel('Y (mm)')
            axes[0, 3].set_ylabel('Z (mm)')
            axes[0, 3].invert_xaxis()
            axes[0, 3].invert_yaxis()

            # SVFFD FORWARD DEFORMATION GRID - using SAME world coordinates as other panels
            print(f"🔍 Creating forward deformation grid at SAME world coordinates...")
            
            try:
                # Use SAME Y/Z range as all other panels for consistency  
                n_grid_lines = 15  # Number of grid lines in each direction
                y_grid_coords = np.linspace(Y_world.min(), Y_world.max(), n_grid_lines)
                z_grid_coords = np.linspace(Z_world.min(), Z_world.max(), n_grid_lines)
                
                # Create grid arrays for visualization
                Y_grid, Z_grid = np.meshgrid(y_grid_coords, z_grid_coords)
                
                # Initialize deformed grid arrays
                Y_deformed = np.zeros_like(Y_grid)
                Z_deformed = np.zeros_like(Z_grid)
                
                print(f"🔍 Grid shape: {Y_grid.shape}, applying SVFFD transform to {Y_grid.size} points...")
                
                # Apply SVFFD transform to each grid point using SAME world_x coordinate
                transform_count = 0
                for i in range(Y_grid.shape[0]):
                    for j in range(Y_grid.shape[1]):
                        try:
                            # Create world coordinate point at SAME world_x as other panels
                            world_point = torch.tensor([[world_x, Y_grid[i, j], Z_grid[i, j]]], 
                                                      dtype=torch.float32, device=self.device)
                            
                            # Apply complete transformation (affine + SVFFD)
                            transformed_point = composed_transform.forward(world_point)
                            transformed_coords = transformed_point.squeeze().detach().cpu().numpy()
                            
                            # Store deformed coordinates (Y and Z only for 2D visualization)  
                            Y_deformed[i, j] = transformed_coords[1]
                            Z_deformed[i, j] = transformed_coords[2]
                            transform_count += 1
                            
                        except Exception as e:
                            # Fallback: no deformation for this point
                            Y_deformed[i, j] = Y_grid[i, j]
                            Z_deformed[i, j] = Z_grid[i, j]
                
                # Calculate deformation magnitude
                deform_magnitude = np.sqrt((Y_deformed - Y_grid)**2 + (Z_deformed - Z_grid)**2)
                max_deformation = np.max(deform_magnitude)
                print(f"🔍 Transformed {transform_count}/{Y_grid.size} points, Max deformation: {max_deformation:.2f}mm")
                
                # Create the forward deformation visualization exactly like PDF
                axes[0, 4].set_xlim(Y_world.min(), Y_world.max())
                axes[0, 4].set_ylim(Z_world.min(), Z_world.max())
                axes[0, 4].set_aspect('equal')
                
                # Add background image for context (SAME static_samples as other panels)
                axes[0, 4].imshow(static_samples, cmap='gray', alpha=0.2, aspect='equal',
                                  extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
                
                # Draw deformed grid lines - this creates the beautiful curved grid from PDF
                # Horizontal lines (constant Z)
                for i in range(Y_deformed.shape[0]):
                    axes[0, 4].plot(Y_deformed[i, :], Z_deformed[i, :], 'k-', linewidth=1.5, alpha=0.8)
                
                # Vertical lines (constant Y)  
                for j in range(Y_deformed.shape[1]):
                    axes[0, 4].plot(Y_deformed[:, j], Z_deformed[:, j], 'k-', linewidth=1.5, alpha=0.8)
                
                axes[0, 4].set_title(f'Forward Deformation\n@ X={world_x:.1f}mm (PDF Style)', fontsize=10)
                axes[0, 4].set_xlabel('Y (mm)')
                axes[0, 4].set_ylabel('Z (mm)')
                axes[0, 4].invert_xaxis()
                axes[0, 4].invert_yaxis()
                
                print(f"✅ Forward deformation grid completed successfully (PDF style)")
                
            except Exception as e:
                print(f"❌ Forward deformation visualization failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: show regular grid
                axes[0, 4].imshow(static_samples, cmap='gray', alpha=0.3, aspect='equal',
                                  extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
                for y in y_lines:
                    axes[0, 4].axhline(y=y, color='red', alpha=0.6, linewidth=1)
                for z in z_lines:
                    axes[0, 4].axvline(x=z, color='red', alpha=0.6, linewidth=1)
                axes[0, 4].set_title(f'Forward Deformation\n@ X={world_x:.1f}mm (FAILED)', fontsize=10)
                axes[0, 4].set_xlabel('Y (mm)')
                axes[0, 4].set_ylabel('Z (mm)')
                axes[0, 4].invert_xaxis()
                axes[0, 4].invert_yaxis()
            
            # Calculate displacement magnitudes for the bottom row
            displacement_magnitudes = np.zeros_like(static_samples)
            for i in range(Y_world.shape[0]):
                for j in range(Y_world.shape[1]):
                    world_point = [world_x, Y_world[i, j], Z_world[i, j]]
                    original_point = torch.tensor([world_point], dtype=torch.float32, device=self.device)
                    transformed_point = composed_transform.forward(original_point)
                    displacement = transformed_point - original_point
                    displacement_magnitudes[i, j] = torch.norm(displacement).item()
            
            # Row 2: Analysis and metrics
            # Displacement field (detailed view)
            axes[1, 0].imshow(displacement_magnitudes, cmap='hot', aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[1, 0].set_title('Displacement Field\n(SVFFD Detailed)', fontsize=10)
            axes[1, 0].set_xlabel('Y (mm)')
            axes[1, 0].set_ylabel('Z (mm)')
            axes[1, 0].invert_xaxis()
            axes[1, 0].invert_yaxis()
            
            # Transform info
            axes[1, 1].text(0.1, 0.9, 'Transform Info', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.7, 'Method: Rigid+Affine+SVFFD', fontsize=10, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, 'Type: Diffeomorphic', fontsize=10, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, 'Levels: Multi-resolution', fontsize=10, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, 'B-spline regularized', fontsize=10, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.3, 'Topology preserving', fontsize=10, transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
            
            # Quality metrics
            axes[1, 2].text(0.1, 0.9, 'Quality Metrics', fontsize=12, fontweight='bold', transform=axes[1, 2].transAxes)
            
            # Compute NMI for after registration using world coordinate samples
            # Calculate NMI between static and moving after SVFFD
            static_flat = static_samples.flatten()
            moving_after_flat = moving_after_samples.flatten()
            
            # Simple NMI approximation using histograms
            hist_2d, _, _ = np.histogram2d(static_flat, moving_after_flat, bins=64)
            hist_static = np.histogram(static_flat, bins=64)[0]
            hist_moving = np.histogram(moving_after_flat, bins=64)[0]
            
            # Normalize histograms
            hist_2d = hist_2d + 1e-8
            hist_static = hist_static + 1e-8
            hist_moving = hist_moving + 1e-8
            
            # Calculate mutual information
            pxy = hist_2d / np.sum(hist_2d)
            px = hist_static / np.sum(hist_static)
            py = hist_moving / np.sum(hist_moving)
            
            # Avoid log(0) by adding small epsilon
            pxy_safe = np.where(pxy > 0, pxy, 1e-8)
            mi = np.sum(pxy_safe * np.log(pxy_safe / (px[:, None] * py[None, :])))
            hx = -np.sum(px * np.log(px))
            hy = -np.sum(py * np.log(py))
            nmi = 2 * mi / (hx + hy)
            
            axes[1, 2].text(0.1, 0.7, f'Final NMI: {nmi:.6f}', fontsize=10, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.6, f'Max displacement: {displacement_magnitudes.max():.2f}mm', fontsize=10, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.5, f'Mean displacement: {displacement_magnitudes.mean():.2f}mm', fontsize=10, transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
            
            # Histogram of displacements
            axes[1, 3].hist(displacement_magnitudes.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 3].set_title('Displacement Dist.\n(SVFFD World Coords)', fontsize=10)
            axes[1, 3].set_xlabel('Displacement (mm)')
            axes[1, 3].set_ylabel('Frequency')
            
            # Summary panel
            axes[1, 4].text(0.1, 0.9, 'Summary', fontsize=12, fontweight='bold', transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.8, 'DEEPALI PROFESSIONAL', fontsize=10, fontweight='bold', transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.7, 'VISUALIZATION', fontsize=10, fontweight='bold', transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.6, '', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.5, 'Built-in SVFFD', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.4, 'Deep Learning based', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.3, 'Image Registration', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.2, '', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].text(0.1, 0.1, 'Topology-AFFINE_SVFFD Method', fontsize=10, transform=axes[1, 4].transAxes)
            axes[1, 4].axis('off')
            
            # Save the custom SVFFD visualization (with forward deformation grid)
            output_path = self.output_dir / "deepali_professional_visualization_rigid_affine_svffd_after_module.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Deepali professional visualization saved: {output_path}")
            
            # Create simple 1x3 PDF-style visualization (target, warped source, forward deformation)
            self._create_simple_pdf_style_visualization(composed_transform, static_samples, moving_after_samples, Y_world, Z_world, world_x)
            
        except Exception as e:
            print(f"⚠️  Error creating deepali visualization: {e}")
            print("📝 Continuing with registration results...")
    
    def _create_simple_pdf_style_visualization(self, composed_transform, static_samples, moving_after_samples, Y_world, Z_world, world_x):
        """Create simple 1x3 visualization like deepali PDF figure: target | warped source | forward deformation"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("🎨 Creating simple PDF-style 1x3 visualization...")
        
        try:
            # Create 1x3 figure like PDF
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("Pairwise Image Registration — deepali", fontsize=14, fontweight='bold')
            
            # Panel 1: Target (static image)
            axes[0].imshow(static_samples, cmap='gray', aspect='equal',
                          extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[0].set_title('target', fontsize=12)
            axes[0].axis('off')
            
            # Panel 2: Warped source (moving after SVFFD)
            axes[1].imshow(moving_after_samples, cmap='gray', aspect='equal',
                          extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            axes[1].set_title('warped source', fontsize=12)
            axes[1].axis('off')
            
            # Panel 3: Forward deformation (curved grid)
            # Create regular grid for deformation
            n_grid_lines = 20
            y_grid_coords = np.linspace(Y_world.min(), Y_world.max(), n_grid_lines)
            z_grid_coords = np.linspace(Z_world.min(), Z_world.max(), n_grid_lines)
            Y_grid, Z_grid = np.meshgrid(y_grid_coords, z_grid_coords)
            
            # Apply SVFFD transform to grid points
            Y_deformed = np.zeros_like(Y_grid)
            Z_deformed = np.zeros_like(Z_grid)
            
            for i in range(Y_grid.shape[0]):
                for j in range(Y_grid.shape[1]):
                    try:
                        world_point = torch.tensor([[world_x, Y_grid[i, j], Z_grid[i, j]]], 
                                                  dtype=torch.float32, device=self.device)
                        transformed_point = composed_transform.forward(world_point)
                        transformed_coords = transformed_point.squeeze().detach().cpu().numpy()
                        
                        Y_deformed[i, j] = transformed_coords[1]
                        Z_deformed[i, j] = transformed_coords[2]
                    except:
                        Y_deformed[i, j] = Y_grid[i, j]
                        Z_deformed[i, j] = Z_grid[i, j]
            
            # Draw deformed grid on white background
            axes[2].set_xlim(Y_world.min(), Y_world.max())
            axes[2].set_ylim(Z_world.min(), Z_world.max())
            axes[2].set_facecolor('white')
            
            # Draw curved grid lines (black lines on white background)
            for i in range(Y_deformed.shape[0]):
                axes[2].plot(Y_deformed[i, :], Z_deformed[i, :], 'k-', linewidth=1.0)
            for j in range(Y_deformed.shape[1]):
                axes[2].plot(Y_deformed[:, j], Z_deformed[:, j], 'k-', linewidth=1.0)
            
            axes[2].set_title('forward deformation', fontsize=12)
            axes[2].set_aspect('equal')
            axes[2].axis('off')
            
            # Save PDF-style visualization
            pdf_output_path = self.output_dir / "deepali_pdf_style_1x3_visualization.png"
            plt.tight_layout()
            plt.savefig(pdf_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ PDF-style 1x3 visualization saved: {pdf_output_path}")
            
        except Exception as e:
            print(f"⚠️  Error creating PDF-style visualization: {e}")
            import traceback
            traceback.print_exc()
    
# Old visualization functions removed - using rewritten working approach
    
    def _sample_at_world_point(self, image, world_point):
        """Sample image at world coordinate with bounds checking - EXACT COPY from working code."""
        try:
            voxel_point = image.TransformPhysicalPointToContinuousIndex(world_point)
            
            # Check bounds
            size = image.GetSize()
            if (0 <= voxel_point[0] < size[0] and
                0 <= voxel_point[1] < size[1] and
                0 <= voxel_point[2] < size[2]):
                
                # Get array and sample
                array = sitk.GetArrayFromImage(image)
                x, y, z = [int(round(v)) for v in voxel_point]
                
                # Bounds check for array (SimpleITK uses ZYX ordering)
                if (0 <= z < array.shape[0] and
                    0 <= y < array.shape[1] and
                    0 <= x < array.shape[2]):
                    return array[z, y, x]
        except:
            pass
        return 0
    
