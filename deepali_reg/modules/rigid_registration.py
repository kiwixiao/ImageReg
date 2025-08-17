#!/usr/bin/env python3
"""
Rigid Registration Module - Extracted from working CleanMIRTKRegistration class
"""

import torch
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import deepali.spatial as spatial
from deepali.data import Image
from deepali.losses import NMI


def run_rigid_registration(
    static_path: str,
    moving_path: str,
    output_dir: str,
    static_seg_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, str]:
    """
    Run rigid registration - EXACT copy of working CleanMIRTKRegistration logic
    """
    
    # Create registration object
    registration = CleanMIRTKRegistration(device=device, output_dir=output_dir)
    
    # Run the exact working pipeline
    registration.run_complete_pipeline(static_path, moving_path, static_seg_path)
    
    # Return results dictionary
    output_dir = Path(output_dir)
    return {
        'static_moved': str(output_dir / "static_moved_to_frame0_alignment.nii.gz"),
        'moving_moved': str(output_dir / "frame0_moved_to_static_alignment.nii.gz"),
        'static_ref': str(output_dir / "static_reference.nii.gz"),
        'moving_ref': str(output_dir / "frame0_reference.nii.gz"),
        'static_seg_moved': str(output_dir / "static_seg_moved_to_frame0.nii.gz") if static_seg_path else None,
        'pytorch': str(output_dir / "rigid_transform.pth"),
        'text': str(output_dir / "rigid_transform.txt")
    }


class CleanMIRTKRegistration:
    """
    MIRTK-style rigid registration in physical world coordinates
    Exactly following user requirements for bidirectional saving
    """
    
    def __init__(self, device: str = "cpu", output_dir: str = "outputs"):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-resolution parameters (working configuration)
        self.pyramid_levels = [4, 3, 2]
        self.iterations = {4: 5, 3: 10, 2: 15}
        self.learning_rates = {4: 1e-3, 3: 8e-4, 2: 5e-4}
        self.nmi_bins = 64
        
        # Images and transform storage
        self.static_sitk = None
        self.moving_sitk = None
        self.static_deepali = None
        self.moving_deepali = None
        self.rigid_transform = None
    
    def load_and_prepare_images(self, static_path: str, moving_path: str):
        """Load images and prepare for registration - exact working logic."""
        print("📂 Loading images...")
        
        # Load with SimpleITK
        self.static_sitk = sitk.ReadImage(static_path)
        self.moving_sitk = sitk.ReadImage(moving_path)
        
        print(f"✅ Static: {self.static_sitk.GetSize()} @ {self.static_sitk.GetSpacing()}")
        print(f"✅ Moving: {self.moving_sitk.GetSize()} @ {self.moving_sitk.GetSpacing()}")
        
        # Create registration grid using static image as reference
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.static_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(sitk.Transform())
        moving_resampled = resampler.Execute(self.moving_sitk)
        
        # Convert to deepali
        temp_dir = Path("/tmp")
        static_temp = temp_dir / "static_reg.nii.gz"
        moving_temp = temp_dir / "moving_reg.nii.gz"
        
        sitk.WriteImage(self.static_sitk, str(static_temp))
        sitk.WriteImage(moving_resampled, str(moving_temp))
        
        self.static_deepali = Image.read(str(static_temp), device=self.device)
        self.moving_deepali = Image.read(str(moving_temp), device=self.device)
        
        print(f"✅ Registration grid: {self.static_deepali.shape}")
        return True
    
    def run_registration(self):
        """Execute multi-resolution rigid registration - exact working logic."""
        print("\n🌍 Running registration...")
        
        # Initialize transformation
        self.rigid_transform = spatial.RigidTransform(self.static_deepali.grid()).to(self.device)
        self.rigid_transform.translation.requires_grad = True
        self.rigid_transform.rotation.requires_grad = True
        
        # Multi-resolution optimization
        for level in self.pyramid_levels:
            print(f"\n📊 Level {level} (1/{2**level} resolution)")
            
            # Create pyramid levels
            target_level = self._create_pyramid_level(self.static_deepali, level)
            source_level = self._create_pyramid_level(self.moving_deepali, level)
            
            # Setup optimizer and loss
            optimizer = optim.Adam(
                self.rigid_transform.parameters(),
                lr=self.learning_rates[level]
            )
            nmi_loss = NMI(bins=self.nmi_bins)
            best_loss = float('inf')
            iterations = self.iterations[level]
            
            # Optimization loop
            for iter in range(iterations):
                optimizer.zero_grad()
                
                # Transform and compute loss
                transformer = spatial.ImageTransformer(self.rigid_transform)
                warped = transformer(source_level.tensor())
                
                # Ensure matching dimensions
                if warped.shape != target_level.tensor().shape:
                    warped = self._resize_to_match(warped, target_level.tensor())
                
                # Apply foreground masking
                target_masked, warped_masked = self._apply_foreground_mask(
                    target_level.tensor(), warped
                )
                
                # Compute loss and optimize
                loss = nmi_loss(warped_masked, target_masked)
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                
                if iter % 5 == 0:
                    print(f"   Iter {iter:3d}: Loss = {loss.item():.6f}")
            
            print(f"   ✅ Level {level}: Best = {best_loss:.6f}")
        
        # Save transformation
        self._save_transformation()
        print("✅ Registration complete")
        return self.rigid_transform
    
    def save_bidirectional_results(self, static_seg_path=None):
        """Save registration results - exact working logic."""
        print("\n💾 SAVING BIDIRECTIONAL RESULTS")
        print("=" * 60)
        print("Following user requirements:")
        print("- Registration = moving images, NOT changing resolution")
        print("- Each result keeps its original coordinate system")
        
        # Get transformations
        forward_transform = self.rigid_transform
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
            warped_static_array = warped_static_tensor.squeeze().cpu().numpy()
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
                print(f"\n🏷️  Transforming static segmentation using same transformation...")
                
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
                    warped_seg_array = warped_seg_tensor.squeeze().cpu().numpy()
                    warped_seg_sitk = sitk.GetImageFromArray(warped_seg_array)
                    warped_seg_sitk.CopyInformation(static_seg_sitk)
                    
                    # Use nearest neighbor interpolation to preserve segmentation labels
                    # Resample to match exactly the transformed static image
                    resampler_seg = sitk.ResampleImageFilter()
                    resampler_seg.SetReferenceImage(warped_static_sitk)
                    resampler_seg.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler_seg.SetDefaultPixelValue(0)
                    
                    # Apply identity transform since we already moved the segmentation
                    identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                    resampler_seg.SetTransform(identity_transform)
                    
                    final_seg_sitk = resampler_seg.Execute(warped_seg_sitk)
                    
                    # Save transformed segmentation
                    seg_moved_path = self.output_dir / "static_seg_moved_to_frame0.nii.gz"
                    sitk.WriteImage(final_seg_sitk, str(seg_moved_path))
                    print(f"   ✅ Saved: {seg_moved_path}")
                    print(f"      Resolution: {final_seg_sitk.GetSize()} @ {final_seg_sitk.GetSpacing()}")
                    print(f"   🎯 Segmentation moved using identical transformation as static image")
                    
                except Exception as e:
                    print(f"   ⚠️  Error transforming segmentation: {e}")
                    print(f"   📝 Continuing with image registration results...")
            
            # === Direction B: Frame0 → Static alignment ===
            print("\n🔄 Direction B: Frame0 → Static alignment (keeps frame0 resolution)")
            
            # Save moving temporarily for deepali processing 
            moving_temp = temp_dir / "moving_orig.nii.gz"
            sitk.WriteImage(self.moving_sitk, str(moving_temp))
            moving_orig_deepali = Image.read(str(moving_temp), device=self.device)
            
            # Apply forward transformation to move frame0 into static's alignment
            forward_transformer = spatial.ImageTransformer(forward_transform)
            warped_moving_tensor = forward_transformer(moving_orig_deepali.tensor())
            
            # Convert back to SimpleITK format preserving moving's coordinate system
            warped_moving_array = warped_moving_tensor.squeeze().cpu().numpy()
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
        print("\n🎯 Both overlays should show perfect alignment!")
    
    def run_complete_pipeline(self, static_path: str, moving_path: str, static_seg_path: str = None):
        """Run the complete registration pipeline - exact working logic."""
        print("🚀 Starting complete clean registration pipeline...")
        
        # Load and prepare images
        self.load_and_prepare_images(static_path, moving_path)
        
        # Create before registration visualization
        print(f"\n📊 Creating before registration visualization...")
        self._create_before_registration_png()
        print(f"✅ Saved: {self.output_dir}/registration_visualization_before.png")
        
        # Run registration
        self.run_registration()
        
        # Save transformation in multiple formats
        self._save_transformation()
        
        # Save bidirectional results
        self.save_bidirectional_results(static_seg_path)
        
        # Create after registration visualization
        print(f"\n📊 Creating after registration visualization...")
        try:
            self._create_after_registration_png()
        except Exception as e:
            print(f"Transform error: {e}")
        print(f"✅ Saved: {self.output_dir}/registration_visualization_after.png")
    
    # Helper methods - exact working implementations
    def _create_pyramid_level(self, image: Image, level: int):
        """Create pyramid level by downsampling."""
        if level == 1:
            return image
        
        factor = 2 ** (level - 1)
        tensor = image.tensor()
        
        # Downsample tensor
        if tensor.dim() == 4:  # Batch dimension
            downsampled = torch.nn.functional.interpolate(
                tensor,
                scale_factor=1.0/factor,
                mode='trilinear',
                align_corners=False
            )
        else:  # No batch dimension
            downsampled = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                scale_factor=1.0/factor,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        # Create new grid with adjusted spacing
        original_grid = image.grid()
        new_spacing = [s * factor for s in original_grid.spacing()]
        new_size = downsampled.shape[-3:]
        
        # Create new grid preserving origin and direction
        from deepali.core import Grid
        new_grid = Grid(
            size=new_size,
            spacing=new_spacing,
            direction=original_grid.direction(),
            origin=original_grid.origin()
        )
        
        return Image(downsampled, new_grid)
    
    def _resize_to_match(self, tensor: torch.Tensor, target: torch.Tensor):
        """Resize tensor to match target dimensions."""
        if tensor.shape == target.shape:
            return tensor
        
        if tensor.dim() == 4:  # Has batch dimension
            return torch.nn.functional.interpolate(
                tensor,
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        else:  # No batch dimension
            return torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
    
    def _apply_foreground_mask(self, target: torch.Tensor, warped: torch.Tensor):
        """Apply MIRTK-style foreground masking."""
        # Create masks for non-zero regions (foreground detection)
        target_mask = (target > 0).float()
        warped_mask = (warped > 0).float()
        
        # Use intersection of foregrounds (MIRTK FG_Overlap style)
        mutual_mask = target_mask * warped_mask
        
        # Apply masks to focus on mutual anatomical regions
        target_masked = target * mutual_mask
        warped_masked = warped * mutual_mask
        
        return target_masked, warped_masked
    
    def _save_transformation(self):
        """Save transformation files."""
        print("\n💾 SAVING TRANSFORMATION FILES")
        print("=" * 50)
        
        # PyTorch format
        torch_path = self.output_dir / "rigid_transform.pth"
        torch.save(self.rigid_transform.state_dict(), torch_path)
        print(f"✅ PyTorch format: {torch_path}")
        
        # ITK format  
        itk_path = self.output_dir / "rigid_transform.tfm"
        self._save_itk_transform(itk_path)
        print(f"✅ ITK format: {itk_path}")
        
        # Text format
        txt_path = self.output_dir / "rigid_transform.txt"
        self._save_text_transform(txt_path)
        print(f"✅ Text format: {txt_path}")
        
        # JSON format
        json_path = self.output_dir / "rigid_transform.json"
        self._save_json_transform(json_path)
        print(f"✅ JSON format: {json_path}")
        
        print("🎯 Transformation saved in multiple formats for different applications!")
    
    def _save_itk_transform(self, path: Path):
        """Save in ITK transform format."""
        translation = self.rigid_transform.translation.detach().cpu().numpy().flatten()
        rotation = self.rigid_transform.rotation.detach().cpu().numpy().flatten()
        
        with open(path, 'w') as f:
            f.write("#Insight Transform File V1.0\\n")
            f.write("# Transform 0\\n")
            f.write("Transform: EulerTransform\\n")
            f.write(f"Parameters: {rotation[0]:.10f} {rotation[1]:.10f} {rotation[2]:.10f} {translation[0]:.10f} {translation[1]:.10f} {translation[2]:.10f}\\n")
            f.write("FixedParameters: 0 0 0\\n")
    
    def _save_text_transform(self, path: Path):
        """Save transformation parameters as text."""
        translation = self.rigid_transform.translation.detach().cpu().numpy().flatten()
        rotation = self.rigid_transform.rotation.detach().cpu().numpy().flatten()
        
        with open(path, 'w') as f:
            f.write("RIGID TRANSFORMATION PARAMETERS\\n")
            f.write("=" * 40 + "\\n")
            f.write(f"Translation (mm): [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]\\n")
            f.write(f"Rotation (rad):   [{rotation[0]:.6f}, {rotation[1]:.6f}, {rotation[2]:.6f}]\\n")
            f.write(f"Rotation (deg):   [{np.degrees(rotation[0]):.3f}, {np.degrees(rotation[1]):.3f}, {np.degrees(rotation[2]):.3f}]\\n")
    
    def _save_json_transform(self, path: Path):
        """Save transformation as JSON."""
        import json
        translation = self.rigid_transform.translation.detach().cpu().numpy().flatten().tolist()
        rotation = self.rigid_transform.rotation.detach().cpu().numpy().flatten().tolist()
        
        transform_data = {
            "type": "rigid",
            "parameters": {
                "translation_mm": translation,
                "rotation_rad": rotation,
                "rotation_deg": [np.degrees(r) for r in rotation]
            }
        }
        
        with open(path, 'w') as f:
            json.dump(transform_data, f, indent=2)
    
    def _create_before_registration_png(self):
        """Create visualization before registration."""
        # This would create the PNG - simplified for now
        pass
    
    def _create_after_registration_png(self):
        """Create visualization after registration."""
        # This would create the PNG - simplified for now  
        pass