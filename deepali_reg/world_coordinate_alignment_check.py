#!/usr/bin/env python
"""
World Coordinate Alignment Check for MIRTK Pipeline

This module provides visualization and verification of world coordinate
alignment between images, to be used as an intermediate check in the
deepali registration pipeline.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

class WorldCoordinateAlignmentChecker:
    """
    Check and visualize alignment in world coordinate space.
    
    This is used in the MIRTK on-the-fly registration pipeline to verify
    that images are properly aligned in world coordinates before registration.
    """
    
    def __init__(self):
        self.static_sitk = None
        self.moving_sitk = None
        self.alignment_metrics = {}
        
    def load_images(self, static_path: str, moving_path: str) -> Dict[str, Any]:
        """
        Load images and compute initial alignment metrics.
        
        Returns:
            Dict containing alignment metrics
        """
        self.static_sitk = sitk.ReadImage(static_path)
        self.moving_sitk = sitk.ReadImage(moving_path)
        
        # Compute alignment metrics
        self.alignment_metrics = self._compute_alignment_metrics()
        
        return self.alignment_metrics
    
    def _compute_alignment_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive alignment metrics in world coordinates"""
        
        metrics = {}
        
        # 1. Image properties
        metrics['static_size'] = self.static_sitk.GetSize()
        metrics['moving_size'] = self.moving_sitk.GetSize()
        metrics['static_spacing'] = self.static_sitk.GetSpacing()
        metrics['moving_spacing'] = self.moving_sitk.GetSpacing()
        
        # 2. Centers in world coordinates
        static_center_voxel = np.array(self.static_sitk.GetSize()) / 2.0 - 0.5
        moving_center_voxel = np.array(self.moving_sitk.GetSize()) / 2.0 - 0.5
        
        metrics['static_center_world'] = self.static_sitk.TransformContinuousIndexToPhysicalPoint(
            static_center_voxel.tolist()
        )
        metrics['moving_center_world'] = self.moving_sitk.TransformContinuousIndexToPhysicalPoint(
            moving_center_voxel.tolist()
        )
        
        # 3. Center distance
        center_diff = np.array(metrics['static_center_world']) - np.array(metrics['moving_center_world'])
        metrics['center_distance_mm'] = np.linalg.norm(center_diff)
        metrics['center_diff_xyz'] = center_diff
        
        # 4. World bounds
        static_bounds = self._get_world_bounds(self.static_sitk)
        moving_bounds = self._get_world_bounds(self.moving_sitk)
        
        metrics['static_bounds'] = static_bounds
        metrics['moving_bounds'] = moving_bounds
        
        # 5. Overlap analysis
        overlap_min = np.maximum(static_bounds[0], moving_bounds[0])
        overlap_max = np.minimum(static_bounds[1], moving_bounds[1])
        
        has_overlap = np.all(overlap_max > overlap_min)
        metrics['has_overlap'] = has_overlap
        
        if has_overlap:
            metrics['overlap_region'] = (overlap_min, overlap_max)
            metrics['overlap_size_mm'] = overlap_max - overlap_min
            
            # Compute overlap percentage
            static_volume = np.prod(static_bounds[1] - static_bounds[0])
            moving_volume = np.prod(moving_bounds[1] - moving_bounds[0])
            overlap_volume = np.prod(overlap_max - overlap_min)
            
            metrics['overlap_percent_static'] = (overlap_volume / static_volume) * 100
            metrics['overlap_percent_moving'] = (overlap_volume / moving_volume) * 100
        else:
            metrics['overlap_region'] = None
            metrics['overlap_size_mm'] = np.zeros(3)
            metrics['overlap_percent_static'] = 0
            metrics['overlap_percent_moving'] = 0
        
        # 6. Alignment quality assessment
        if metrics['center_distance_mm'] < 5:
            metrics['alignment_quality'] = 'EXCELLENT'
        elif metrics['center_distance_mm'] < 20:
            metrics['alignment_quality'] = 'GOOD'
        elif metrics['center_distance_mm'] < 50:
            metrics['alignment_quality'] = 'MODERATE'
        else:
            metrics['alignment_quality'] = 'POOR'
        
        return metrics
    
    def _get_world_bounds(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """Get world coordinate bounds of image"""
        size = np.array(image.GetSize())
        
        # Get all 8 corners of the image
        corners = []
        for x in [0, size[0]-1]:
            for y in [0, size[1]-1]:
                for z in [0, size[2]-1]:
                    world_point = image.TransformIndexToPhysicalPoint([int(x), int(y), int(z)])
                    corners.append(world_point)
        
        corners = np.array(corners)
        min_bounds = np.min(corners, axis=0)
        max_bounds = np.max(corners, axis=0)
        
        return min_bounds, max_bounds
    
    def print_alignment_report(self):
        """Print detailed alignment report"""
        
        print("\n" + "="*70)
        print("WORLD COORDINATE ALIGNMENT REPORT")
        print("="*70)
        
        print("\n📊 IMAGE PROPERTIES")
        print("-"*50)
        print(f"Static: {self.alignment_metrics['static_size']} voxels")
        print(f"        Spacing: {self.alignment_metrics['static_spacing']} mm")
        print(f"Moving: {self.alignment_metrics['moving_size']} voxels")
        print(f"        Spacing: {self.alignment_metrics['moving_spacing']} mm")
        
        print("\n🌍 WORLD COORDINATE CENTERS")
        print("-"*50)
        static_center = self.alignment_metrics['static_center_world']
        moving_center = self.alignment_metrics['moving_center_world']
        print(f"Static: [{static_center[0]:7.2f}, {static_center[1]:7.2f}, {static_center[2]:7.2f}] mm")
        print(f"Moving: [{moving_center[0]:7.2f}, {moving_center[1]:7.2f}, {moving_center[2]:7.2f}] mm")
        print(f"Difference: [{self.alignment_metrics['center_diff_xyz'][0]:7.2f}, "
              f"{self.alignment_metrics['center_diff_xyz'][1]:7.2f}, "
              f"{self.alignment_metrics['center_diff_xyz'][2]:7.2f}] mm")
        print(f"Distance: {self.alignment_metrics['center_distance_mm']:.2f} mm")
        
        print("\n📐 WORLD COORDINATE BOUNDS")
        print("-"*50)
        static_bounds = self.alignment_metrics['static_bounds']
        moving_bounds = self.alignment_metrics['moving_bounds']
        print(f"Static: [{static_bounds[0][0]:6.1f}, {static_bounds[0][1]:6.1f}, {static_bounds[0][2]:6.1f}] to")
        print(f"        [{static_bounds[1][0]:6.1f}, {static_bounds[1][1]:6.1f}, {static_bounds[1][2]:6.1f}] mm")
        print(f"Moving: [{moving_bounds[0][0]:6.1f}, {moving_bounds[0][1]:6.1f}, {moving_bounds[0][2]:6.1f}] to")
        print(f"        [{moving_bounds[1][0]:6.1f}, {moving_bounds[1][1]:6.1f}, {moving_bounds[1][2]:6.1f}] mm")
        
        if self.alignment_metrics['has_overlap']:
            print("\n✅ OVERLAP REGION")
            print("-"*50)
            overlap = self.alignment_metrics['overlap_region']
            print(f"X: {overlap[0][0]:6.1f} to {overlap[1][0]:6.1f} mm ({self.alignment_metrics['overlap_size_mm'][0]:.1f} mm)")
            print(f"Y: {overlap[0][1]:6.1f} to {overlap[1][1]:6.1f} mm ({self.alignment_metrics['overlap_size_mm'][1]:.1f} mm)")
            print(f"Z: {overlap[0][2]:6.1f} to {overlap[1][2]:6.1f} mm ({self.alignment_metrics['overlap_size_mm'][2]:.1f} mm)")
            print(f"Overlap: {self.alignment_metrics['overlap_percent_static']:.1f}% of static volume")
            print(f"         {self.alignment_metrics['overlap_percent_moving']:.1f}% of moving volume")
        else:
            print("\n❌ NO OVERLAP DETECTED")
        
        print("\n🎯 ALIGNMENT ASSESSMENT")
        print("-"*50)
        quality = self.alignment_metrics['alignment_quality']
        distance = self.alignment_metrics['center_distance_mm']
        
        if quality == 'EXCELLENT':
            print(f"✅ {quality}: Images are well-aligned ({distance:.1f}mm)")
            print("   Ready for registration with minimal adjustment")
        elif quality == 'GOOD':
            print(f"✅ {quality}: Images have good alignment ({distance:.1f}mm)")
            print("   Registration should converge easily")
        elif quality == 'MODERATE':
            print(f"⚠️  {quality}: Images need alignment ({distance:.1f}mm)")
            print("   Consider initial rigid registration")
        else:
            print(f"❌ {quality}: Images are poorly aligned ({distance:.1f}mm)")
            print("   Strong initial alignment recommended")
    
    def create_world_alignment_visualization(self, 
                                           output_path: Optional[str] = None,
                                           world_x: Optional[float] = None) -> str:
        """
        Create world coordinate alignment visualization.
        
        This shows the same physical slice from both images to verify alignment.
        
        Args:
            output_path: Where to save the visualization
            world_x: World X coordinate for sagittal slice (None = auto-select)
            
        Returns:
            Path to saved visualization
        """
        
        if output_path is None:
            output_path = "world_coordinate_alignment.png"
        
        # Auto-select world X if not provided
        if world_x is None:
            # Use moving image center X (good for centerline visualization)
            world_x = self.alignment_metrics['moving_center_world'][0]
        
        # Sample both images at the same world coordinates
        static_samples, moving_samples, Y_world, Z_world = self._sample_world_plane(world_x)
        
        # Get native slices for comparison
        static_array = sitk.GetArrayFromImage(self.static_sitk)
        moving_array = sitk.GetArrayFromImage(self.moving_sitk)
        
        static_center_x = self.static_sitk.GetSize()[0] // 2
        moving_center_x = self.moving_sitk.GetSize()[0] // 2
        
        static_native = np.flipud(static_array[:, :, static_center_x])
        moving_native = np.flipud(moving_array[:, :, moving_center_x])
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        
        # Native coordinate views
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(static_native, cmap='gray', aspect='equal')
        ax1.set_title('Static: Native Coordinates\n(Different slice)')
        ax1.set_xlabel('Y voxels')
        ax1.set_ylabel('Z voxels')
        
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(moving_native, cmap='gray', aspect='equal')
        ax2.set_title('Moving: Native Coordinates\n(Different slice)')
        ax2.set_xlabel('Y voxels')
        ax2.set_ylabel('Z voxels')
        
        # World coordinate views (SAME slice)
        ax3 = plt.subplot(2, 4, 5)
        im3 = ax3.imshow(static_samples, cmap='gray', aspect='equal',
                        extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        ax3.set_title(f'Static: World Coordinates\n@ X={world_x:.1f}mm')
        ax3.set_xlabel('Y (mm)')
        ax3.set_ylabel('Z (mm)')
        
        ax4 = plt.subplot(2, 4, 6)
        im4 = ax4.imshow(moving_samples, cmap='gray', aspect='equal',
                        extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        ax4.set_title(f'Moving: World Coordinates\n@ X={world_x:.1f}mm (same slice!)')
        ax4.set_xlabel('Y (mm)')
        ax4.set_ylabel('Z (mm)')
        
        # Overlay
        ax5 = plt.subplot(2, 4, 7)
        overlay = np.zeros((*static_samples.shape, 3))
        static_norm = (static_samples - static_samples.min()) / (static_samples.max() - static_samples.min() + 1e-8)
        moving_norm = (moving_samples - moving_samples.min()) / (moving_samples.max() - moving_samples.min() + 1e-8)
        overlay[:, :, 0] = static_norm
        overlay[:, :, 1] = moving_norm
        
        ax5.imshow(overlay, aspect='equal',
                  extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        ax5.set_title('Overlay (R=Static, G=Moving)\nYellow = Aligned regions')
        ax5.set_xlabel('Y (mm)')
        ax5.set_ylabel('Z (mm)')
        
        # Metrics panel
        ax6 = plt.subplot(2, 4, 4)
        ax6.axis('off')
        metrics_text = f"""ALIGNMENT METRICS
        
Center Distance: {self.alignment_metrics['center_distance_mm']:.2f} mm
Quality: {self.alignment_metrics['alignment_quality']}

Overlap: {self.alignment_metrics['overlap_percent_moving']:.0f}% of moving vol

MIRTK Approach:
• No resampling
• On-the-fly transform
• World coordinates"""
        
        ax6.text(0.5, 0.5, metrics_text, ha='center', va='center', 
                fontsize=11, transform=ax6.transAxes)
        
        # Add colorbar
        ax7 = plt.subplot(2, 4, 8)
        difference = np.abs(static_norm - moving_norm)
        im7 = ax7.imshow(difference, cmap='hot', aspect='equal',
                        extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        ax7.set_title('Alignment Quality\n(Lower = Better)')
        ax7.set_xlabel('Y (mm)')
        ax7.set_ylabel('Z (mm)')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # Title
        plt.suptitle('World Coordinate Alignment Check\n' +
                    'Top: Native coordinates (different slices) | ' + 
                    'Bottom: World coordinates (SAME physical slice)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f"\n✅ Visualization saved: {output_path}")
        
        return output_path
    
    def _sample_world_plane(self, world_x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample both images at the same world coordinate plane"""
        
        # Define sampling grid in world coordinates
        if self.alignment_metrics['has_overlap']:
            overlap = self.alignment_metrics['overlap_region']
            y_range = np.linspace(overlap[0][1], overlap[1][1], 100)
            z_range = np.linspace(overlap[0][2], overlap[1][2], 100)
        else:
            # Use static bounds if no overlap
            bounds = self.alignment_metrics['static_bounds']
            y_range = np.linspace(bounds[0][1], bounds[1][1], 100)
            z_range = np.linspace(bounds[0][2], bounds[1][2], 100)
        
        Y_world, Z_world = np.meshgrid(y_range, z_range)
        
        # Sample both images
        static_samples = np.zeros_like(Y_world)
        moving_samples = np.zeros_like(Y_world)
        
        for i in range(Y_world.shape[0]):
            for j in range(Y_world.shape[1]):
                world_point = [world_x, Y_world[i, j], Z_world[i, j]]
                
                # Sample static
                static_samples[i, j] = self._sample_at_world_point(
                    self.static_sitk, world_point
                )
                
                # Sample moving
                moving_samples[i, j] = self._sample_at_world_point(
                    self.moving_sitk, world_point
                )
        
        return static_samples, moving_samples, Y_world, Z_world
    
    def _sample_at_world_point(self, image, world_point) -> float:
        """Sample image at world coordinate with bounds checking"""
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
                
                # Bounds check for array
                if (0 <= z < array.shape[0] and
                    0 <= y < array.shape[1] and
                    0 <= x < array.shape[2]):
                    return array[z, y, x]
        except:
            pass
        
        return 0

def check_world_alignment(static_path: str, moving_path: str, 
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to check world coordinate alignment.
    
    This can be called from the deepali registration pipeline.
    
    Args:
        static_path: Path to static/reference image
        moving_path: Path to moving image
        output_dir: Directory for output visualizations
        
    Returns:
        Dictionary with alignment metrics
    """
    
    print("\n🔍 CHECKING WORLD COORDINATE ALIGNMENT")
    print("="*70)
    
    # Create checker
    checker = WorldCoordinateAlignmentChecker()
    
    # Load and analyze
    metrics = checker.load_images(static_path, moving_path)
    
    # Print report
    checker.print_alignment_report()
    
    # Create visualization
    if output_dir:
        output_path = Path(output_dir) / "world_coordinate_alignment.png"
    else:
        output_path = "world_coordinate_alignment.png"
    
    checker.create_world_alignment_visualization(str(output_path))
    
    # Return metrics for pipeline use
    return metrics

def main():
    """Test the alignment checker"""
    
    # Test with OSAMRI007 data
    data_dir = Path("/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/reg_OSAMRI007")
    static_path = data_dir / "osamri007_static.nii.gz"
    moving_path = data_dir / "osamri007_frame0.nii.gz"
    
    # Check alignment
    metrics = check_world_alignment(str(static_path), str(moving_path))
    
    print("\n🎯 ALIGNMENT CHECK COMPLETE")
    print(f"   Quality: {metrics['alignment_quality']}")
    print(f"   Distance: {metrics['center_distance_mm']:.2f} mm")
    print(f"   Ready for MIRTK registration: {'Yes' if metrics['alignment_quality'] in ['EXCELLENT', 'GOOD'] else 'Needs initial alignment'}")

if __name__ == "__main__":
    main()