"""
SVFFD Visualization - Complete Rewrite
Using the EXACT SAME approach as the working pre-registration visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path


def create_working_svffd_visualization(static_sitk, moving_sitk, composed_transform, output_dir, device):
    """
    Create visualization using the EXACT SAME approach as the working pre-registration
    The key insight: Use the saved bidirectional results (SimpleITK images), not applying transforms
    """
    print("🎨 Creating SVFFD visualization using WORKING approach...")
    
    try:
        # CRITICAL: Use the saved bidirectional results that were just created
        # These are SimpleITK images that can be sampled properly
        static_moved_path = output_dir / "static_moved_to_frame0_alignment.nii.gz"
        frame0_moved_path = output_dir / "frame0_moved_to_static_alignment.nii.gz"
        static_ref_path = output_dir / "static_reference.nii.gz"
        frame0_ref_path = output_dir / "frame0_reference.nii.gz"
        
        # Load the WORKING results (these are guaranteed to be correct)
        static_ref_sitk = sitk.ReadImage(str(static_ref_path))
        frame0_ref_sitk = sitk.ReadImage(str(frame0_ref_path))
        static_moved_sitk = sitk.ReadImage(str(static_moved_path))
        frame0_moved_sitk = sitk.ReadImage(str(frame0_moved_path))
        
        print("✅ Loaded bidirectional results for visualization")
        
        # Create figure following the WORKING world coordinate approach
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle("DEEPALI Professional Visualization - RIGID_AFFINE_SVFFD (after)", fontsize=16, fontweight='bold')
        
        # Use the SAME world coordinate sampling approach as the working visualization
        # Get world coordinate for visualization (use center of static)
        static_center = static_ref_sitk.TransformIndexToPhysicalPoint(
            [s//2 for s in static_ref_sitk.GetSize()]
        )
        world_x = static_center[0]
        
        # Sample images at same world coordinate plane - EXACT COPY of working approach
        static_samples, moving_after_samples, Y_world, Z_world = sample_world_plane_working_approach(
            world_x, static_ref_sitk, frame0_moved_sitk
        )
        
        # Also sample the original moving for comparison
        _, moving_before_samples, _, _ = sample_world_plane_working_approach(
            world_x, static_ref_sitk, frame0_ref_sitk
        )
        
        print(f"✅ World coordinate sampling completed")
        print(f"   Static range: [{static_samples.min():.1f}, {static_samples.max():.1f}]")
        print(f"   Moving before range: [{moving_before_samples.min():.1f}, {moving_before_samples.max():.1f}]")  
        print(f"   Moving after range: [{moving_after_samples.min():.1f}, {moving_after_samples.max():.1f}]")
        
        # Check if all images are valid (not all zeros)
        if static_samples.max() == 0 or moving_after_samples.max() == 0:
            raise ValueError("Sampled images are all zeros - coordinate sampling failed")
        
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

        # SVFFD FORWARD DEFORMATION GRID - using SAME world coordinates as other panels
        print(f"🔍 Creating forward deformation grid at SAME world coordinates...")
        
        try:
            # Use SAME Y/Z range as all other panels for consistency  
            n_grid = 15
            grid_y = np.linspace(Y_world.min(), Y_world.max(), n_grid)
            grid_z = np.linspace(Z_world.min(), Z_world.max(), n_grid)
            
            # Create grid points in 3D world coordinates (same X as visualization)
            grid_points_list = []
            for y in grid_y:
                for z in grid_z:
                    grid_points_list.append([world_x, y, z])
            
            # Convert to tensor for deepali
            grid_points = torch.tensor(grid_points_list, dtype=torch.float32, device=device)
            
            # Apply SVFFD transform using deepali's forward method
            print(f"🔍 Grid shape: {(len(grid_y), len(grid_z))}, applying SVFFD transform to {len(grid_points)} points...")
            with torch.no_grad():
                transformed_points = composed_transform.forward(grid_points)
                transformed_points = transformed_points.cpu().numpy()
            
            # Calculate displacement magnitudes
            original_points = np.array(grid_points_list)
            displacement_vectors = transformed_points - original_points
            displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)
            
            print(f"🔍 Transformed {len(transformed_points)}/{len(grid_points)} points, Max deformation: {displacement_magnitudes.max():.2f}mm")
            
            # Draw on axes[0, 4] - forward deformation
            # Background image (static) for context
            axes[0, 4].imshow(static_samples, cmap='gray', alpha=0.3, aspect='equal',
                              extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
            
            # Draw original grid (blue lines)
            for y in grid_y:
                axes[0, 4].plot([Y_world.min(), Y_world.max()], [y, y], 'b-', alpha=0.5, linewidth=1)
            for z in grid_z:
                axes[0, 4].plot([z, z], [Z_world.min(), Z_world.max()], 'b-', alpha=0.5, linewidth=1)
            
            # Draw transformed grid (red lines) 
            for i, y in enumerate(grid_y):
                row_points = transformed_points[i*len(grid_z):(i+1)*len(grid_z)]
                if len(row_points) > 1:
                    axes[0, 4].plot(row_points[:, 2], row_points[:, 1], 'r-', alpha=0.9, linewidth=1.5)
            
            for j, z in enumerate(grid_z):
                col_points = transformed_points[j::len(grid_z)]
                if len(col_points) > 1:
                    axes[0, 4].plot(col_points[:, 2], col_points[:, 1], 'r-', alpha=0.9, linewidth=1.5)
            
            # Add displacement arrows for enhanced visualization (every 3rd point to avoid overcrowding)
            arrow_scale = 3.0  # Scale arrows for better visibility
            for i in range(0, len(original_points), 3):
                orig_pt = original_points[i]
                trans_pt = transformed_points[i]
                displacement = trans_pt - orig_pt
                disp_magnitude = np.linalg.norm(displacement)
                
                # Only show arrows for significant displacement (>0.3mm)
                if disp_magnitude > 0.3:
                    # Scale displacement for visualization
                    scaled_displacement = displacement * arrow_scale
                    # Arrow from original to enhanced transformed position (Z, Y mapping for plot)
                    axes[0, 4].annotate('', 
                                       xy=(orig_pt[2] + scaled_displacement[2], orig_pt[1] + scaled_displacement[1]),
                                       xytext=(orig_pt[2], orig_pt[1]),
                                       arrowprops=dict(arrowstyle='->', 
                                                     color='orange', 
                                                     lw=2.5, 
                                                     alpha=0.9,
                                                     shrinkA=0, 
                                                     shrinkB=0))
                    
                    # Add magnitude labels for largest displacements
                    if disp_magnitude > 1.0:
                        mid_y = orig_pt[1] + scaled_displacement[1] / 2
                        mid_z = orig_pt[2] + scaled_displacement[2] / 2
                        axes[0, 4].text(mid_z, mid_y, f'{disp_magnitude:.1f}mm', 
                                       fontsize=7, color='orange', ha='center', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            axes[0, 4].set_title(f'Forward Deformation\n@ X={world_x:.1f}mm (PDF Style)', fontsize=10)
            
        except Exception as grid_error:
            print(f"⚠️  Error in SVFFD grid deformation: {grid_error}")
            # Fallback: show uniform grid
            n_lines = 15
            y_lines = np.linspace(Y_world.min(), Y_world.max(), n_lines)
            z_lines = np.linspace(Z_world.min(), Z_world.max(), n_lines)
            
            for y in y_lines:
                axes[0, 4].axhline(y=y, color='blue', alpha=0.6, linewidth=1)
            for z in z_lines:
                axes[0, 4].axvline(x=z, color='blue', alpha=0.6, linewidth=1)
            axes[0, 4].set_title('Grid (Transform Failed)', fontsize=10)
            displacement_magnitudes = np.zeros(1)  # Dummy for metrics
            
        axes[0, 4].set_xlabel('Y (mm)')
        axes[0, 4].set_ylabel('Z (mm)')
        axes[0, 4].invert_xaxis()
        axes[0, 4].invert_yaxis()
        
        # Displacement Field (bottom left)
        axes[1, 0].imshow(static_samples, cmap='gray', alpha=0.3, aspect='equal',
                          extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        
        # Sample displacement field at lower resolution for enhanced visualization
        disp_sample_factor = 4
        y_disp = np.linspace(Y_world.min(), Y_world.max(), static_samples.shape[0]//disp_sample_factor)
        z_disp = np.linspace(Z_world.min(), Z_world.max(), static_samples.shape[1]//disp_sample_factor)
        
        # Create enhanced displacement vectors for visualization  
        vector_scale = 5.0  # Scale factor for better visibility
        for i, y in enumerate(y_disp[::2]):  # Sample every other point
            for j, z in enumerate(z_disp[::2]):
                world_point = torch.tensor([[world_x, y, z]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    transformed_point = composed_transform.forward(world_point).cpu().numpy()[0]
                    displacement = transformed_point - [world_x, y, z]
                    disp_magnitude = np.linalg.norm(displacement)
                    
                    # Only show significant displacements with proportional scaling
                    if disp_magnitude > 0.1:
                        # Scale displacement proportionally for visualization
                        scaled_displacement = displacement * vector_scale
                        
                        # Enhanced arrow with proportional head size
                        head_width = min(3.0, disp_magnitude * 2.0)  # Proportional to magnitude
                        head_length = min(2.5, disp_magnitude * 1.5)
                        
                        axes[1, 0].arrow(z, y, scaled_displacement[2], scaled_displacement[1], 
                                       head_width=head_width, head_length=head_length, 
                                       fc='red', ec='darkred', alpha=0.9, linewidth=1.5)
        
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
        output_path = output_dir / "deepali_professional_visualization_rigid_affine_svffd_after_module.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Deepali professional visualization saved: {output_path}")
        
        # Create simple 1x3 PDF-style visualization (target, warped source, forward deformation)
        create_simple_pdf_style_visualization(composed_transform, static_samples, moving_after_samples, Y_world, Z_world, world_x, displacement_magnitudes, output_dir, device)
        
    except Exception as e:
        print(f"⚠️  Error creating deepali visualization: {e}")
        import traceback
        traceback.print_exc()
        print("📝 Continuing with registration results...")


def sample_world_plane_working_approach(world_x, static_sitk, moving_sitk):
    """EXACT COPY of the working _sample_world_plane_for_viz approach."""
    # Define common world coordinate grid
    y_min, y_max = -100, 100  # mm
    z_min, z_max = -100, 100  # mm
    
    y_range = np.linspace(y_min, y_max, 200)
    z_range = np.linspace(z_min, z_max, 200)
    Y_world, Z_world = np.meshgrid(y_range, z_range)
    
    # Sample both images
    static_samples = np.zeros_like(Y_world)
    moving_samples = np.zeros_like(Y_world)
    
    for i in range(Y_world.shape[0]):
        for j in range(Y_world.shape[1]):
            world_point = [world_x, Y_world[i, j], Z_world[i, j]]
            
            # Sample static
            static_samples[i, j] = sample_at_world_point_working(static_sitk, world_point)
            
            # Sample moving  
            moving_samples[i, j] = sample_at_world_point_working(moving_sitk, world_point)
    
    return static_samples, moving_samples, Y_world, Z_world


def sample_at_world_point_working(image, world_point):
    """EXACT COPY of working _sample_at_world_point."""
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


def create_simple_pdf_style_visualization(composed_transform, static_samples, moving_after_samples, Y_world, Z_world, world_x, displacement_magnitudes, output_dir, device):
    """Create simple 1x3 visualization like deepali PDF figure: target | warped source | forward deformation"""
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
        
        # Panel 3: Forward deformation (grid)
        # Background image for context
        axes[2].imshow(static_samples, cmap='gray', alpha=0.2, aspect='equal',
                      extent=[Y_world.min(), Y_world.max(), Z_world.min(), Z_world.max()])
        
        # Create forward deformation grid
        n_grid = 20
        grid_y = np.linspace(Y_world.min(), Y_world.max(), n_grid)
        grid_z = np.linspace(Z_world.min(), Z_world.max(), n_grid)
        
        # Create grid points in 3D world coordinates (same X as visualization)
        grid_points_list = []
        for y in grid_y:
            for z in grid_z:
                grid_points_list.append([world_x, y, z])
        
        # Convert to tensor for deepali
        grid_points = torch.tensor(grid_points_list, dtype=torch.float32, device=device)
        
        # Apply SVFFD transform using deepali's forward method
        with torch.no_grad():
            transformed_points = composed_transform.forward(grid_points)
            transformed_points = transformed_points.cpu().numpy()
        
        # Draw original grid (thin gray lines)
        for y in grid_y:
            axes[2].plot([Y_world.min(), Y_world.max()], [y, y], 'gray', alpha=0.3, linewidth=0.5)
        for z in grid_z:
            axes[2].plot([z, z], [Z_world.min(), Z_world.max()], 'gray', alpha=0.3, linewidth=0.5)
        
        # Draw transformed grid (blue lines) 
        for i, y in enumerate(grid_y):
            row_points = transformed_points[i*len(grid_z):(i+1)*len(grid_z)]
            if len(row_points) > 1:
                axes[2].plot(row_points[:, 2], row_points[:, 1], 'b-', alpha=0.8, linewidth=1.0)
        
        for j, z in enumerate(grid_z):
            col_points = transformed_points[j::len(grid_z)]
            if len(col_points) > 1:
                axes[2].plot(col_points[:, 2], col_points[:, 1], 'b-', alpha=0.8, linewidth=1.0)
        
        # Add displacement arrows to PDF-style visualization (every 4th point)
        original_points = np.array(grid_points_list)
        displacement_vectors = transformed_points - original_points
        arrow_scale_pdf = 2.0  # Smaller scale for PDF figure
        
        for i in range(0, len(original_points), 4):
            orig_pt = original_points[i]
            displacement = displacement_vectors[i]
            disp_magnitude = np.linalg.norm(displacement)
            
            # Only show arrows for significant displacement (>0.5mm)
            if disp_magnitude > 0.5:
                # Scale displacement for visualization
                scaled_displacement = displacement * arrow_scale_pdf
                # Arrow from original to enhanced transformed position (Z, Y mapping for plot)
                axes[2].annotate('', 
                               xy=(orig_pt[2] + scaled_displacement[2], orig_pt[1] + scaled_displacement[1]),
                               xytext=(orig_pt[2], orig_pt[1]),
                               arrowprops=dict(arrowstyle='->', 
                                             color='green', 
                                             lw=2.0, 
                                             alpha=0.8,
                                             shrinkA=0, 
                                             shrinkB=0))
        
        axes[2].set_title('forward deformation', fontsize=12)
        axes[2].axis('off')
        axes[2].invert_xaxis()
        axes[2].invert_yaxis()
        
        # Save PDF-style visualization
        pdf_output_path = output_dir / "deepali_pdf_style_1x3_visualization.png"
        plt.tight_layout()
        plt.savefig(pdf_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PDF-style 1x3 visualization saved: {pdf_output_path}")
        
    except Exception as e:
        print(f"⚠️  Error creating PDF-style visualization: {e}")
        import traceback
        traceback.print_exc()