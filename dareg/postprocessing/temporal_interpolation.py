#!/usr/bin/env python3
"""
Temporal Interpolation for 4D Motion Registration

Provides temporal interpolation between discrete time frames to create
smooth 4D animations of segmentation surfaces.

Features:
- B-spline temporal interpolation of displacement fields
- Support for any temporal resolution output
- Mesh vertex interpolation with topology preservation
- MIRTK-compatible temporal parameterization
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import torch

from .stl_generator import STLGenerator, SurfaceMesh, generate_stl_from_segmentation


@dataclass
class TemporalPoint:
    """Represents a point in time with associated transform data"""
    time: float  # Time in seconds
    frame_index: int  # Original frame index
    displacement_field: Optional[np.ndarray] = None  # Displacement field if available


class TemporalInterpolator:
    """
    Temporal Interpolator for 4D Motion

    Interpolates between discrete time frames using B-spline or linear
    interpolation to produce smooth motion at arbitrary temporal resolution.
    """

    def __init__(
        self,
        method: str = "linear",
        smoothing_factor: float = 0.0,
    ):
        """
        Initialize temporal interpolator.

        Args:
            method: Interpolation method ("linear", "cubic", "bspline")
            smoothing_factor: Optional temporal smoothing (0 = no smoothing)
        """
        self.method = method
        self.smoothing_factor = smoothing_factor

    def get_frame_times(
        self,
        num_frames: int,
        temporal_resolution: float,
        start_time: float = 0.0,
    ) -> List[float]:
        """
        Get time points for each frame.

        Args:
            num_frames: Number of frames
            temporal_resolution: Time between frames in seconds
            start_time: Start time in seconds

        Returns:
            List of time points in seconds
        """
        return [start_time + i * temporal_resolution for i in range(num_frames)]

    def interpolate_vertices(
        self,
        base_vertices: np.ndarray,
        displacement_fields: List[np.ndarray],
        frame_times: List[float],
        target_time: float,
    ) -> np.ndarray:
        """
        Interpolate vertex positions at arbitrary time.

        Uses displacement field interpolation to compute vertex positions
        at any time between frames.

        Args:
            base_vertices: Base mesh vertices (frame 0) [N, 3]
            displacement_fields: Displacement fields for each frame [N, 3]
            frame_times: Time points for each frame
            target_time: Target time for interpolation

        Returns:
            Interpolated vertex positions
        """
        # Find bracketing frames
        frame_times = np.array(frame_times)
        n_frames = len(frame_times)

        if target_time <= frame_times[0]:
            return base_vertices + displacement_fields[0]
        if target_time >= frame_times[-1]:
            return base_vertices + displacement_fields[-1]

        # Find frames to interpolate between
        idx_after = np.searchsorted(frame_times, target_time)
        idx_before = idx_after - 1

        t_before = frame_times[idx_before]
        t_after = frame_times[idx_after]

        # Calculate interpolation weight
        alpha = (target_time - t_before) / (t_after - t_before)

        if self.method == "linear":
            # Linear interpolation of displacements
            disp_interp = (1 - alpha) * displacement_fields[idx_before] + \
                         alpha * displacement_fields[idx_after]

        elif self.method in ("cubic", "bspline"):
            # Cubic B-spline interpolation using 4 control points
            # Get 4 frames for cubic interpolation
            idx0 = max(0, idx_before - 1)
            idx1 = idx_before
            idx2 = idx_after
            idx3 = min(n_frames - 1, idx_after + 1)

            # Cubic B-spline basis functions
            t = alpha
            t2 = t * t
            t3 = t2 * t

            b0 = (-t3 + 3*t2 - 3*t + 1) / 6.0
            b1 = (3*t3 - 6*t2 + 4) / 6.0
            b2 = (-3*t3 + 3*t2 + 3*t + 1) / 6.0
            b3 = t3 / 6.0

            disp_interp = (b0 * displacement_fields[idx0] +
                          b1 * displacement_fields[idx1] +
                          b2 * displacement_fields[idx2] +
                          b3 * displacement_fields[idx3])
        else:
            # Default to linear
            disp_interp = (1 - alpha) * displacement_fields[idx_before] + \
                         alpha * displacement_fields[idx_after]

        return base_vertices + disp_interp

    def interpolate_mesh_sequence(
        self,
        meshes: List[SurfaceMesh],
        frame_times: List[float],
        target_times: List[float],
    ) -> List[SurfaceMesh]:
        """
        Interpolate mesh sequence at target times.

        Assumes all meshes have the same topology (same number of vertices/faces).
        Interpolates vertex positions between frames.

        Args:
            meshes: List of SurfaceMesh objects with same topology
            frame_times: Time points for input meshes
            target_times: Target time points for output

        Returns:
            List of interpolated SurfaceMesh objects
        """
        if len(meshes) < 2:
            return meshes

        # Use first mesh as reference
        base_vertices = meshes[0].vertices.copy()
        base_faces = meshes[0].faces.copy()

        # Compute displacement fields relative to frame 0
        displacement_fields = []
        for mesh in meshes:
            if mesh.vertices.shape[0] != base_vertices.shape[0]:
                raise ValueError("All meshes must have same number of vertices")
            disp = mesh.vertices - base_vertices
            displacement_fields.append(disp)

        # Interpolate at target times
        interpolated_meshes = []
        for t in target_times:
            verts = self.interpolate_vertices(
                base_vertices,
                displacement_fields,
                frame_times,
                t,
            )

            mesh = SurfaceMesh(
                vertices=verts,
                faces=base_faces.copy(),
                time_point=t,
            )
            interpolated_meshes.append(mesh)

        return interpolated_meshes


def get_temporal_resolution_from_nifti(image_path: Path) -> Optional[float]:
    """
    Extract temporal resolution from NIfTI header.

    Reads pixdim[4] which contains the TR or temporal spacing.

    Args:
        image_path: Path to NIfTI file

    Returns:
        Temporal resolution in seconds, or None if not available
    """
    try:
        import nibabel as nib
        img = nib.load(str(image_path))
        header = img.header

        # Get pixdim - TR is in pixdim[4]
        pixdim = header.get_zooms()
        if len(pixdim) >= 4:
            tr = pixdim[3]
            if tr > 0:
                # Check units - xyzt_units
                units = header.get_xyzt_units()
                time_unit = units[1] if len(units) > 1 else 'unknown'

                # Convert to seconds based on unit
                if time_unit == 'msec':
                    return tr / 1000.0
                elif time_unit == 'usec':
                    return tr / 1000000.0
                else:
                    # Assume seconds
                    return float(tr)

        return None
    except Exception as e:
        print(f"   Warning: Could not read temporal resolution: {e}")
        return None


def generate_interpolated_stl_sequence(
    segmentation_sequence: List,
    output_dir: Path,
    frame_times: List[float],
    target_temporal_resolution: float,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    smoothing_iterations: int = 2,
    interpolation_method: str = "linear",
    prefix: str = "surface",
) -> List[Path]:
    """
    Generate temporally interpolated STL sequence.

    First generates STL meshes for each frame, then interpolates between
    them at the target temporal resolution.

    Args:
        segmentation_sequence: List of segmentation images
        output_dir: Output directory
        frame_times: Time points for input frames (seconds)
        target_temporal_resolution: Desired output temporal resolution (seconds)
        spacing: Voxel spacing in mm
        origin: World coordinate origin
        smoothing_iterations: Mesh smoothing iterations
        interpolation_method: Interpolation method ("linear", "cubic", "bspline")
        prefix: Filename prefix

    Returns:
        List of paths to saved STL files
    """
    output_dir = Path(output_dir)
    stl_dir = output_dir / "stl_interpolated"
    stl_dir.mkdir(parents=True, exist_ok=True)

    # Generate base meshes
    stl_generator = STLGenerator(smoothing_iterations=smoothing_iterations)
    base_meshes = []

    print(f"   Step 1: Extracting surfaces from {len(segmentation_sequence)} frames...")
    for frame_idx, seg in enumerate(segmentation_sequence):
        # Extract segmentation data
        if hasattr(seg, 'tensor'):
            seg_np = seg.tensor().squeeze().cpu().numpy()
            # Get spacing from image if available
            if frame_idx == 0 and hasattr(seg, 'grid'):
                grid = seg.grid()
                if hasattr(grid, 'spacing'):
                    sp = grid.spacing()
                    if hasattr(sp, 'cpu'):
                        spacing = tuple(sp.cpu().numpy())
                    else:
                        spacing = tuple(sp)
                if hasattr(grid, 'origin'):
                    orig = grid.origin()
                    if hasattr(orig, 'cpu'):
                        origin = tuple(orig.cpu().numpy())
                    else:
                        origin = tuple(orig)
        elif isinstance(seg, torch.Tensor):
            seg_np = seg.squeeze().cpu().numpy()
        else:
            seg_np = np.array(seg).squeeze()

        mesh = stl_generator.extract_surface(
            seg_np,
            spacing=spacing,
            origin=origin,
            frame_index=frame_idx,
            time_point=frame_times[frame_idx] if frame_idx < len(frame_times) else 0.0,
        )

        if mesh is not None:
            base_meshes.append(mesh)
        else:
            print(f"   Warning: No surface found for frame {frame_idx}")

    if len(base_meshes) < 2:
        print("   Warning: Not enough meshes for interpolation")
        return []

    # Generate target time points
    start_time = frame_times[0]
    end_time = frame_times[-1]
    num_output_frames = int((end_time - start_time) / target_temporal_resolution) + 1
    target_times = [start_time + i * target_temporal_resolution for i in range(num_output_frames)]

    print(f"   Step 2: Interpolating to {num_output_frames} frames at {target_temporal_resolution:.3f}s resolution...")

    # Check if meshes have same topology
    n_verts = base_meshes[0].vertices.shape[0]
    same_topology = all(m.vertices.shape[0] == n_verts for m in base_meshes)

    if same_topology:
        # Use mesh interpolation
        interpolator = TemporalInterpolator(method=interpolation_method)
        mesh_times = [m.time_point for m in base_meshes]
        interpolated_meshes = interpolator.interpolate_mesh_sequence(
            base_meshes, mesh_times, target_times
        )
    else:
        # Topology changes between frames - just use nearest frame
        print("   Warning: Mesh topology changes between frames, using nearest-frame interpolation")
        interpolated_meshes = []
        for t in target_times:
            # Find nearest frame
            idx = np.argmin([abs(m.time_point - t) for m in base_meshes])
            interpolated_meshes.append(base_meshes[idx])

    # Save interpolated meshes
    print(f"   Step 3: Saving {len(interpolated_meshes)} interpolated STL files...")
    saved_paths = []
    for i, mesh in enumerate(interpolated_meshes):
        output_path = stl_dir / f"{prefix}_t{target_times[i]:.3f}.stl"
        stl_generator.save_stl(mesh, output_path)
        saved_paths.append(output_path)

    print(f"   Generated {len(saved_paths)} interpolated STL files in {stl_dir}")
    return saved_paths


def animate_stl_sequence(
    stl_dir: Path,
    output_path: Path,
    fps: int = 10,
) -> Optional[Path]:
    """
    Create animation from STL sequence (requires external tools).

    This is a placeholder for potential future implementation using
    tools like VTK, PyVista, or external renderers.

    Args:
        stl_dir: Directory containing STL sequence
        output_path: Output animation file path
        fps: Frames per second

    Returns:
        Path to animation file, or None if not supported
    """
    try:
        import pyvista as pv

        stl_files = sorted(stl_dir.glob("*.stl"))
        if not stl_files:
            print("   No STL files found for animation")
            return None

        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.open_gif(str(output_path))

        for stl_file in stl_files:
            mesh = pv.read(str(stl_file))
            plotter.clear()
            plotter.add_mesh(mesh, color='lightblue')
            plotter.camera_position = 'iso'
            plotter.write_frame()

        plotter.close()
        print(f"   Created animation: {output_path}")
        return output_path

    except ImportError:
        print("   Animation requires PyVista. Install with: pip install pyvista")
        return None
    except Exception as e:
        print(f"   Animation creation failed: {e}")
        return None
