#!/usr/bin/env python3
"""
DAREG Post-Processing Pipeline

Generates STL surfaces, temporal interpolation, and videos from motion registration results.
This is designed to run AFTER main_motion.py completes.

Usage:
    dareg postprocess --output_dir <registration_output_folder>
    dareg postprocess --output_dir outputs_motion_test --stl --video
    dareg postprocess --output_dir outputs_motion_test --interpolate --temporal_resolution 0.01

Features:
    - STL surface generation from segmentation masks
    - Temporal interpolation for smoother animations
    - Video/GIF generation (requires pyvista)
    - Automatic input file detection and validation
"""

import sys
import os
import platform
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import json

# Fix macOS threading crash (condition_variable wait failed)
# This must be done BEFORE importing numpy/torch
if platform.system() == "Darwin":
    # macOS: conservative thread count to avoid condition_variable crash
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
else:
    # Linux: use 20 cores for parallel computation
    os.environ.setdefault("OMP_NUM_THREADS", "20")
    os.environ.setdefault("MKL_NUM_THREADS", "20")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")

import numpy as np
import nibabel as nib


@dataclass
class PostProcessConfig:
    """Configuration for post-processing pipeline"""
    # Input/Output
    output_dir: Path

    # STL Generation
    generate_stl: bool = True
    stl_smoothing_iterations: int = 2
    stl_use_frame_grid: bool = False  # Use original segmentations/ folder (full resolution)

    # Temporal Interpolation
    interpolate: bool = False
    interpolation_method: str = "cubic"  # linear, cubic (MIRTK default)
    total_duration_ms: Optional[float] = None  # Total duration in ms for all frames
    interp_step_ms: float = 1.0  # Interpolation step in ms (default 1ms)

    # Motion Table Export
    export_motion_table: bool = False  # Export STAR format motion table

    # Video Generation
    generate_video: bool = True  # Generate video by default
    video_fps: int = 10
    video_duration_sec: float = 10.0  # Target video duration in seconds (fps auto-calculated)
    video_2x2_view: bool = True  # Create 2x2 subfigure (sagittal, axial, coronal, perspective)

    # Verbosity
    verbose: bool = False


@dataclass
class RequiredFiles:
    """Container for required input files with validation"""
    segmentations: List[Path] = field(default_factory=list)
    segmentations_frame_grid: List[Path] = field(default_factory=list)
    frames: List[Path] = field(default_factory=list)

    # Metadata
    num_frames: int = 0
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    temporal_resolution: Optional[float] = None


def detect_output_structure(output_dir: Path, verbose: bool = False) -> Dict[str, Path]:
    """
    Detect the folder structure of registration output.

    Args:
        output_dir: Root output directory from main_motion.py
        verbose: Print detailed information

    Returns:
        Dictionary mapping folder types to paths
    """
    structure = {}

    # Expected subdirectories
    expected_dirs = {
        'segmentations': output_dir / 'segmentations',
        'segmentations_frame_grid': output_dir / 'segmentations_frame_grid',
        'frames': output_dir / 'frames',
        'transforms': output_dir / 'transforms',
        'visualizations': output_dir / 'visualizations',
        'alignment': output_dir / 'alignment',
        'longitudinal': output_dir / 'longitudinal',
    }

    for name, path in expected_dirs.items():
        if path.exists():
            structure[name] = path
            if verbose:
                print(f"   Found: {name}/ ({len(list(path.glob('*')))} files)")
        else:
            if verbose:
                print(f"   Missing: {name}/")

    return structure


def validate_required_files(output_dir: Path, config: PostProcessConfig) -> RequiredFiles:
    """
    Validate that all required files exist for post-processing.

    Args:
        output_dir: Root output directory
        config: Post-processing configuration

    Returns:
        RequiredFiles object with validated file paths

    Raises:
        FileNotFoundError: If required files are missing
    """
    required = RequiredFiles()
    missing_files = []
    missing_dirs = []

    print(f"\n{'='*60}")
    print("VALIDATING REGISTRATION OUTPUT")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Check output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    # Detect structure
    structure = detect_output_structure(output_dir, config.verbose)

    # ========== REQUIRED: Segmentations ==========
    # Naming convention: seg_f###_t###.nii.gz (f=absolute, t=relative, both 1-indexed)
    if config.stl_use_frame_grid:
        seg_dir = structure.get('segmentations_frame_grid')
        seg_pattern = "seg_f*_t*_frame_grid.nii.gz"
    else:
        seg_dir = structure.get('segmentations')
        seg_pattern = "seg_f*_t*.nii.gz"

    if seg_dir is None:
        dir_name = 'segmentations_frame_grid' if config.stl_use_frame_grid else 'segmentations'
        missing_dirs.append(dir_name)
    else:
        seg_files = sorted(seg_dir.glob(seg_pattern))
        # Filter out frame_grid files if using high-res
        if not config.stl_use_frame_grid:
            seg_files = [f for f in seg_files if '_frame_grid' not in f.name]

        if not seg_files:
            missing_files.append(f"{seg_dir.name}/{seg_pattern}")
        else:
            if config.stl_use_frame_grid:
                required.segmentations_frame_grid = seg_files
            else:
                required.segmentations = seg_files
            required.num_frames = len(seg_files)
            print(f"   Found {len(seg_files)} segmentation files")

    # ========== OPTIONAL: Frames (for overlay/video) ==========
    # Naming convention: dynamic_extracted_f###_t###.nii.gz (f=absolute, t=relative, both 1-indexed)
    frames_dir = structure.get('frames')
    if frames_dir:
        frame_files = sorted(frames_dir.glob("dynamic_extracted_f*_t*.nii.gz"))
        if frame_files:
            required.frames = frame_files
            print(f"   Found {len(frame_files)} extracted frame files")

    # ========== Extract metadata from first segmentation ==========
    seg_files_to_check = required.segmentations_frame_grid or required.segmentations
    if seg_files_to_check:
        first_seg = seg_files_to_check[0]
        try:
            nii = nib.load(str(first_seg))
            header = nii.header
            affine = nii.affine

            # Get spacing
            spacing = header.get_zooms()[:3]
            required.spacing = tuple(float(s) for s in spacing)

            # Get origin from affine
            origin = affine[:3, 3]
            required.origin = tuple(float(o) for o in origin)

            # Try to get temporal resolution from 4D header
            if len(header.get_zooms()) >= 4:
                tr = header.get_zooms()[3]
                if tr > 0:
                    required.temporal_resolution = float(tr)

            print(f"   Spacing: {required.spacing} mm")
            print(f"   Origin: {required.origin} mm")
            if required.temporal_resolution:
                print(f"   Temporal resolution: {required.temporal_resolution} s")

        except Exception as e:
            print(f"   Warning: Could not read metadata from {first_seg}: {e}")

    # ========== Report missing files ==========
    if missing_dirs or missing_files:
        print(f"\n{'='*60}")
        print("ERROR: MISSING REQUIRED FILES")
        print(f"{'='*60}")

        if missing_dirs:
            print("\nMissing directories:")
            for d in missing_dirs:
                print(f"   - {output_dir}/{d}/")

        if missing_files:
            print("\nMissing files:")
            for f in missing_files:
                print(f"   - {output_dir}/{f}")

        print("\nPlease run main_motion.py first to generate registration results.")
        print(f"{'='*60}\n")

        raise FileNotFoundError(
            f"Missing required files for post-processing. "
            f"Missing dirs: {missing_dirs}, Missing files: {missing_files}"
        )

    print(f"\n   Validation PASSED: All required files found")
    print(f"{'='*60}\n")

    return required


def load_segmentations(required: RequiredFiles, use_frame_grid: bool = True) -> List:
    """
    Load segmentation files into memory.

    Args:
        required: RequiredFiles object with file paths
        use_frame_grid: Use frame_grid segmentations

    Returns:
        List of loaded segmentation data (numpy arrays or nibabel images)
    """
    seg_files = required.segmentations_frame_grid if use_frame_grid else required.segmentations

    segmentations = []
    for seg_path in seg_files:
        nii = nib.load(str(seg_path))
        seg_data = nii.get_fdata()
        segmentations.append({
            'data': seg_data,
            'affine': nii.affine,
            'header': nii.header,
            'path': seg_path,
        })

    return segmentations


def run_stl_generation(
    required: RequiredFiles,
    config: PostProcessConfig,
) -> List[Path]:
    """
    Generate STL surfaces from segmentation masks.

    Uses full affine matrices for proper world coordinate transformation,
    which includes rotation (for oblique scans) in addition to spacing and origin.

    Args:
        required: RequiredFiles object with validated paths
        config: Post-processing configuration

    Returns:
        List of generated STL file paths
    """
    # Import directly from stl_generator to avoid complex dependency chain
    from .postprocessing.stl_generator import generate_stl_sequence

    print(f"\n{'='*60}")
    print("STL SURFACE GENERATION")
    print(f"{'='*60}")

    # Load segmentations
    print("Loading segmentations...")
    segmentations = load_segmentations(required, config.stl_use_frame_grid)

    # Extract data arrays and affine matrices
    seg_arrays = [s['data'] for s in segmentations]
    affine_matrices = [s['affine'] for s in segmentations]

    # Generate STL sequence with full affine transformation
    print(f"Generating STL surfaces ({len(seg_arrays)} frames)...")
    print(f"   Smoothing iterations: {config.stl_smoothing_iterations}")
    print(f"   Using full affine transformation (includes rotation)")

    stl_paths = generate_stl_sequence(
        segmentation_sequence=seg_arrays,
        output_dir=config.output_dir,
        spacing=required.spacing,
        origin=required.origin,
        affine_matrices=affine_matrices,  # Pass affine for proper world coords
        smoothing_iterations=config.stl_smoothing_iterations,
        temporal_resolution=required.temporal_resolution,
        prefix="seg_surface",
    )

    print(f"\n   Generated {len(stl_paths)} STL files")
    print(f"   Output: {config.output_dir / 'stl_surfaces'}")

    return stl_paths


def _load_longitudinal_transforms(output_dir: Path, num_frames: int) -> List[Optional[Path]]:
    """
    Find longitudinal transform files for each frame.

    MIRTK-style: transforms map from frame 0 to frame N.
    Frame 0 has identity transform (None).

    Args:
        output_dir: Registration output directory
        num_frames: Number of frames

    Returns:
        List of transform paths (None for frame 0)
    """
    transforms_dir = output_dir / 'longitudinal'
    if not transforms_dir.exists():
        transforms_dir = output_dir / 'transforms'

    transform_paths = [None]  # Frame 0 is identity

    for i in range(1, num_frames):
        # Try different naming conventions
        # Prefer _refined (single FFD, works with apply_ffd_transform_mirtk_style)
        # over _composed (SequentialTransform chain, requires different loading)
        candidates = [
            transforms_dir / f'longitudinal_0_to_{i}_refined.pth',
            transforms_dir / f'longitudinal_0_to_{i}.pth',
            output_dir / 'transforms' / f'longitudinal_0_to_{i}.pth',
            transforms_dir / f'longitudinal_0_to_{i}_composed.pth',
        ]

        found = None
        for path in candidates:
            if path.exists():
                found = path
                break

        transform_paths.append(found)

    return transform_paths


def _apply_transform_to_vertices(vertices: np.ndarray, transform_path: Path,
                                  reference_image_path: Optional[Path] = None) -> np.ndarray:
    """
    Apply a saved FFD transform to mesh vertices (MIRTK-style).

    MIRTK approach: The transform file contains the grid domain info (like DOF files).
    We convert world coordinates to the transform's lattice coordinates, sample the
    B-spline, and return the displacement in world space.

    This works for vertices in ANY coordinate space because the transform knows its
    own domain - just like MIRTK's transform-points command.

    Uses the new MIRTK-style transform utilities from transform_utils.py.

    Args:
        vertices: Vertex positions [num_vertices, 3] in world coordinates (mm)
        transform_path: Path to saved transform (.pth file)
        reference_image_path: Path to reference image for grid info (fallback, unused)

    Returns:
        Transformed vertex positions [num_vertices, 3]
    """
    from .postprocessing.transform_utils import (
        apply_ffd_transform_mirtk_style,
        apply_sequential_transform_to_vertices,
    )
    import torch

    # Load transform to check its type
    checkpoint = torch.load(transform_path, weights_only=False, map_location='cpu')
    transform_type = checkpoint.get('type', 'unknown')

    # Use appropriate function based on transform type
    if transform_type == 'SequentialTransform':
        # Alignment transform: rigid + affine + FFD
        return apply_sequential_transform_to_vertices(vertices, transform_path)
    else:
        # Pure FFD transform (longitudinal)
        return apply_ffd_transform_mirtk_style(vertices, transform_path)


def run_temporal_interpolation(
    required: RequiredFiles,
    config: PostProcessConfig,
) -> Tuple[List[Path], Optional[Path]]:
    """
    Generate temporally interpolated STL sequence using MIRTK-style approach.

    MIRTK-style workflow:
    1. Load HIGH-RES reference mesh (original segmentation geometry)
    2. Apply ALIGNMENT transform (segmentation space -> frame 0 space)
    3. Apply longitudinal transforms to deform to each frame
    4. Interpolate vertex positions using cubic spline
    5. Export motion table with consistent vertex correspondence

    This guarantees consistent vertex count and enables proper motion tracking.
    Using high-res STLs preserves the original segmentation quality.

    Args:
        required: RequiredFiles object with validated paths
        config: Post-processing configuration

    Returns:
        Tuple of (list of interpolated STL file paths, motion table path or None)
    """
    from scipy.interpolate import interp1d
    from .postprocessing.transform_utils import (
        apply_sequential_transform_to_vertices,
        apply_ffd_transform_mirtk_style,
    )
    import pandas as pd
    import math
    import torch

    print(f"\n{'='*60}")
    print("TEMPORAL INTERPOLATION (MIRTK-style: Reference Mesh + Transforms)")
    print(f"{'='*60}")

    # Determine source STL files
    # PREFER high-res STLs (stl_surfaces) for original geometry quality
    # We will apply alignment transform to bring them into frame space
    use_highres = False
    stl_dir = config.output_dir / 'stl_surfaces'  # Try high-res first
    if stl_dir.exists():
        use_highres = True
        print("   Using HIGH-RES STLs (original segmentation geometry)")
    else:
        # Fall back to frame_grid STLs
        stl_dir = config.output_dir / 'stl_surfaces_frame_grid'
        if stl_dir.exists():
            print("   Using frame_grid STLs (already in frame coordinate space)")
    if not stl_dir.exists():
        print("   ERROR: No STL files found. Run with --stl first.")
        return [], None

    print(f"   Using STLs from: {stl_dir.name}")

    stl_files = sorted(stl_dir.glob("*.stl"))
    if len(stl_files) < 2:
        print("   ERROR: Need at least 2 STL files for interpolation.")
        return [], None

    num_frames = len(stl_files)
    print(f"   Found {num_frames} frames")

    # Check for longitudinal transforms
    transform_paths = _load_longitudinal_transforms(config.output_dir, num_frames)
    has_transforms = any(p is not None for p in transform_paths[1:])  # Skip frame 0

    # Check for alignment transform (needed for high-res STLs)
    alignment_transform_path = None
    alignment_candidates = [
        config.output_dir / 'transforms' / 'alignment_composed.pth',
        config.output_dir / 'alignment' / 'alignment_transform.pth',
        config.output_dir / 'transforms' / 'alignment_transform.pth',
    ]
    for candidate in alignment_candidates:
        if candidate.exists():
            alignment_transform_path = candidate
            break

    if use_highres and alignment_transform_path:
        print(f"   Found alignment transform: {alignment_transform_path.name}")
        print(f"   Will apply: alignment (seg->frame0) + longitudinal (frame0->frameN)")
    elif use_highres:
        print("   WARNING: Using high-res STLs but no alignment transform found!")
        print("            STL coordinates may not match frame space.")

    if has_transforms:
        print(f"   Found longitudinal transforms - using MIRTK-style deformation")
        for i, p in enumerate(transform_paths):
            if p:
                print(f"      Frame {i}: {p.name}")
            else:
                print(f"      Frame {i}: identity (reference)")
    else:
        print(f"   No longitudinal transforms found - checking mesh topology...")

    # Load reference mesh (frame 0)
    try:
        import trimesh
    except ImportError:
        print("   ERROR: trimesh required. Install with: pip install trimesh")
        return [], None

    reference_mesh = trimesh.load(str(stl_files[0]))
    num_vertices = reference_mesh.vertices.shape[0]
    print(f"   Reference mesh: {num_vertices} vertices, {len(reference_mesh.faces)} faces")

    # Compute frame times
    if config.total_duration_ms is not None:
        total_duration_s = config.total_duration_ms / 1000.0
        frame_times_s = np.linspace(0, total_duration_s, num_frames)
        print(f"   Total duration: {config.total_duration_ms:.1f}ms")
    else:
        default_tr_ms = 100.0
        total_duration_s = (num_frames - 1) * default_tr_ms / 1000.0
        frame_times_s = np.linspace(0, total_duration_s, num_frames)
        print(f"   Warning: No total_duration_ms specified, using {default_tr_ms}ms between frames")

    # Compute target times
    interp_step_s = config.interp_step_ms / 1000.0
    num_output_frames = int(total_duration_s / interp_step_s) + 1
    target_times_s = np.linspace(0, total_duration_s, num_output_frames)
    decimal_precision = max(3, int(math.ceil(-math.log10(interp_step_s))))

    print(f"   Interpolation step: {config.interp_step_ms}ms")
    print(f"   Output frames: {num_output_frames}")
    print(f"   Filename format: t_X.{decimal_precision}fs.stl")

    # Get vertex positions at each frame
    all_vertices = []

    if has_transforms:
        # MIRTK-style: Apply transforms to reference mesh
        print(f"\n   Deforming reference mesh using transforms...")

        # Step 1: Get base vertices (after alignment if using high-res)
        base_vertices = reference_mesh.vertices.copy()

        if use_highres and alignment_transform_path:
            # Apply alignment transform: segmentation space -> frame 0 space
            print(f"   Applying alignment transform to bring STL into frame space...")
            try:
                aligned_vertices = apply_sequential_transform_to_vertices(
                    base_vertices, alignment_transform_path
                )
                base_vertices = aligned_vertices
                print(f"      Aligned {len(base_vertices)} vertices to frame space")

                # Report displacement statistics
                disp = aligned_vertices - reference_mesh.vertices
                disp_mag = np.linalg.norm(disp, axis=1)
                print(f"      Alignment displacement: min={disp_mag.min():.2f}, max={disp_mag.max():.2f}, mean={disp_mag.mean():.2f} mm")
            except Exception as e:
                print(f"      WARNING: Alignment transform failed ({e})")
                print(f"      Continuing with original vertices (may have coordinate mismatch)")

        # Frame 0: aligned base vertices
        all_vertices.append(base_vertices.copy())

        # Step 2: Apply longitudinal transforms for subsequent frames
        for i in range(1, num_frames):
            if transform_paths[i] is not None:
                try:
                    # Apply longitudinal FFD transform (frame 0 -> frame i)
                    transformed_verts = apply_ffd_transform_mirtk_style(
                        base_vertices, transform_paths[i]
                    )
                    all_vertices.append(transformed_verts)

                    # Report displacement statistics
                    disp = transformed_verts - base_vertices
                    disp_mag = np.linalg.norm(disp, axis=1)
                    print(f"      Frame {i}: {len(transformed_verts)} vertices, disp: min={disp_mag.min():.2f}, max={disp_mag.max():.2f}, mean={disp_mag.mean():.2f} mm")
                except Exception as e:
                    print(f"      Frame {i}: transform failed ({e}), using base vertices")
                    all_vertices.append(base_vertices.copy())
            else:
                # No transform - use base vertices (frame 0 position)
                all_vertices.append(base_vertices.copy())
                print(f"      Frame {i}: no transform, using base vertices")
    else:
        # Fallback: Load meshes directly and check topology
        print(f"\n   Loading meshes directly (no transforms available)...")

        meshes = []
        for stl_path in stl_files:
            mesh = trimesh.load(str(stl_path))
            meshes.append(mesh)
            all_vertices.append(mesh.vertices.copy())

        # Check topology consistency
        same_topology = all(len(v) == num_vertices for v in all_vertices)
        if not same_topology:
            print("   WARNING: Mesh topology varies between frames.")
            print(f"   Vertex counts: {[len(v) for v in all_vertices]}")
            print("   Using nearest-frame interpolation (no motion table).")

            # Nearest-frame interpolation fallback
            output_dir = config.output_dir / 'stl_interpolated'
            output_dir.mkdir(parents=True, exist_ok=True)

            saved_paths = []
            for i, t_s in enumerate(target_times_s):
                idx = np.argmin(np.abs(frame_times_s - t_s))
                interp_mesh = trimesh.Trimesh(
                    vertices=all_vertices[idx],
                    faces=meshes[idx].faces.copy(),
                )
                output_path = output_dir / f"t_{t_s:.{decimal_precision}f}s.stl"
                interp_mesh.export(str(output_path), file_type='stl')
                saved_paths.append(output_path)

            print(f"\n   Generated {len(saved_paths)} interpolated STL files (nearest-frame)")

            if config.export_motion_table:
                print("   WARNING: Motion table skipped - inconsistent topology")

            return saved_paths, None

    # Stack vertices for interpolation [num_frames, num_vertices, 3]
    mesh_points = np.array(all_vertices)
    print(f"\n   Vertex array shape: {mesh_points.shape}")

    # Create spline interpolator
    interp_kind = config.interpolation_method if config.interpolation_method in ['linear', 'cubic'] else 'cubic'

    # Fall back to linear interpolation if we have fewer than 4 frames (cubic requires ≥4 points)
    num_frames = len(frame_times_s)
    if interp_kind == 'cubic' and num_frames < 4:
        interp_kind = 'linear'
        print(f"   Note: Using linear interpolation (cubic requires ≥4 frames, have {num_frames})")
    else:
        print(f"   Creating {interp_kind} spline interpolator...")

    mesh_points_interpolator = interp1d(
        frame_times_s, mesh_points, axis=0,
        bounds_error=True, assume_sorted=True, kind=interp_kind
    )

    # Interpolate at target times
    print(f"   Interpolating {num_output_frames} frames...")
    interpolated_vertices = mesh_points_interpolator(target_times_s)

    # Save interpolated STL files
    output_dir = config.output_dir / 'stl_interpolated'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Saving {num_output_frames} interpolated STL files (binary)...")
    saved_paths = []
    for i, t_s in enumerate(target_times_s):
        interp_mesh = trimesh.Trimesh(
            vertices=interpolated_vertices[i],
            faces=reference_mesh.faces.copy(),  # Always use reference topology
        )
        output_path = output_dir / f"t_{t_s:.{decimal_precision}f}s.stl"
        interp_mesh.export(str(output_path), file_type='stl')
        saved_paths.append(output_path)

    print(f"\n   Generated {len(saved_paths)} interpolated STL files")
    print(f"   Output: {output_dir}")

    # Export motion table
    motion_table_path = None
    if config.export_motion_table:
        motion_table_path = _export_star_motion_table(
            interpolated_vertices, target_times_s * 1000.0, config.output_dir
        )

    return saved_paths, motion_table_path


def _export_star_motion_table(
    interpolated_vertices: np.ndarray,
    target_times_ms: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Export STAR format motion table.

    Format: rows = vertices, columns = XYZ at each time point
    Header row has double quotes, no index column.

    Args:
        interpolated_vertices: [num_frames, num_vertices, 3] array
        target_times_ms: [num_frames] array of time points in ms
        output_dir: Output directory

    Returns:
        Path to saved motion table
    """
    import pandas as pd

    print(f"\n   Exporting STAR format motion table...")

    num_frames, num_vertices, _ = interpolated_vertices.shape

    # Build table: columns = XYZ for each time point
    tables = []
    for i, t_ms in enumerate(target_times_ms):
        points = interpolated_vertices[i]  # [num_vertices, 3]
        df = pd.DataFrame({
            f'"X[t={t_ms:.1f}ms] (mm)"': points[:, 0],
            f'"Y[t={t_ms:.1f}ms] (mm)"': points[:, 1],
            f'"Z[t={t_ms:.1f}ms] (mm)"': points[:, 2],
        })
        tables.append(df)

    # Concatenate all columns
    motion_table = pd.concat(tables, axis=1)

    # Save as CSV without index, header already has quotes
    table_path = output_dir / 'motion_table.csv'
    motion_table.to_csv(str(table_path), index=False)

    print(f"   Motion table: {table_path}")
    print(f"   Table shape: {motion_table.shape[0]} vertices × {motion_table.shape[1]} columns")
    print(f"   Time range: {target_times_ms[0]:.1f}ms to {target_times_ms[-1]:.1f}ms")

    return table_path


def run_video_generation(
    required: RequiredFiles,
    config: PostProcessConfig,
    stl_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate video (MP4 or GIF) from STL sequence with 2x2 subfigure view.

    Creates a 2x2 layout:
    - Top left: Sagittal view (side)
    - Top right: Axial view (top)
    - Bottom left: Coronal view (front)
    - Bottom right: Perspective (3D oblique)

    Args:
        required: RequiredFiles object
        config: Post-processing configuration
        stl_dir: Directory containing STL files (auto-detect if None)

    Returns:
        Path to generated video file, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"VIDEO GENERATION (2x2 Subfigure View, MP4)")
    print(f"{'='*60}")

    # Check for pyvista
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True  # Enable off-screen rendering
    except ImportError:
        print("   Video generation requires PyVista. Install with: pip install pyvista")
        return None

    # Determine STL directory
    if stl_dir is None:
        # Try interpolated first, then regular
        interp_dir = config.output_dir / 'stl_interpolated'
        regular_dir = config.output_dir / 'stl_surfaces'

        if interp_dir.exists() and list(interp_dir.glob("*.stl")):
            stl_dir = interp_dir
            print(f"   Using interpolated STL files")
        elif regular_dir.exists() and list(regular_dir.glob("*.stl")):
            stl_dir = regular_dir
            print(f"   Using regular STL files")
        else:
            print(f"   ERROR: No STL files found. Run with --stl first.")
            return None

    # Find all STL files
    stl_files = sorted(stl_dir.glob("*.stl"))
    if not stl_files:
        print(f"   ERROR: No STL files found in {stl_dir}")
        return None

    print(f"   STL directory: {stl_dir}")
    print(f"   Found {len(stl_files)} STL files")

    # Auto-calculate FPS to achieve target video duration
    num_frames = len(stl_files)
    if config.video_duration_sec > 0 and num_frames > 1:
        auto_fps = max(1, round(num_frames / config.video_duration_sec))
        print(f"   Auto FPS: {auto_fps} ({num_frames} frames / {config.video_duration_sec}s target)")
        config.video_fps = auto_fps
    print(f"   FPS: {config.video_fps}")

    # Output paths
    output_dir = config.output_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / 'motion_2x2_view.mp4'

    print(f"   Output: {video_path}")

    # Generate 2x2 view frames
    frames_dir = output_dir / 'video_frames'
    frames_dir.mkdir(exist_ok=True)

    print(f"   Rendering {len(stl_files)} frames...")

    frame_paths = []
    for i, stl_path in enumerate(stl_files):
        frame_path = frames_dir / f"frame_{i:04d}.png"

        try:
            # Load mesh
            mesh = pv.read(str(stl_path))

            # Create 2x2 plotter
            plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1200, 1200))

            # Get mesh center and bounds for camera positioning
            center = mesh.center
            bounds = mesh.bounds
            extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

            # ITK-SNAP NIfTI convention for anatomical views:
            # Sagittal (top-left): Looking from right (-X direction), posterior up
            plotter.subplot(0, 0)
            plotter.add_mesh(mesh, color='lightblue', opacity=0.9, smooth_shading=True)
            cam_dist = extent * 2.0
            # Camera at +X looking toward -X, with +Z as up vector
            plotter.camera_position = [
                (center[0] + cam_dist, center[1], center[2]),  # Position: right side
                center,                                         # Focal point: center
                (0, 0, 1)                                       # Up vector: +Z (superior)
            ]
            plotter.add_text("Sagittal", font_size=12)

            # Axial (top-right): Looking from superior (+Z), anterior at top
            plotter.subplot(0, 1)
            plotter.add_mesh(mesh, color='lightblue', opacity=0.9, smooth_shading=True)
            # Camera at +Z looking toward -Z, with -Y as up vector (anterior up)
            plotter.camera_position = [
                (center[0], center[1], center[2] + cam_dist),  # Position: above
                center,                                         # Focal point: center
                (0, -1, 0)                                      # Up vector: -Y (anterior)
            ]
            plotter.add_text("Axial", font_size=12)

            # Coronal (bottom-left): Looking from anterior (-Y), superior up
            plotter.subplot(1, 0)
            plotter.add_mesh(mesh, color='lightblue', opacity=0.9, smooth_shading=True)
            # Camera at -Y looking toward +Y, with +Z as up vector
            plotter.camera_position = [
                (center[0], center[1] - cam_dist, center[2]),  # Position: front
                center,                                         # Focal point: center
                (0, 0, 1)                                       # Up vector: +Z (superior)
            ]
            plotter.add_text("Coronal", font_size=12)

            # Perspective view (bottom-right) - oblique 3D view
            plotter.subplot(1, 1)
            plotter.add_mesh(mesh, color='lightblue', opacity=0.9, smooth_shading=True)
            # Set oblique camera position
            cam_dist = extent * 2.5
            plotter.camera_position = [
                (center[0] + cam_dist * 0.7, center[1] + cam_dist * 0.7, center[2] + cam_dist * 0.5),
                center,
                (0, 0, 1)
            ]
            plotter.add_text("Perspective", font_size=12)

            # Save frame
            plotter.screenshot(str(frame_path))
            plotter.close()

            frame_paths.append(frame_path)

            if config.verbose or (i + 1) % 10 == 0:
                print(f"      Frame {i + 1}/{len(stl_files)}: {stl_path.name}")

        except Exception as e:
            print(f"      ERROR rendering {stl_path.name}: {e}")
            continue

    if not frame_paths:
        print("   ERROR: No frames were rendered successfully")
        return None

    # Create MP4 video from frames
    video_path = _create_mp4_from_frames(frame_paths, video_path, config.video_fps)

    if video_path:
        # Clean up frame images
        for p in frame_paths:
            p.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass  # Directory not empty

    return video_path


def _create_mp4_from_frames(frame_paths: List[Path], output_path: Path, fps: int) -> Optional[Path]:
    """Create MP4 video from frame images using imageio or cv2."""
    print(f"   Creating MP4 video...")

    try:
        import imageio.v2 as imageio

        # Read frames
        frames = [imageio.imread(str(p)) for p in frame_paths]

        # Write MP4 using imageio with explicit format
        imageio.mimwrite(str(output_path), frames, fps=fps, format='FFMPEG')

        duration = len(frames) / fps
        print(f"\n   Video saved: {output_path}")
        print(f"   Duration: {duration:.2f}s ({len(frames)} frames @ {fps} fps)")
        return output_path

    except ImportError:
        # Try cv2 as fallback
        try:
            import cv2
            from PIL import Image

            # Read first frame to get dimensions
            first_img = Image.open(str(frame_paths[0]))
            width, height = first_img.size

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame_path in frame_paths:
                img = cv2.imread(str(frame_path))
                writer.write(img)

            writer.release()

            duration = len(frame_paths) / fps
            print(f"\n   Video saved: {output_path}")
            print(f"   Duration: {duration:.2f}s ({len(frame_paths)} frames @ {fps} fps)")
            return output_path

        except ImportError:
            print("   MP4 creation requires imageio or opencv-python.")
            print("   Install with: pip install imageio[ffmpeg] or pip install opencv-python")
            return None

    except Exception as e:
        print(f"   ERROR creating MP4: {e}")
        return None


def run_postprocessing(config: PostProcessConfig):
    """
    Run the complete post-processing pipeline.

    Args:
        config: Post-processing configuration
    """
    print(f"\n{'='*70}")
    print("DAREG POST-PROCESSING PIPELINE")
    print(f"{'='*70}")

    # Validate inputs
    required = validate_required_files(config.output_dir, config)

    # Track generated files
    generated_files = {
        'stl': [],
        'interpolated': [],
        'motion_table': None,
        'video': None,
    }

    # Run STL generation
    if config.generate_stl:
        try:
            generated_files['stl'] = run_stl_generation(required, config)
        except Exception as e:
            print(f"   ERROR in STL generation: {e}")
            if config.verbose:
                import traceback
                traceback.print_exc()

    # Run temporal interpolation
    if config.interpolate:
        try:
            interp_paths, motion_table_path = run_temporal_interpolation(required, config)
            generated_files['interpolated'] = interp_paths
            generated_files['motion_table'] = motion_table_path
        except Exception as e:
            print(f"   ERROR in temporal interpolation: {e}")
            if config.verbose:
                import traceback
                traceback.print_exc()

    # Run video generation
    if config.generate_video:
        try:
            generated_files['video'] = run_video_generation(required, config)
        except Exception as e:
            print(f"   ERROR in video generation: {e}")
            if config.verbose:
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("POST-PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {config.output_dir}")

    if generated_files['stl']:
        print(f"   STL surfaces: {len(generated_files['stl'])} files (binary)")
    if generated_files['interpolated']:
        print(f"   Interpolated STL: {len(generated_files['interpolated'])} files")
    if generated_files['motion_table']:
        print(f"   Motion table: {generated_files['motion_table']}")
    if generated_files['video']:
        print(f"   Video: {generated_files['video']}")

    print(f"{'='*70}\n")

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="DAREG Post-Processing Pipeline - Generate STL surfaces and videos from registration results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (STL + 2x2 MP4 video)
  dareg postprocess --output_dir outputs_motion_test

  # Full pipeline with temporal interpolation and motion table
  dareg postprocess --output_dir outputs_motion_test --all --total_duration 500 --motion_table

  # STL only (no video)
  dareg postprocess --output_dir outputs_motion_test --stl --no_video

  # Interpolation with custom settings (500ms total, 1ms step, cubic spline)
  dareg postprocess --output_dir outputs_motion_test --interpolate \\
      --total_duration 500 --interp_step 1 --motion_table

  # Use frame-grid aligned segmentations (lower resolution)
  dareg postprocess --output_dir outputs_motion_test --use_frame_grid
        """
    )

    # Required arguments
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Path to registration output directory (from main_motion.py)'
    )

    # Processing options
    parser.add_argument(
        '--stl',
        action='store_true',
        help='Generate STL surface meshes from segmentations (binary format)'
    )
    parser.add_argument(
        '--interpolate',
        action='store_true',
        help='Generate temporally interpolated STL sequence (cubic spline)'
    )
    parser.add_argument(
        '--video',
        action='store_true',
        help='Generate video animation (requires pyvista)'
    )
    parser.add_argument(
        '--no_video',
        action='store_true',
        help='Disable video generation'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all post-processing steps (stl + interpolate + video)'
    )

    # STL options
    parser.add_argument(
        '--smoothing',
        type=int,
        default=2,
        help='Number of Laplacian smoothing iterations for STL (default: 2)'
    )
    parser.add_argument(
        '--use_frame_grid',
        action='store_true',
        help='Use frame-grid aligned segmentations (lower res). Default uses full-resolution segmentations/'
    )

    # Interpolation options
    parser.add_argument(
        '--total_duration',
        type=float,
        default=None,
        help='Total duration in milliseconds for all STL frames (e.g., 500 for 500ms)'
    )
    parser.add_argument(
        '--interp_step',
        type=float,
        default=1.0,
        help='Interpolation step in milliseconds (default: 1ms)'
    )
    parser.add_argument(
        '--interpolation_method',
        type=str,
        default='cubic',
        choices=['linear', 'cubic'],
        help='Temporal interpolation method (default: cubic - MIRTK style)'
    )
    parser.add_argument(
        '--motion_table',
        action='store_true',
        help='Export STAR format motion table (no index, quoted header)'
    )

    # Video options
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Video frames per second (default: 10, overridden by --video_duration)'
    )
    parser.add_argument(
        '--video_duration',
        type=float,
        default=10.0,
        help='Target video duration in seconds. FPS is auto-calculated to fit all frames (default: 10s)'
    )
    # General options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        args.stl = True
        args.interpolate = True
        args.video = True

    # Handle --no_video flag
    if args.no_video:
        args.video = False

    # If no processing options specified, default to STL + video
    if not (args.stl or args.interpolate or args.video):
        print("No processing options specified. Use --stl, --interpolate, --video, or --all")
        print("Defaulting to --stl --video")
        args.stl = True
        args.video = True

    # Build config
    config = PostProcessConfig(
        output_dir=Path(args.output_dir),
        generate_stl=args.stl,
        stl_smoothing_iterations=args.smoothing,
        stl_use_frame_grid=args.use_frame_grid,  # Default False = use segmentations/ folder
        interpolate=args.interpolate,
        interpolation_method=args.interpolation_method,
        total_duration_ms=args.total_duration,
        interp_step_ms=args.interp_step,
        export_motion_table=args.motion_table,
        generate_video=args.video and not args.no_video,
        video_fps=args.fps,
        video_duration_sec=args.video_duration,
        verbose=args.verbose,
    )

    # Run pipeline
    try:
        run_postprocessing(config)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
