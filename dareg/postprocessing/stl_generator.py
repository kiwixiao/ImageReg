#!/usr/bin/env python3
"""
STL Surface Generator from Segmentation Masks

Converts binary segmentation masks to 3D surface meshes (STL format) using
marching cubes algorithm. Supports world coordinate transformation for
proper physical dimensions.

Features:
- Marching cubes surface extraction
- Optional Laplacian smoothing
- World coordinate transformation (voxels → mm)
- Batch processing for segmentation sequences
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from dataclasses import dataclass

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from stl import mesh as stl_mesh
    HAS_NUMPY_STL = True
except ImportError:
    HAS_NUMPY_STL = False


@dataclass
class SurfaceMesh:
    """Container for extracted surface mesh data"""
    vertices: np.ndarray  # (N, 3) array of vertex positions in world coords (mm)
    faces: np.ndarray     # (M, 3) array of face vertex indices
    normals: Optional[np.ndarray] = None  # (M, 3) array of face normals
    frame_index: int = 0  # Frame index for sequence
    time_point: float = 0.0  # Time point in seconds


class STLGenerator:
    """
    STL Surface Generator

    Converts segmentation masks to STL surface meshes using marching cubes.
    Handles world coordinate transformation to produce physically correct meshes.
    """

    def __init__(
        self,
        smoothing_iterations: int = 0,
        level: float = 0.5,
        step_size: int = 1,
    ):
        """
        Initialize STL generator.

        Args:
            smoothing_iterations: Number of Laplacian smoothing iterations (0 = no smoothing)
            level: Contour level for marching cubes (default 0.5 for binary masks)
            step_size: Step size for marching cubes (1 = full resolution)
        """
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image is required for STL generation. Install with: pip install scikit-image")

        self.smoothing_iterations = smoothing_iterations
        self.level = level
        self.step_size = step_size

    def extract_surface(
        self,
        segmentation: Union[np.ndarray, torch.Tensor],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        affine: Optional[np.ndarray] = None,
        frame_index: int = 0,
        time_point: float = 0.0,
    ) -> Optional[SurfaceMesh]:
        """
        Extract surface mesh from segmentation mask.

        Args:
            segmentation: 3D binary segmentation array [D, H, W]
            spacing: Voxel spacing in mm (z, y, x) - used if affine is None
            origin: World coordinate origin in mm - used if affine is None
            affine: 4x4 affine matrix for voxel-to-world transformation (preferred)
                   If provided, spacing and origin are ignored.
            frame_index: Frame index for sequence tracking
            time_point: Time point in seconds

        Returns:
            SurfaceMesh object, or None if no surface found
        """
        # Convert to numpy if needed
        if isinstance(segmentation, torch.Tensor):
            seg_np = segmentation.squeeze().cpu().numpy()
        else:
            seg_np = np.array(segmentation).squeeze()

        # Ensure float type for marching cubes
        seg_np = seg_np.astype(np.float32)

        # Check if there's any segmentation
        if seg_np.max() < self.level:
            return None

        try:
            # Run marching cubes with unit spacing (we'll apply affine after)
            if affine is not None:
                # Use unit spacing, apply full affine transformation after
                verts, faces, normals, values = marching_cubes(
                    seg_np,
                    level=self.level,
                    spacing=(1.0, 1.0, 1.0),
                    step_size=self.step_size,
                )

                # Apply full affine transformation: world = affine[:3,:3] @ voxel + affine[:3,3]
                # This correctly handles rotation, scaling, and translation
                verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
            else:
                # Legacy mode: use spacing and origin (no rotation)
                verts, faces, normals, values = marching_cubes(
                    seg_np,
                    level=self.level,
                    spacing=spacing,
                    step_size=self.step_size,
                )
                # Transform vertices to world coordinates (translation only)
                verts = verts + np.array(origin)

            # Apply smoothing if requested
            if self.smoothing_iterations > 0:
                verts = self._laplacian_smooth(verts, faces, self.smoothing_iterations)

            return SurfaceMesh(
                vertices=verts,
                faces=faces,
                normals=normals,
                frame_index=frame_index,
                time_point=time_point,
            )

        except Exception as e:
            print(f"   Warning: Failed to extract surface: {e}")
            return None

    def _laplacian_smooth(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int,
        lambda_factor: float = 0.5,
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to mesh vertices.

        Args:
            vertices: (N, 3) vertex positions
            faces: (M, 3) face vertex indices
            iterations: Number of smoothing iterations
            lambda_factor: Smoothing factor (0-1)

        Returns:
            Smoothed vertex positions
        """
        # Build adjacency list
        n_verts = len(vertices)
        neighbors = [set() for _ in range(n_verts)]

        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                neighbors[v1].add(v2)
                neighbors[v2].add(v1)

        # Smooth iteratively
        smoothed = vertices.copy()
        for _ in range(iterations):
            new_verts = smoothed.copy()
            for i in range(n_verts):
                if neighbors[i]:
                    neighbor_mean = np.mean(smoothed[list(neighbors[i])], axis=0)
                    new_verts[i] = smoothed[i] + lambda_factor * (neighbor_mean - smoothed[i])
            smoothed = new_verts

        return smoothed

    def save_stl(
        self,
        mesh: SurfaceMesh,
        output_path: Path,
        binary: bool = True,
    ) -> Path:
        """
        Save surface mesh to STL file.

        Args:
            mesh: SurfaceMesh to save
            output_path: Output file path
            binary: Save as binary STL (True) or ASCII (False)

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_TRIMESH:
            # Use trimesh (preferred)
            tri_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
            )
            tri_mesh.export(str(output_path), file_type='stl')

        elif HAS_NUMPY_STL:
            # Use numpy-stl
            stl_data = stl_mesh.Mesh(np.zeros(len(mesh.faces), dtype=stl_mesh.Mesh.dtype))
            for i, face in enumerate(mesh.faces):
                for j in range(3):
                    stl_data.vectors[i][j] = mesh.vertices[face[j]]
            if binary:
                stl_data.save(str(output_path), mode=stl_mesh.Mode.BINARY)
            else:
                stl_data.save(str(output_path), mode=stl_mesh.Mode.ASCII)

        else:
            # Fallback: write simple ASCII STL manually
            with open(output_path, 'w') as f:
                f.write("solid surface\n")
                for face in mesh.faces:
                    # Calculate face normal
                    v0, v1, v2 = mesh.vertices[face]
                    normal = np.cross(v1 - v0, v2 - v0)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    else:
                        normal = np.array([0, 0, 1])

                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    for vertex_idx in face:
                        v = mesh.vertices[vertex_idx]
                        f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                f.write("endsolid surface\n")

        return output_path


def generate_stl_from_segmentation(
    segmentation,
    output_path: Path,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    smoothing_iterations: int = 2,
) -> Optional[Path]:
    """
    Generate STL file from a single segmentation mask.

    Args:
        segmentation: 3D segmentation (deepali Image, tensor, or numpy array)
        output_path: Output STL file path
        spacing: Voxel spacing in mm
        origin: World coordinate origin
        smoothing_iterations: Number of smoothing iterations

    Returns:
        Path to saved STL file, or None if failed
    """
    generator = STLGenerator(smoothing_iterations=smoothing_iterations)

    # Extract segmentation data
    if hasattr(segmentation, 'tensor'):
        seg_np = segmentation.tensor().squeeze().cpu().numpy()
        # Get spacing from image if available
        if hasattr(segmentation, 'grid'):
            grid = segmentation.grid()
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
    elif isinstance(segmentation, torch.Tensor):
        seg_np = segmentation.squeeze().cpu().numpy()
    else:
        seg_np = np.array(segmentation).squeeze()

    mesh = generator.extract_surface(seg_np, spacing=spacing, origin=origin)

    if mesh is None:
        print(f"   Warning: No surface found in segmentation")
        return None

    return generator.save_stl(mesh, output_path)


def generate_stl_sequence(
    segmentation_sequence: List,
    output_dir: Path,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    affine_matrices: Optional[List[np.ndarray]] = None,
    smoothing_iterations: int = 2,
    temporal_resolution: Optional[float] = None,
    prefix: str = "seg_frame",
) -> List[Path]:
    """
    Generate STL sequence from segmentation sequence.

    Args:
        segmentation_sequence: List of segmentation images/tensors
        output_dir: Output directory for STL files
        spacing: Voxel spacing in mm (used if affine_matrices is None)
        origin: World coordinate origin (used if affine_matrices is None)
        affine_matrices: List of 4x4 affine matrices for each segmentation.
                        If provided, these are used for proper world coordinate
                        transformation including rotation (preferred over spacing/origin).
        smoothing_iterations: Number of smoothing iterations
        temporal_resolution: Time between frames in seconds (for metadata)
        prefix: Filename prefix

    Returns:
        List of paths to saved STL files
    """
    output_dir = Path(output_dir)
    stl_dir = output_dir / "stl_surfaces"
    stl_dir.mkdir(parents=True, exist_ok=True)

    generator = STLGenerator(smoothing_iterations=smoothing_iterations)
    saved_paths = []

    # Determine if using affine matrices
    use_affine = affine_matrices is not None and len(affine_matrices) == len(segmentation_sequence)
    if use_affine:
        print(f"   Using full affine transformation for world coordinates")
    else:
        print(f"   Using spacing/origin for world coordinates (no rotation)")

    print(f"   Generating STL surfaces for {len(segmentation_sequence)} frames...")

    for frame_idx, seg in enumerate(segmentation_sequence):
        # Extract segmentation data
        if hasattr(seg, 'tensor'):
            seg_np = seg.tensor().squeeze().cpu().numpy()
            # Get spacing from first image if available (fallback)
            if frame_idx == 0 and hasattr(seg, 'grid') and not use_affine:
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

        # Calculate time point
        time_point = frame_idx * temporal_resolution if temporal_resolution else 0.0

        # Extract surface with appropriate transformation
        if use_affine:
            mesh = generator.extract_surface(
                seg_np,
                affine=affine_matrices[frame_idx],
                frame_index=frame_idx,
                time_point=time_point,
            )
        else:
            mesh = generator.extract_surface(
                seg_np,
                spacing=spacing,
                origin=origin,
                frame_index=frame_idx,
                time_point=time_point,
            )

        if mesh is not None:
            output_path = stl_dir / f"{prefix}_{frame_idx:03d}.stl"
            saved_path = generator.save_stl(mesh, output_path)
            saved_paths.append(saved_path)
            print(f"   Created {output_path.name} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
        else:
            print(f"   Warning: No surface found for frame {frame_idx}")

    print(f"   Generated {len(saved_paths)} STL files in {stl_dir}")
    return saved_paths
