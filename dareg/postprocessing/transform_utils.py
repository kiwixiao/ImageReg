"""
MIRTK-Style Transform Utilities for STL Deformation

This module implements MIRTK-equivalent transform application for mesh vertices.
Key algorithms:
1. WorldToLattice matrix construction - exactly like MIRTK's GetWorldToLatticeMatrix()
2. B-spline evaluation at arbitrary points
3. Sequential transform application (rigid + affine + FFD)

Reference: MIRTK/Modules/Image/src/BSplineInterpolateImageFunction.cc
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


def build_world_to_lattice_matrix(domain: Dict[str, Any]) -> np.ndarray:
    """
    Build WorldToLattice 4x4 matrix.

    This converts world coordinates (mm) to lattice/array indices where:
    - Origin (world) maps to index (0, 0, 0)
    - Opposite corner maps to index (size-1, size-1, size-1)

    The transformation is:
        lattice = (world - origin) / spacing

    For oblique grids with non-identity direction matrix:
        lattice = R^T @ (world - origin) / spacing

    Where R is the direction cosine matrix (columns are axis vectors).

    Args:
        domain: Dictionary with keys (all in X, Y, Z order as returned by deepali):
            - 'size': (X, Y, Z) grid dimensions
            - 'spacing': (dx, dy, dz) in mm
            - 'origin': (ox, oy, oz) in mm
            - 'direction': 3x3 orientation matrix (optional, defaults to identity)

    Returns:
        4x4 WorldToLattice matrix (converts world XYZ to array index XYZ)
    """
    # Domain info from deepali is in (X, Y, Z) order
    size_xyz = np.array(domain['size'])      # (X, Y, Z)
    spacing_xyz = np.array(domain['spacing'])  # (dx, dy, dz)
    origin_xyz = np.array(domain['origin'])    # (ox, oy, oz)
    direction = np.array(domain.get('direction', np.eye(3)))  # 3x3

    # WorldToLattice construction:
    # Step 1: Translate by negative origin (move origin to world origin)
    T_inv = np.eye(4)
    T_inv[:3, 3] = -origin_xyz

    # Step 2: Rotate by transpose of direction (undo orientation)
    # Direction matrix columns are axis vectors; transpose gives inverse
    R_inv = np.eye(4)
    R_inv[:3, :3] = direction.T

    # Step 3: Scale by inverse spacing (convert mm to voxel indices)
    S_inv = np.eye(4)
    S_inv[0, 0] = 1.0 / spacing_xyz[0]
    S_inv[1, 1] = 1.0 / spacing_xyz[1]
    S_inv[2, 2] = 1.0 / spacing_xyz[2]

    # Compose: W2L = S_inv @ R_inv @ T_inv
    # This gives: lattice = (1/spacing) * R^T * (world - origin)
    # Result: origin maps to (0,0,0), opposite corner maps to (size-1)
    w2l = S_inv @ R_inv @ T_inv

    return w2l


def world_to_lattice(vertices: np.ndarray, w2l_matrix: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates to lattice coordinates using W2L matrix.

    Args:
        vertices: [N, 3] array of world coordinates (X, Y, Z) in mm
        w2l_matrix: 4x4 WorldToLattice transformation matrix

    Returns:
        [N, 3] array of lattice coordinates (X, Y, Z)
    """
    # Add homogeneous coordinate
    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    vertices_h = np.hstack([vertices, ones])

    # Apply transformation
    lattice_h = (w2l_matrix @ vertices_h.T).T

    return lattice_h[:, :3]


def lattice_to_array_index(lattice_xyz: np.ndarray, size_xyz: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert lattice coordinates (X, Y, Z) to array indices for numpy.

    The W2L matrix produces array indices in (X, Y, Z) order.
    But numpy arrays (from torch tensors) are stored in (D, H, W) = (Z, Y, X) order.

    So we need to reverse the order for array indexing.

    Args:
        lattice_xyz: [N, 3] lattice coordinates in (X, Y, Z) order
        size_xyz: (X, Y, Z) grid dimensions (not used, kept for API compatibility)

    Returns:
        [N, 3] array indices in (Z, Y, X) order for numpy array indexing
    """
    # W2L gives indices in (X, Y, Z) order
    # Numpy arrays from torch are in (D, H, W) = (Z, Y, X) order
    # Reverse the order: (X, Y, Z) -> (Z, Y, X)
    array_zyx = lattice_xyz[:, [2, 1, 0]]

    return array_zyx


def evaluate_bspline_displacement(
    lattice_coords_xyz: np.ndarray,
    control_points: np.ndarray,
    domain: Dict[str, Any],
    boundary_value: float = 0.0,
) -> np.ndarray:
    """
    Evaluate B-spline displacement at lattice coordinates.

    Uses scipy's map_coordinates with cubic spline interpolation (order=3),
    which is equivalent to MIRTK's cubic B-spline evaluation.

    Note: The control_points array shape [3, D, H, W] uses torch ordering
    where D=Z, H=Y, W=X. The lattice_coords are in (X, Y, Z) order from
    the W2L matrix, so we reverse them for array indexing.

    Args:
        lattice_coords_xyz: [N, 3] lattice coordinates in (X, Y, Z) order
        control_points: [3, D, H, W] displacement control points (torch order)
        domain: Domain info dict with 'size' key (X, Y, Z order)
        boundary_value: Value for extrapolation outside domain (MIRTK's _CPValue)

    Returns:
        [N, 3] displacement vectors in normalized units (X, Y, Z)
    """
    # Control points shape [3, D, H, W] where D=Z, H=Y, W=X (torch order)
    # Lattice coords are in (X, Y, Z) order from W2L matrix
    # Need to convert to (Z, Y, X) = (D, H, W) order for array indexing

    size_xyz = tuple(domain['size'])

    # Convert lattice (X, Y, Z) to array indices (Z, Y, X) = (D, H, W)
    array_indices = lattice_to_array_index(lattice_coords_xyz, size_xyz)

    # Sample each displacement component
    N = lattice_coords_xyz.shape[0]
    displacements = np.zeros((N, 3), dtype=np.float32)

    # Control points channel order: params[0]=X, params[1]=Y, params[2]=Z displacement
    for c in range(3):
        # map_coordinates expects [ndim, N] for coordinates
        displacements[:, c] = map_coordinates(
            control_points[c],  # [D, H, W] = [Z, Y, X]
            array_indices.T,    # [3, N] in (Z, Y, X) order
            order=3,            # Cubic B-spline
            mode='constant',    # Extrapolate with constant value
            cval=boundary_value,
        )

    return displacements


def normalized_disp_to_world_disp(
    disp_normalized: np.ndarray,
    domain: Dict[str, Any],
) -> np.ndarray:
    """
    Convert normalized displacement (deepali internal) to world displacement (mm).

    Deepali stores FFD displacements in normalized [-1, 1] grid space.
    To convert to world units (mm), multiply by (extent / 2.0).

    For a grid with N points spaced by S, extent = (N-1) * S.

    Args:
        disp_normalized: [N, 3] normalized displacements (X, Y, Z)
        domain: Domain info with 'size' and 'spacing' in (X, Y, Z) order

    Returns:
        [N, 3] world displacements in mm (X, Y, Z)
    """
    # Domain from deepali is already in (X, Y, Z) order
    size_xyz = np.array(domain['size'])       # (X, Y, Z) - dense grid size
    spacing_xyz = np.array(domain['spacing'])  # (dx, dy, dz) - voxel spacing in mm

    # Compute extent in each axis
    # For N grid points with spacing S, extent = (N-1) * S
    extent_xyz = (size_xyz - 1) * spacing_xyz  # (X, Y, Z) extent in mm

    # Scale factor: extent / 2.0 (from normalized [-1,1] to world)
    scale = extent_xyz / 2.0

    return disp_normalized * scale


def apply_ffd_transform_mirtk_style(
    vertices: np.ndarray,
    transform_path: Union[str, Path],
    reference_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply FFD transform to vertices using deepali's native methods.

    This reconstructs the deepali FFD object and uses its transformation
    methods to ensure correct B-spline interpolation and coordinate handling.

    Algorithm:
    1. Load transform and reconstruct deepali FFD
    2. Convert world coordinates to normalized grid coordinates
    3. Apply FFD transformation using deepali's forward()
    4. Convert back to world coordinates

    Args:
        vertices: [N, 3] vertex positions in world coordinates (mm)
        transform_path: Path to saved transform (.pth file)
        reference_affine: Optional 4x4 affine matrix (unused, kept for API compatibility)

    Returns:
        [N, 3] transformed vertex positions in world coordinates
    """
    from deepali.core import Grid
    from deepali.spatial import FreeFormDeformation, StationaryVelocityFreeFormDeformation

    transform_path = Path(transform_path)

    # Load transform
    checkpoint = torch.load(transform_path, weights_only=False, map_location='cpu')
    state_dict = checkpoint['state_dict']
    transform_type = checkpoint.get('type', 'FreeFormDeformation')

    # Get domain info - REQUIRED for reconstruction
    domain = checkpoint.get('domain') or checkpoint.get('grid')
    if domain is None:
        raise ValueError(
            f"Transform has no domain/grid info. Re-run registration to save domain info: {transform_path}"
        )

    # Reconstruct the Grid from domain info
    # Domain is in (X, Y, Z) order
    size_xyz = domain['size']
    spacing_xyz = domain['spacing']
    origin_xyz = domain['origin']
    direction = np.array(domain.get('direction', np.eye(3)))

    # Create deepali Grid (expects shape in (D, H, W) = (Z, Y, X) order)
    shape_dhw = (size_xyz[2], size_xyz[1], size_xyz[0])  # Z, Y, X
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    grid = Grid(
        shape=shape_dhw,
        spacing=spacing_zyx,
        origin=origin_xyz,  # deepali accepts in (X, Y, Z) order
        direction=direction,
        align_corners=True,
    )

    # Get stride from checkpoint if available
    stride = checkpoint.get('stride', 4)

    # Reconstruct FFD or SVFFD and load state
    if transform_type == 'StationaryVelocityFreeFormDeformation':
        # SVFFD requires integration steps for ExpFlow
        steps = domain.get('integration_steps', checkpoint.get('integration_steps', 5))
        ffd = StationaryVelocityFreeFormDeformation(grid, stride=stride, steps=steps)
    else:
        # Standard FFD (no steps parameter)
        ffd = FreeFormDeformation(grid, stride=stride)

    # Load state dict
    ffd.load_state_dict(state_dict)
    ffd.eval()

    # Convert vertices to torch tensor
    vertices_torch = torch.from_numpy(vertices.astype(np.float32))

    # Convert world coordinates (mm) to normalized grid coordinates [-1, 1]
    # Using deepali's Grid.world_to_cube() method
    with torch.no_grad():
        # vertices_torch shape: [N, 3] in (X, Y, Z) order
        # Need to reshape for deepali: [1, N, 3]
        points = vertices_torch.unsqueeze(0)  # [1, N, 3]

        # World to normalized cube coordinates
        points_normalized = grid.world_to_cube(points)

        # Apply FFD transformation
        # FFD expects points in normalized coords, returns transformed normalized coords
        transformed_normalized = ffd.forward(points_normalized, grid=False)

        # Convert back to world coordinates
        transformed_world = grid.cube_to_world(transformed_normalized)

        # Remove batch dimension
        transformed_world = transformed_world.squeeze(0)  # [N, 3]

    return transformed_world.numpy()


def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (radians) to 3x3 rotation matrix.

    Uses ZYX convention (yaw-pitch-roll) which is common in medical imaging.

    Args:
        euler_angles: [3] array of (rx, ry, rz) in radians

    Returns:
        3x3 rotation matrix
    """
    # scipy uses 'xyz' lowercase for intrinsic rotations
    # deepali uses ZYX extrinsic, which is equivalent to xyz intrinsic
    rot = Rotation.from_euler('xyz', euler_angles, degrees=False)
    return rot.as_matrix()


def apply_rigid_transform(
    vertices: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply rigid transform: R @ (v - center) + center + t

    Args:
        vertices: [N, 3] vertex positions
        rotation: [3] Euler angles or [3, 3] rotation matrix
        translation: [3] translation vector
        center: [3] center of rotation (default: origin)

    Returns:
        [N, 3] transformed vertices
    """
    if rotation.ndim == 1:
        R = euler_to_rotation_matrix(rotation)
    else:
        R = rotation

    if center is None:
        center = np.zeros(3)

    # Rigid: rotate around center, then translate
    centered = vertices - center
    rotated = (R @ centered.T).T
    transformed = rotated + center + translation

    return transformed


def apply_affine_transform(
    vertices: np.ndarray,
    rotation: np.ndarray,
    scaling: np.ndarray,
    translation: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply affine transform: S @ R @ (v - center) + center + t

    Args:
        vertices: [N, 3] vertex positions
        rotation: [3] Euler angles or [3, 3] rotation matrix
        scaling: [3] scaling factors or [3, 3] diagonal matrix
        translation: [3] translation vector
        center: [3] center of transformation (default: origin)

    Returns:
        [N, 3] transformed vertices
    """
    if rotation.ndim == 1:
        R = euler_to_rotation_matrix(rotation)
    else:
        R = rotation

    if scaling.ndim == 1:
        S = np.diag(scaling)
    else:
        S = scaling

    if center is None:
        center = np.zeros(3)

    # Affine: scale, rotate around center, then translate
    centered = vertices - center
    scaled_rotated = (S @ R @ centered.T).T
    transformed = scaled_rotated + center + translation

    return transformed


def apply_sequential_transform_to_vertices(
    vertices: np.ndarray,
    transform_path: Union[str, Path],
) -> np.ndarray:
    """
    Apply SequentialTransform (rigid + affine + FFD) to vertices.

    This handles the alignment transform which composes:
    - RigidTransform (6 DOF: rotation + translation)
    - AffineTransform (12 DOF: scaling + rotation + translation)
    - FFD (B-spline control points)

    Args:
        vertices: [N, 3] vertex positions in world coordinates
        transform_path: Path to saved SequentialTransform

    Returns:
        [N, 3] transformed vertices
    """
    transform_path = Path(transform_path)
    checkpoint = torch.load(transform_path, weights_only=False, map_location='cpu')
    state_dict = checkpoint['state_dict']
    transform_type = checkpoint.get('type', 'unknown')

    # Handle different transform types
    if transform_type == 'SequentialTransform':
        # Sequential: apply each sub-transform in order
        vertices = _apply_sequential_subtransforms(vertices, state_dict, checkpoint)
    elif transform_type in ('FreeFormDeformation', 'StationaryVelocityFreeFormDeformation'):
        # Pure FFD
        vertices = apply_ffd_transform_mirtk_style(vertices, transform_path)
    elif transform_type == 'RigidTransform':
        vertices = _apply_rigid_from_state(vertices, state_dict)
    elif transform_type == 'AffineTransform':
        vertices = _apply_affine_from_state(vertices, state_dict)
    else:
        # Try to detect structure
        if any('_transforms.0' in k for k in state_dict):
            vertices = _apply_sequential_subtransforms(vertices, state_dict, checkpoint)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    return vertices


def _apply_sequential_subtransforms(
    vertices: np.ndarray,
    state_dict: Dict[str, torch.Tensor],
    checkpoint: Dict[str, Any],
) -> np.ndarray:
    """
    Apply sub-transforms in a SequentialTransform.

    Deepali SequentialTransform stores sub-transforms as:
    - _transforms.0.* : First transform (usually rigid)
    - _transforms.1.* : Second transform (usually affine)
    - _transforms.2.* : Third transform (usually FFD)

    CRITICAL: FFD sub-transforms are reconstructed using deepali's native
    FreeFormDeformation to ensure correct B-spline interpolation.
    """
    from deepali.core import Grid
    from deepali.spatial import FreeFormDeformation, StationaryVelocityFreeFormDeformation

    # Find all sub-transforms
    transform_indices = set()
    for key in state_dict:
        if key.startswith('_transforms.'):
            idx = int(key.split('.')[1])
            transform_indices.add(idx)

    # Get FFD-specific domain info if available
    ffd_domains = checkpoint.get('ffd_domains', {})

    for idx in sorted(transform_indices):
        prefix = f'_transforms.{idx}.'

        # Extract sub-state_dict
        sub_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        if not sub_state:
            continue

        # Determine sub-transform type
        if '_transforms.rotation.params' in sub_state or 'rotation.params' in sub_state:
            # Nested parametric transform (RigidTransform or AffineTransform)
            if '_transforms.scaling.params' in sub_state or 'scaling.params' in sub_state:
                vertices = _apply_affine_from_state(vertices, sub_state, prefix='')
            else:
                vertices = _apply_rigid_from_state(vertices, sub_state, prefix='')
        elif 'params' in sub_state:
            # FFD transform - reconstruct and apply using deepali
            sub_domain = ffd_domains.get(str(idx))  # Key is string (from ModuleDict)
            if sub_domain is None:
                # Fallback: try parent domain (legacy transforms)
                sub_domain = checkpoint.get('domain') or checkpoint.get('grid')
            if sub_domain is None:
                raise ValueError(
                    f"FFD sub-transform {idx} missing domain info. "
                    "Re-run registration to save FFD domain info."
                )

            # Reconstruct the Grid from domain info (X, Y, Z order)
            size_xyz = sub_domain['size']
            spacing_xyz = sub_domain['spacing']
            origin_xyz = sub_domain['origin']
            direction = np.array(sub_domain.get('direction', np.eye(3)))

            # Create deepali Grid (expects shape in (D, H, W) = (Z, Y, X) order)
            shape_dhw = (size_xyz[2], size_xyz[1], size_xyz[0])
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

            grid = Grid(
                shape=shape_dhw,
                spacing=spacing_zyx,
                origin=origin_xyz,
                direction=direction,
                align_corners=True,
            )

            # Get stride
            stride = sub_domain.get('stride', checkpoint.get('stride', 4))

            # Determine sub-transform type (FFD vs SVFFD)
            sub_transform_type = sub_domain.get('transform_type', 'FreeFormDeformation')

            # Reconstruct FFD or SVFFD based on type
            if sub_transform_type == 'StationaryVelocityFreeFormDeformation':
                # SVFFD requires integration steps for ExpFlow
                steps = sub_domain.get('integration_steps', 5)
                ffd = StationaryVelocityFreeFormDeformation(grid, stride=stride, steps=steps)
            else:
                # Standard FFD (no steps parameter)
                ffd = FreeFormDeformation(grid, stride=stride)

            # Create sub-state dict with proper keys for load_state_dict
            ffd_state = {'params': sub_state['params']}
            ffd.load_state_dict(ffd_state)
            ffd.eval()

            # Convert vertices to torch and apply
            vertices_torch = torch.from_numpy(vertices.astype(np.float32))

            with torch.no_grad():
                points = vertices_torch.unsqueeze(0)  # [1, N, 3]
                points_normalized = grid.world_to_cube(points)
                transformed_normalized = ffd.forward(points_normalized, grid=False)
                transformed_world = grid.cube_to_world(transformed_normalized)
                vertices = transformed_world.squeeze(0).numpy()

    return vertices


def _apply_rigid_from_state(
    vertices: np.ndarray,
    state_dict: Dict[str, torch.Tensor],
    prefix: str = '',
) -> np.ndarray:
    """Extract and apply rigid transform from state dict."""
    # Try different key patterns
    rot_key = f'{prefix}_transforms.rotation.params' if prefix else '_transforms.rotation.params'
    trans_key = f'{prefix}_transforms.translation.params' if prefix else '_transforms.translation.params'

    if rot_key not in state_dict:
        rot_key = f'{prefix}rotation.params' if prefix else 'rotation.params'
        trans_key = f'{prefix}translation.params' if prefix else 'translation.params'

    if rot_key not in state_dict:
        return vertices

    rotation = state_dict[rot_key].numpy().flatten()
    translation = state_dict[trans_key].numpy().flatten()

    return apply_rigid_transform(vertices, rotation, translation)


def _apply_affine_from_state(
    vertices: np.ndarray,
    state_dict: Dict[str, torch.Tensor],
    prefix: str = '',
) -> np.ndarray:
    """Extract and apply affine transform from state dict."""
    # Try different key patterns
    rot_key = f'{prefix}_transforms.rotation.params' if prefix else '_transforms.rotation.params'
    scale_key = f'{prefix}_transforms.scaling.params' if prefix else '_transforms.scaling.params'
    trans_key = f'{prefix}_transforms.translation.params' if prefix else '_transforms.translation.params'

    if rot_key not in state_dict:
        rot_key = f'{prefix}rotation.params' if prefix else 'rotation.params'
        scale_key = f'{prefix}scaling.params' if prefix else 'scaling.params'
        trans_key = f'{prefix}translation.params' if prefix else 'translation.params'

    if rot_key not in state_dict:
        return vertices

    rotation = state_dict[rot_key].numpy().flatten()
    scaling = state_dict.get(scale_key, torch.ones(3)).numpy().flatten()
    translation = state_dict[trans_key].numpy().flatten()

    return apply_affine_transform(vertices, rotation, scaling, translation)


# Convenience functions for the main postprocess pipeline

def load_and_apply_alignment_transform(
    vertices: np.ndarray,
    alignment_transform_path: Union[str, Path],
) -> np.ndarray:
    """
    Load and apply the alignment transform (segmentation space -> frame space).

    Args:
        vertices: [N, 3] vertex positions from segmentation (high-res)
        alignment_transform_path: Path to alignment_composed.pth or similar

    Returns:
        [N, 3] transformed vertices in frame space
    """
    return apply_sequential_transform_to_vertices(vertices, alignment_transform_path)


def load_and_apply_longitudinal_transform(
    vertices: np.ndarray,
    longitudinal_transform_path: Union[str, Path],
) -> np.ndarray:
    """
    Load and apply a longitudinal FFD transform (frame 0 -> frame N).

    Args:
        vertices: [N, 3] vertex positions (after alignment to frame 0)
        longitudinal_transform_path: Path to longitudinal_0_to_N.pth

    Returns:
        [N, 3] transformed vertices at frame N
    """
    return apply_ffd_transform_mirtk_style(vertices, longitudinal_transform_path)
