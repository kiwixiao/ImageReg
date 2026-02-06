"""
DAREG Transform Composition

Compose, invert, and chain transforms for motion tracking.
Replicates MIRTK's compose-dofs functionality using deepali.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch

import deepali.spatial as spatial
from deepali.spatial import (
    SpatialTransform,
    SequentialTransform,
    RigidTransform,
    AffineTransform,
    FreeFormDeformation,
    StationaryVelocityFreeFormDeformation,
)
from deepali.data import Image, FlowFields
from deepali.core import Grid

from ..utils.logging_config import get_logger

logger = get_logger("composer")


@dataclass
class ComposedTransform:
    """
    Result of transform composition

    Attributes:
        transform: The composed transform
        source_idx: Source frame index
        target_idx: Target frame index
        intermediate_indices: List of intermediate frame indices
    """
    transform: SpatialTransform
    source_idx: int
    target_idx: int
    intermediate_indices: List[int]


class TransformComposer:
    """
    Compose and chain transforms for motion tracking

    Implements the MIRTK composition strategy:
    1. Register consecutive frames: T(0→1), T(1→2), T(2→3), ...
    2. Compose to get longitudinal: T(0→2) = T(0→1) ∘ T(1→2)
    3. Refine each longitudinal transform

    In deepali, composition is done by:
    - SequentialTransform for chaining multiple transforms
    - Flow field composition via displacement field addition
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize composer

        Args:
            device: Computation device
        """
        self.device = torch.device(device)

    def compose_sequential(
        self,
        transforms: List[SpatialTransform],
        grid: Grid,
    ) -> SpatialTransform:
        """
        Compose a sequence of transforms: T_total = T_1 ∘ T_2 ∘ ... ∘ T_n

        For motion tracking: T(0→N) = T(0→1) ∘ T(1→2) ∘ ... ∘ T(N-1→N)

        Args:
            transforms: List of transforms in order of application
            grid: Reference grid for composition

        Returns:
            Composed transform
        """
        if not transforms:
            raise ValueError("Need at least one transform to compose")

        if len(transforms) == 1:
            return transforms[0]

        # Create SequentialTransform
        composed = SequentialTransform(*transforms)
        composed = composed.to(self.device)

        logger.info(f"Composed {len(transforms)} transforms")

        return composed

    def compose_flow_fields(
        self,
        flow1: FlowFields,
        flow2: FlowFields,
        grid: Grid,
    ) -> FlowFields:
        """
        Compose two flow fields: u_composed(x) = u1(x) + u2(x + u1(x))

        This is the correct composition for displacement fields.

        Args:
            flow1: First flow field (applied first)
            flow2: Second flow field (applied second)
            grid: Reference grid

        Returns:
            Composed flow field
        """
        # Get displacement tensors
        u1 = flow1.tensor()  # [N, D, H, W, 3] or [N, 3, D, H, W]
        u2 = flow2.tensor()

        # Ensure consistent format [N, 3, D, H, W]
        if u1.shape[-1] == 3:
            u1 = u1.permute(0, 4, 1, 2, 3)
        if u2.shape[-1] == 3:
            u2 = u2.permute(0, 4, 1, 2, 3)

        # Get grid coordinates
        coords = grid.coords(device=self.device)  # [D, H, W, 3]
        coords = coords.unsqueeze(0)  # [1, D, H, W, 3]

        # Warp u2 by u1: sample u2 at (x + u1(x))
        # Convert u1 from [N, 3, D, H, W] to [N, D, H, W, 3] for grid_sample
        u1_sample = u1.permute(0, 2, 3, 4, 1)

        # Create warped coordinates
        warped_coords = coords + u1_sample

        # Normalize to [-1, 1] for grid_sample
        warped_coords_norm = grid.world_to_cube(warped_coords)

        # Sample u2 at warped coordinates
        u2_warped = torch.nn.functional.grid_sample(
            u2,
            warped_coords_norm,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        # Compose: u_total = u1 + u2(x + u1)
        u_composed = u1 + u2_warped

        # Convert back to FlowFields format
        u_composed = u_composed.permute(0, 2, 3, 4, 1)  # [N, D, H, W, 3]

        return FlowFields(u_composed, grid=grid)

    def compose_pairwise_to_longitudinal(
        self,
        pairwise_transforms: List[Tuple[int, int, SpatialTransform]],
        grid: Grid,
    ) -> List[ComposedTransform]:
        """
        Compose pairwise transforms to get longitudinal transforms to frame 0

        Given: T(0→1), T(1→2), T(2→3), ...
        Output: T(0→1), T(0→2), T(0→3), ...

        This replicates MIRTK's compose_longitudinal_dofs

        Args:
            pairwise_transforms: List of (source_idx, target_idx, transform)
                Expected to be consecutive: (1,0,T01), (2,1,T12), ...
            grid: Reference grid

        Returns:
            List of ComposedTransform from frame 0 to each frame
        """
        # Sort by source index
        sorted_pairs = sorted(pairwise_transforms, key=lambda x: x[0])

        longitudinal = []

        # First transform is already from 1 to 0
        first_src, first_tgt, first_transform = sorted_pairs[0]
        longitudinal.append(ComposedTransform(
            transform=first_transform,
            source_idx=first_src,
            target_idx=first_tgt,
            intermediate_indices=[],
        ))

        # Compose subsequent transforms
        for i in range(1, len(sorted_pairs)):
            src_idx, tgt_idx, current_transform = sorted_pairs[i]

            # Get previous longitudinal transform (0 → i)
            prev_composed = longitudinal[-1]

            # Compose: T(0→i+1) = T(0→i) ∘ T(i→i+1)
            # Note: current_transform is T(i+1 → i), we need to chain it
            composed = self.compose_sequential(
                [prev_composed.transform, current_transform],
                grid,
            )

            # Track intermediate indices
            intermediates = prev_composed.intermediate_indices + [tgt_idx]

            longitudinal.append(ComposedTransform(
                transform=composed,
                source_idx=src_idx,
                target_idx=0,  # All longitudinal go to frame 0
                intermediate_indices=intermediates,
            ))

        logger.info(f"Composed {len(longitudinal)} longitudinal transforms")

        return longitudinal

    def extract_flow_field(
        self,
        transform: SpatialTransform,
        grid: Grid,
    ) -> FlowFields:
        """
        Extract flow/displacement field from any transform type

        Args:
            transform: Any spatial transform
            grid: Grid to sample displacement field on

        Returns:
            FlowFields representing the displacement
        """
        transform = transform.to(self.device)

        with torch.no_grad():
            # Different transform types have different methods
            if hasattr(transform, 'flow'):
                # SVFFD and some others have flow() method
                flow = transform.flow(grid, device=self.device)
            elif hasattr(transform, 'disp'):
                # FFD and others have disp() method
                flow = transform.disp(grid)
                flow = FlowFields(flow, grid=grid)
            else:
                # For rigid/affine, compute displacement manually
                coords = grid.coords(device=self.device)
                coords = coords.unsqueeze(0)  # [1, D, H, W, 3]

                # Apply transform
                warped_coords = transform(coords)

                # Displacement = warped - original
                displacement = warped_coords - coords

                flow = FlowFields(displacement, grid=grid)

        return flow

    def save_transform(
        self,
        transform: SpatialTransform,
        path: Union[str, Path],
        metadata: Optional[dict] = None,
    ):
        """
        Save transform to file with complete grid domain info (MIRTK DOF-style).

        IMPORTANT: For FFD transforms, we save the FULL grid domain info so the
        transform can be applied to points in ANY coordinate system. This
        replicates how MIRTK DOF files are self-contained.

        Saved domain info includes:
        - size: Grid dimensions (D, H, W) - deepali order
        - spacing: Grid spacing in mm (dz, dy, dx)
        - origin: Grid origin in world coords (oz, oy, ox)
        - direction: 3x3 orientation matrix for rotated/oblique grids

        This enables MIRTK-style WorldToLattice matrix construction for
        applying transforms to arbitrary points (e.g., STL vertices).

        Args:
            transform: Transform to save
            path: Output path
            metadata: Optional metadata dict
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'state_dict': transform.state_dict(),
            'type': type(transform).__name__,
        }

        # Save COMPLETE grid info for FFD transforms (MIRTK DOF-style)
        # This enables WorldToLattice matrix construction for transform application
        if hasattr(transform, 'grid') and callable(transform.grid):
            grid = transform.grid()

            # Extract direction matrix (3x3 orientation)
            direction = grid.direction()
            if hasattr(direction, 'tolist'):
                direction_list = direction.tolist()
            else:
                direction_list = direction.cpu().numpy().tolist()

            # Save in 'domain' key (MIRTK-style) for clarity
            # Also keep 'grid' key for backward compatibility
            # NOTE: deepali returns size/spacing/origin in (X, Y, Z) order!
            # Handle both torch.Size (no .tolist()) and torch.Tensor (has .tolist())
            size_val = grid.size()
            spacing_val = grid.spacing()
            origin_val = grid.origin()

            domain_info = {
                'size': tuple(size_val) if isinstance(size_val, torch.Size) else tuple(size_val.tolist()),
                'spacing': tuple(spacing_val) if isinstance(spacing_val, torch.Size) else tuple(spacing_val.tolist()),
                'origin': tuple(origin_val) if isinstance(origin_val, torch.Size) else tuple(origin_val.tolist()),
                'direction': direction_list,                # 3x3 orientation matrix
            }
            save_dict['domain'] = domain_info
            save_dict['grid'] = domain_info  # Backward compatibility

            # Also save stride if available (for control point spacing)
            if hasattr(transform, 'stride'):
                save_dict['stride'] = transform.stride
                # Compute control point spacing in mm for reference
                stride = transform.stride
                spacing = grid.spacing()
                cp_spacing = tuple((s * st).item() if hasattr(s * st, 'item') else s * st
                                   for s, st in zip(spacing, stride))
                save_dict['domain']['control_point_spacing'] = cp_spacing

            # SVFFD-specific: Save integration steps for ExpFlow
            if hasattr(transform, 'exp') and hasattr(transform.exp, 'steps'):
                save_dict['integration_steps'] = transform.exp.steps
                save_dict['domain']['integration_steps'] = transform.exp.steps
                logger.debug(f"Saved SVFFD integration steps: {transform.exp.steps}")

        # CRITICAL: For SequentialTransform containing FFD, save FFD's domain separately
        # The main 'domain' key holds the first sub-transform's grid (image domain),
        # but FFD sub-transforms need their control point grid for WorldToLattice matrix
        if type(transform).__name__ == 'SequentialTransform':
            ffd_domains = {}
            if hasattr(transform, '_transforms'):
                for name, sub_transform in transform._transforms.items():
                    sub_type = type(sub_transform).__name__
                    if sub_type in ('FreeFormDeformation', 'StationaryVelocityFreeFormDeformation'):
                        sub_grid = sub_transform.grid()
                        sub_direction = sub_grid.direction()
                        if hasattr(sub_direction, 'tolist'):
                            sub_dir_list = sub_direction.tolist()
                        else:
                            sub_dir_list = sub_direction.cpu().numpy().tolist()

                        # Handle both torch.Size and torch.Tensor
                        sub_size = sub_grid.size()
                        sub_spacing = sub_grid.spacing()
                        sub_origin = sub_grid.origin()

                        ffd_domain = {
                            'size': tuple(sub_size) if isinstance(sub_size, torch.Size) else tuple(sub_size.tolist()),
                            'spacing': tuple(sub_spacing) if isinstance(sub_spacing, torch.Size) else tuple(sub_spacing.tolist()),
                            'origin': tuple(sub_origin) if isinstance(sub_origin, torch.Size) else tuple(sub_origin.tolist()),
                            'direction': sub_dir_list,
                            'transform_type': sub_type,  # FFD vs SVFFD distinction
                        }
                        if hasattr(sub_transform, 'stride'):
                            ffd_domain['stride'] = sub_transform.stride
                        # SVFFD-specific: Save integration steps
                        if hasattr(sub_transform, 'exp') and hasattr(sub_transform.exp, 'steps'):
                            ffd_domain['integration_steps'] = sub_transform.exp.steps
                        ffd_domains[name] = ffd_domain

            if ffd_domains:
                save_dict['ffd_domains'] = ffd_domains
                logger.debug(f"Saved FFD domain info for {len(ffd_domains)} sub-transforms")

        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)
        logger.debug(f"Saved transform: {path.name}")

    def load_transform(
        self,
        path: Union[str, Path],
        grid: Optional[Grid] = None,
    ) -> SpatialTransform:
        """
        Load transform from file

        Args:
            path: Path to saved transform
            grid: Optional grid for reconstruction

        Returns:
            Loaded transform
        """
        path = Path(path)
        saved = torch.load(path, map_location=self.device)

        transform_type = saved.get('type', 'unknown')
        state_dict = saved['state_dict']

        # Reconstruct based on type
        if transform_type == 'RigidTransform':
            transform = RigidTransform(grid)
        elif transform_type == 'AffineTransform':
            transform = AffineTransform(grid)
        elif transform_type == 'FreeFormDeformation':
            transform = FreeFormDeformation(grid)
        elif transform_type == 'StationaryVelocityFreeFormDeformation':
            transform = StationaryVelocityFreeFormDeformation(grid)
        elif transform_type == 'SequentialTransform':
            # SequentialTransform needs special handling
            logger.warning("SequentialTransform loading requires manual reconstruction")
            return None
        else:
            logger.warning(f"Unknown transform type: {transform_type}")
            return None

        transform.load_state_dict(state_dict)
        transform = transform.to(self.device)

        logger.debug(f"Loaded transform: {path.name} ({transform_type})")

        return transform


def compose_transforms(
    transforms: List[SpatialTransform],
    grid: Grid,
    device: str = "cpu",
) -> SpatialTransform:
    """
    Convenience function to compose multiple transforms

    Args:
        transforms: List of transforms to compose
        grid: Reference grid
        device: Computation device

    Returns:
        Composed transform
    """
    composer = TransformComposer(device=device)
    return composer.compose_sequential(transforms, grid)
