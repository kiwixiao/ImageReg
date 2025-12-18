"""
DAREG Motion Registration Pipeline

Implements MIRTK-style motion tracking for 4D image sequences:
1. Alignment: Register static high-res to frame 0
2. Pairwise: Register consecutive frames (0→1, 1→2, ...)
3. Compose: Build longitudinal transforms (0→N)
4. Refine: Fine-tune each longitudinal transform
5. Propagate: Transform segmentation through all frames
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deepali" / "src"))

import deepali.spatial as spatial
from deepali.spatial import SpatialTransform, ImageTransformer
from deepali.data import Image
from deepali.core import Grid

from .base import BaseRegistration
from .rigid import RigidRegistration
from .affine import AffineRegistration
from .ffd import FFDRegistration
from .composer import TransformComposer, ComposedTransform
from ..data.image_4d import Image4D, MotionSequence, create_frame_pairs
from ..data.saver import save_nifti, save_segmentation
from ..preprocessing import normalize_intensity, create_common_grid
from ..postprocessing import transform_segmentation
from ..postprocessing.transformer import apply_transform_preserve_resolution
from ..utils.logging_config import get_logger, Timer

logger = get_logger("motion")


def _log_image_info(image: Image, name: str, level: str = "debug") -> None:
    """Log detailed image information for debugging

    Args:
        image: deepali Image object
        name: Name/description of the image
        level: Log level ("debug" or "info")
    """
    tensor = image.tensor()
    grid = image.grid()

    # Get grid properties
    shape = tuple(grid.shape)
    spacing = tuple(s.item() if hasattr(s, 'item') else s for s in grid.spacing())
    origin = tuple(o.item() if hasattr(o, 'item') else o for o in grid.origin())

    # Get tensor stats (handle integer tensors like segmentations)
    t_min = float(tensor.min())
    t_max = float(tensor.max())
    # mean() doesn't work on integer tensors, convert to float first
    t_mean = float(tensor.float().mean())

    log_fn = logger.debug if level == "debug" else logger.info
    log_fn(f"  [{name}]")
    log_fn(f"    Tensor shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")
    log_fn(f"    Grid shape:   {shape}")
    log_fn(f"    Spacing (mm): ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})")
    log_fn(f"    Origin (mm):  ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    log_fn(f"    Intensity:    min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}")


def _log_transform_info(transform, name: str) -> None:
    """Log detailed transform information for debugging

    Args:
        transform: deepali transform object
        name: Name/description of the transform
    """
    logger.debug(f"  [{name}]")
    logger.debug(f"    Type: {type(transform).__name__}")

    if hasattr(transform, 'grid'):
        try:
            grid = transform.grid()
            if grid is not None:
                logger.debug(f"    Grid shape: {tuple(grid.shape)}")
        except Exception:
            pass

    # Log number of parameters
    total_params = 0
    for p in transform.parameters():
        total_params += p.numel()
    logger.debug(f"    Parameters: {total_params}")


def _normalize_image(image: Image) -> Image:
    """Normalize image intensity to [0, 1]"""
    tensor = image.tensor()
    normalized_tensor = normalize_intensity(tensor, method="minmax")
    return Image(data=normalized_tensor, grid=image.grid())


def _resample_to_grid(image: Image, target_grid: Grid) -> Image:
    """Resample image to target grid using deepali's built-in sample() method

    Uses deepali's Image.sample() which properly handles:
    - All coordinate transformations
    - World space resampling
    - Correct interpolation
    """
    # Use deepali's built-in sample() method - handles ALL coordinate transforms correctly
    return image.sample(target_grid)


def _construct_affine_from_image4d(image_4d: Image4D) -> np.ndarray:
    """
    Construct 4x4 affine matrix from Image4D spacing, origin, and direction.

    The affine matrix encodes the mapping from voxel indices to world coordinates:
        world_coord = affine @ [voxel_i, voxel_j, voxel_k, 1]^T

    Args:
        image_4d: Image4D object with spacing, origin, and direction

    Returns:
        4x4 affine matrix as numpy array
    """
    # Get spacing (stored as dZ, dY, dX in Image4D)
    dz, dy, dx = image_4d.spacing

    # Get origin
    ox, oy, oz = image_4d.origin

    # Get direction cosine matrix (flattened 9 elements, row-major)
    direction = image_4d.direction
    if len(direction) == 9:
        R = np.array(direction).reshape(3, 3)
    else:
        R = np.eye(3)

    # Construct affine: R * diag(spacing) with origin
    # NIfTI convention: spacing in X, Y, Z order
    affine = np.eye(4)
    affine[:3, :3] = R @ np.diag([dx, dy, dz])
    affine[:3, 3] = [ox, oy, oz]

    return affine


def _construct_affine_from_grid(grid: Grid) -> np.ndarray:
    """
    Construct 4x4 affine matrix from deepali Grid object.

    This is used to save segmentation files with the correct affine
    that matches their actual grid (data coordinate space).

    Args:
        grid: deepali Grid object

    Returns:
        4x4 affine matrix as numpy array (in LPS convention, same as deepali)

    Note:
        deepali Grid stores spacing/origin in (X, Y, Z) order (same as SimpleITK/ITK).
        Grid.shape returns (D, H, W) = (Z, Y, X) - reversed order for torch tensors.
        So spacing() and origin() already return (X, Y, Z) order - no reordering needed!
    """
    # deepali Grid returns spacing/origin in (X, Y, Z) order (SimpleITK convention)
    spacing = grid.spacing()  # [dX, dY, dZ]
    origin = grid.origin()    # [oX, oY, oZ] in world coords
    direction = grid.direction()  # 3x3 direction cosine matrix

    # Convert tensors to numpy
    if hasattr(spacing, 'cpu'):
        spacing = spacing.cpu().numpy()
    if hasattr(origin, 'cpu'):
        origin = origin.cpu().numpy()
    if hasattr(direction, 'cpu'):
        direction = direction.cpu().numpy()

    # Grid returns in (X, Y, Z) order - no reordering needed!
    dx, dy, dz = spacing[0], spacing[1], spacing[2]
    ox, oy, oz = origin[0], origin[1], origin[2]

    # Direction is already in (X, Y, Z) order - no reordering needed!
    R = direction

    # Construct affine: R * diag(spacing) with origin
    affine = np.eye(4)
    affine[:3, :3] = R @ np.diag([dx, dy, dz])
    affine[:3, 3] = [ox, oy, oz]

    return affine


@dataclass
class PairwiseResult:
    """Result of pairwise registration between two frames"""
    source_idx: int
    target_idx: int
    transform: SpatialTransform
    final_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlignmentIntermediates:
    """Intermediate results from alignment stage (for saving like old code)"""
    source_image: Optional[Image] = None  # Original static image
    target_image: Optional[Image] = None  # Frame 0
    rigid_warped: Optional[Image] = None  # After rigid registration
    affine_warped: Optional[Image] = None  # After affine registration
    ffd_warped: Optional[Image] = None  # After FFD/SVFFD registration
    rigid_transform: Optional[SpatialTransform] = None
    affine_transform: Optional[SpatialTransform] = None
    ffd_transform: Optional[SpatialTransform] = None
    common_grid: Optional[Grid] = None
    convergence_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionResult:
    """
    Complete result of motion registration pipeline

    Attributes:
        alignment_transform: Static→Frame0 alignment (if performed)
        alignment_intermediates: Per-stage intermediate results from alignment
        pairwise_transforms: List of consecutive pairwise transforms
        longitudinal_transforms: List of Frame0→FrameN transforms
        segmentation_sequence: Segmentation at each frame
        frame_metrics: Quality metrics at each frame
        total_time: Total processing time
    """
    alignment_transform: Optional[SpatialTransform] = None
    alignment_intermediates: Optional[AlignmentIntermediates] = None
    pairwise_transforms: List[PairwiseResult] = field(default_factory=list)
    longitudinal_transforms: List[ComposedTransform] = field(default_factory=list)
    segmentation_sequence: List[Image] = field(default_factory=list)
    frame_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    total_time: float = 0.0


class MotionRegistration:
    """
    Motion Registration Pipeline

    Implements the full MIRTK motion tracking workflow using deepali.

    Workflow:
    1. ALIGNMENT (optional): Register static 3D to first frame of 4D
       - Static image has manual segmentation
       - Brings segmentation into 4D coordinate space

    2. PAIRWISE REGISTRATION: Register consecutive frames
       - Frame 0→1, 1→2, 2→3, ...
       - Small deformations between adjacent frames
       - Uses FFD or SVFFD

    3. COMPOSE LONGITUDINAL: Chain pairwise to get 0→N
       - T(0→2) = T(0→1) ∘ T(1→2)
       - T(0→3) = T(0→2) ∘ T(2→3)
       - ...

    4. REFINE LONGITUDINAL: Fine-tune each 0→N transform
       - Use composed transform as initialization
       - Run 1 level of pyramid for refinement

    5. PROPAGATE SEGMENTATION: Transform seg through all frames
       - Apply each longitudinal transform to seg_0
       - Get seg_1, seg_2, ..., seg_N
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Any] = None,
        registration_model: str = "rigid+affine+ffd",
    ):
        """
        Initialize motion registration pipeline

        Args:
            device: Computation device
            config: Registration configuration
            registration_model: Model for alignment ("rigid", "rigid+affine", "rigid+affine+ffd", "rigid+affine+svffd")
        """
        self.device = torch.device(device)
        self.config = config
        self.registration_model = registration_model

        # Initialize transform composer
        self.composer = TransformComposer(device=device)

        logger.info(f"Motion Registration initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Registration model: {registration_model}")

    def register_alignment(
        self,
        static_image: Image,
        frame_0: Image,
        segmentation: Optional[Image] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[SpatialTransform, Image, Optional[Image]]:
        """
        Step 1: Alignment - Register static high-res image to frame 0

        This brings the manual segmentation into the 4D coordinate space.
        Uses sequential registration: Rigid → Affine → FFD (based on registration_model)

        Args:
            static_image: 3D static high-resolution image
            frame_0: First frame of 4D sequence
            segmentation: Optional segmentation on static image
            output_dir: Optional output directory for saving intermediate visualizations

        Returns:
            Tuple of (alignment_transform, aligned_static, aligned_segmentation)
        """
        logger.info("=" * 60)
        logger.info("STEP 1: ALIGNMENT (Static → Frame 0)")
        logger.info("=" * 60)

        with Timer("Alignment registration"):
            # Log input images
            logger.debug("INPUT IMAGES:")
            _log_image_info(static_image, "Static (source)", level="debug")
            _log_image_info(frame_0, "Frame 0 (target)", level="debug")
            if segmentation is not None:
                _log_image_info(segmentation, "Segmentation", level="debug")

            # Normalize images
            static_norm = _normalize_image(static_image)
            frame0_norm = _normalize_image(frame_0)

            # Create common grid from image grids (use target=frame0 grid by default)
            common_grid = create_common_grid(
                static_norm.grid(), frame0_norm.grid(), method="target"
            )
            logger.info(f"Common grid shape: {tuple(common_grid.shape)}")
            logger.debug(f"  Common grid spacing: {tuple(s.item() for s in common_grid.spacing())}")
            logger.debug(f"  Common grid origin: {tuple(o.item() for o in common_grid.origin())}")

            # Resample both images to common grid
            static_resampled = _resample_to_grid(static_norm, common_grid)
            frame0_resampled = _resample_to_grid(frame0_norm, common_grid)
            logger.info(f"Resampled static: {static_resampled.tensor().shape}")
            logger.info(f"Resampled frame0: {frame0_resampled.tensor().shape}")
            logger.debug("RESAMPLED IMAGES:")
            _log_image_info(static_resampled, "Static resampled", level="debug")
            _log_image_info(frame0_resampled, "Frame 0 resampled", level="debug")

            # Initialize transforms list and intermediate results for visualization
            transforms = []
            current_source = static_resampled
            target_image = frame0_resampled

            # Store intermediate results for visualization and saving (like old code)
            intermediate_results = {
                'rigid': None,
                'affine': None,
                'ffd': None,
            }
            intermediate_transforms = {
                'rigid': None,
                'affine': None,
                'ffd': None,
            }
            convergence_data = {}

            # Parse registration model
            stages = self.registration_model.split("+")

            # Rigid stage
            if "rigid" in stages:
                logger.info("Alignment: Rigid registration...")
                logger.debug("  RIGID STAGE INPUT:")
                logger.debug(f"    Source tensor shape: {current_source.tensor().shape}")
                logger.debug(f"    Target tensor shape: {target_image.tensor().shape}")

                # Extract rigid config if available, else use defaults
                rigid_kwargs = {"device": self.device}
                if self.config and hasattr(self.config, "rigid"):
                    rc = self.config.rigid
                    rigid_kwargs.update({
                        "pyramid_levels": rc.pyramid_levels,
                        "iterations_per_level": rc.iterations_per_level,
                        "learning_rates": rc.learning_rates_per_level,
                    })
                    if hasattr(rc, "convergence"):
                        rigid_kwargs["convergence_delta"] = rc.convergence.min_delta
                        rigid_kwargs["convergence_patience"] = rc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    rigid_kwargs["num_bins"] = sc.num_bins
                    rigid_kwargs["foreground_threshold"] = sc.foreground_threshold
                logger.debug(f"  RIGID CONFIG: {rigid_kwargs}")

                rigid_reg = RigidRegistration(**rigid_kwargs)
                rigid_result = rigid_reg.register(current_source, target_image)
                transforms.append(rigid_result.transform)
                current_source = rigid_result.warped_source
                intermediate_results['rigid'] = current_source  # Store for visualization
                intermediate_transforms['rigid'] = rigid_result.transform
                if hasattr(rigid_result, 'loss_history'):
                    convergence_data['rigid'] = rigid_result.loss_history

                logger.debug("  RIGID STAGE OUTPUT:")
                logger.debug(f"    Warped source shape: {current_source.tensor().shape}")
                _log_transform_info(rigid_result.transform, "Rigid transform")
                if hasattr(rigid_result, 'final_loss'):
                    logger.info(f"  Rigid final loss: {rigid_result.final_loss:.6f}")

                # INCREMENTAL SAVE: Save rigid result immediately
                if output_dir is not None:
                    self._save_alignment_stage(
                        current_source, "rigid", output_dir, target_image
                    )

            # Affine stage
            if "affine" in stages:
                logger.info("Alignment: Affine registration...")
                logger.debug("  AFFINE STAGE INPUT:")
                logger.debug(f"    Source tensor shape: {current_source.tensor().shape}")
                logger.debug(f"    Target tensor shape: {target_image.tensor().shape}")

                # Extract affine config if available, else use defaults
                affine_kwargs = {"device": self.device}
                if self.config and hasattr(self.config, "affine"):
                    ac = self.config.affine
                    affine_kwargs.update({
                        "pyramid_levels": ac.pyramid_levels,
                        "iterations_per_level": ac.iterations_per_level,
                        "learning_rates": ac.learning_rates_per_level,
                    })
                    if hasattr(ac, "convergence"):
                        affine_kwargs["convergence_delta"] = ac.convergence.min_delta
                        affine_kwargs["convergence_patience"] = ac.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    affine_kwargs["num_bins"] = sc.num_bins
                    affine_kwargs["foreground_threshold"] = sc.foreground_threshold
                logger.debug(f"  AFFINE CONFIG: {affine_kwargs}")

                affine_reg = AffineRegistration(**affine_kwargs)
                init_transform = transforms[-1] if transforms else None
                affine_result = affine_reg.register(
                    current_source, target_image,
                    initial_transform=init_transform
                )
                transforms.append(affine_result.transform)
                current_source = affine_result.warped_source
                intermediate_results['affine'] = current_source  # Store for visualization
                intermediate_transforms['affine'] = affine_result.transform
                if hasattr(affine_result, 'loss_history'):
                    convergence_data['affine'] = affine_result.loss_history

                logger.debug("  AFFINE STAGE OUTPUT:")
                logger.debug(f"    Warped source shape: {current_source.tensor().shape}")
                _log_transform_info(affine_result.transform, "Affine transform")
                if hasattr(affine_result, 'final_loss'):
                    logger.info(f"  Affine final loss: {affine_result.final_loss:.6f}")

                # INCREMENTAL SAVE: Save affine result immediately
                if output_dir is not None:
                    self._save_alignment_stage(
                        current_source, "affine", output_dir, target_image
                    )

            # FFD/SVFFD stage
            if "ffd" in stages or "svffd" in stages:
                ffd_type = "svffd" if "svffd" in stages else "ffd"
                logger.info(f"Alignment: {ffd_type.upper()} registration...")
                logger.debug(f"  {ffd_type.upper()} STAGE INPUT:")
                logger.debug(f"    Source tensor shape: {current_source.tensor().shape}")
                logger.debug(f"    Target tensor shape: {target_image.tensor().shape}")

                # Extract FFD config if available, else use defaults
                ffd_kwargs = {"device": self.device, "model": ffd_type}
                if self.config and hasattr(self.config, "ffd"):
                    fc = self.config.ffd
                    ffd_kwargs.update({
                        "control_point_spacing": fc.control_point_spacing,
                        "pyramid_levels": fc.pyramid_levels,
                        "iterations_per_level": fc.iterations_per_level,
                        "learning_rates": fc.learning_rates_per_level,
                    })
                    if hasattr(fc, "regularization"):
                        ffd_kwargs["bending_weight"] = fc.regularization.bending_weight
                        ffd_kwargs["diffusion_weight"] = fc.regularization.diffusion_weight
                    if hasattr(fc, "convergence"):
                        ffd_kwargs["convergence_delta"] = fc.convergence.min_delta
                        ffd_kwargs["convergence_patience"] = fc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    ffd_kwargs["num_bins"] = sc.num_bins
                    ffd_kwargs["foreground_threshold"] = sc.foreground_threshold
                logger.debug(f"  {ffd_type.upper()} CONFIG: {ffd_kwargs}")

                ffd_reg = FFDRegistration(**ffd_kwargs)
                init_transform = transforms[-1] if transforms else None
                ffd_result = ffd_reg.register(
                    current_source, target_image,
                    initial_transform=init_transform
                )
                transforms.append(ffd_result.transform)
                current_source = ffd_result.warped_source
                intermediate_results['ffd'] = current_source  # Store for visualization
                intermediate_transforms['ffd'] = ffd_result.transform
                if hasattr(ffd_result, 'convergence_data'):
                    convergence_data['ffd'] = ffd_result.convergence_data
                elif hasattr(ffd_result, 'loss_history'):
                    convergence_data['ffd'] = ffd_result.loss_history

                logger.debug(f"  {ffd_type.upper()} STAGE OUTPUT:")
                logger.debug(f"    Warped source shape: {current_source.tensor().shape}")
                _log_transform_info(ffd_result.transform, f"{ffd_type.upper()} transform")
                if hasattr(ffd_result, 'final_loss'):
                    logger.info(f"  {ffd_type.upper()} final loss: {ffd_result.final_loss:.6f}")

                # INCREMENTAL SAVE: Save FFD result immediately
                if output_dir is not None:
                    self._save_alignment_stage(
                        current_source, ffd_type, output_dir, target_image
                    )

            # Compose all alignment transforms
            alignment_transform = self.composer.compose_sequential(transforms, common_grid)

            # Apply to static image
            # Use correct deepali API: target=grid for output sampling grid
            transformer = ImageTransformer(alignment_transform, target=common_grid)
            aligned_static = transformer(static_norm.tensor().unsqueeze(0))
            aligned_static = Image(data=aligned_static.squeeze(0), grid=common_grid)

            # Transform segmentation if provided
            # CRITICAL: Keep original segmentation resolution (MIRTK convention)
            # The transform moves the seg to frame 0's coordinate space,
            # but preserves the original high-resolution grid
            aligned_seg = None
            if segmentation is not None:
                logger.info("Transforming segmentation to frame 0 space (keeping original resolution)...")
                aligned_seg = transform_segmentation(
                    segmentation, alignment_transform, segmentation.grid()  # Keep original resolution!
                )
                logger.info(f"  Aligned segmentation resolution: {tuple(aligned_seg.grid().shape)}")
                logger.info(f"  Aligned segmentation spacing: {[f'{s:.4f}' for s in aligned_seg.grid().spacing().tolist()]}")

            # Create alignment overlay visualization if output_dir provided
            if output_dir is not None:
                try:
                    from ..visualization import save_alignment_progression
                    vis_dir = Path(output_dir) / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    save_alignment_progression(
                        target_image=frame0_resampled,  # Target = Frame 0
                        rigid_result=intermediate_results.get('rigid'),
                        affine_result=intermediate_results.get('affine'),
                        ffd_result=intermediate_results.get('ffd'),
                        output_dir=vis_dir,
                        prefix="alignment",
                        # Pass transforms for grid deformation overlay
                        rigid_transform=intermediate_transforms.get('rigid'),
                        affine_transform=intermediate_transforms.get('affine'),
                        ffd_transform=intermediate_transforms.get('ffd'),
                        show_grid=True,
                    )
                    logger.info(f"Alignment overlay visualizations saved to {vis_dir}")
                except Exception as e:
                    logger.warning(f"Could not create alignment visualizations: {e}")

            # Build AlignmentIntermediates for saving (like old code output structure)
            self._alignment_intermediates = AlignmentIntermediates(
                source_image=static_resampled,
                target_image=frame0_resampled,
                rigid_warped=intermediate_results.get('rigid'),
                affine_warped=intermediate_results.get('affine'),
                ffd_warped=intermediate_results.get('ffd'),
                rigid_transform=intermediate_transforms.get('rigid'),
                affine_transform=intermediate_transforms.get('affine'),
                ffd_transform=intermediate_transforms.get('ffd'),
                common_grid=common_grid,
                convergence_data=convergence_data,
            )

        logger.info("Alignment complete")

        return alignment_transform, aligned_static, aligned_seg

    def register_pairwise(
        self,
        image_4d: Image4D,
        reference_segmentation: Optional[Image] = None,
    ) -> List[PairwiseResult]:
        """
        Step 2: Pairwise registration between consecutive frames

        Registers: 0→1, 1→2, 2→3, ..., (N-1)→N

        Args:
            image_4d: 4D image container
            reference_segmentation: Segmentation at frame 0 (for masking)

        Returns:
            List of PairwiseResult for each consecutive pair
        """
        logger.info("=" * 60)
        logger.info("STEP 2: PAIRWISE REGISTRATION (Consecutive Frames)")
        logger.info("=" * 60)

        # Create frame pairs
        pairs = create_frame_pairs(image_4d, mode="sequential")

        pairwise_results = []

        for i, (src_idx, tgt_idx, source, target) in enumerate(pairs):
            logger.info(f"\nPair {i+1}/{len(pairs)}: Frame {src_idx} → Frame {tgt_idx}")

            with Timer(f"Pairwise {src_idx}→{tgt_idx}"):
                # Normalize
                source_norm = _normalize_image(source)
                target_norm = _normalize_image(target)

                # Create common grid for this pair
                common_grid = create_common_grid(source_norm, target_norm)

                # For pairwise, we typically use just FFD (no rigid/affine)
                # since frames should already be spatially aligned
                ffd_type = "svffd" if "svffd" in self.registration_model else "ffd"
                ffd_kwargs = {"device": self.device, "model": ffd_type}
                if self.config and hasattr(self.config, "ffd"):
                    fc = self.config.ffd
                    ffd_kwargs.update({
                        "control_point_spacing": fc.control_point_spacing,
                        "pyramid_levels": fc.pyramid_levels,
                        "iterations_per_level": fc.iterations_per_level,
                        "learning_rates": fc.learning_rates_per_level,
                    })
                    if hasattr(fc, "regularization"):
                        ffd_kwargs["bending_weight"] = fc.regularization.bending_weight
                        ffd_kwargs["diffusion_weight"] = fc.regularization.diffusion_weight
                    if hasattr(fc, "convergence"):
                        ffd_kwargs["convergence_delta"] = fc.convergence.min_delta
                        ffd_kwargs["convergence_patience"] = fc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    ffd_kwargs["num_bins"] = sc.num_bins
                    ffd_kwargs["foreground_threshold"] = sc.foreground_threshold
                ffd_reg = FFDRegistration(**ffd_kwargs)

                ffd_result = ffd_reg.register(source_norm, target_norm)

                pairwise_results.append(PairwiseResult(
                    source_idx=src_idx,
                    target_idx=tgt_idx,
                    transform=ffd_result.transform,
                    final_loss=ffd_result.final_loss,
                    metrics=ffd_result.quality_metrics,
                ))

                logger.info(f"  Loss: {ffd_result.final_loss:.6f}")

        logger.info(f"\nCompleted {len(pairwise_results)} pairwise registrations")

        return pairwise_results

    def compose_longitudinal(
        self,
        pairwise_results: List[PairwiseResult],
        reference_grid: Grid,
    ) -> List[ComposedTransform]:
        """
        Step 3: Compose pairwise transforms to get longitudinal (0→N)

        Given T(0→1), T(1→2), T(2→3), ...
        Compute T(0→1), T(0→2) = T(0→1)∘T(1→2), T(0→3) = T(0→2)∘T(2→3), ...

        Args:
            pairwise_results: List of consecutive pairwise results
            reference_grid: Grid for composition

        Returns:
            List of ComposedTransform from frame 0 to each frame
        """
        logger.info("=" * 60)
        logger.info("STEP 3: COMPOSE LONGITUDINAL TRANSFORMS")
        logger.info("=" * 60)

        # Log input
        logger.debug("COMPOSE LONGITUDINAL INPUT:")
        logger.debug(f"  Number of pairwise transforms: {len(pairwise_results)}")
        logger.debug(f"  Reference grid shape: {tuple(reference_grid.shape)}")
        for i, pr in enumerate(pairwise_results):
            logger.debug(f"  Pairwise {i+1}: Frame {pr.source_idx} → Frame {pr.target_idx}, loss={pr.final_loss:.6f}")

        with Timer("Composition"):
            # Convert to format expected by composer
            transform_tuples = [
                (r.source_idx, r.target_idx, r.transform)
                for r in pairwise_results
            ]

            longitudinal = self.composer.compose_pairwise_to_longitudinal(
                transform_tuples,
                reference_grid,
            )

        # Log output
        logger.debug("COMPOSE LONGITUDINAL OUTPUT:")
        logger.debug(f"  Number of longitudinal transforms: {len(longitudinal)}")
        for lt in longitudinal:
            logger.debug(f"  Longitudinal: Frame 0 → Frame {lt.source_idx}")
            logger.debug(f"    Intermediate indices: {lt.intermediate_indices}")
            _log_transform_info(lt.transform, f"Longitudinal 0→{lt.source_idx} transform")

        logger.info(f"Composed {len(longitudinal)} longitudinal transforms")

        return longitudinal

    def refine_longitudinal(
        self,
        image_4d: Image4D,
        longitudinal_transforms: List[ComposedTransform],
        reference_grid: Grid,
    ) -> List[ComposedTransform]:
        """
        Step 4: Refine each longitudinal transform

        Use composed transform as initialization, then run 1-level
        registration for fine-tuning.

        This replicates MIRTK's refine_longitudinal_dofs

        Args:
            image_4d: 4D image container
            longitudinal_transforms: Composed transforms to refine
            reference_grid: Reference grid

        Returns:
            List of refined ComposedTransform
        """
        logger.info("=" * 60)
        logger.info("STEP 4: REFINE LONGITUDINAL TRANSFORMS")
        logger.info("=" * 60)

        frame_0 = image_4d.get_frame(0)
        frame0_norm = _normalize_image(frame_0)

        refined = []

        for i, composed in enumerate(longitudinal_transforms):
            src_idx = composed.source_idx

            # Skip first frame (no refinement needed for identity)
            if src_idx == 1:
                # First longitudinal is just the pairwise, no refinement needed
                refined.append(composed)
                continue

            logger.info(f"\nRefining Frame 0 → Frame {src_idx}...")

            with Timer(f"Refine 0→{src_idx}"):
                # Get source frame
                source = image_4d.get_frame(src_idx)
                source_norm = _normalize_image(source)

                # Create registration with 1 pyramid level (refinement only)
                ffd_type = "svffd" if "svffd" in self.registration_model else "ffd"
                ffd_kwargs = {"device": self.device, "model": ffd_type, "pyramid_levels": 1}
                if self.config and hasattr(self.config, "ffd"):
                    fc = self.config.ffd
                    ffd_kwargs.update({
                        "control_point_spacing": fc.control_point_spacing,
                        "iterations_per_level": [fc.iterations_per_level[0]],  # Single level
                        "learning_rates": [fc.learning_rates_per_level[0]],
                    })
                    if hasattr(fc, "regularization"):
                        ffd_kwargs["bending_weight"] = fc.regularization.bending_weight
                        ffd_kwargs["diffusion_weight"] = fc.regularization.diffusion_weight
                    if hasattr(fc, "convergence"):
                        ffd_kwargs["convergence_delta"] = fc.convergence.min_delta
                        ffd_kwargs["convergence_patience"] = fc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    ffd_kwargs["num_bins"] = sc.num_bins
                    ffd_kwargs["foreground_threshold"] = sc.foreground_threshold
                ffd_reg = FFDRegistration(**ffd_kwargs)

                # Use composed transform as initialization
                ffd_result = ffd_reg.register(
                    source_norm, frame0_norm,
                    initial_transform=composed.transform,
                )

                # Create refined composed transform
                refined.append(ComposedTransform(
                    transform=ffd_result.transform,
                    source_idx=src_idx,
                    target_idx=0,
                    intermediate_indices=composed.intermediate_indices,
                ))

                logger.info(f"  Refined loss: {ffd_result.final_loss:.6f}")

        logger.info(f"\nRefined {len(refined)} longitudinal transforms")

        return refined

    def propagate_segmentation(
        self,
        segmentation_0: Image,
        longitudinal_transforms: List[ComposedTransform],
        reference_grid: Grid,
    ) -> List[Image]:
        """
        Step 5: Propagate segmentation through all frames

        Apply each longitudinal transform to seg_0 to get seg_N

        Args:
            segmentation_0: Segmentation at frame 0
            longitudinal_transforms: Longitudinal transforms
            reference_grid: Reference grid

        Returns:
            List of segmentations [seg_1, seg_2, ..., seg_N]
        """
        logger.info("=" * 60)
        logger.info("STEP 5: PROPAGATE SEGMENTATION")
        logger.info("=" * 60)

        segmentations = []

        for composed in longitudinal_transforms:
            logger.info(f"Propagating to frame {composed.source_idx}...")

            with Timer(f"Propagate to frame {composed.source_idx}"):
                # Use inverse of longitudinal transform
                # Longitudinal is source→target (N→0), we need inverse (0→N)
                # For segmentation, we want to warp frame 0 seg TO frame N
                transform = composed.transform

                # Compute inverse transform for segmentation propagation
                # Direction: We have transforms that map frame_i -> frame_0
                # For segmentation, we need to map frame_0 -> frame_i (inverse direction)
                #
                # CRITICAL: Do NOT fall back to forward transform - that would warp
                # the segmentation in the WRONG direction (approximately 15 voxel offset)
                #
                # SVFFD has analytical inverse (negated velocity field)
                # FFD does NOT have inverse - MIRTK uses Newton-Raphson approximation
                try:
                    inverse_transform = transform.inverse()
                except NotImplementedError as e:
                    # FFD (non-diffeomorphic) doesn't support analytical inverse
                    # Use MIRTK-style Newton-Raphson approximation for FFD inverse
                    logger.warning(
                        f"Transform type '{type(transform).__name__}' does not support inverse(). "
                        f"Using Newton-Raphson approximation (MIRTK-style) for FFD inverse."
                    )
                    from ..postprocessing.segmentation import approximate_ffd_inverse
                    inverse_transform = approximate_ffd_inverse(transform, max_iterations=10, tolerance=1e-5)
                except Exception as e:
                    # Unexpected error during inverse computation
                    error_msg = (
                        f"Error computing inverse transform for frame {composed.source_idx}: {e}. "
                        f"Segmentation propagation requires a valid inverse transform."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # Transform segmentation using nearest neighbor
                # CRITICAL: Use segmentation_0.grid() to preserve original high-res resolution
                # The transform moves the seg to frame N's space, but keeps original resolution
                seg_n = transform_segmentation(
                    segmentation_0,
                    inverse_transform,
                    segmentation_0.grid(),  # Keep original segmentation resolution
                )

                segmentations.append(seg_n)

        logger.info(f"\nPropagated segmentation to {len(segmentations)} frames")

        return segmentations

    def _save_frames(self, image_4d: Image4D, output_dir: Path):
        """Save extracted frames from 4D image using nibabel for correct affine preservation"""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Construct affine matrix from Image4D (stored in LPS after loading)
        affine = _construct_affine_from_image4d(image_4d)
        # Convert LPS to RAS for NIfTI saving
        affine[:2] *= -1

        for i in range(image_4d.num_frames):
            frame = image_4d.get_frame(i)
            frame_np = frame.tensor().cpu().numpy()
            if frame_np.ndim == 4:
                frame_np = frame_np.squeeze(0)

            # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
            frame_nifti = np.transpose(frame_np, (2, 1, 0))

            # Use nibabel-based save_nifti for correct affine preservation
            save_nifti(
                frame_nifti,
                frames_dir / f"frame_{i:03d}.nii.gz",
                affine,
                description=f"frame {i}"
            )

        logger.info(f"Saved {image_4d.num_frames} frames to {frames_dir}")

    def _save_alignment_stage(self, warped_image: Image, stage_name: str,
                               output_dir: Path, target_image: Image):
        """Save intermediate alignment result after each stage (rigid, affine, ffd).

        This allows monitoring progress during long-running alignment registration.

        Args:
            warped_image: Warped source image after this stage
            stage_name: Name of the stage ("rigid", "affine", "ffd", "svffd")
            output_dir: Output directory
            target_image: Target image for reference affine
        """
        alignment_dir = output_dir / "alignment"
        alignment_dir.mkdir(parents=True, exist_ok=True)

        # Get image data
        warped_np = warped_image.tensor().cpu().numpy()
        if warped_np.ndim == 4:
            warped_np = warped_np.squeeze(0)

        # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
        warped_nifti = np.transpose(warped_np, (2, 1, 0))

        # Construct affine from target grid (common grid)
        target_grid = target_image.grid()
        spacing = target_grid.spacing().cpu().numpy()
        origin = target_grid.origin().cpu().numpy()
        direction = target_grid.direction().cpu().numpy()

        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        affine[:2] *= -1  # LPS to RAS conversion for NIfTI

        # Save with stage name (like old code: source_after_rigid_common.nii.gz)
        save_nifti(
            warped_nifti,
            alignment_dir / f"source_after_{stage_name}_common.nii.gz",
            affine,
            description=f"source after {stage_name} (common grid)"
        )
        logger.info(f"  INCREMENTAL SAVE: source_after_{stage_name}_common.nii.gz")

    def _save_alignment(self, alignment_transform: SpatialTransform,
                        aligned_seg: Optional[Image], image_4d: Image4D, output_dir: Path,
                        static_image: Optional[Image] = None, frame_0: Optional[Image] = None,
                        segmentation: Optional[Image] = None):
        """Save alignment results using nibabel for correct affine preservation.

        MIRTK Convention:
        - Aligned static image keeps original resolution but moves to frame 0 space
        - Segmentation keeps original resolution (same as static) with nearest neighbor interpolation
        """
        alignment_dir = output_dir / "alignment"
        alignment_dir.mkdir(parents=True, exist_ok=True)

        # Save alignment transform
        self.composer.save_transform(
            alignment_transform,
            alignment_dir / "alignment_transform.pth",
            metadata={"type": "alignment", "model": self.registration_model},
        )

        # =====================================================
        # MIRTK CONVENTION: Save aligned static at ORIGINAL resolution
        # The static image is moved to frame 0 space but keeps its resolution
        # =====================================================
        if static_image is not None and frame_0 is not None:
            logger.info("Saving aligned static image at original resolution (MIRTK convention)...")

            # Apply transform to static image, keeping static's original resolution
            aligned_static_original_res = apply_transform_preserve_resolution(
                source_image=static_image,
                transform=alignment_transform,
                target_grid=frame_0.grid(),  # Reference for coordinate space
                sampling="linear",  # deepali uses "linear" not "bilinear"
            )

            # Construct 4x4 affine from grid components
            # deepali grid stores in LPS, need to convert to RAS for NIfTI
            aligned_grid = aligned_static_original_res.grid()
            spacing = aligned_grid.spacing().cpu().numpy()
            origin = aligned_grid.origin().cpu().numpy()
            direction = aligned_grid.direction().cpu().numpy()

            static_affine = np.eye(4)
            static_affine[:3, :3] = direction @ np.diag(spacing)
            static_affine[:3, 3] = origin
            static_affine[:2] *= -1  # LPS to RAS conversion for NIfTI

            # Save aligned static image
            aligned_np = aligned_static_original_res.tensor().cpu().numpy()
            if aligned_np.ndim == 4:
                aligned_np = aligned_np.squeeze(0)

            # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
            aligned_nifti = np.transpose(aligned_np, (2, 1, 0))

            save_nifti(
                aligned_nifti,
                alignment_dir / "aligned_static_original_resolution.nii.gz",
                static_affine,
                description="aligned static (original resolution, frame 0 space)"
            )
            spacing = aligned_grid.spacing()
            logger.info(f"  Saved aligned_static_original_resolution.nii.gz")
            logger.info(f"  Shape: {aligned_nifti.shape}, Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f})mm")

        # Save aligned segmentation at ORIGINAL resolution (same as static image)
        # This ensures the segmentation and aligned static have the same geometry for overlay
        if segmentation is not None and static_image is not None and frame_0 is not None:
            logger.info("Saving aligned segmentation at original resolution (MIRTK convention)...")

            # Apply transform to segmentation, keeping original resolution with nearest neighbor
            aligned_seg_original_res = apply_transform_preserve_resolution(
                source_image=segmentation,
                transform=alignment_transform,
                target_grid=frame_0.grid(),  # Reference for coordinate space
                sampling="nearest",  # Nearest neighbor for label maps
            )

            seg_tensor = aligned_seg_original_res.tensor()
            seg_np = seg_tensor.cpu().numpy()
            if seg_np.ndim == 4:
                seg_np = seg_np.squeeze(0)

            # Check if segmentation has any content (non-zero voxels)
            num_nonzero = np.count_nonzero(seg_np)
            if num_nonzero == 0:
                logger.warning("WARNING: Aligned segmentation is EMPTY (all zeros)!")
                logger.warning("  This may indicate registration failed or segmentation was not provided.")
            else:
                logger.info(f"  Segmentation has {num_nonzero} non-zero voxels")

            # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
            seg_nifti = np.transpose(seg_np, (2, 1, 0))

            # Use the SAME affine as aligned static (they share the same geometry)
            # This was already computed above for aligned_static_original_res
            seg_grid = aligned_seg_original_res.grid()
            seg_spacing = seg_grid.spacing().cpu().numpy()
            seg_origin = seg_grid.origin().cpu().numpy()
            seg_direction = seg_grid.direction().cpu().numpy()

            seg_affine = np.eye(4)
            seg_affine[:3, :3] = seg_direction @ np.diag(seg_spacing)
            seg_affine[:3, 3] = seg_origin
            seg_affine[:2] *= -1  # LPS to RAS conversion for NIfTI

            # Use nibabel-based save_segmentation for correct affine preservation
            save_segmentation(
                seg_nifti,
                alignment_dir / "aligned_segmentation.nii.gz",
                seg_affine,
                description="aligned segmentation (original resolution)"
            )
            seg_spacing_tensor = seg_grid.spacing()
            logger.info(f"  Saved aligned_segmentation.nii.gz")
            logger.info(f"  Shape: {seg_nifti.shape}, Spacing: ({seg_spacing_tensor[0]:.3f}, {seg_spacing_tensor[1]:.3f}, {seg_spacing_tensor[2]:.3f})mm")

        logger.info(f"Saved alignment results to {alignment_dir}")

    def _save_pairwise(self, pairwise_result: PairwiseResult, output_dir: Path):
        """Save single pairwise registration result"""
        pairwise_dir = output_dir / "pairwise"
        pairwise_dir.mkdir(parents=True, exist_ok=True)

        self.composer.save_transform(
            pairwise_result.transform,
            pairwise_dir / f"pairwise_{pairwise_result.source_idx}_{pairwise_result.target_idx}.pth",
            metadata={
                "type": "pairwise",
                "source_idx": pairwise_result.source_idx,
                "target_idx": pairwise_result.target_idx,
                "loss": pairwise_result.final_loss,
                "metrics": pairwise_result.metrics,
            },
        )
        logger.info(f"Saved pairwise {pairwise_result.source_idx}→{pairwise_result.target_idx}")

    def _save_longitudinal(self, longitudinal_transforms: List[ComposedTransform],
                           output_dir: Path, suffix: str = ""):
        """Save longitudinal transforms"""
        longitudinal_dir = output_dir / "longitudinal"
        longitudinal_dir.mkdir(parents=True, exist_ok=True)

        for lt in longitudinal_transforms:
            filename = f"longitudinal_0_to_{lt.source_idx}{suffix}.pth"
            self.composer.save_transform(
                lt.transform,
                longitudinal_dir / filename,
                metadata={
                    "type": "longitudinal",
                    "source_idx": lt.source_idx,
                    "target_idx": lt.target_idx,
                    "intermediates": lt.intermediate_indices,
                },
            )
        logger.info(f"Saved {len(longitudinal_transforms)} longitudinal transforms to {longitudinal_dir}")

    def _save_segmentation(self, seg: Image, frame_idx: int, image_4d: Image4D, output_dir: Path):
        """Save single segmentation using nibabel with CORRECT affine from segmentation's grid.

        CRITICAL: High-res segmentations must use their OWN grid's affine, NOT the frame's.
        Otherwise, the saved NIfTI will have wrong spacing (e.g., 3mm instead of 1mm)
        and will display incorrectly in ITK-SNAP.

        This function saves the "moved-only" high-resolution segmentation that preserves
        the original segmentation resolution while being spatially transformed.
        For the "resampled to frame grid" version, use _save_segmentation_frame_grid().
        """
        seg_dir = output_dir / "segmentations"
        seg_dir.mkdir(parents=True, exist_ok=True)

        seg_np = seg.tensor().cpu().numpy()
        if seg_np.ndim == 4:
            seg_np = seg_np.squeeze(0)

        # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
        seg_nifti = np.transpose(seg_np, (2, 1, 0))

        # CRITICAL: Use the SEGMENTATION's actual grid for affine construction
        # This preserves the high-resolution spacing in the saved file
        affine = _construct_affine_from_grid(seg.grid())
        # Convert LPS to RAS for NIfTI saving
        affine[:2] *= -1

        # Log for debugging - show actual segmentation resolution
        grid = seg.grid()
        logger.info(f"Saving seg_frame_{frame_idx:03d} (high-res):")
        logger.info(f"  Grid shape: {tuple(grid.shape)}")
        logger.info(f"  Spacing (mm): {[f'{s:.4f}' for s in grid.spacing().tolist()]}")

        # Use nibabel-based save_segmentation for correct affine preservation
        save_segmentation(
            seg_nifti,
            seg_dir / f"seg_frame_{frame_idx:03d}.nii.gz",
            affine,
            description=f"segmentation frame {frame_idx} (high-res)"
        )
        logger.info(f"  Saved: seg_frame_{frame_idx:03d}.nii.gz")

    def _save_segmentation_frame_grid(self, seg: Image, frame_idx: int, image_4d: Image4D, output_dir: Path):
        """Save segmentation resampled to frame grid for intermediate sanity check.

        This creates a SEPARATE set of segmentation files that are resampled to match
        the extracted frame grid (lower resolution). These are NOT the final results
        but are useful for:
        1. Quick visual verification in ITK-SNAP (exact overlay with frame_XXX.nii.gz)
        2. Intermediate quality check during motion tracking

        The high-res segmentations (original resolution) are saved by _save_segmentation()
        and should be used for final analysis.

        Args:
            seg: Segmentation image (already resampled to frame grid)
            frame_idx: Frame index
            image_4d: 4D image for header information
            output_dir: Output directory
        """
        seg_dir = output_dir / "segmentations_frame_grid"
        seg_dir.mkdir(parents=True, exist_ok=True)

        seg_np = seg.tensor().cpu().numpy()
        if seg_np.ndim == 4:
            seg_np = seg_np.squeeze(0)

        # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
        seg_nifti = np.transpose(seg_np, (2, 1, 0))

        # Use SAME affine construction as _save_frames() for consistency
        # This ensures seg overlays correctly with frame_XXX.nii.gz in ITK-SNAP
        affine = _construct_affine_from_image4d(image_4d)
        # Convert LPS to RAS for NIfTI saving (same as frame saving)
        affine[:2] *= -1

        save_segmentation(
            seg_nifti,
            seg_dir / f"seg_frame_{frame_idx:03d}_frame_grid.nii.gz",
            affine,
            description=f"segmentation frame {frame_idx} (frame grid)"
        )
        logger.debug(f"Saved frame-grid segmentation for frame {frame_idx}")

    def run_full_pipeline(
        self,
        image_4d: Image4D,
        static_image: Optional[Image] = None,
        segmentation: Optional[Image] = None,
        skip_alignment: bool = False,
        skip_refinement: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> MotionResult:
        """
        Run complete motion registration pipeline with incremental saving

        Args:
            image_4d: 4D image sequence
            static_image: Optional static high-res image for alignment
            segmentation: Optional segmentation on static image
            skip_alignment: Skip alignment step (if static already aligned)
            skip_refinement: Skip refinement step (faster but less accurate)
            output_dir: Output directory for incremental saves (optional)

        Returns:
            MotionResult with all transforms and segmentations
        """
        logger.info("=" * 70)
        logger.info("DAREG MOTION REGISTRATION PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Frames: {image_4d.num_frames}")
        logger.info(f"Frame shape: {image_4d.frame_shape}")
        logger.info(f"Model: {self.registration_model}")
        if output_dir:
            logger.info(f"Output: {output_dir} (incremental saving enabled)")
        logger.info("=" * 70)

        result = MotionResult()

        # Setup output directory for incremental saving
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save extracted frames immediately
            logger.info("Saving extracted frames...")
            self._save_frames(image_4d, output_dir)

        with Timer("Total pipeline") as total_timer:
            # Reference grid from frame 0
            frame_0 = image_4d.get_frame(0)
            reference_grid = frame_0.grid()

            # Step 1: Alignment (optional)
            seg_0 = segmentation
            if static_image is not None and not skip_alignment:
                alignment, aligned_static, aligned_seg = self.register_alignment(
                    static_image, frame_0, segmentation,
                    output_dir=output_dir,  # Pass output_dir for visualization
                )
                result.alignment_transform = alignment
                seg_0 = aligned_seg

                # INCREMENTAL SAVE: Save alignment results
                if output_dir is not None:
                    self._save_alignment(
                        alignment, aligned_seg, image_4d, output_dir,
                        static_image=static_image, frame_0=frame_0,
                        segmentation=segmentation  # Pass original segmentation for proper resolution saving
                    )

            elif segmentation is not None:
                # When alignment is skipped, keep segmentation at ORIGINAL resolution
                # MIRTK convention: transforms move the segmentation spatially,
                # but preserve its original high-resolution grid
                #
                # The frame-grid version (for ITK-SNAP sanity check) is created
                # separately during propagation in _propagate_segmentation_incremental
                seg_grid = segmentation.grid()
                frame0_grid = frame_0.grid()

                logger.info("Using segmentation at original resolution (no alignment)")
                logger.info(f"  Segmentation resolution: {tuple(seg_grid.shape)}")
                logger.info(f"  Segmentation spacing: {[f'{s:.4f}' for s in seg_grid.spacing().tolist()]}")
                logger.info(f"  Frame 0 resolution: {tuple(frame0_grid.shape)}")
                logger.info(f"  Frame 0 spacing: {[f'{s:.4f}' for s in frame0_grid.spacing().tolist()]}")

                # Keep original resolution - do NOT resample
                seg_0 = segmentation

            # Step 2: Pairwise registration (with incremental saving)
            pairwise_results = self._register_pairwise_incremental(
                image_4d, seg_0, output_dir
            )
            result.pairwise_transforms = pairwise_results

            # Step 3: Compose longitudinal
            longitudinal = self.compose_longitudinal(pairwise_results, reference_grid)

            # INCREMENTAL SAVE: Save composed transforms
            if output_dir is not None:
                self._save_longitudinal(longitudinal, output_dir, suffix="_composed")

            # Step 4: Refine longitudinal (optional)
            if not skip_refinement:
                longitudinal = self._refine_longitudinal_incremental(
                    image_4d, longitudinal, reference_grid, output_dir
                )
            result.longitudinal_transforms = longitudinal

            # INCREMENTAL SAVE: Save refined transforms (overwrites composed)
            if output_dir is not None and not skip_refinement:
                self._save_longitudinal(longitudinal, output_dir, suffix="_refined")

            # Step 5: Propagate segmentation (if available) with incremental saving
            if seg_0 is not None:
                segmentations = self._propagate_segmentation_incremental(
                    seg_0, longitudinal, reference_grid, image_4d, output_dir
                )
                # Include frame 0 segmentation
                result.segmentation_sequence = [seg_0] + segmentations

                # Save frame 0 segmentation
                if output_dir is not None:
                    self._save_segmentation(seg_0, 0, image_4d, output_dir)

            result.total_time = total_timer.elapsed

        logger.info("=" * 70)
        logger.info("MOTION REGISTRATION COMPLETE")
        logger.info(f"Total time: {result.total_time:.1f}s")
        logger.info(f"Frames processed: {image_4d.num_frames}")
        if result.segmentation_sequence:
            logger.info(f"Segmentations generated: {len(result.segmentation_sequence)}")
        logger.info("=" * 70)

        return result

    def _register_pairwise_incremental(
        self,
        image_4d: Image4D,
        reference_segmentation: Optional[Image] = None,
        output_dir: Optional[Path] = None,
    ) -> List[PairwiseResult]:
        """
        Step 2: Pairwise registration with incremental saving
        """
        logger.info("=" * 60)
        logger.info("STEP 2: PAIRWISE REGISTRATION (Consecutive Frames)")
        logger.info("=" * 60)

        # Log step overview
        logger.debug("PAIRWISE REGISTRATION OVERVIEW:")
        logger.debug(f"  Number of frames: {image_4d.num_frames}")
        logger.debug(f"  Frame shape: {image_4d.frame_shape}")
        logger.debug(f"  Expected pairs: {image_4d.num_frames - 1}")

        # Create frame pairs
        pairs = create_frame_pairs(image_4d, mode="sequential")
        logger.debug(f"  Created {len(pairs)} frame pairs")

        pairwise_results = []

        for i, (src_idx, tgt_idx, source, target) in enumerate(pairs):
            logger.info(f"\nPair {i+1}/{len(pairs)}: Frame {src_idx} → Frame {tgt_idx}")

            with Timer(f"Pairwise {src_idx}→{tgt_idx}"):
                # Log input frames
                logger.debug(f"  PAIRWISE INPUT (Pair {i+1}):")
                _log_image_info(source, f"Source frame {src_idx}", level="debug")
                _log_image_info(target, f"Target frame {tgt_idx}", level="debug")

                # Normalize
                source_norm = _normalize_image(source)
                target_norm = _normalize_image(target)

                # Create common grid for this pair
                common_grid = create_common_grid(source_norm, target_norm)
                logger.debug(f"  Common grid shape: {tuple(common_grid.shape)}")

                # For pairwise, we typically use just FFD (no rigid/affine)
                ffd_type = "svffd" if "svffd" in self.registration_model else "ffd"
                ffd_kwargs = {"device": self.device, "model": ffd_type}
                if self.config and hasattr(self.config, "ffd"):
                    fc = self.config.ffd
                    ffd_kwargs.update({
                        "control_point_spacing": fc.control_point_spacing,
                        "pyramid_levels": fc.pyramid_levels,
                        "iterations_per_level": fc.iterations_per_level,
                        "learning_rates": fc.learning_rates_per_level,
                    })
                    if hasattr(fc, "regularization"):
                        ffd_kwargs["bending_weight"] = fc.regularization.bending_weight
                        ffd_kwargs["diffusion_weight"] = fc.regularization.diffusion_weight
                    if hasattr(fc, "convergence"):
                        ffd_kwargs["convergence_delta"] = fc.convergence.min_delta
                        ffd_kwargs["convergence_patience"] = fc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    ffd_kwargs["num_bins"] = sc.num_bins
                    ffd_kwargs["foreground_threshold"] = sc.foreground_threshold
                ffd_reg = FFDRegistration(**ffd_kwargs)

                ffd_result = ffd_reg.register(source_norm, target_norm)

                pw_result = PairwiseResult(
                    source_idx=src_idx,
                    target_idx=tgt_idx,
                    transform=ffd_result.transform,
                    final_loss=ffd_result.final_loss,
                    metrics=ffd_result.quality_metrics,
                )
                pairwise_results.append(pw_result)

                # Log output
                logger.debug(f"  PAIRWISE OUTPUT (Pair {i+1}):")
                _log_transform_info(ffd_result.transform, f"Pairwise {src_idx}→{tgt_idx} transform")
                logger.debug(f"    Final loss: {ffd_result.final_loss:.6f}")
                if ffd_result.quality_metrics:
                    for key, value in ffd_result.quality_metrics.items():
                        logger.debug(f"    {key}: {value}")

                # INCREMENTAL SAVE: Save this pairwise result immediately
                if output_dir is not None:
                    self._save_pairwise(pw_result, output_dir)

                logger.info(f"  Loss: {ffd_result.final_loss:.6f}")

        logger.info(f"\nCompleted {len(pairwise_results)} pairwise registrations")

        return pairwise_results

    def _refine_longitudinal_incremental(
        self,
        image_4d: Image4D,
        longitudinal_transforms: List[ComposedTransform],
        reference_grid: Grid,
        output_dir: Optional[Path] = None,
    ) -> List[ComposedTransform]:
        """
        Step 4: Refine longitudinal transforms with incremental saving
        """
        logger.info("=" * 60)
        logger.info("STEP 4: REFINE LONGITUDINAL TRANSFORMS")
        logger.info("=" * 60)

        # Log input
        logger.debug("REFINE LONGITUDINAL INPUT:")
        logger.debug(f"  Number of longitudinal transforms: {len(longitudinal_transforms)}")
        logger.debug(f"  Reference grid shape: {tuple(reference_grid.shape)}")
        logger.debug(f"  Note: Frame 1 will be skipped (no refinement needed)")

        frame_0 = image_4d.get_frame(0)
        frame0_norm = _normalize_image(frame_0)
        logger.debug("FRAME 0 (TARGET):")
        _log_image_info(frame0_norm, "Frame 0 normalized", level="debug")

        refined = []

        for i, composed in enumerate(longitudinal_transforms):
            src_idx = composed.source_idx

            # Skip first frame (no refinement needed for identity)
            if src_idx == 1:
                logger.debug(f"  Skipping Frame 1 (no refinement needed - using pairwise directly)")
                refined.append(composed)
                continue

            logger.info(f"\nRefining Frame 0 → Frame {src_idx}...")

            with Timer(f"Refine 0→{src_idx}"):
                source = image_4d.get_frame(src_idx)
                source_norm = _normalize_image(source)

                logger.debug(f"  REFINEMENT INPUT (Frame {src_idx}):")
                _log_image_info(source_norm, f"Source frame {src_idx} normalized", level="debug")
                _log_transform_info(composed.transform, f"Initial transform (composed 0→{src_idx})")

                ffd_type = "svffd" if "svffd" in self.registration_model else "ffd"
                ffd_kwargs = {"device": self.device, "model": ffd_type, "pyramid_levels": 1}
                logger.debug(f"  Refinement config: model={ffd_type}, pyramid_levels=1")
                if self.config and hasattr(self.config, "ffd"):
                    fc = self.config.ffd
                    ffd_kwargs.update({
                        "control_point_spacing": fc.control_point_spacing,
                        "iterations_per_level": [fc.iterations_per_level[0]],
                        "learning_rates": [fc.learning_rates_per_level[0]],
                    })
                    if hasattr(fc, "regularization"):
                        ffd_kwargs["bending_weight"] = fc.regularization.bending_weight
                        ffd_kwargs["diffusion_weight"] = fc.regularization.diffusion_weight
                    if hasattr(fc, "convergence"):
                        ffd_kwargs["convergence_delta"] = fc.convergence.min_delta
                        ffd_kwargs["convergence_patience"] = fc.convergence.patience
                if self.config and hasattr(self.config, "similarity"):
                    sc = self.config.similarity
                    ffd_kwargs["num_bins"] = sc.num_bins
                    ffd_kwargs["foreground_threshold"] = sc.foreground_threshold
                ffd_reg = FFDRegistration(**ffd_kwargs)

                ffd_result = ffd_reg.register(
                    source_norm, frame0_norm,
                    initial_transform=composed.transform,
                )

                refined_transform = ComposedTransform(
                    transform=ffd_result.transform,
                    source_idx=src_idx,
                    target_idx=0,
                    intermediate_indices=composed.intermediate_indices,
                )
                refined.append(refined_transform)

                # Log refinement output
                logger.debug(f"  REFINEMENT OUTPUT (Frame {src_idx}):")
                _log_transform_info(ffd_result.transform, f"Refined transform 0→{src_idx}")
                logger.debug(f"    Final loss: {ffd_result.final_loss:.6f}")
                if hasattr(ffd_result, 'quality_metrics') and ffd_result.quality_metrics:
                    for key, value in ffd_result.quality_metrics.items():
                        logger.debug(f"    {key}: {value}")

                logger.info(f"  Refined loss: {ffd_result.final_loss:.6f}")

        # Log summary
        logger.debug("REFINEMENT SUMMARY:")
        for rt in refined:
            logger.debug(f"  Refined 0→{rt.source_idx}: type={type(rt.transform).__name__}")

        logger.info(f"\nRefined {len(refined)} longitudinal transforms")

        return refined

    def _propagate_segmentation_incremental(
        self,
        segmentation_0: Image,
        longitudinal_transforms: List[ComposedTransform],
        reference_grid: Grid,
        image_4d: Image4D,
        output_dir: Optional[Path] = None,
    ) -> List[Image]:
        """
        Step 5: Propagate segmentation with incremental saving
        """
        logger.info("=" * 60)
        logger.info("STEP 5: PROPAGATE SEGMENTATION")
        logger.info("=" * 60)

        # Log segmentation input info
        logger.debug("SEGMENTATION PROPAGATION INPUT:")
        _log_image_info(segmentation_0, "Segmentation at frame 0", level="debug")
        logger.debug(f"  Number of longitudinal transforms: {len(longitudinal_transforms)}")
        logger.debug(f"  Reference grid shape: {tuple(reference_grid.shape)}")

        # Log unique labels in segmentation
        seg_tensor = segmentation_0.tensor()
        unique_labels = torch.unique(seg_tensor).tolist()
        logger.debug(f"  Unique labels in segmentation: {unique_labels}")

        segmentations = []

        for composed in longitudinal_transforms:
            logger.info(f"Propagating to frame {composed.source_idx}...")

            with Timer(f"Propagate to frame {composed.source_idx}"):
                transform = composed.transform

                logger.debug(f"  FRAME {composed.source_idx} PROPAGATION:")
                logger.debug(f"    Forward transform type: {type(transform).__name__}")
                logger.debug(f"    Direction: Frame 0 → Frame {composed.source_idx} (need inverse)")

                # Compute inverse transform for segmentation propagation
                # Direction: We have transforms that map frame_i -> frame_0
                # For segmentation, we need to map frame_0 -> frame_i (inverse direction)
                #
                # CRITICAL: Do NOT fall back to forward transform - that would warp
                # the segmentation in the WRONG direction (approximately 15 voxel offset)
                #
                # SVFFD has analytical inverse (negated velocity field)
                # FFD does NOT have inverse - MIRTK uses Newton-Raphson approximation
                try:
                    inverse_transform = transform.inverse()
                    logger.debug(f"    Inverse method: Analytical (SVFFD)")
                except NotImplementedError as e:
                    # FFD (non-diffeomorphic) doesn't support analytical inverse
                    # Use MIRTK-style Newton-Raphson approximation for FFD inverse
                    logger.debug(f"    Inverse method: Newton-Raphson approximation (FFD)")
                    logger.warning(
                        f"Transform type '{type(transform).__name__}' does not support inverse(). "
                        f"Using Newton-Raphson approximation (MIRTK-style) for FFD inverse."
                    )
                    from ..postprocessing.segmentation import approximate_ffd_inverse
                    inverse_transform = approximate_ffd_inverse(transform, max_iterations=10, tolerance=1e-5)
                except Exception as e:
                    # Unexpected error during inverse computation
                    error_msg = (
                        f"Error computing inverse transform for frame {composed.source_idx}: {e}. "
                        f"Segmentation propagation requires a valid inverse transform."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # CRITICAL: Use segmentation_0.grid() to preserve original high-res resolution
                # The transform moves the seg to frame N's space, but keeps original resolution
                seg_n = transform_segmentation(
                    segmentation_0,
                    inverse_transform,
                    segmentation_0.grid(),  # Keep original segmentation resolution
                )

                # Log output segmentation info (verify label preservation)
                seg_n_tensor = seg_n.tensor()
                output_labels = torch.unique(seg_n_tensor).tolist()
                logger.debug(f"    Output segmentation shape: {tuple(seg_n_tensor.shape)}")
                logger.debug(f"    Output unique labels: {output_labels}")
                if set(output_labels) != set(unique_labels):
                    logger.warning(f"    WARNING: Labels changed! Input: {unique_labels}, Output: {output_labels}")

                segmentations.append(seg_n)

                # INCREMENTAL SAVE: Save this segmentation immediately
                if output_dir is not None:
                    # Save high-res version (original resolution) - for final use
                    self._save_segmentation(seg_n, composed.source_idx, image_4d, output_dir)

                    # Also save frame-grid version for intermediate sanity check (easy ITK-SNAP overlay)
                    # This resamples to match frame_XXX.nii.gz grid for quick visual verification
                    frame_grid = image_4d.get_frame(composed.source_idx).grid()
                    seg_n_frame_grid = transform_segmentation(
                        segmentation_0,
                        inverse_transform,
                        frame_grid,  # Resample to frame grid for easy overlay
                    )
                    self._save_segmentation_frame_grid(
                        seg_n_frame_grid, composed.source_idx, image_4d, output_dir
                    )

        logger.info(f"\nPropagated segmentation to {len(segmentations)} frames")

        return segmentations

    def save_results(
        self,
        result: MotionResult,
        output_dir: Union[str, Path],
        image_4d: Image4D,
        static_image: Optional[Image] = None,
        segmentation: Optional[Image] = None,
    ):
        """
        Save motion registration results (matching old code output structure)

        Creates output structure like the old sequential_reg_clean code:
        - transforms/: Per-stage transforms (.pth files)
        - intermediate_results/: Warped images after each stage
        - final_results/: Final warped images and references
        - debug_analysis/: Visualizations (convergence, grid deformation, etc.)
        - segmentation_results/: Segmentation transformations

        Args:
            result: MotionResult from pipeline
            output_dir: Output directory
            image_4d: Original 4D image for reference
            static_image: Original static image (for reference saving)
            segmentation: Original segmentation (for reference saving)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure matching old code
        transforms_dir = output_dir / "transforms"
        final_dir = output_dir / "final_results"
        debug_dir = output_dir / "debug_analysis"
        seg_dir = output_dir / "segmentation_results"

        for d in [transforms_dir, final_dir, debug_dir, seg_dir]:
            d.mkdir(exist_ok=True)

        # ================================================================
        # ALIGNMENT RESULTS (like old code's rigid/affine/ffd stages)
        # ================================================================
        if hasattr(self, '_alignment_intermediates') and self._alignment_intermediates is not None:
            ai = self._alignment_intermediates
            logger.info("Saving alignment intermediate results (old code format)...")

            # Save per-stage transforms
            if ai.rigid_transform is not None:
                self.composer.save_transform(
                    ai.rigid_transform,
                    transforms_dir / "rigid_transform.pth",
                    metadata={"type": "rigid", "stage": "alignment"},
                )
            if ai.affine_transform is not None:
                self.composer.save_transform(
                    ai.affine_transform,
                    transforms_dir / "affine_transform.pth",
                    metadata={"type": "affine", "stage": "alignment"},
                )
            if ai.ffd_transform is not None:
                self.composer.save_transform(
                    ai.ffd_transform,
                    transforms_dir / "ffd_transform.pth",
                    metadata={"type": "ffd", "stage": "alignment"},
                )

            # NOTE: Intermediate images (source_after_rigid_common.nii.gz, etc.)
            # are already saved incrementally in the alignment/ folder during registration.
            # No need to save duplicates here.

            # Save final results (like old code)
            # Source reference
            if ai.source_image is not None:
                self._save_image_nifti(
                    ai.source_image,
                    final_dir / "source_reference.nii.gz",
                    ai.source_image,
                )
            # Target reference
            if ai.target_image is not None:
                self._save_image_nifti(
                    ai.target_image,
                    final_dir / "target_reference.nii.gz",
                    ai.target_image,
                )

            # Save final warped results per stage
            if ai.rigid_warped is not None:
                self._save_image_nifti(
                    ai.rigid_warped,
                    final_dir / "source_moved_to_target_rigid.nii.gz",
                    ai.target_image,
                )
            if ai.affine_warped is not None:
                self._save_image_nifti(
                    ai.affine_warped,
                    final_dir / "source_moved_to_target_affine.nii.gz",
                    ai.target_image,
                )
            if ai.ffd_warped is not None:
                self._save_image_nifti(
                    ai.ffd_warped,
                    final_dir / "source_moved_to_target_ffd.nii.gz",
                    ai.target_image,
                )

            # Create debug visualizations
            self._create_alignment_visualizations(ai, debug_dir)

        # Save alignment composed transform
        if result.alignment_transform is not None:
            self.composer.save_transform(
                result.alignment_transform,
                transforms_dir / "alignment_composed.pth",
                metadata={"type": "alignment_composed", "model": self.registration_model},
            )

        # ================================================================
        # PAIRWISE TRANSFORMS
        # ================================================================
        for pw in result.pairwise_transforms:
            self.composer.save_transform(
                pw.transform,
                transforms_dir / f"pairwise_{pw.source_idx}_{pw.target_idx}.pth",
                metadata={
                    "type": "pairwise",
                    "source_idx": pw.source_idx,
                    "target_idx": pw.target_idx,
                    "loss": pw.final_loss,
                },
            )

        # ================================================================
        # LONGITUDINAL TRANSFORMS
        # ================================================================
        for long in result.longitudinal_transforms:
            self.composer.save_transform(
                long.transform,
                transforms_dir / f"longitudinal_0_to_{long.source_idx}.pth",
                metadata={
                    "type": "longitudinal",
                    "source_idx": long.source_idx,
                    "target_idx": long.target_idx,
                    "intermediates": long.intermediate_indices,
                },
            )

        # ================================================================
        # SEGMENTATION RESULTS
        # ================================================================
        # Original segmentation
        if segmentation is not None:
            self._save_image_nifti(
                segmentation,
                seg_dir / "source_seg_original.nii.gz",
                segmentation,
            )

        # Note: Per-frame segmentations are already saved incrementally
        # during run_full_pipeline() as seg_frame_000.nii.gz, etc.

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"  - Transforms: {transforms_dir}")
        logger.info(f"  - Final results: {final_dir}")
        logger.info(f"  - Debug visualizations: {debug_dir}")
        logger.info(f"  - Segmentations: {seg_dir}")

    def _save_image_nifti(self, image: Image, path: Path, reference: Image = None):
        """Save deepali Image as NIfTI with proper header.

        Uses the SAME affine construction method as _save_alignment_stage()
        to ensure correct orientation in ITK-SNAP.
        """
        try:
            tensor = image.tensor()
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            tensor_np = tensor.detach().cpu().numpy()

            # Transpose from torch [D, H, W] to NIfTI [X, Y, Z] order
            # This is CRITICAL - without this, the orientation is wrong
            tensor_nifti = np.transpose(tensor_np, (2, 1, 0))

            # Get grid info for affine
            grid = image.grid()
            spacing = grid.spacing().cpu().numpy()
            origin = grid.origin().cpu().numpy()
            direction = grid.direction().cpu().numpy()

            # Construct affine from grid components (same as _save_alignment_stage)
            affine = np.eye(4)
            affine[:3, :3] = direction @ np.diag(spacing)
            affine[:3, 3] = origin
            affine[:2] *= -1  # LPS to RAS conversion for NIfTI

            import nibabel as nib
            nifti_img = nib.Nifti1Image(tensor_nifti, affine)
            nib.save(nifti_img, str(path))
            logger.debug(f"Saved: {path}")
        except Exception as e:
            logger.warning(f"Could not save {path}: {e}")

    def _create_alignment_visualizations(self, ai: AlignmentIntermediates, debug_dir: Path):
        """Create alignment visualizations like old code"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            logger.info("Creating alignment visualizations...")

            # Get middle slice for 2D visualization
            if ai.source_image is not None:
                source_np = ai.source_image.tensor().squeeze().cpu().numpy()
                d_mid = source_np.shape[0] // 2

                # Side-by-side initial alignment
                if ai.target_image is not None:
                    target_np = ai.target_image.tensor().squeeze().cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(source_np[d_mid], cmap='gray')
                    axes[0].set_title('Source (Static)')
                    axes[0].axis('off')

                    axes[1].imshow(target_np[d_mid], cmap='gray')
                    axes[1].set_title('Target (Frame 0)')
                    axes[1].axis('off')

                    # Overlay
                    overlay = np.zeros((*source_np[d_mid].shape, 3))
                    overlay[:, :, 0] = source_np[d_mid] / (source_np[d_mid].max() + 1e-8)
                    overlay[:, :, 1] = target_np[d_mid] / (target_np[d_mid].max() + 1e-8)
                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay (Red=Source, Green=Target)')
                    axes[2].axis('off')

                    plt.tight_layout()
                    plt.savefig(debug_dir / 'side_by_side_common_coords_initial_alignment.png', dpi=150)
                    plt.close()

                # Per-stage results
                for stage, warped in [('rigid', ai.rigid_warped),
                                       ('affine', ai.affine_warped),
                                       ('ffd', ai.ffd_warped)]:
                    if warped is not None and ai.target_image is not None:
                        warped_np = warped.tensor().squeeze().cpu().numpy()
                        target_np = ai.target_image.tensor().squeeze().cpu().numpy()

                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        axes[0].imshow(warped_np[d_mid], cmap='gray')
                        axes[0].set_title(f'Warped Source (After {stage.upper()})')
                        axes[0].axis('off')

                        axes[1].imshow(target_np[d_mid], cmap='gray')
                        axes[1].set_title('Target (Frame 0)')
                        axes[1].axis('off')

                        # Overlay
                        overlay = np.zeros((*warped_np[d_mid].shape, 3))
                        overlay[:, :, 0] = warped_np[d_mid] / (warped_np[d_mid].max() + 1e-8)
                        overlay[:, :, 1] = target_np[d_mid] / (target_np[d_mid].max() + 1e-8)
                        axes[2].imshow(overlay)
                        axes[2].set_title(f'Overlay After {stage.upper()}')
                        axes[2].axis('off')

                        plt.tight_layout()
                        plt.savefig(debug_dir / f'side_by_side_common_coords_{stage}_result.png', dpi=150)
                        plt.close()

            # Convergence plots
            if ai.convergence_data:
                for stage, data in ai.convergence_data.items():
                    if data is not None:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        if isinstance(data, dict):
                            for level, losses in data.items():
                                if isinstance(losses, (list, np.ndarray)):
                                    ax.plot(losses, label=str(level))
                            ax.legend()
                        elif isinstance(data, (list, np.ndarray)):
                            ax.plot(data)
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Loss')
                        ax.set_title(f'{stage.upper()} Convergence')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(debug_dir / f'{stage}_convergence_analysis.png', dpi=150)
                        plt.close()

            logger.info(f"Visualizations saved to {debug_dir}")

        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


# Convenience function
def run_motion_registration(
    image_4d_path: Union[str, Path],
    static_image_path: Optional[Union[str, Path]] = None,
    segmentation_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "./motion_output",
    device: str = "cpu",
    registration_model: str = "rigid+affine+ffd",
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> MotionResult:
    """
    Convenience function to run full motion registration pipeline

    Args:
        image_4d_path: Path to 4D NIfTI image
        static_image_path: Optional path to static high-res image
        segmentation_path: Optional path to segmentation on static
        output_dir: Output directory
        device: Computation device
        registration_model: Registration model
        start_frame: Starting frame in 4D
        num_frames: Number of frames to process

    Returns:
        MotionResult with all outputs
    """
    from ..data.image_4d import load_image_4d
    from ..data import load_image

    # Load 4D image
    image_4d = load_image_4d(image_4d_path, start_frame, num_frames)

    # Load static and segmentation if provided
    # load_image returns (Image, sitk_image, metadata) tuple
    static_image = None
    if static_image_path:
        static_image, _, _ = load_image(static_image_path)

    segmentation = None
    if segmentation_path:
        segmentation, _, _ = load_image(segmentation_path)

    # Create pipeline
    pipeline = MotionRegistration(
        device=device,
        registration_model=registration_model,
    )

    # Run pipeline with incremental saving enabled
    result = pipeline.run_full_pipeline(
        image_4d=image_4d,
        static_image=static_image,
        segmentation=segmentation,
        output_dir=output_dir,  # Enable incremental saving
    )

    # Final save (for any remaining results not saved incrementally)
    pipeline.save_results(result, output_dir, image_4d)

    return result
