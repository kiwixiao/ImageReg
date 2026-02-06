"""
DAREG FFD Registration

MIRTK-equivalent Free-Form Deformation (B-spline) registration.
Uses Conjugate Gradient Descent with Adaptive Line Search
matching MIRTK behavior exactly.

Features:
- NMI similarity + foreground overlap masking (FG_Overlap)
- Bending + diffusion energy regularization
- World coordinate regularization for anisotropic images
- Multi-resolution coarse-to-fine optimization
- MIRTK-equivalent optimizer (CG + Adaptive Line Search)

Reference:
- MIRTK/Modules/Numerics/src/ConjugateGradientDescent.cc
- MIRTK/Modules/Numerics/src/AdaptiveLineSearch.cc
- mirtk_binary_reg_pipeline_demo/REGTOOL/register.cfg
"""

import torch
from typing import Optional, Dict, List, Tuple
import numpy as np

from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial
from deepali.spatial import FreeFormDeformation, SequentialTransform
from deepali.losses import functional as L

from .base import BaseRegistration, RegistrationResult
from .optimizers import ConjugateGradientOptimizer
from ..preprocessing.pyramid import create_pyramid
from ..preprocessing.grid_manager import get_anisotropic_pyramid_dims
from ..utils.logging_config import get_logger
from ..utils.progress_tracker import ProgressTracker

logger = get_logger("ffd")


class FFDRegistration(BaseRegistration):
    """
    MIRTK-equivalent FFD (Free-Form Deformation) Registration

    Uses B-spline control points for non-rigid deformation.
    Implements MIRTK behavior exactly:
    - Conjugate Gradient Descent with Adaptive Line Search
    - NMI similarity with FG_Overlap foreground masking
    - Bending + diffusion energy regularization
    - World coordinate regularization (fixes anisotropic images)
    - Multi-resolution pyramid optimization
    - Support region constraint (MIRTK line 195 equivalent)

    MIRTK Reference (register.cfg):
        Energy function = NMI + 0.001 BE(T) + 0.0005 LE(T)
        Control point spacing = 4 mm
        No. of bins = 256
        Maximum streak of rejected steps = 1
        Maximum no. of iterations = 100
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        control_point_spacing: int = 4,
        pyramid_levels: int = 4,
        iterations_per_level: List[int] = None,
        learning_rates: List[float] = None,
        num_bins: int = 256,  # MIRTK default
        foreground_threshold: float = 0.01,
        bending_weight: float = 0.001,
        diffusion_weight: float = 0.0005,
        laplacian_weight: float = 0.0005,
        world_coord_regularization: bool = True,
        max_rejected_streak: int = 1,
        epsilon: float = 1e-4,
        delta: float = 1e-12,
        support_region_threshold: float = 0.3,
        roi_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ):
        """
        Initialize MIRTK-equivalent FFD registration.

        Args:
            device: Computation device
            control_point_spacing: Spacing between control points in mm (MIRTK: 4)
            pyramid_levels: Number of multi-resolution levels (MIRTK: 4)
            iterations_per_level: Max iterations per level (MIRTK: 100)
            learning_rates: Learning rate per level (MIRTK: adaptive)
            num_bins: NMI histogram bins (MIRTK: 256)
            foreground_threshold: Threshold for FG_Overlap mask
            bending_weight: Bending energy weight (MIRTK: 0.001)
            diffusion_weight: Diffusion/linear energy weight (MIRTK: 0.0005)
            laplacian_weight: Laplacian energy weight (MIRTK: 0.0005 LE(T))
            world_coord_regularization: Enable world coordinate scaling for regularization
            max_rejected_streak: Stop after N consecutive rejections (MIRTK: 1)
            epsilon: Minimum relative function change (MIRTK: 1e-4)
            delta: Minimum DoF change (MIRTK: 1e-12)
            support_region_threshold: Minimum foreground fraction in control point support region
            roi_mask: Optional binary ROI mask to restrict registration region.
            num_samples: Optional fixed number of voxels to sample for NMI speedup
            sample_ratio: Optional ratio of voxels to sample for NMI speedup
        """
        super().__init__(device)

        self.control_point_spacing = control_point_spacing
        self.pyramid_levels = pyramid_levels
        self.iterations_per_level = iterations_per_level or [100] * pyramid_levels
        self.learning_rates = learning_rates or [0.01] * pyramid_levels
        self.num_bins = num_bins
        self.foreground_threshold = foreground_threshold
        self.bending_weight = bending_weight
        self.diffusion_weight = diffusion_weight
        self.laplacian_weight = laplacian_weight  # MIRTK default: 0.0005 LE(T)
        self.world_coord_regularization = world_coord_regularization
        self.max_rejected_streak = max_rejected_streak
        self.epsilon = epsilon
        self.delta = delta
        self.support_region_threshold = support_region_threshold
        self.roi_mask = roi_mask
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio

    def register(
        self,
        source: Image,
        target: Image,
        initial_transform: Optional[spatial.AffineTransform] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> RegistrationResult:
        """
        Perform MIRTK-equivalent FFD registration.

        Args:
            source: Source (moving) image (typically after affine alignment)
            target: Target (fixed) image
            initial_transform: Optional initial affine transform (for composition)
            progress_tracker: Optional progress tracker for monitoring

        Returns:
            RegistrationResult with transform and warped source
        """
        logger.info("=" * 60)
        logger.info("FFD REGISTRATION (MIRTK-equivalent)")
        logger.info("=" * 60)
        logger.info(f"Control point spacing: {self.control_point_spacing}mm")
        logger.info(f"Regularization: bending={self.bending_weight}, diffusion={self.diffusion_weight}, laplacian={self.laplacian_weight}")
        logger.info(f"Optimizer: Conjugate Gradient + Adaptive Line Search")
        logger.info(f"Convergence: max_rejected_streak={self.max_rejected_streak}")
        if self.world_coord_regularization:
            logger.info("World coordinate regularization: ENABLED")

        # Determine pyramid dimensions for anisotropic handling
        pyramid_dims = get_anisotropic_pyramid_dims(target.grid())

        # Create image pyramids
        source_pyramid = create_pyramid(source, self.pyramid_levels)
        target_pyramid = create_pyramid(target, self.pyramid_levels)

        logger.info(f"Created {self.pyramid_levels}-level pyramid")
        if pyramid_dims:
            logger.info(f"  Anisotropic mode: downsampling dims {pyramid_dims}")

        # Get coarsest level grid for initial transform
        coarsest_level = self.pyramid_levels - 1
        coarsest_grid = target_pyramid[coarsest_level].grid

        # Create FFD transform
        transform = self._create_transform(coarsest_grid)
        transform = transform.to(self.device).train()

        # Wrap in SequentialTransform for proper grid management (deepali pattern)
        grid_transform = SequentialTransform(transform)
        grid_transform = grid_transform.to(self.device)

        logger.info("Created FFD transform")

        # Compute world scale for regularization (once)
        world_scale = None
        if self.world_coord_regularization:
            world_scale = self._compute_world_scale(target.grid())
            logger.info(f"World scale factors: {[f'{s:.2f}' for s in world_scale.squeeze().tolist()]}")

        # Track convergence
        loss_history = {}
        best_overall_loss = float('inf')

        # Multi-resolution optimization (coarse to fine)
        for level_idx in range(self.pyramid_levels):
            level = self.pyramid_levels - 1 - level_idx  # Coarse to fine

            logger.info(f"\nLevel {level_idx + 1}/{self.pyramid_levels} (pyramid {level})")

            # Report progress
            if progress_tracker:
                max_iters_for_level = self.iterations_per_level[level_idx]
                progress_tracker.set_level(level_idx, max_iters_for_level)

            # Get images at this level
            source_level = source_pyramid[level].image
            target_level = target_pyramid[level].image

            logger.info(f"  Shape: {tuple(source_level.grid().shape)}")

            # Update transform grid (subdivide control points for finer levels)
            if level_idx > 0:
                transform.grid_(target_level.grid())
                logger.debug("  Control points subdivided")

            # Set output grid
            grid_transform.grid_(target_level.grid())

            # Get level parameters
            max_iters = self.iterations_per_level[level_idx]
            lr = self.learning_rates[level_idx]

            # Create MIRTK-equivalent optimizer
            optimizer = ConjugateGradientOptimizer(
                grid_transform.parameters(),
                lr=lr,
                max_rejected_streak=self.max_rejected_streak,
                epsilon=self.epsilon,
                delta=self.delta,
            )
            logger.info(f"  Optimizer: Conjugate Gradient, lr={lr}")

            # Create transformer
            transformer = spatial.ImageTransformer(grid_transform)

            # Prepare tensors
            source_tensor = self._ensure_5d(source_level.tensor()).to(self.device)
            target_tensor = self._ensure_5d(target_level.tensor()).to(self.device)

            # Compute STATIC foreground overlap mask (MIRTK FG_Overlap)
            fg_mask = self._compute_foreground_overlap_mask(
                source_tensor, target_tensor, self.foreground_threshold,
                roi_mask=self.roi_mask
            )
            fg_coverage = fg_mask.sum() / fg_mask.numel() * 100
            logger.info(f"  Foreground overlap: {fg_coverage:.1f}%")

            # Pre-mask images for cleaner gradients
            masked_source = (source_tensor * fg_mask).detach().requires_grad_(True)
            masked_target = (target_tensor * fg_mask).detach()

            # Compute world scale for this level
            level_world_scale = None
            if self.world_coord_regularization:
                level_world_scale = self._compute_world_scale(target_level.grid())

            # Optimization loop with MIRTK-style CG optimizer
            level_losses = []
            best_level_loss = float('inf')

            # Store for logging
            current_similarity_loss = torch.tensor(0.0)
            current_bending_loss = torch.tensor(0.0)

            for iteration in range(max_iters):
                # Define closure for optimizer
                def closure():
                    nonlocal current_similarity_loss, current_bending_loss

                    # Warp masked source
                    warped_source = transformer(masked_source)

                    # Compute NMI loss with mask
                    similarity_loss = self._compute_nmi_loss(
                        warped_source, masked_target,
                        num_bins=self.num_bins,
                        mask=fg_mask,
                        num_samples=self.num_samples,
                        sample_ratio=self.sample_ratio
                    )

                    # Compute regularization (bending + diffusion + laplacian)
                    bending_loss, diffusion_loss, laplacian_loss = self._compute_regularization(
                        transform, level_world_scale
                    )

                    # Total loss (MIRTK style: NMI + BE + LE)
                    total_loss = (
                        similarity_loss
                        + self.bending_weight * bending_loss
                        + self.diffusion_weight * diffusion_loss
                        + self.laplacian_weight * laplacian_loss
                    )

                    # Backward
                    total_loss.backward()

                    # Apply MIRTK-style gradient processing
                    self._process_gradients(transform, fg_mask)

                    # Store for logging
                    current_similarity_loss = similarity_loss
                    current_bending_loss = bending_loss

                    return total_loss

                # Optimization step with MIRTK-style CG + Adaptive Line Search
                loss_val, converged = optimizer.step(closure)
                level_losses.append(loss_val)

                # Track best
                if loss_val < best_level_loss:
                    best_level_loss = loss_val

                # Report iteration progress
                if progress_tracker:
                    progress_tracker.update_iteration(
                        iteration=iteration + 1,
                        loss=loss_val,
                        nmi_loss=current_similarity_loss.item(),
                        reg_loss=self.bending_weight * current_bending_loss.item()
                    )

                    # Generate visualization every 20 iterations
                    if (iteration + 1) % 20 == 0 and progress_tracker.output_dir:
                        with torch.no_grad():
                            vis_warped = transformer(source_tensor)
                        vis_path = str(progress_tracker.output_dir / "progress_overlay.png")
                        saved_path = self._generate_progress_visualization(
                            vis_warped, target_tensor, vis_path,
                            level=level_idx, iteration=iteration + 1
                        )
                        if saved_path:
                            progress_tracker.set_visualization("progress_overlay.png")

                # Log progress
                if (iteration + 1) % 20 == 0 or iteration == 0:
                    logger.info(
                        f"  Iter {iteration+1:3d}: loss={loss_val:.6f}, "
                        f"nmi={current_similarity_loss.item():.6f}, "
                        f"bend={self.bending_weight * current_bending_loss.item():.6f}"
                    )

                # MIRTK-style convergence
                if converged:
                    logger.info(f"  Converged at iteration {iteration + 1}")
                    break

            loss_history[f"level_{level}"] = level_losses
            logger.info(f"  Level complete: best_loss={best_level_loss:.6f}")

            if best_level_loss < best_overall_loss:
                best_overall_loss = best_level_loss

        # Final warping at original resolution
        grid_transform.eval()
        grid_transform.grid_(target.grid())
        final_transformer = spatial.ImageTransformer(grid_transform)

        with torch.no_grad():
            source_tensor = self._ensure_5d(source.tensor()).to(self.device)
            warped_result = final_transformer(source_tensor)

        # Create result image
        warped_image = Image(
            data=warped_result.squeeze(0),
            grid=target.grid()
        )

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(transform, target.grid())

        logger.info("\n" + "=" * 60)
        logger.info("FFD REGISTRATION COMPLETE")
        logger.info(f"  Final loss: {best_overall_loss:.6f}")
        logger.info(f"  Max displacement: {quality_metrics.get('max_displacement_mm', 0):.3f}mm")
        logger.info("=" * 60)

        return RegistrationResult(
            transform=transform,
            warped_source=warped_image,
            final_loss=best_overall_loss,
            loss_history=loss_history,
            quality_metrics=quality_metrics,
            metadata={
                "method": "ffd",
                "levels": self.pyramid_levels,
                "control_point_spacing": self.control_point_spacing,
                "optimizer": "conjugate_gradient",
            },
        )

    def _create_transform(self, grid: Grid):
        """Create FFD transform with MIRTK-style stride computation."""
        stride = self._compute_stride(grid)
        transform = FreeFormDeformation(
            grid=grid,
            stride=stride,
        )
        return transform

    def _compute_stride(self, grid: Grid) -> Tuple[int, ...]:
        """
        Compute control point stride from spacing (mm to voxels).

        MIRTK-style adaptive minimum stride per dimension.
        """
        spacing = grid.spacing()
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy()

        shape = grid.shape
        if isinstance(shape, torch.Size):
            shape = tuple(shape)

        strides = []
        for i, (dim_spacing, dim_size) in enumerate(zip(spacing, shape)):
            # Compute stride
            computed_stride = int(round(self.control_point_spacing / float(dim_spacing)))

            # MIRTK adaptive: ensure at least 3 control points per dimension
            max_stride = max(1, (int(dim_size) - 1) // 2)
            stride = max(1, min(computed_stride, max_stride))
            strides.append(stride)

        logger.debug(f"Control point stride: {tuple(strides)}")
        return tuple(strides)

    def _compute_world_scale(self, grid: Grid) -> torch.Tensor:
        """
        Compute world coordinate scale for regularization.

        Converts normalized coordinates to mm for proper anisotropic handling.
        """
        extent = grid.size()  # Physical size in mm
        if isinstance(extent, torch.Tensor):
            extent = extent.float()
        else:
            extent = torch.tensor(extent, dtype=torch.float32)

        # Scale: normalized [-1,1] -> world mm
        scale = extent / 2.0

        # Shape for broadcasting [1, 3, 1, 1, 1]
        return scale.view(1, 3, 1, 1, 1).to(self.device)

    def _compute_laplacian_loss(
        self,
        displacement: torch.Tensor,
        world_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Laplacian energy: Σ(∇²u)² where ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z².

        This is different from bending energy which sums all 2nd derivatives separately.
        Laplacian energy penalizes the sum of second derivatives (Laplacian operator),
        matching MIRTK's LE(T) term.

        MIRTK Reference:
            Energy function = NMI + 0.001 BE(T) + 0.0005 LE(T)

        Args:
            displacement: Displacement field (N, 3, D, H, W) or (3, D, H, W)
            world_scale: Optional scale tensor for world coordinate regularization

        Returns:
            Scalar Laplacian energy
        """
        # Ensure 5D
        if displacement.dim() == 4:
            displacement = displacement.unsqueeze(0)

        # Apply world coordinate scaling
        if world_scale is not None:
            displacement = displacement * world_scale

        N, C, D, H, W = displacement.shape

        # Compute Laplacian for each displacement component
        # Laplacian = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        laplacian_total = torch.zeros(N, 1, D, H, W, device=displacement.device)

        for c in range(C):
            u = displacement[:, c:c+1]

            # Second derivatives using central differences: d²u/dx² = u[i+1] - 2*u[i] + u[i-1]
            laplacian_c = torch.zeros_like(u)

            # d²/dz² (dim 2)
            if D >= 3:
                laplacian_c[:, :, 1:-1, :, :] += (
                    u[:, :, 2:, :, :] - 2 * u[:, :, 1:-1, :, :] + u[:, :, :-2, :, :]
                )

            # d²/dy² (dim 3)
            if H >= 3:
                laplacian_c[:, :, :, 1:-1, :] += (
                    u[:, :, :, 2:, :] - 2 * u[:, :, :, 1:-1, :] + u[:, :, :, :-2, :]
                )

            # d²/dx² (dim 4)
            if W >= 3:
                laplacian_c[:, :, :, :, 1:-1] += (
                    u[:, :, :, :, 2:] - 2 * u[:, :, :, :, 1:-1] + u[:, :, :, :, :-2]
                )

            laplacian_total = laplacian_total + laplacian_c

        # Laplacian energy = mean of squared Laplacian
        return (laplacian_total ** 2).mean()

    def _compute_regularization(
        self,
        transform,
        world_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute MIRTK-style regularization: Bending + Diffusion + Laplacian energy.

        Args:
            transform: FFD transform
            world_scale: Optional scale tensor for world coordinate regularization

        Returns:
            Tuple of (bending_loss, diffusion_loss, laplacian_loss)
        """
        # Get control point data
        field = transform.data()

        # Apply world coordinate scaling
        if world_scale is not None:
            if field.dim() == 4:  # [C, D, H, W]
                field_scaled = field * world_scale.squeeze(0)
            else:  # [N, C, D, H, W]
                field_scaled = field * world_scale
        else:
            field_scaled = field

        # Compute bending energy (2nd order derivatives)
        bending = L.bending_loss(field_scaled)

        # Compute diffusion energy (1st order derivatives)
        diffusion = L.diffusion_loss(field_scaled)

        # Compute Laplacian energy (MIRTK LE(T))
        laplacian = self._compute_laplacian_loss(field, world_scale)

        return bending, diffusion, laplacian

    def _process_gradients(self, transform, fg_mask: torch.Tensor):
        """
        Apply MIRTK-style gradient processing.

        Includes:
        - Gradient normalization per control point
        - Boundary constraint
        - Gradient thresholding
        - Support region constraint (MIRTK line 195 equivalent)
        """
        if transform.params.grad is None:
            return

        grad = transform.params.grad

        # 1. Gradient normalization (MIRTK ImageSimilarity.cc)
        grad_norm = torch.sqrt((grad ** 2).sum(dim=1, keepdim=True) + 1e-10)
        max_norm = grad_norm.max()
        sigma = 0.5 * max_norm  # Preconditioning factor
        transform.params.grad = grad / (grad_norm + sigma)

        # 2. Boundary constraint (zero gradients near edges)
        if grad.dim() == 5:
            _, _, D, H, W = grad.shape
            support_radius = 2
            mask = torch.ones_like(grad)
            if D > 2 * support_radius:
                mask[:, :, :support_radius, :, :] = 0
                mask[:, :, -support_radius:, :, :] = 0
            if H > 2 * support_radius:
                mask[:, :, :, :support_radius, :] = 0
                mask[:, :, :, -support_radius:, :] = 0
            if W > 2 * support_radius:
                mask[:, :, :, :, :support_radius] = 0
                mask[:, :, :, :, -support_radius:] = 0
            transform.params.grad = transform.params.grad * mask

        # 3. Threshold small gradients (MIRTK line 195 equivalent)
        grad_magnitude = torch.sqrt((transform.params.grad ** 2).sum(dim=1, keepdim=True))
        threshold_mask = (grad_magnitude > 1e-8).float()
        transform.params.grad = transform.params.grad * threshold_mask

        # 4. Support region constraint
        self._apply_support_region_constraint(transform, fg_mask)

    def _apply_support_region_constraint(
        self,
        transform,
        overlap_mask: torch.Tensor,
        threshold: float = None,
    ):
        """
        Control Point Support Region Masking (MIRTK line 195 equivalent).

        Zero gradients for control points whose B-spline support region
        has less than threshold fraction within the foreground overlap mask.
        """
        import torch.nn.functional as F

        if threshold is None:
            threshold = self.support_region_threshold

        with torch.no_grad():
            params = transform.params
            if params.grad is None:
                return

            # Get control point grid shape
            if params.dim() != 5:
                return

            _, _, cp_D, cp_H, cp_W = params.shape
            stride = transform.stride

            # Handle stride as int or tuple
            if isinstance(stride, int):
                stride = (stride, stride, stride)
            elif hasattr(stride, '__iter__'):
                stride = tuple(stride)

            # Compute kernel size based on B-spline support
            kernel_size = tuple(
                min(4 * s, overlap_mask.shape[i + 2])
                for i, s in enumerate(stride)
            )

            # Ensure overlap_mask is 5D [1, 1, D, H, W]
            if overlap_mask.dim() == 3:
                mask_5d = overlap_mask.unsqueeze(0).unsqueeze(0)
            elif overlap_mask.dim() == 4:
                mask_5d = overlap_mask.unsqueeze(0)
            else:
                mask_5d = overlap_mask

            # Pad mask for boundary control points
            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            padded_mask = F.pad(
                mask_5d.float(),
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                value=0
            )

            # Compute fraction of foreground in each control point's support region
            overlap_fractions = F.avg_pool3d(
                padded_mask,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )

            # Create control point mask: 1 where >= threshold foreground
            cp_mask = (overlap_fractions > threshold).float()

            # Resize to match control point grid if needed
            if cp_mask.shape[-3:] != (cp_D, cp_H, cp_W):
                cp_mask = F.interpolate(cp_mask, size=(cp_D, cp_H, cp_W), mode='nearest')

            # Expand mask to match gradient shape [1, 3, D, H, W]
            cp_mask_expanded = cp_mask.expand_as(params.grad)

            # Apply mask
            params.grad = params.grad * cp_mask_expanded

    def _compute_quality_metrics(self, transform, grid: Grid) -> Dict[str, float]:
        """Compute quality metrics for FFD transform."""
        try:
            transform.update()

            # Get displacement field
            if hasattr(transform, 'u'):
                u = transform.u
            elif hasattr(transform, 'disp'):
                u = transform.disp(grid)
            else:
                return {"max_displacement_mm": 0.0, "mean_displacement_mm": 0.0}

            if u is None:
                return {"max_displacement_mm": 0.0, "mean_displacement_mm": 0.0}

            # Compute displacement magnitude
            if u.dim() == 4:
                u_mag = torch.sqrt((u ** 2).sum(dim=0))
            else:
                u_mag = torch.sqrt((u ** 2).sum(dim=1))

            max_disp = float(u_mag.max())
            mean_disp = float(u_mag.mean())

            return {
                "max_displacement_mm": max_disp,
                "mean_displacement_mm": mean_disp,
            }

        except Exception:
            return {"max_displacement_mm": 0.0, "mean_displacement_mm": 0.0}
