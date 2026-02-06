"""
DAREG SVFFD Registration

Stationary Velocity Free-Form Deformation (diffeomorphic) registration.
Uses Adam optimizer with SVFFD-specific tuning parameters for velocity field
optimization.

Features:
- Diffeomorphic (topology-preserving) transformations
- Velocity field with exponential integration (scaling-and-squaring)
- SVFFD-specific regularization:
  - Velocity smoothing (Gaussian blur)
  - Laplacian regularization (∇²v)
  - Jacobian penalty (topology preservation)
- NMI similarity + foreground overlap masking
- Multi-resolution coarse-to-fine optimization

Key difference from FFD:
- FFD: Directly optimizes displacement control points
- SVFFD: Optimizes velocity field, integrates via ExpFlow for smooth deformation
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np

from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial
from deepali.spatial import (
    StationaryVelocityFreeFormDeformation as SVFFD,
    SequentialTransform,
)
from deepali.losses import functional as L

from .base import BaseRegistration, RegistrationResult
from ..preprocessing.pyramid import create_pyramid
from ..preprocessing.grid_manager import get_anisotropic_pyramid_dims
from ..utils.logging_config import get_logger
from ..utils.progress_tracker import ProgressTracker

logger = get_logger("svffd")


class SVFFDRegistration(BaseRegistration):
    """
    SVFFD (Stationary Velocity Free-Form Deformation) Registration

    Diffeomorphic registration using velocity field parameterization.
    Uses Adam optimizer (appropriate for velocity field optimization).

    Features:
    - Topology-preserving (diffeomorphic) transformations
    - Velocity field smoothing for regularization
    - Laplacian regularization for smoother velocity fields
    - Jacobian penalty to prevent folding
    - NMI similarity with FG_Overlap foreground masking

    SVFFD-specific parameters (tunable):
    - velocity_smoothing_sigma: Gaussian blur sigma for velocity field (mm)
    - laplacian_weight: Weight for ∇²v regularization
    - jacobian_penalty: Weight for Jacobian determinant penalty
    - integration_steps: Number of scaling-and-squaring steps
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
        bending_weight: float = 0.0005,
        diffusion_weight: float = 0.00025,
        world_coord_regularization: bool = True,
        integration_steps: int = 5,
        velocity_smoothing_sigma: float = 0.0,
        laplacian_weight: float = 0.0,
        jacobian_penalty: float = 0.0,
        convergence_delta: float = 1e-6,
        convergence_patience: int = 20,
        support_region_threshold: float = 0.3,
        roi_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ):
        """
        Initialize SVFFD registration with tunable parameters.

        Args:
            device: Computation device
            control_point_spacing: Spacing between control points in mm
            pyramid_levels: Number of multi-resolution levels
            iterations_per_level: Max iterations per level
            learning_rates: Learning rate per level (recommended: 0.005, half of FFD)
            num_bins: NMI histogram bins
            foreground_threshold: Threshold for FG_Overlap mask
            bending_weight: Bending energy weight (recommended: 0.0005 for SVFFD)
            diffusion_weight: Diffusion energy weight (recommended: 0.00025 for SVFFD)
            world_coord_regularization: Enable world coordinate scaling for regularization
            integration_steps: Scaling-and-squaring steps for ExpFlow (default: 5)
            velocity_smoothing_sigma: Gaussian blur sigma for velocity field (mm, 0=disabled)
            laplacian_weight: Weight for Laplacian regularization (0=disabled)
            jacobian_penalty: Weight for Jacobian determinant penalty (0=disabled)
            convergence_delta: Minimum loss change for convergence
            convergence_patience: Iterations without improvement before stopping
            support_region_threshold: Minimum foreground fraction in control point support region
            roi_mask: Optional binary ROI mask to restrict registration region.
            num_samples: Optional fixed number of voxels to sample for NMI speedup
            sample_ratio: Optional ratio of voxels to sample for NMI speedup
        """
        super().__init__(device)

        self.control_point_spacing = control_point_spacing
        self.pyramid_levels = pyramid_levels
        self.iterations_per_level = iterations_per_level or [100] * pyramid_levels
        # SVFFD typically uses lower learning rates than FFD
        self.learning_rates = learning_rates or [0.005] * pyramid_levels
        self.num_bins = num_bins
        self.foreground_threshold = foreground_threshold
        # SVFFD typically uses lower regularization weights
        self.bending_weight = bending_weight
        self.diffusion_weight = diffusion_weight
        self.world_coord_regularization = world_coord_regularization
        self.integration_steps = integration_steps
        # SVFFD-specific tuning parameters
        self.velocity_smoothing_sigma = velocity_smoothing_sigma
        self.laplacian_weight = laplacian_weight
        self.jacobian_penalty = jacobian_penalty
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience
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
        Perform SVFFD (diffeomorphic) registration.

        Args:
            source: Source (moving) image (typically after affine alignment)
            target: Target (fixed) image
            initial_transform: Optional initial affine transform (for composition)
            progress_tracker: Optional progress tracker for monitoring

        Returns:
            RegistrationResult with transform and warped source
        """
        logger.info("=" * 60)
        logger.info("SVFFD REGISTRATION (Diffeomorphic)")
        logger.info("=" * 60)
        logger.info(f"Control point spacing: {self.control_point_spacing}mm")
        logger.info(f"Regularization: bending={self.bending_weight}, diffusion={self.diffusion_weight}")
        logger.info(f"Integration steps: {self.integration_steps}")
        logger.info(f"Optimizer: Adam")
        if self.velocity_smoothing_sigma > 0:
            logger.info(f"Velocity smoothing: sigma={self.velocity_smoothing_sigma}mm")
        if self.laplacian_weight > 0:
            logger.info(f"Laplacian weight: {self.laplacian_weight}")
        if self.jacobian_penalty > 0:
            logger.info(f"Jacobian penalty: {self.jacobian_penalty}")
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

        # Create SVFFD transform
        transform = self._create_transform(coarsest_grid)
        transform = transform.to(self.device).train()

        # Wrap in SequentialTransform for proper grid management
        grid_transform = SequentialTransform(transform)
        grid_transform = grid_transform.to(self.device)

        logger.info(f"Created SVFFD transform (steps={self.integration_steps})")

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

            # Use Adam optimizer (appropriate for velocity fields)
            optimizer = optim.Adam(grid_transform.parameters(), lr=lr)
            logger.info(f"  Optimizer: Adam, lr={lr}")

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

            # Optimization loop
            level_losses = []
            best_level_loss = float('inf')
            no_improve_count = 0

            # Store for logging
            current_similarity_loss = torch.tensor(0.0)
            current_bending_loss = torch.tensor(0.0)

            for iteration in range(max_iters):
                optimizer.zero_grad()

                # Apply velocity smoothing if enabled
                if self.velocity_smoothing_sigma > 0:
                    self._smooth_velocity_field(transform, target_level.grid())

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

                # Compute regularization on velocity field
                bending_loss, diffusion_loss = self._compute_regularization(
                    transform, level_world_scale
                )

                # Total loss
                total_loss = (
                    similarity_loss
                    + self.bending_weight * bending_loss
                    + self.diffusion_weight * diffusion_loss
                )

                # Add Laplacian regularization if enabled
                if self.laplacian_weight > 0:
                    laplacian_loss = self._compute_laplacian_loss(transform)
                    total_loss = total_loss + self.laplacian_weight * laplacian_loss

                # Add Jacobian penalty if enabled
                if self.jacobian_penalty > 0:
                    jac_loss = self._compute_jacobian_loss(transform)
                    total_loss = total_loss + self.jacobian_penalty * jac_loss

                # Backward
                total_loss.backward()

                # Apply MIRTK-style gradient processing
                self._process_gradients(transform, fg_mask)

                # Optimizer step
                optimizer.step()

                loss_val = float(total_loss)
                level_losses.append(loss_val)

                # Store for logging
                current_similarity_loss = similarity_loss
                current_bending_loss = bending_loss

                # Track best
                if loss_val < best_level_loss - self.convergence_delta:
                    best_level_loss = loss_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

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

                # Early stopping
                if no_improve_count >= self.convergence_patience:
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

        # Compute quality metrics (including Jacobian for SVFFD)
        quality_metrics = self._compute_quality_metrics(transform, target.grid())

        logger.info("\n" + "=" * 60)
        logger.info("SVFFD REGISTRATION COMPLETE")
        logger.info(f"  Final loss: {best_overall_loss:.6f}")
        logger.info(f"  Max displacement: {quality_metrics.get('max_displacement_mm', 0):.3f}mm")
        if 'min_jacobian' in quality_metrics:
            logger.info(f"  Min Jacobian: {quality_metrics['min_jacobian']:.4f}")
            logger.info(f"  Folding: {quality_metrics['folding_percentage']:.2f}%")
        logger.info("=" * 60)

        return RegistrationResult(
            transform=transform,
            warped_source=warped_image,
            final_loss=best_overall_loss,
            loss_history=loss_history,
            quality_metrics=quality_metrics,
            metadata={
                "method": "svffd",
                "levels": self.pyramid_levels,
                "control_point_spacing": self.control_point_spacing,
                "integration_steps": self.integration_steps,
                "velocity_smoothing_sigma": self.velocity_smoothing_sigma,
                "laplacian_weight": self.laplacian_weight,
                "jacobian_penalty": self.jacobian_penalty,
                "optimizer": "adam",
            },
        )

    def _create_transform(self, grid: Grid):
        """Create SVFFD transform with ExpFlow integration."""
        stride = self._compute_stride(grid)
        transform = SVFFD(
            grid=grid,
            stride=stride,
            steps=self.integration_steps,
        )
        return transform

    def _compute_stride(self, grid: Grid) -> Tuple[int, ...]:
        """
        Compute control point stride from spacing (mm to voxels).
        """
        spacing = grid.spacing()
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy()

        shape = grid.shape
        if isinstance(shape, torch.Size):
            shape = tuple(shape)

        strides = []
        for i, (dim_spacing, dim_size) in enumerate(zip(spacing, shape)):
            computed_stride = int(round(self.control_point_spacing / float(dim_spacing)))
            max_stride = max(1, (int(dim_size) - 1) // 2)
            stride = max(1, min(computed_stride, max_stride))
            strides.append(stride)

        logger.debug(f"Control point stride: {tuple(strides)}")
        return tuple(strides)

    def _compute_world_scale(self, grid: Grid) -> torch.Tensor:
        """Compute world coordinate scale for regularization."""
        extent = grid.size()
        if isinstance(extent, torch.Tensor):
            extent = extent.float()
        else:
            extent = torch.tensor(extent, dtype=torch.float32)

        scale = extent / 2.0
        return scale.view(1, 3, 1, 1, 1).to(self.device)

    def _compute_regularization(
        self,
        transform,
        world_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute regularization on velocity field.

        For SVFFD, regularization is applied to the velocity field (v),
        not the displacement field (u).
        """
        # Get velocity field (SVFFD-specific)
        if hasattr(transform, 'v'):
            field = transform.v
        else:
            field = transform.data()

        # Apply world coordinate scaling
        if world_scale is not None:
            if field.dim() == 4:
                field_scaled = field * world_scale.squeeze(0)
            else:
                field_scaled = field * world_scale
        else:
            field_scaled = field

        # Compute bending energy
        bending = L.bending_loss(field_scaled)

        # Compute diffusion energy
        diffusion = L.diffusion_loss(field_scaled)

        return bending, diffusion

    def _smooth_velocity_field(self, transform, grid: Grid):
        """
        Apply Gaussian smoothing to velocity field.

        SVFFD-specific: Smooths the velocity field to encourage
        smoother deformations.

        Args:
            transform: SVFFD transform
            grid: Current grid for spacing information
        """
        if not hasattr(transform, 'v') or transform.v is None:
            return

        spacing = grid.spacing()
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy()

        # Convert sigma from mm to voxels for each dimension
        sigma_voxels = [self.velocity_smoothing_sigma / s for s in spacing]

        # Apply separable 3D Gaussian blur to velocity field
        with torch.no_grad():
            v = transform.v
            original_shape = v.shape
            if v.dim() == 4:
                v = v.unsqueeze(0)  # [1, C, D, H, W]

            N, C, D, H, W = v.shape
            smoothed = v.clone()

            # Apply separable Gaussian along each spatial dimension
            for dim, sigma in enumerate(sigma_voxels):
                if sigma > 0.5:  # Only blur if sigma > 0.5 voxels
                    kernel_size = int(4 * sigma) | 1  # Ensure odd
                    kernel_size = max(3, min(kernel_size, 15))

                    # Create 1D Gaussian kernel
                    x = torch.arange(kernel_size, dtype=torch.float32, device=v.device)
                    x = x - kernel_size // 2
                    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                    kernel = kernel / kernel.sum()

                    # Reshape kernel for conv3d based on dimension
                    pad_size = kernel_size // 2
                    if dim == 0:  # Z dimension
                        kernel_3d = kernel.view(1, 1, -1, 1, 1)
                        padding = (0, 0, 0, 0, pad_size, pad_size)
                    elif dim == 1:  # Y dimension
                        kernel_3d = kernel.view(1, 1, 1, -1, 1)
                        padding = (0, 0, pad_size, pad_size, 0, 0)
                    else:  # X dimension
                        kernel_3d = kernel.view(1, 1, 1, 1, -1)
                        padding = (pad_size, pad_size, 0, 0, 0, 0)

                    # Apply to each channel
                    for c in range(C):
                        channel = smoothed[:, c:c+1]
                        padded = F.pad(channel, padding, mode='replicate')
                        smoothed[:, c:c+1] = F.conv3d(padded, kernel_3d)

            # Write back to transform (same shape as original)
            if original_shape != smoothed.shape:
                smoothed = smoothed.squeeze(0)
            transform.v.copy_(smoothed)

    def _compute_laplacian_loss(self, transform) -> torch.Tensor:
        """
        Compute Laplacian regularization (∇²v).

        Encourages smoother velocity fields by penalizing
        second-order spatial derivatives.
        """
        if hasattr(transform, 'v'):
            v = transform.v
        else:
            v = transform.data()

        if v.dim() == 4:
            v = v.unsqueeze(0)

        # Compute Laplacian (second derivatives)
        laplacian = torch.zeros_like(v)

        # For each spatial dimension
        for dim in range(3):
            # Shift indices for this dimension
            idx_dim = dim + 2  # Skip N, C dimensions

            # Second derivative: v[i+1] - 2*v[i] + v[i-1]
            if v.shape[idx_dim] >= 3:
                # Forward difference
                slices_f = [slice(None)] * 5
                slices_f[idx_dim] = slice(2, None)
                slices_c = [slice(None)] * 5
                slices_c[idx_dim] = slice(1, -1)
                slices_b = [slice(None)] * 5
                slices_b[idx_dim] = slice(None, -2)

                laplacian[tuple(slices_c)] += (
                    v[tuple(slices_f)] - 2 * v[tuple(slices_c)] + v[tuple(slices_b)]
                )

        return (laplacian ** 2).mean()

    def _compute_jacobian_loss(self, transform) -> torch.Tensor:
        """
        Compute Jacobian determinant penalty using MIRTK-style soft log-barrier.

        MIRTK uses a log-barrier penalty for topology preservation:
        - penalty = (log(J))^2 for J < 1 (penalizes both compression and folding)
        - Soft continuous gradients (unlike ReLU which has discontinuous gradients)
        - Provides smooth optimization landscape

        The log-barrier naturally penalizes:
        - J < 0: folding (topology violation) - strongly penalized
        - J << 1: severe compression - moderately penalized
        - J ≈ 1: volume-preserving - zero penalty
        - J > 1: expansion - allowed (no penalty)
        """
        try:
            transform.update()

            if hasattr(transform, 'u'):
                u = transform.u
            else:
                return torch.tensor(0.0, device=self.device)

            if u is None:
                return torch.tensor(0.0, device=self.device)

            # Compute Jacobian determinant
            jac_det = L.jacobian_det(u, add_identity=True)

            # MIRTK-style soft log-barrier penalty
            # Clamp to small positive value to avoid log(0) = -inf
            epsilon = 1e-6
            jac_clamped = torch.clamp(jac_det, min=epsilon)

            # Log-barrier: penalize J < 1 (compression and folding)
            # log(J) < 0 when J < 1, so we penalize negative log values
            log_jac = torch.log(jac_clamped)

            # Only penalize compression/folding (log(J) < 0), not expansion
            # penalty = (min(0, log(J)))^2 = (max(0, -log(J)))^2
            neg_log_jac = F.relu(-log_jac)
            penalty = (neg_log_jac ** 2).mean()

            return penalty

        except Exception:
            return torch.tensor(0.0, device=self.device)

    def _process_gradients(self, transform, fg_mask: torch.Tensor):
        """Apply gradient processing (similar to FFD but for velocity field)."""
        if transform.params.grad is None:
            return

        grad = transform.params.grad

        # Gradient normalization
        grad_norm = torch.sqrt((grad ** 2).sum(dim=1, keepdim=True) + 1e-10)
        max_norm = grad_norm.max()
        sigma = 0.5 * max_norm
        transform.params.grad = grad / (grad_norm + sigma)

        # Boundary constraint
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

        # Threshold small gradients
        grad_magnitude = torch.sqrt((transform.params.grad ** 2).sum(dim=1, keepdim=True))
        threshold_mask = (grad_magnitude > 1e-8).float()
        transform.params.grad = transform.params.grad * threshold_mask

        # Support region constraint
        self._apply_support_region_constraint(transform, fg_mask)

    def _apply_support_region_constraint(
        self,
        transform,
        overlap_mask: torch.Tensor,
        threshold: float = None,
    ):
        """Control Point Support Region Masking."""
        if threshold is None:
            threshold = self.support_region_threshold

        with torch.no_grad():
            params = transform.params
            if params.grad is None:
                return

            if params.dim() != 5:
                return

            _, _, cp_D, cp_H, cp_W = params.shape
            stride = transform.stride

            if isinstance(stride, int):
                stride = (stride, stride, stride)
            elif hasattr(stride, '__iter__'):
                stride = tuple(stride)

            kernel_size = tuple(
                min(4 * s, overlap_mask.shape[i + 2])
                for i, s in enumerate(stride)
            )

            if overlap_mask.dim() == 3:
                mask_5d = overlap_mask.unsqueeze(0).unsqueeze(0)
            elif overlap_mask.dim() == 4:
                mask_5d = overlap_mask.unsqueeze(0)
            else:
                mask_5d = overlap_mask

            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            padded_mask = F.pad(
                mask_5d.float(),
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                value=0
            )

            overlap_fractions = F.avg_pool3d(
                padded_mask,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )

            cp_mask = (overlap_fractions > threshold).float()

            if cp_mask.shape[-3:] != (cp_D, cp_H, cp_W):
                cp_mask = F.interpolate(cp_mask, size=(cp_D, cp_H, cp_W), mode='nearest')

            cp_mask_expanded = cp_mask.expand_as(params.grad)
            params.grad = params.grad * cp_mask_expanded

    def _compute_quality_metrics(self, transform, grid: Grid) -> Dict[str, float]:
        """Compute quality metrics for SVFFD transform including Jacobian analysis."""
        try:
            transform.update()

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

            # Compute Jacobian for SVFFD (topology analysis)
            try:
                jac = L.jacobian_det(u, add_identity=True)
                min_jac = float(jac.min())
                folding_pct = float((jac < 0).float().mean() * 100)
            except Exception:
                min_jac = 1.0
                folding_pct = 0.0

            return {
                "max_displacement_mm": max_disp,
                "mean_displacement_mm": mean_disp,
                "min_jacobian": min_jac,
                "folding_percentage": folding_pct,
                "diffeomorphic": folding_pct < 1.0,
            }

        except Exception:
            return {"max_displacement_mm": 0.0, "mean_displacement_mm": 0.0}
