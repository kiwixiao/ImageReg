"""
DAREG Affine Registration

12-DOF affine registration (translation, rotation, scaling, shearing).
Implements MIRTK-equivalent multi-resolution optimization with NMI.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, List

from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial

from .base import BaseRegistration, RegistrationResult
from ..preprocessing.pyramid import create_pyramid
from ..utils.logging_config import get_logger

logger = get_logger("affine")


class AffineRegistration(BaseRegistration):
    """
    Affine Registration (12 DOF)

    Implements MIRTK-equivalent affine registration with:
    - Multi-resolution pyramid optimization
    - NMI similarity with foreground overlap masking
    - Initialization from rigid transform
    - Early stopping convergence
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        pyramid_levels: int = 4,
        iterations_per_level: List[int] = None,
        learning_rates: List[float] = None,
        num_bins: int = 64,
        foreground_threshold: float = 0.01,
        convergence_delta: float = 1e-6,
        convergence_patience: int = 20,
    ):
        """
        Initialize affine registration

        Args:
            device: Computation device
            pyramid_levels: Number of multi-resolution levels
            iterations_per_level: Max iterations per level
            learning_rates: Learning rate per level (lower than rigid for stability)
            num_bins: NMI histogram bins
            foreground_threshold: Threshold for FG_Overlap mask
            convergence_delta: Minimum loss change for convergence
            convergence_patience: Iterations without improvement before stopping
        """
        super().__init__(device)

        self.pyramid_levels = pyramid_levels
        self.iterations_per_level = iterations_per_level or [100] * pyramid_levels
        self.learning_rates = learning_rates or [0.005] * pyramid_levels
        self.num_bins = num_bins
        self.foreground_threshold = foreground_threshold
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience

    def register(
        self,
        source: Image,
        target: Image,
        initial_transform: Optional[spatial.AffineTransform] = None,
    ) -> RegistrationResult:
        """
        Perform affine registration

        Args:
            source: Source (moving) image
            target: Target (fixed) image
            initial_transform: Optional initial affine transform (or rigid to initialize from)

        Returns:
            RegistrationResult with transform and warped source
        """
        logger.info("=" * 60)
        logger.info("AFFINE REGISTRATION (12 DOF)")
        logger.info("=" * 60)

        # Create image pyramids
        source_pyramid = create_pyramid(source, self.pyramid_levels)
        target_pyramid = create_pyramid(target, self.pyramid_levels)

        logger.info(f"Created {self.pyramid_levels}-level pyramid")

        # Initialize transform
        transform = spatial.AffineTransform(target.grid())
        transform = transform.to(self.device).train()

        # Initialize from rigid if provided
        if initial_transform is not None:
            if isinstance(initial_transform, spatial.RigidTransform):
                # Initialize affine from rigid
                self._initialize_from_rigid(transform, initial_transform)
                logger.info("Initialized from rigid transform")
            elif isinstance(initial_transform, spatial.AffineTransform):
                transform.load_state_dict(initial_transform.state_dict())
                logger.info("Initialized from previous affine transform")

        # Track convergence
        loss_history = {}
        best_overall_loss = float('inf')

        # Multi-resolution optimization (coarse to fine)
        for level_idx, level_data in enumerate(source_pyramid.coarse_to_fine()):
            level = level_data.level
            logger.info(f"\nLevel {level_idx + 1}/{self.pyramid_levels} (pyramid {level})")

            # Get images at this level
            source_level = source_pyramid[level].image
            target_level = target_pyramid[level].image

            logger.info(f"  Shape: {tuple(source_level.grid().shape)}")

            # Update transform grid for this level
            transform.grid_(target_level.grid())

            # Get level parameters
            max_iters = self.iterations_per_level[level_idx]
            lr = self.learning_rates[level_idx]

            # Setup optimizer
            optimizer = optim.Adam(transform.parameters(), lr=lr)

            # Create transformer for warping
            transformer = spatial.ImageTransformer(transform)

            # Prepare tensors
            source_tensor = self._ensure_5d(source_level.tensor()).to(self.device)
            target_tensor = self._ensure_5d(target_level.tensor()).to(self.device)

            # Compute static foreground overlap mask (MIRTK FG_Overlap)
            fg_mask = self._compute_foreground_overlap_mask(
                source_tensor, target_tensor, self.foreground_threshold
            )
            fg_coverage = fg_mask.sum() / fg_mask.numel() * 100
            logger.info(f"  Foreground overlap: {fg_coverage:.1f}%")

            # Optimization loop
            level_losses = []
            best_level_loss = float('inf')
            no_improve_count = 0

            for iteration in range(max_iters):
                optimizer.zero_grad()

                # Warp source
                warped_source = transformer(source_tensor)

                # Compute NMI loss with foreground mask
                loss = self._compute_nmi_loss(
                    warped_source, target_tensor,
                    num_bins=self.num_bins,
                    mask=fg_mask
                )

                # Backward and step
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                level_losses.append(loss_val)

                # Track best
                if loss_val < best_level_loss - self.convergence_delta:
                    best_level_loss = loss_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Log progress
                if (iteration + 1) % 20 == 0 or iteration == 0:
                    grad_norm = sum(p.grad.norm().item() for p in transform.parameters() if p.grad is not None)
                    logger.info(f"  Iter {iteration+1:3d}: loss={loss_val:.6f}, grad={grad_norm:.4f}")

                # Early stopping
                if no_improve_count >= self.convergence_patience:
                    logger.info(f"  Converged at iteration {iteration + 1}")
                    break

            loss_history[f"level_{level}"] = level_losses
            logger.info(f"  Level complete: best_loss={best_level_loss:.6f}")

            if best_level_loss < best_overall_loss:
                best_overall_loss = best_level_loss

        # Final warping at original resolution
        transform.eval()
        transform.grid_(target.grid())
        final_transformer = spatial.ImageTransformer(transform)

        with torch.no_grad():
            source_tensor = self._ensure_5d(source.tensor()).to(self.device)
            warped_result = final_transformer(source_tensor)

        # Create result image
        warped_image = Image(
            data=warped_result.squeeze(0),
            grid=target.grid()
        )

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(transform)

        logger.info("\n" + "=" * 60)
        logger.info("AFFINE REGISTRATION COMPLETE")
        logger.info(f"  Final loss: {best_overall_loss:.6f}")
        logger.info("=" * 60)

        return RegistrationResult(
            transform=transform,
            warped_source=warped_image,
            final_loss=best_overall_loss,
            loss_history=loss_history,
            quality_metrics=quality_metrics,
            metadata={"method": "affine", "levels": self.pyramid_levels},
        )

    def _initialize_from_rigid(
        self,
        affine: spatial.AffineTransform,
        rigid: spatial.RigidTransform,
    ):
        """
        Initialize affine transform from rigid transform

        The affine matrix is initialized to match the rigid transformation,
        with scaling and shearing parameters set to identity.
        """
        with torch.no_grad():
            # Copy rotation parameters from rigid to affine
            # Both have EulerRotation components
            if hasattr(rigid, 'rotation') and hasattr(affine, 'rotation'):
                rigid_rotation = rigid.rotation.data()
                if rigid_rotation is not None:
                    affine.rotation.data_(rigid_rotation.clone())

            # Copy translation parameters from rigid to affine
            if hasattr(rigid, 'translation') and hasattr(affine, 'translation'):
                rigid_translation = rigid.translation.data()
                if rigid_translation is not None:
                    affine.translation.data_(rigid_translation.clone())

            # Scaling is already initialized to identity (1.0) by default
            # which is what we want for initializing from rigid

    def _compute_quality_metrics(self, transform: spatial.AffineTransform) -> Dict[str, float]:
        """Compute quality metrics for affine transform"""
        import numpy as np

        try:
            # Get affine tensor representation - tensor() returns homogeneous matrix
            # Shape: (N, D, D+1) where D=3 for 3D, so (1, 3, 4) for single 3D transform
            tensor = transform.tensor().detach().cpu().numpy()

            # Handle batch dimension
            if tensor.ndim == 3:
                tensor = tensor[0]  # Remove batch dim: (1, 3, 4) -> (3, 4)

            # Extract components
            # tensor shape is (D, D+1) = (3, 4) for 3D
            if tensor.shape[0] >= 3 and tensor.shape[1] >= 4:
                rotation_scale = tensor[:3, :3]
                translation = tensor[:3, 3]
            elif tensor.shape[0] >= 3 and tensor.shape[1] >= 3:
                rotation_scale = tensor[:3, :3]
                translation = np.zeros(3)
            else:
                rotation_scale = np.eye(3)
                translation = np.zeros(3)

            # Compute metrics
            # Translation magnitude
            trans_mag = float(np.linalg.norm(translation))

            # Scale (determinant of rotation-scale matrix)
            det = float(np.linalg.det(rotation_scale))

            # Shear estimate (deviation from orthogonal)
            u, s, vh = np.linalg.svd(rotation_scale)
            scale_factors = s
            mean_scale = float(np.mean(scale_factors))

            return {
                "translation_mm": trans_mag,
                "determinant": det,
                "mean_scale": mean_scale,
                "scale_x": float(scale_factors[0]),
                "scale_y": float(scale_factors[1]),
                "scale_z": float(scale_factors[2]) if len(scale_factors) > 2 else 1.0,
            }

        except Exception as e:
            logger.warning(f"Could not compute affine quality metrics: {e}")
            return {
                "translation_mm": 0.0,
                "determinant": 1.0,
                "mean_scale": 1.0,
            }
