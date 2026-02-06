"""
DAREG Rigid Registration

6-DOF rigid registration (3 translation + 3 rotation).
Implements MIRTK-equivalent multi-resolution optimization with NMI.
"""

import torch
from typing import Optional, Dict, List
from dataclasses import dataclass

from deepali.data import Image
from deepali.core import Grid
import deepali.spatial as spatial

from .base import BaseRegistration, RegistrationResult
from .optimizers import ConjugateGradientOptimizer
from ..preprocessing.pyramid import create_pyramid
from ..utils.logging_config import get_logger
from ..utils.progress_tracker import ProgressTracker

logger = get_logger("rigid")


class RigidRegistration(BaseRegistration):
    """
    Rigid Registration (6 DOF)

    Implements MIRTK-equivalent rigid registration with:
    - Multi-resolution pyramid optimization
    - NMI similarity with foreground overlap masking
    - Early stopping convergence
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        pyramid_levels: int = 4,
        iterations_per_level: List[int] = None,
        learning_rates: List[float] = None,
        num_bins: int = 256,  # MIRTK default
        foreground_threshold: float = 0.01,
        convergence_delta: float = 1e-6,
        convergence_patience: int = 20,
        roi_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ):
        """
        Initialize rigid registration

        Args:
            device: Computation device
            pyramid_levels: Number of multi-resolution levels
            iterations_per_level: Max iterations per level
            learning_rates: Learning rate per level
            num_bins: NMI histogram bins
            foreground_threshold: Threshold for FG_Overlap mask
            convergence_delta: Minimum loss change for convergence
            convergence_patience: Iterations without improvement before stopping
            roi_mask: Optional binary ROI mask to restrict registration region.
                      If provided, similarity is computed only within this mask.
            num_samples: Optional fixed number of voxels to sample for NMI speedup
            sample_ratio: Optional ratio of voxels to sample for NMI speedup
        """
        super().__init__(device)

        self.pyramid_levels = pyramid_levels
        self.iterations_per_level = iterations_per_level or [100] * pyramid_levels
        self.learning_rates = learning_rates or [0.01] * pyramid_levels
        self.num_bins = num_bins
        self.foreground_threshold = foreground_threshold
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience
        self.roi_mask = roi_mask
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio

    def register(
        self,
        source: Image,
        target: Image,
        initial_transform: Optional[spatial.RigidTransform] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> RegistrationResult:
        """
        Perform rigid registration

        Args:
            source: Source (moving) image
            target: Target (fixed) image
            initial_transform: Optional initial rigid transform
            progress_tracker: Optional progress tracker for monitoring

        Returns:
            RegistrationResult with transform and warped source
        """
        logger.info("=" * 60)
        logger.info("RIGID REGISTRATION (6 DOF)")
        logger.info("=" * 60)

        # Create image pyramids
        source_pyramid = create_pyramid(source, self.pyramid_levels)
        target_pyramid = create_pyramid(target, self.pyramid_levels)

        logger.info(f"Created {self.pyramid_levels}-level pyramid")

        # Initialize transform
        transform = spatial.RigidTransform(target.grid())
        transform = transform.to(self.device).train()

        if initial_transform is not None:
            transform.load_state_dict(initial_transform.state_dict())
            logger.info("Initialized from previous transform")

        # Track convergence
        loss_history = {}
        best_overall_loss = float('inf')

        # Multi-resolution optimization (coarse to fine)
        for level_idx, level_data in enumerate(source_pyramid.coarse_to_fine()):
            level = level_data.level
            logger.info(f"\nLevel {level_idx + 1}/{self.pyramid_levels} (pyramid {level})")

            # Report progress
            if progress_tracker:
                max_iters_for_level = self.iterations_per_level[level_idx]
                progress_tracker.set_level(level_idx, max_iters_for_level)

            # Get images at this level
            source_level = source_pyramid[level].image
            target_level = target_pyramid[level].image

            logger.info(f"  Shape: {tuple(source_level.grid().shape)}")

            # Update transform grid for this level
            transform.grid_(target_level.grid())

            # Get level parameters
            max_iters = self.iterations_per_level[level_idx]
            lr = self.learning_rates[level_idx]

            # Setup MIRTK-equivalent Conjugate Gradient optimizer
            optimizer = ConjugateGradientOptimizer(
                transform.parameters(),
                lr=lr,
                max_rejected_streak=1,  # MIRTK default
                epsilon=1e-4,
                delta=1e-12,
            )
            logger.info(f"  Optimizer: Conjugate Gradient, lr={lr}")

            # Create transformer for warping
            transformer = spatial.ImageTransformer(transform)

            # Prepare tensors
            source_tensor = self._ensure_5d(source_level.tensor()).to(self.device)
            target_tensor = self._ensure_5d(target_level.tensor()).to(self.device)

            # Compute static foreground overlap mask (MIRTK FG_Overlap)
            # Optionally intersects with user-provided ROI mask
            fg_mask = self._compute_foreground_overlap_mask(
                source_tensor, target_tensor, self.foreground_threshold,
                roi_mask=self.roi_mask
            )
            fg_coverage = fg_mask.sum() / fg_mask.numel() * 100
            logger.info(f"  Foreground overlap: {fg_coverage:.1f}%")

            # Optimization loop with MIRTK-style CG optimizer
            level_losses = []
            best_level_loss = float('inf')

            for iteration in range(max_iters):
                # Define closure for CG optimizer
                def closure():
                    # Warp source
                    warped_source = transformer(source_tensor)

                    # Compute NMI loss with foreground mask
                    loss = self._compute_nmi_loss(
                        warped_source, target_tensor,
                        num_bins=self.num_bins,
                        mask=fg_mask,
                        num_samples=self.num_samples,
                        sample_ratio=self.sample_ratio
                    )

                    # Backward
                    loss.backward()
                    return loss

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
                        nmi_loss=loss_val,
                        reg_loss=0.0
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
                    logger.info(f"  Iter {iteration+1:3d}: loss={loss_val:.6f}")

                # MIRTK-style convergence
                if converged:
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
        logger.info("RIGID REGISTRATION COMPLETE")
        logger.info(f"  Final loss: {best_overall_loss:.6f}")
        logger.info("=" * 60)

        return RegistrationResult(
            transform=transform,
            warped_source=warped_image,
            final_loss=best_overall_loss,
            loss_history=loss_history,
            quality_metrics=quality_metrics,
            metadata={"method": "rigid", "levels": self.pyramid_levels},
        )

    def _compute_quality_metrics(self, transform: spatial.RigidTransform) -> Dict[str, float]:
        """Compute quality metrics for rigid transform"""
        import numpy as np

        # deepali RigidTransform uses parameters() generator returning separate tensors
        # for translation and rotation components
        try:
            params = list(transform.parameters())
            if len(params) >= 2:
                # First tensor is translation [1, 3], second is rotation [1, 3]
                translation = params[0].detach().cpu().numpy().flatten()
                rotation = params[1].detach().cpu().numpy().flatten()

                return {
                    "translation_mm": float(np.linalg.norm(translation)),
                    "rotation_deg": float(np.linalg.norm(rotation) * 180 / np.pi),
                }
        except Exception as e:
            logger.warning(f"Could not extract transform parameters: {e}")

        return {"translation_mm": 0.0, "rotation_deg": 0.0}
