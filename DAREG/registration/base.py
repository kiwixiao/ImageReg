"""
DAREG Base Registration

Abstract base class for all registration methods.
Defines common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch
from deepali.data import Image
from deepali.core import Grid

from ..utils.logging_config import get_logger

logger = get_logger("registration")


@dataclass
class RegistrationResult:
    """
    Container for registration results

    Attributes:
        transform: The learned transformation
        warped_source: Source image warped to target space
        final_loss: Final loss value
        loss_history: Loss values per iteration (per level)
        quality_metrics: Dictionary of quality metrics
        metadata: Additional metadata
    """
    transform: Any  # Specific transform type
    warped_source: Image
    final_loss: float
    loss_history: Dict[str, List[float]] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRegistration(ABC):
    """
    Abstract base class for registration methods

    All registration methods (Rigid, Affine, FFD) inherit from this class
    and implement the common interface.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        config: Optional[Dict] = None,
    ):
        """
        Initialize registration

        Args:
            device: Computation device
            config: Registration configuration dictionary
        """
        self.device = device
        self.config = config or {}
        self._transform = None

    @property
    def name(self) -> str:
        """Registration method name"""
        return self.__class__.__name__

    @abstractmethod
    def register(
        self,
        source: Image,
        target: Image,
        initial_transform: Optional[Any] = None,
    ) -> RegistrationResult:
        """
        Perform registration

        Args:
            source: Source (moving) image
            target: Target (fixed) image
            initial_transform: Optional initial transform

        Returns:
            RegistrationResult with transform and warped source
        """
        pass

    def _compute_nmi_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        num_bins: int = 64,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NMI similarity loss (MIRTK-equivalent)

        MIRTK formula: NMI = 2 - (H(X) + H(Y)) / H(X,Y)

        Args:
            source: Warped source tensor [N, C, D, H, W] or [C, D, H, W]
            target: Target tensor
            num_bins: Number of histogram bins
            mask: Optional foreground mask

        Returns:
            NMI loss (lower is better - negated for minimization)
        """
        from deepali.losses import functional as L

        # Ensure 5D tensors
        if source.dim() == 3:
            source = source.unsqueeze(0).unsqueeze(0)
        elif source.dim() == 4:
            source = source.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 4:
            target = target.unsqueeze(0)

        # Compute NMI using deepali
        nmi = L.nmi_loss(source, target, mask=mask, num_bins=num_bins)
        return nmi

    def _compute_foreground_overlap_mask(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute foreground overlap mask (MIRTK FG_Overlap equivalent)

        This is the complete MIRTK-style foreground masking that:
        1. Normalizes intensities to [0,1] so threshold works consistently
        2. Detects foreground where intensity > threshold
        3. Filters out zero-gradient regions (flat areas with no useful info)

        MIRTK explicitly skips zero-gradient voxels:
            if (*gx == .0 && *gy == .0 && *gz == .0) continue;  // line 195

        Args:
            source: Source tensor [N, C, D, H, W] or [D, H, W]
            target: Target tensor
            threshold: Foreground intensity threshold (after normalization)

        Returns:
            Float mask tensor [N, 1, D, H, W]
        """
        # Ensure proper dimensions
        if source.dim() == 3:
            source = source.unsqueeze(0).unsqueeze(0)
        elif source.dim() == 4:
            source = source.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 4:
            target = target.unsqueeze(0)

        # Normalize to [0, 1] range so threshold works consistently
        source_norm = (source - source.min()) / (source.max() - source.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

        # Foreground detection
        source_fg = source_norm > threshold
        target_fg = target_norm > threshold

        # MIRTK line 195: Skip zero-gradient voxels
        # Compute gradient magnitude for both images
        source_grad = self._compute_gradient_magnitude(source_norm)
        target_grad = self._compute_gradient_magnitude(target_norm)
        grad_threshold = 1e-5
        source_grad_mask = source_grad > grad_threshold
        target_grad_mask = target_grad > grad_threshold

        # MIRTK FG_Overlap: INTERSECTION of foregrounds AND gradient masks
        overlap_mask = source_fg & target_fg
        final_mask = overlap_mask.float() * source_grad_mask.float() * target_grad_mask.float()

        return final_mask

    def _compute_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude using central differences

        This is used for MIRTK line 195 equivalent:
        Skip voxels where gradient is zero (flat regions with no useful info)

        Args:
            image: Image tensor [N, C, D, H, W]

        Returns:
            Gradient magnitude tensor [N, C, D, H, W]
        """
        if image.dim() == 5:  # [N, C, D, H, W]
            # Central differences in each dimension
            grad_d = torch.abs(image[:, :, 2:, :, :] - image[:, :, :-2, :, :])
            grad_h = torch.abs(image[:, :, :, 2:, :] - image[:, :, :, :-2, :])
            grad_w = torch.abs(image[:, :, :, :, 2:] - image[:, :, :, :, :-2])

            # Pad back to original size
            grad_d = torch.nn.functional.pad(grad_d, (0, 0, 0, 0, 1, 1))
            grad_h = torch.nn.functional.pad(grad_h, (0, 0, 1, 1, 0, 0))
            grad_w = torch.nn.functional.pad(grad_w, (1, 1, 0, 0, 0, 0))

            # Compute magnitude
            grad_mag = torch.sqrt(grad_d**2 + grad_h**2 + grad_w**2 + 1e-8)
            return grad_mag
        else:
            # For lower dimensional tensors, return ones (no filtering)
            return torch.ones_like(image)

    def _ensure_5d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 5D [N, C, D, H, W]"""
        if tensor.dim() == 3:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 4:
            return tensor.unsqueeze(0)
        return tensor

    def _log_iteration(
        self,
        iteration: int,
        total: int,
        loss: float,
        grad_norm: Optional[float] = None,
        extra: str = "",
    ):
        """Log iteration progress"""
        msg = f"Iter {iteration:3d}/{total}: loss={loss:.6f}"
        if grad_norm is not None:
            msg += f", grad={grad_norm:.4f}"
        if extra:
            msg += f", {extra}"
        logger.debug(msg)
