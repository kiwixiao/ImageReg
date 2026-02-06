"""
DAREG Quality Metrics

Compute registration quality metrics including NMI, Dice, and Jacobian analysis.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from deepali.data import Image
from deepali.losses import functional as L

from ..utils.logging_config import get_logger

logger = get_logger("metrics")


def compute_quality_metrics(
    warped_source: Image,
    target: Image,
    source_segmentation: Optional[Image] = None,
    target_segmentation: Optional[Image] = None,
    displacement_field: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive registration quality metrics

    Args:
        warped_source: Warped source image
        target: Target image
        source_segmentation: Optional warped source segmentation
        target_segmentation: Optional target segmentation
        displacement_field: Optional displacement field for Jacobian analysis

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Image similarity metrics
    metrics["nmi"] = compute_nmi(warped_source, target)
    metrics["ncc"] = compute_ncc(warped_source, target)
    metrics["mse"] = compute_mse(warped_source, target)

    # Segmentation overlap (if available)
    if source_segmentation is not None and target_segmentation is not None:
        dice_metrics = compute_dice(source_segmentation, target_segmentation)
        metrics.update(dice_metrics)

    # Jacobian analysis (if displacement field available)
    if displacement_field is not None:
        jac_metrics = compute_jacobian_metrics(displacement_field)
        metrics.update(jac_metrics)

    return metrics


def compute_nmi(
    image1: Image,
    image2: Image,
    num_bins: int = 64,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Normalized Mutual Information

    MIRTK formula: NMI = 2 - (H(X) + H(Y)) / H(X,Y)
    Range: [0, 2] where 2 = perfect alignment

    Args:
        image1: First image
        image2: Second image
        num_bins: Histogram bins
        mask: Optional mask

    Returns:
        NMI value (higher = better alignment)
    """
    tensor1 = image1.tensor()
    tensor2 = image2.tensor()

    # Ensure 5D
    if tensor1.dim() == 3:
        tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
    elif tensor1.dim() == 4:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 3:
        tensor2 = tensor2.unsqueeze(0).unsqueeze(0)
    elif tensor2.dim() == 4:
        tensor2 = tensor2.unsqueeze(0)

    # Compute NMI (loss form, negate for metric)
    nmi_loss = L.nmi_loss(tensor1, tensor2, mask=mask, num_bins=num_bins)

    # Convert to metric (higher = better)
    # NMI loss returns 2 - NMI_value, so NMI_value = 2 - loss
    nmi_value = 2.0 - float(nmi_loss)

    return nmi_value


def compute_ncc(
    image1: Image,
    image2: Image,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Normalized Cross Correlation

    Range: [-1, 1] where 1 = perfect correlation

    Args:
        image1: First image
        image2: Second image
        mask: Optional mask

    Returns:
        NCC value
    """
    tensor1 = image1.tensor()
    tensor2 = image2.tensor()

    # Ensure 5D
    if tensor1.dim() == 3:
        tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
    elif tensor1.dim() == 4:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 3:
        tensor2 = tensor2.unsqueeze(0).unsqueeze(0)
    elif tensor2.dim() == 4:
        tensor2 = tensor2.unsqueeze(0)

    # Compute NCC
    ncc_loss = L.ncc_loss(tensor1, tensor2, mask=mask)

    # Convert to metric (higher = better)
    # NCC loss = 1 - NCC, so NCC = 1 - loss
    ncc_value = 1.0 - float(ncc_loss)

    return ncc_value


def compute_mse(
    image1: Image,
    image2: Image,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Mean Squared Error

    Args:
        image1: First image
        image2: Second image
        mask: Optional mask

    Returns:
        MSE value (lower = better)
    """
    tensor1 = image1.tensor()
    tensor2 = image2.tensor()

    if mask is not None:
        diff = (tensor1 - tensor2) ** 2
        mse = float((diff * mask).sum() / mask.sum())
    else:
        mse = float(((tensor1 - tensor2) ** 2).mean())

    return mse


def compute_dice(
    segmentation1: Image,
    segmentation2: Image,
    labels: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute Dice coefficient for segmentation overlap

    Args:
        segmentation1: First segmentation
        segmentation2: Second segmentation
        labels: Optional list of labels (default: all non-zero)

    Returns:
        Dictionary with per-label and mean Dice
    """
    seg1 = segmentation1.tensor()
    seg2 = segmentation2.tensor()

    # Get unique labels
    if labels is None:
        all_labels = torch.unique(torch.cat([seg1.flatten(), seg2.flatten()]))
        labels = [int(l) for l in all_labels if l != 0]

    results = {}
    dice_values = []

    for label in labels:
        mask1 = (seg1 == label).float()
        mask2 = (seg2 == label).float()

        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum()

        if union > 0:
            dice = float(2 * intersection / union)
        else:
            dice = 1.0 if intersection == 0 else 0.0

        results[f"dice_label_{label}"] = dice
        dice_values.append(dice)

    # Mean Dice
    if dice_values:
        results["mean_dice"] = sum(dice_values) / len(dice_values)

    return results


def compute_jacobian_metrics(
    displacement_field: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Jacobian determinant metrics for deformation quality

    Negative Jacobian = folding (topology violation)
    Jacobian near 0 = extreme compression
    Jacobian >> 1 = extreme expansion

    Args:
        displacement_field: Displacement field [3, D, H, W] or [D, H, W, 3]

    Returns:
        Dictionary with Jacobian statistics
    """
    # Ensure proper format
    if displacement_field.dim() == 4 and displacement_field.shape[-1] == 3:
        # [D, H, W, 3] -> [3, D, H, W]
        displacement_field = displacement_field.permute(3, 0, 1, 2)

    if displacement_field.dim() == 4:
        displacement_field = displacement_field.unsqueeze(0)  # Add batch

    try:
        # Compute Jacobian determinant
        jac_det = L.jacobian_det(displacement_field, add_identity=True)

        # Compute statistics
        jac_np = jac_det.detach().cpu().numpy().flatten()

        metrics = {
            "jacobian_min": float(np.min(jac_np)),
            "jacobian_max": float(np.max(jac_np)),
            "jacobian_mean": float(np.mean(jac_np)),
            "jacobian_std": float(np.std(jac_np)),
            "folding_percentage": float(np.mean(jac_np < 0) * 100),
        }

        # Quality assessment
        metrics["diffeomorphic"] = metrics["folding_percentage"] < 1.0

    except Exception as e:
        logger.warning(f"Could not compute Jacobian metrics: {e}")
        metrics = {
            "jacobian_min": 0.0,
            "jacobian_max": 0.0,
            "jacobian_mean": 1.0,
            "jacobian_std": 0.0,
            "folding_percentage": 0.0,
            "diffeomorphic": True,
        }

    return metrics
