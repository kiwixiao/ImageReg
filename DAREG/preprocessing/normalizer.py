"""
DAREG Intensity Normalization

Intensity preprocessing for registration.
"""

import torch
import numpy as np
from typing import Optional, Tuple

from ..utils.logging_config import get_logger

logger = get_logger("normalizer")


def normalize_intensity(
    tensor: torch.Tensor,
    method: str = "minmax",
    percentile_range: Tuple[float, float] = (1, 99),
    target_range: Tuple[float, float] = (0, 1),
) -> torch.Tensor:
    """
    Normalize image intensity

    Args:
        tensor: Image tensor
        method: Normalization method
            - "minmax": Scale to [0, 1] using min/max
            - "percentile": Scale using percentile values
            - "zscore": Standardize to mean=0, std=1
        percentile_range: Percentiles for clipping (for "percentile" method)
        target_range: Target intensity range

    Returns:
        Normalized tensor
    """
    if method == "minmax":
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            normalized = (tensor - min_val) / (max_val - min_val)
        else:
            normalized = tensor - min_val
        # Scale to target range
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

    elif method == "percentile":
        flat = tensor.flatten()
        low = torch.quantile(flat, percentile_range[0] / 100)
        high = torch.quantile(flat, percentile_range[1] / 100)
        tensor_clipped = torch.clamp(tensor, low, high)
        if high - low > 0:
            normalized = (tensor_clipped - low) / (high - low)
        else:
            normalized = tensor_clipped - low
        # Scale to target range
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

    elif method == "zscore":
        mean = tensor.mean()
        std = tensor.std()
        if std > 0:
            normalized = (tensor - mean) / std
        else:
            normalized = tensor - mean

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.debug(f"Normalized with {method}: [{normalized.min():.3f}, {normalized.max():.3f}]")
    return normalized


def match_histograms(
    source: torch.Tensor,
    reference: torch.Tensor,
    num_bins: int = 256,
) -> torch.Tensor:
    """
    Match source histogram to reference histogram

    Args:
        source: Source image tensor
        reference: Reference image tensor
        num_bins: Number of histogram bins

    Returns:
        Source tensor with matched histogram
    """
    # Convert to numpy for histogram matching
    src_np = source.cpu().numpy().flatten()
    ref_np = reference.cpu().numpy().flatten()

    # Compute histograms and CDFs
    src_values, src_counts = np.unique(src_np, return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]

    ref_values, ref_counts = np.unique(ref_np, return_counts=True)
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Map source values to reference values
    interp_values = np.interp(src_cdf, ref_cdf, ref_values)

    # Create lookup table
    matched = np.interp(src_np, src_values, interp_values)

    # Reshape and convert back to tensor
    matched_tensor = torch.from_numpy(matched.reshape(source.shape)).to(source.device)

    logger.debug("Histogram matching completed")
    return matched_tensor.float()


def compute_foreground_mask(
    tensor: torch.Tensor,
    threshold: float = 0.01,
    method: str = "threshold",
) -> torch.Tensor:
    """
    Compute foreground mask for image

    Args:
        tensor: Image tensor
        threshold: Intensity threshold (fraction of range)
        method: Masking method
            - "threshold": Simple threshold
            - "otsu": Otsu's method

    Returns:
        Boolean mask tensor
    """
    if method == "threshold":
        # Compute absolute threshold from relative
        min_val = tensor.min()
        max_val = tensor.max()
        abs_threshold = min_val + (max_val - min_val) * threshold
        mask = tensor > abs_threshold

    elif method == "otsu":
        # Otsu's method
        flat = tensor.cpu().numpy().flatten()

        # Compute histogram
        hist, bin_edges = np.histogram(flat, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Otsu's algorithm
        total_weight = flat.size
        current_max = 0
        threshold_value = 0
        sum_total = np.dot(bin_centers, hist)
        sum_bg = 0
        weight_bg = 0

        for i, (center, count) in enumerate(zip(bin_centers, hist)):
            weight_bg += count
            if weight_bg == 0:
                continue

            weight_fg = total_weight - weight_bg
            if weight_fg == 0:
                break

            sum_bg += center * count
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if variance_between > current_max:
                current_max = variance_between
                threshold_value = center

        mask = tensor > threshold_value

    else:
        raise ValueError(f"Unknown masking method: {method}")

    fg_fraction = mask.float().mean().item() * 100
    logger.debug(f"Foreground mask: {fg_fraction:.1f}% of voxels")

    return mask
