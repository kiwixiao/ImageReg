"""
DAREG Image Saver

Save medical images with CORRECT header preservation.
Uses nibabel to maintain affine matrix exactly like ITK-SNAP.

GOLDEN RULE: Never lose the original affine matrix.
"""

from pathlib import Path
from typing import Union, Optional, Any, Tuple
import numpy as np
import torch

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

import SimpleITK as sitk

from ..utils.logging_config import get_logger

logger = get_logger("saver")


def save_nifti(
    data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    affine: np.ndarray,
    header: Optional[nib.Nifti1Header] = None,
    description: str = "",
) -> Path:
    """
    Save image data to NIfTI with EXACT affine preservation.

    This is the PRIMARY save function - use this whenever possible.

    Args:
        data: Image data in NIfTI order [X, Y, Z] or [X, Y, Z, T]
        output_path: Output file path
        affine: 4x4 affine matrix (MUST be from original image)
        header: Optional NIfTI header (preserves additional metadata)
        description: Description for logging

    Returns:
        Path to saved file
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for correct NIfTI saving")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Ensure float32 for medical images
    data = data.astype(np.float32)

    # Create NIfTI image with exact affine
    if header is not None:
        # Update header for data shape
        header = header.copy()
        header.set_data_shape(data.shape)
        nii = nib.Nifti1Image(data, affine, header)
    else:
        nii = nib.Nifti1Image(data, affine)

    # Save
    nib.save(nii, str(output_path))

    if description:
        logger.info(f"Saved {description}: {output_path.name}")
    else:
        logger.info(f"Saved: {output_path.name}")

    return output_path


def save_image_like_reference(
    data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    reference_path: Union[str, Path],
    description: str = "",
) -> Path:
    """
    Save image data with SAME affine/header as a reference NIfTI file.

    Use this when you have transformed data that should match a reference's
    coordinate system (e.g., warped source should match target's space).

    Args:
        data: Image data (will be saved in reference's coordinate system)
        output_path: Output file path
        reference_path: Path to reference NIfTI (affine/header copied from here)
        description: Description for logging

    Returns:
        Path to saved file
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for correct NIfTI saving")

    # Load reference to get affine and header
    ref_nii = nib.load(str(reference_path))
    affine = ref_nii.affine
    header = ref_nii.header.copy()

    return save_nifti(data, output_path, affine, header, description)


def save_from_deepali_image(
    image,  # deepali.data.Image
    output_path: Union[str, Path],
    reference_path: Optional[Union[str, Path]] = None,
    description: str = "",
) -> Path:
    """
    Save a deepali Image to NIfTI file correctly.

    CRITICAL: deepali stores tensors in [C, D, H, W] order (torch convention).
    NIfTI expects [X, Y, Z] order. We must transpose correctly.

    Args:
        image: deepali Image object
        output_path: Output file path
        reference_path: Optional reference NIfTI for affine (recommended)
        description: Description for logging

    Returns:
        Path to saved file
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for correct NIfTI saving")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get tensor data from deepali Image
    tensor = image.tensor()  # [C, D, H, W] or [D, H, W]

    # Remove channel dimension if present
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # [D, H, W]

    # Convert to numpy
    data = tensor.detach().cpu().numpy()  # [D, H, W] = [Z, Y, X] in torch

    # Transpose from torch [Z, Y, X] to NIfTI [X, Y, Z]
    data_nifti = np.transpose(data, (2, 1, 0))  # [X, Y, Z]

    # Get affine from reference or construct from grid
    if reference_path is not None:
        ref_nii = nib.load(str(reference_path))
        affine = ref_nii.affine
        header = ref_nii.header.copy()
        header.set_data_shape(data_nifti.shape)
    else:
        # Construct affine from deepali grid (less reliable)
        grid = image.grid()
        affine = _grid_to_affine(grid)
        header = None
        logger.warning("No reference provided - affine constructed from grid (may be inaccurate)")

    return save_nifti(data_nifti, output_path, affine, header, description)


def _grid_to_affine(grid) -> np.ndarray:
    """
    Construct affine matrix from deepali Grid.

    WARNING: This is a fallback. Always prefer using reference affine.

    Args:
        grid: deepali Grid object

    Returns:
        4x4 affine matrix
    """
    # Get grid properties
    spacing = grid.spacing()  # [dD, dH, dW] in deepali = [dZ, dY, dX]
    origin = grid.origin()    # [oX, oY, oZ]

    if isinstance(spacing, torch.Tensor):
        spacing = spacing.cpu().numpy()
    if isinstance(origin, torch.Tensor):
        origin = origin.cpu().numpy()

    # Convert spacing from [dZ, dY, dX] to [dX, dY, dZ]
    spacing_xyz = np.array([spacing[2], spacing[1], spacing[0]])

    # Get direction if available
    try:
        direction = grid.direction()
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()
        # Reshape to 3x3 if flattened
        if direction.size == 9:
            direction = direction.reshape(3, 3)
    except:
        direction = np.eye(3)

    # Construct affine: rotation*diag(spacing) with origin
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing_xyz)
    affine[:3, 3] = origin

    return affine


def save_segmentation(
    data: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    affine: np.ndarray,
    header: Optional[nib.Nifti1Header] = None,
    description: str = "",
) -> Path:
    """
    Save segmentation mask to NIfTI.

    Segmentations are saved as integer type to preserve discrete labels.

    Args:
        data: Segmentation data in NIfTI order [X, Y, Z]
        output_path: Output file path
        affine: 4x4 affine matrix
        header: Optional NIfTI header
        description: Description for logging

    Returns:
        Path to saved file
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for correct NIfTI saving")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Convert to integer type for segmentation
    data = data.astype(np.int16)

    # Create NIfTI image
    if header is not None:
        header = header.copy()
        header.set_data_shape(data.shape)
        header.set_data_dtype(np.int16)
        nii = nib.Nifti1Image(data, affine, header)
    else:
        nii = nib.Nifti1Image(data, affine)

    # Save
    nib.save(nii, str(output_path))

    if description:
        logger.info(f"Saved segmentation {description}: {output_path.name}")
    else:
        logger.info(f"Saved segmentation: {output_path.name}")

    return output_path


def save_transform(
    transform: Any,
    output_path: Union[str, Path],
    metadata: Optional[dict] = None,
    description: str = "",
) -> Path:
    """
    Save transform to PyTorch file.

    Args:
        transform: Transform object (must have state_dict())
        output_path: Output file path (.pth)
        metadata: Optional metadata dictionary
        description: Description for logging

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": transform.state_dict(),
        "type": type(transform).__name__,
    }

    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, output_path)

    if description:
        logger.info(f"Saved {description}: {output_path.name}")
    else:
        logger.info(f"Saved transform: {output_path.name}")

    return output_path


def create_output_directories(base_path: Union[str, Path]) -> dict:
    """
    Create standard output directory structure.

    Args:
        base_path: Base output directory

    Returns:
        Dictionary with paths to subdirectories
    """
    base_path = Path(base_path)

    dirs = {
        "base": base_path,
        "transforms": base_path / "transforms",
        "frames": base_path / "frames",
        "segmentations": base_path / "segmentations",
        "alignment": base_path / "alignment",
        "pairwise": base_path / "pairwise",
        "longitudinal": base_path / "longitudinal",
        "visualizations": base_path / "visualizations",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directories at: {base_path}")

    return dirs


# =============================================================================
# DEPRECATED - Keep for backward compatibility but warn users
# =============================================================================

def save_image(
    tensor: torch.Tensor,
    output_path: Union[str, Path],
    reference_sitk: Optional[sitk.Image] = None,
    spacing: Optional[tuple] = None,
    origin: Optional[tuple] = None,
    direction: Optional[tuple] = None,
    description: str = "",
) -> Path:
    """
    DEPRECATED: Use save_nifti() or save_from_deepali_image() instead.

    This function uses SimpleITK which can cause orientation issues.
    Kept for backward compatibility only.
    """
    logger.warning("save_image() is deprecated - use save_nifti() with nibabel for correct orientation")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.array(tensor)

    # Handle dimensions
    if array.ndim == 4 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim == 5:
        array = array.squeeze(0).squeeze(0)

    # Create SimpleITK image
    sitk_image = sitk.GetImageFromArray(array)

    if reference_sitk is not None:
        sitk_image.SetSpacing(reference_sitk.GetSpacing())
        sitk_image.SetOrigin(reference_sitk.GetOrigin())
        sitk_image.SetDirection(reference_sitk.GetDirection())
    else:
        if spacing:
            sitk_image.SetSpacing(spacing[::-1])
        if origin:
            sitk_image.SetOrigin(origin[::-1])
        if direction:
            sitk_image.SetDirection(direction)

    sitk.WriteImage(sitk_image, str(output_path))

    if description:
        logger.info(f"Saved {description}: {output_path.name}")

    return output_path
