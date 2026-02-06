"""
DAREG Image Loader

Load medical images using deepali's built-in Image.read() method.
This properly handles all coordinate transformations, axis ordering,
and world-space operations.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch

import SimpleITK as sitk
from deepali.data import Image
from deepali.core import Grid

from .image_pair import ImagePair, ImageMetadata
from ..utils.logging_config import get_logger

logger = get_logger("loader")


def load_image(
    path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
    normalize_intensity: bool = False,
) -> Tuple[Image, sitk.Image, ImageMetadata]:
    """
    Load a medical image from NIfTI file using deepali's built-in loader

    Uses Image.read() which properly handles:
    - RAS to LPS coordinate conversion
    - Axis order reversal for tensor compatibility
    - Direction matrix handling
    - All coordinate system transformations

    Args:
        path: Path to NIfTI file (.nii or .nii.gz)
        device: Target device for tensor
        normalize_intensity: Whether to normalize intensity to [0, 1]

    Returns:
        Tuple of (deepali Image, SimpleITK Image, ImageMetadata)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.info(f"Loading image: {path.name}")

    # Use deepali's built-in Image.read() - handles ALL coordinate transforms correctly
    image = Image.read(str(path), device=device)

    # Optional intensity normalization using deepali's method
    if normalize_intensity:
        image = image.normalize(mode="unit")

    # Also load with SimpleITK for metadata extraction (header preservation for saving)
    sitk_image = sitk.ReadImage(str(path))

    # Extract metadata from SimpleITK for backwards compatibility
    metadata = ImageMetadata(
        path=str(path),
        shape=tuple(image.grid().shape),  # Use deepali grid shape (D, H, W)
        spacing=tuple(s.item() if isinstance(s, torch.Tensor) else s
                     for s in image.spacing()),
        origin=tuple(image.grid().origin().tolist()),
        direction=tuple(image.grid().direction().flatten().tolist()),
        dtype=str(sitk_image.GetPixelIDTypeAsString()),
    )

    logger.debug(f"  Shape: {metadata.shape}")
    logger.debug(f"  Spacing: {[f'{s:.3f}mm' for s in metadata.spacing]}")

    return image, sitk_image, metadata


def load_image_pair(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    segmentation_path: Optional[Union[str, Path]] = None,
    device: torch.device = torch.device("cpu"),
) -> ImagePair:
    """
    Load source and target images as an ImagePair

    Uses deepali's built-in methods for proper coordinate handling:
    - Image.read() for loading
    - Image.sample() for resampling

    Args:
        source_path: Path to source (moving) image
        target_path: Path to target (fixed) image
        segmentation_path: Optional path to segmentation
        device: Target device

    Returns:
        ImagePair with loaded images
    """
    logger.info("Loading image pair...")

    # Load source image using deepali
    source_image, source_sitk, source_meta = load_image(source_path, device)

    # Load target image using deepali
    target_image, target_sitk, target_meta = load_image(target_path, device)

    # Load optional segmentation
    segmentation = None
    segmentation_sitk = None
    if segmentation_path:
        segmentation, segmentation_sitk, _ = load_image(
            segmentation_path, device, normalize_intensity=False
        )

    # Create common grid (use target grid as reference - medical imaging convention)
    common_grid = target_image.grid()

    # Resample source to common grid using deepali's built-in sample() method
    # This properly handles all coordinate transformations
    source_normalized = source_image.sample(common_grid)
    target_normalized = target_image  # Target is already on its own grid

    logger.info(f"Image pair loaded:")
    logger.info(f"  Source: {source_meta.shape} @ {[f'{s:.2f}mm' for s in source_meta.spacing]}")
    logger.info(f"  Target: {target_meta.shape} @ {[f'{s:.2f}mm' for s in target_meta.spacing]}")
    logger.info(f"  Common grid: {tuple(common_grid.shape)}")

    return ImagePair(
        source_original=source_image,
        target_original=target_image,
        source_normalized=source_normalized,
        target_normalized=target_normalized,
        common_grid=common_grid,
        source_metadata=source_meta,
        target_metadata=target_meta,
        segmentation=segmentation,
        source_sitk=source_sitk,
        target_sitk=target_sitk,
        segmentation_sitk=segmentation_sitk,
    )


def load_4d_sequence(
    path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
) -> Tuple[list, sitk.Image, ImageMetadata]:
    """
    Load a 4D image sequence (e.g., dynamic MRI)

    Args:
        path: Path to 4D NIfTI file
        device: Target device

    Returns:
        Tuple of (list of 3D Images, original SimpleITK Image, metadata)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.info(f"Loading 4D sequence: {path.name}")

    # Load with SimpleITK first to get dimensions
    sitk_image = sitk.ReadImage(str(path))
    size = sitk_image.GetSize()

    if len(size) != 4:
        raise ValueError(f"Expected 4D image, got {len(size)}D")

    num_frames = size[3]  # Time dimension is last in SimpleITK
    logger.info(f"  Found {num_frames} frames")

    # Extract each frame as a 3D Image using deepali
    frames = []
    for i in range(num_frames):
        # Extract 3D slice using SimpleITK
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, i])
        frame_sitk = extractor.Execute(sitk_image)

        # Convert to deepali Image using from_sitk
        frame = Image.from_sitk(frame_sitk, device=device)
        frames.append(frame)

    # Get metadata from first frame
    metadata = ImageMetadata(
        path=str(path),
        shape=tuple(frames[0].grid().shape),
        spacing=tuple(s.item() if isinstance(s, torch.Tensor) else s
                     for s in frames[0].spacing()),
        origin=tuple(frames[0].grid().origin().tolist()),
        direction=tuple(frames[0].grid().direction().flatten().tolist()),
        dtype=str(sitk_image.GetPixelIDTypeAsString()),
    )

    logger.info(f"  Frame shape: {metadata.shape}")
    logger.info(f"  Frame spacing: {[f'{s:.3f}mm' for s in metadata.spacing]}")

    return frames, sitk_image, metadata


def save_image(
    image: Image,
    path: Union[str, Path],
    reference_sitk: Optional[sitk.Image] = None,
) -> None:
    """
    Save deepali Image to file

    Uses deepali's built-in write() method for proper coordinate handling.
    Optionally uses reference SimpleITK image for header preservation.

    Args:
        image: deepali Image to save
        path: Output path
        reference_sitk: Optional reference SimpleITK image for header
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving image: {path.name}")

    # Use deepali's built-in write method
    image.write(str(path))

    logger.debug(f"  Saved: {path}")
