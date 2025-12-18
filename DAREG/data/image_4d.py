"""
DAREG 4D Image Handling

Load, extract, and manage 4D (3D + time) medical image sequences.
Replicates MIRTK's extract-image-volume functionality.

Supports:
- Standard 4D NIfTI [X, Y, Z, T]
- 5D NIfTI with singleton dimension [X, Y, Z, 1, T] (common in dynamic imaging)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import torch
import SimpleITK as sitk

# Use nibabel for better handling of non-standard dimensions
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deepali" / "src"))

from deepali.data import Image
from deepali.core import Grid

from ..utils.logging_config import get_logger

logger = get_logger("image_4d")


@dataclass
class Image4D:
    """
    Container for 4D image data (3D spatial + time dimension)

    Attributes:
        data: 4D tensor [T, D, H, W] or [T, C, D, H, W]
        spacing: Voxel spacing (dx, dy, dz) in mm
        origin: Image origin (x, y, z)
        direction: Direction cosine matrix (flattened 9 elements)
        temporal_spacing: Time between frames (ms or arbitrary units)
        num_frames: Number of time frames
        frame_shape: Shape of each 3D frame (D, H, W)
    """
    data: torch.Tensor
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...] = field(default_factory=lambda: (1, 0, 0, 0, 1, 0, 0, 0, 1))
    temporal_spacing: float = 1.0

    @property
    def num_frames(self) -> int:
        """Number of time frames"""
        return self.data.shape[0]

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Shape of each 3D frame (D, H, W) - for tensor access"""
        if self.data.dim() == 4:
            return tuple(self.data.shape[1:])
        else:  # 5D with channel
            return tuple(self.data.shape[2:])

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Grid size for deepali Grid (X, Y, Z) = (W, H, D) order"""
        # deepali Grid expects size in (X, Y, Z) order
        # Grid.shape property then returns it as (..., X) = (D, H, W) for tensor compatibility
        d, h, w = self.frame_shape
        return (w, h, d)  # (X, Y, Z) = (W, H, D)

    @property
    def spacing_xyz(self) -> Tuple[float, float, float]:
        """Spacing in (X, Y, Z) = (dW, dH, dD) order for deepali Grid"""
        # self.spacing is stored as (dZ, dY, dX) = (dD, dH, dW) order
        # deepali Grid expects spacing in same order as size: (X, Y, Z)
        dz, dy, dx = self.spacing
        return (dx, dy, dz)  # (dX, dY, dZ) = (dW, dH, dD)

    def get_frame(self, t: int) -> Image:
        """
        Extract a single 3D frame as deepali Image

        Uses the same coordinate convention as deepali's Image.read() to ensure
        consistency with static images loaded via Image.read().

        Args:
            t: Time index (0-based)

        Returns:
            deepali Image object for frame t
        """
        if t < 0 or t >= self.num_frames:
            raise IndexError(f"Frame index {t} out of range [0, {self.num_frames})")

        # Extract frame data - already in [T, Z, Y, X] torch order
        if self.data.dim() == 4:
            frame_data = self.data[t]  # [Z, Y, X] = [D, H, W]
            frame_data = frame_data.unsqueeze(0)  # [1, D, H, W]
        else:
            frame_data = self.data[t]  # [C, D, H, W]

        # Create Grid using the stored origin/direction (already in LPS convention after loading)
        # Grid size is (X, Y, Z) = (W, H, D) so Grid.shape returns (D, H, W)
        d, h, w = self.frame_shape
        grid = Grid(
            size=(w, h, d),  # (X, Y, Z)
            spacing=self.spacing_xyz,  # (dX, dY, dZ)
            origin=self.origin,
            direction=self.direction,
        )

        return Image(data=frame_data, grid=grid)

    def get_frames(self, start: int = 0, end: Optional[int] = None) -> List[Image]:
        """
        Extract multiple frames as list of deepali Images

        Args:
            start: Starting frame index (inclusive)
            end: Ending frame index (exclusive), None for all remaining

        Returns:
            List of deepali Image objects
        """
        if end is None:
            end = self.num_frames

        return [self.get_frame(t) for t in range(start, end)]

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Image:
        return self.get_frame(idx)


def load_image_4d(
    path: Union[str, Path],
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> Image4D:
    """
    Load 4D NIfTI image and optionally extract subset of frames

    Replicates: mirtk extract-image-volume <input> -t <start> -n <num>

    Handles:
    - Standard 4D NIfTI [X, Y, Z, T]
    - 5D NIfTI with singleton dimension [X, Y, Z, 1, T]

    Args:
        path: Path to 4D NIfTI file
        start_frame: Starting frame index (default 0)
        num_frames: Number of frames to extract (None = all remaining)

    Returns:
        Image4D container with extracted frames
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"4D image not found: {path}")

    logger.info(f"Loading 4D image: {path.name}")

    # Use nibabel for better handling of non-standard dimensions
    if HAS_NIBABEL:
        return _load_image_4d_nibabel(path, start_frame, num_frames)
    else:
        return _load_image_4d_sitk(path, start_frame, num_frames)


def _load_image_4d_nibabel(
    path: Path,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> Image4D:
    """Load 4D image using nibabel (better 5D handling)"""

    nii = nib.load(str(path))
    header = nii.header
    affine = nii.affine
    data = nii.get_fdata()

    original_shape = data.shape
    logger.info(f"Nibabel loaded shape: {original_shape}")

    # Handle different dimensionalities
    if data.ndim == 3:
        # Single 3D volume - treat as 1-frame 4D
        logger.warning("Loaded 3D image, treating as single frame")
        data = data[..., np.newaxis]  # Add time dimension

    elif data.ndim == 4:
        # Standard 4D [X, Y, Z, T] - keep as is
        pass

    elif data.ndim == 5:
        # 5D image - common format: [X, Y, Z, 1, T]
        # Find singleton dimension and squeeze it
        if data.shape[3] == 1:
            # Format: [X, Y, Z, 1, T] -> squeeze dim 3
            logger.info(f"5D image with singleton dim 3: {original_shape} -> squeezing")
            data = data.squeeze(axis=3)  # -> [X, Y, Z, T]
        elif data.shape[4] == 1:
            # Format: [X, Y, Z, T, 1] -> squeeze dim 4
            logger.info(f"5D image with singleton dim 4: {original_shape} -> squeezing")
            data = data.squeeze(axis=4)  # -> [X, Y, Z, T]
        else:
            raise ValueError(f"5D image has no singleton dimension: {original_shape}")
    else:
        raise ValueError(f"Unexpected image dimension: {data.ndim}D (shape: {original_shape})")

    logger.info(f"Final 4D shape after processing: {data.shape}")

    # Now data is [X, Y, Z, T]
    total_frames = data.shape[3]
    spatial_shape = data.shape[:3]  # [X, Y, Z]

    # Get spacing from header
    # NIfTI zooms are in [X, Y, Z] order, but deepali uses [Z, Y, X] (D, H, W) order
    zooms = header.get_zooms()
    spacing_xyz = tuple(float(z) for z in zooms[:3])  # [dx, dy, dz]
    spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # [dz, dy, dx] for deepali
    temporal_spacing = float(zooms[3]) if len(zooms) > 3 else 1.0

    # Get origin from affine
    origin_arr = np.array([float(affine[i, 3]) for i in range(3)])

    # Get direction from affine (rotation part)
    # Normalize columns of rotation matrix
    rotation = affine[:3, :3].copy()
    for i in range(3):
        col_norm = np.linalg.norm(rotation[:, i])
        if col_norm > 0:
            rotation[:, i] /= col_norm

    # CRITICAL: Convert from NIfTI RAS to ITK/deepali LPS convention
    # This matches what deepali's read_nifti_image() does in nifti.py lines 51-53
    # Without this conversion, static images (loaded via Image.read()) and 4D frames
    # will have different coordinate conventions, causing orientation mismatches.
    origin_arr[:2] *= -1
    rotation[:2] *= -1

    origin = tuple(origin_arr)
    direction_3x3 = tuple(rotation.flatten())

    # Calculate frame range
    if num_frames is None:
        num_frames = total_frames - start_frame

    end_frame = start_frame + num_frames
    if end_frame > total_frames:
        logger.warning(f"Requested frames exceed available ({total_frames}), truncating")
        end_frame = total_frames
        num_frames = end_frame - start_frame

    logger.info(f"Extracting frames {start_frame} to {end_frame-1} ({num_frames} frames)")
    logger.info(f"Frame shape: {spatial_shape}, spacing: {spacing}")

    # Extract frames: [X, Y, Z, T] -> select time range
    data = data[..., start_frame:end_frame]  # [X, Y, Z, num_frames]

    # Transpose to [T, Z, Y, X] for torch convention
    # NIfTI: [X, Y, Z, T] -> torch: [T, Z, Y, X]
    data = np.transpose(data, (3, 2, 1, 0))  # [T, Z, Y, X]

    # Convert to torch tensor
    tensor_data = torch.from_numpy(data.astype(np.float32))

    return Image4D(
        data=tensor_data,
        spacing=spacing,
        origin=origin,
        direction=direction_3x3,
        temporal_spacing=temporal_spacing,
    )


def _load_image_4d_sitk(
    path: Path,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> Image4D:
    """Load 4D image using SimpleITK (fallback)"""

    # Load with SimpleITK
    sitk_image = sitk.ReadImage(str(path))

    # Verify 4D
    dimension = sitk_image.GetDimension()
    if dimension != 4:
        raise ValueError(f"Expected 4D image, got {dimension}D. Install nibabel for 5D support.")

    # Get metadata
    size = sitk_image.GetSize()  # (X, Y, Z, T)
    spacing = sitk_image.GetSpacing()[:3]  # Spatial spacing only
    origin = sitk_image.GetOrigin()[:3]

    # Handle direction - SimpleITK returns flat array for 4D
    direction_flat = sitk_image.GetDirection()
    # Extract 3x3 spatial direction from 4x4
    direction_3x3 = (
        direction_flat[0], direction_flat[1], direction_flat[2],
        direction_flat[4], direction_flat[5], direction_flat[6],
        direction_flat[8], direction_flat[9], direction_flat[10],
    )

    total_frames = size[3]

    # Calculate frame range
    if num_frames is None:
        num_frames = total_frames - start_frame

    end_frame = start_frame + num_frames
    if end_frame > total_frames:
        logger.warning(f"Requested frames exceed available ({total_frames}), truncating")
        end_frame = total_frames
        num_frames = end_frame - start_frame

    logger.info(f"Extracting frames {start_frame} to {end_frame-1} ({num_frames} frames)")
    logger.info(f"Frame shape: ({size[2]}, {size[1]}, {size[0]}), spacing: {spacing}")

    # Convert to numpy array [X, Y, Z, T]
    array = sitk.GetArrayFromImage(sitk_image)  # [T, Z, Y, X]

    # Extract frames
    array = array[start_frame:end_frame]  # [num_frames, Z, Y, X]

    # Convert to torch tensor
    data = torch.from_numpy(array.astype(np.float32))

    # Get temporal spacing if available
    temporal_spacing = spacing[3] if len(sitk_image.GetSpacing()) > 3 else 1.0

    return Image4D(
        data=data,
        spacing=tuple(spacing),
        origin=tuple(origin),
        direction=direction_3x3,
        temporal_spacing=temporal_spacing,
    )


def extract_frames_to_files(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "frame_",
    start_frame: int = 0,
    num_frames: Optional[int] = None,
) -> List[Path]:
    """
    Extract 4D frames to individual 3D NIfTI files

    Replicates: mirtk extract-image-volume + renaming

    CRITICAL: Uses nibabel to preserve the original affine matrix exactly.
    This ensures ITK-SNAP displays frames correctly with proper orientation.

    Args:
        input_path: Path to 4D NIfTI
        output_dir: Output directory for extracted frames
        prefix: Filename prefix (default: "frame_")
        start_frame: Starting frame index
        num_frames: Number of frames to extract

    Returns:
        List of paths to extracted frame files
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for correct frame extraction. Install with: pip install nibabel")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original 4D/5D NIfTI with nibabel to preserve affine exactly
    nii = nib.load(str(input_path))
    data = nii.get_fdata()
    affine = nii.affine  # Original affine - MUST preserve this!
    header = nii.header.copy()

    original_shape = data.shape
    logger.info(f"Original shape: {original_shape}, affine preserved")

    # Handle 5D: [X, Y, Z, 1, T] -> squeeze singleton dim
    if data.ndim == 5:
        if data.shape[3] == 1:
            data = data.squeeze(axis=3)  # -> [X, Y, Z, T]
        elif data.shape[4] == 1:
            data = data.squeeze(axis=4)  # -> [X, Y, Z, T]
        else:
            raise ValueError(f"5D image has no singleton dimension: {original_shape}")

    # Handle 3D (single frame)
    if data.ndim == 3:
        data = data[..., np.newaxis]  # -> [X, Y, Z, 1]

    # Now data is [X, Y, Z, T]
    total_frames = data.shape[3]

    # Calculate frame range
    if num_frames is None:
        num_frames = total_frames - start_frame

    end_frame = min(start_frame + num_frames, total_frames)

    logger.info(f"Extracting frames {start_frame} to {end_frame-1} ({end_frame - start_frame} frames)")

    # Update header for 3D output
    header.set_data_shape(data.shape[:3])  # Set to 3D shape
    header['dim'][0] = 3  # Set number of dimensions to 3

    output_paths = []

    for t in range(start_frame, end_frame):
        # Extract frame - keep in original [X, Y, Z] orientation
        frame_data = data[..., t].astype(np.float32)

        # Create new NIfTI with SAME affine as original
        # This is the KEY - preserve the exact affine matrix
        frame_nii = nib.Nifti1Image(frame_data, affine, header)

        # Save with proper naming (3 digits)
        frame_idx = t - start_frame
        output_path = output_dir / f"{prefix}{frame_idx:03d}.nii.gz"
        nib.save(frame_nii, str(output_path))
        output_paths.append(output_path)

        logger.debug(f"Saved frame {t} -> {output_path.name}")

    logger.info(f"Extracted {len(output_paths)} frames to {output_dir} (affine preserved)")

    return output_paths


def create_frame_pairs(
    image_4d: Image4D,
    mode: str = "sequential",
) -> List[Tuple[int, int, Image, Image]]:
    """
    Create pairs of frames for registration

    Args:
        image_4d: 4D image container
        mode: Pairing mode
            - "sequential": (0,1), (1,2), (2,3), ... (for motion tracking)
            - "to_first": (0,1), (0,2), (0,3), ... (direct to reference)

    Returns:
        List of tuples (source_idx, target_idx, source_image, target_image)
    """
    pairs = []

    if mode == "sequential":
        # Consecutive pairs for composition strategy
        for i in range(image_4d.num_frames - 1):
            target = image_4d.get_frame(i)      # Earlier frame is target
            source = image_4d.get_frame(i + 1)  # Later frame is source
            pairs.append((i + 1, i, source, target))

    elif mode == "to_first":
        # All frames to first frame
        target = image_4d.get_frame(0)
        for i in range(1, image_4d.num_frames):
            source = image_4d.get_frame(i)
            pairs.append((i, 0, source, target))

    else:
        raise ValueError(f"Unknown pairing mode: {mode}")

    logger.info(f"Created {len(pairs)} frame pairs (mode: {mode})")

    return pairs


@dataclass
class MotionSequence:
    """
    Container for a complete motion tracking sequence

    Holds the 4D image, extracted frames, alignment info, and registration results.
    """
    image_4d: Image4D
    static_image: Optional[Image] = None
    segmentation: Optional[Image] = None
    alignment_transform: Optional[object] = None  # Rigid+Affine+FFD
    frame_transforms: List[object] = field(default_factory=list)  # Frame0→FrameN
    aligned_segmentations: List[Image] = field(default_factory=list)

    @property
    def reference_frame(self) -> Image:
        """First frame (time 0) - the reference"""
        return self.image_4d.get_frame(0)

    @property
    def num_frames(self) -> int:
        return self.image_4d.num_frames

    def get_segmentation_at_frame(self, t: int) -> Optional[Image]:
        """Get propagated segmentation at frame t"""
        if t == 0:
            return self.segmentation
        if t - 1 < len(self.aligned_segmentations):
            return self.aligned_segmentations[t - 1]
        return None
