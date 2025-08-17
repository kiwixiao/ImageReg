"""I/O utilities for images and meshes."""

from pathlib import Path
from typing import Optional, Union, List
import torch
from torch import Tensor

from deepali.data import Image
from deepali.core import Grid
import SimpleITK as sitk


def read_image(
    path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Image:
    """Read image from file.
    
    Equivalent to reading images in MIRTK pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = Image.read(path, device=device)
    return image


def write_image(
    image: Image,
    path: Union[str, Path]
) -> None:
    """Write image to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.write(path)


def extract_volume_from_4d(
    image_4d: Image,
    time_point: int,
    num_frames: Optional[int] = None
) -> List[Image]:
    """Extract 3D volumes from 4D image.
    
    Equivalent to: mirtk extract-image-volume
    
    Args:
        image_4d: 4D image tensor
        time_point: Starting time point (0-indexed)
        num_frames: Number of frames to extract
    
    Returns:
        List of 3D Image objects
    """
    data = image_4d.tensor()
    
    # Assuming time is the first dimension after channels
    if data.ndim == 5:  # (C, T, X, Y, Z)
        time_dim = 1
    elif data.ndim == 4:  # (T, X, Y, Z) - no channels
        time_dim = 0
        data = data.unsqueeze(0)  # Add channel dimension
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")
    
    total_frames = data.shape[time_dim]
    
    if num_frames is None:
        num_frames = total_frames - time_point
    
    end_point = min(time_point + num_frames, total_frames)
    
    volumes = []
    grid_3d = image_4d.grid()
    if grid_3d.ndim == 4:  # Remove time dimension from grid
        grid_3d = Grid(
            size=grid_3d.size()[1:],  # Remove T dimension
            spacing=grid_3d.spacing()[1:],
            origin=grid_3d.origin()[1:],
            direction=grid_3d.direction()[3:],  # Adjust direction matrix
            device=grid_3d.device(),
        )
    
    for t in range(time_point, end_point):
        if time_dim == 1:
            vol_data = data[:, t, ...]  # (C, X, Y, Z)
        else:
            vol_data = data[t, ...]
        
        volume = Image(vol_data, grid_3d)
        volumes.append(volume)
    
    return volumes


def combine_volumes_to_4d(
    volumes: List[Image]
) -> Image:
    """Combine 3D volumes into a 4D image.
    
    Equivalent to: mirtk combine-images
    """
    if not volumes:
        raise ValueError("No volumes provided")
    
    # Stack along time dimension
    tensors = [vol.tensor() for vol in volumes]
    
    # Ensure all have same shape
    shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != shape:
            raise ValueError("All volumes must have the same shape")
    
    # Stack along new dimension (time)
    data_4d = torch.stack(tensors, dim=1)  # (C, T, X, Y, Z)
    
    # Create 4D grid
    grid_3d = volumes[0].grid()
    grid_4d = Grid(
        size=(len(volumes),) + grid_3d.size(),
        spacing=(1.0,) + grid_3d.spacing(),  # Time spacing = 1
        origin=(0.0,) + grid_3d.origin(),
        direction=torch.eye(4, device=grid_3d.device()),
        device=grid_3d.device(),
    )
    
    return Image(data_4d, grid_4d)


def read_stl_mesh(path: Union[str, Path]):
    """Read STL mesh file.
    
    Uses VTK for mesh I/O.
    """
    try:
        from vtk import vtkSTLReader
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("VTK is required for mesh I/O. Install with: pip install vtk")
    
    path = str(Path(path))
    reader = vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    
    mesh = reader.GetOutput()
    return mesh


def write_stl_mesh(mesh, path: Union[str, Path]) -> None:
    """Write mesh to STL file."""
    try:
        from vtk import vtkSTLWriter
    except ImportError:
        raise ImportError("VTK is required for mesh I/O. Install with: pip install vtk")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mesh)
    writer.Write()