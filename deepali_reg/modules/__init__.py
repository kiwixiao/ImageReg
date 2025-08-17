"""Deepali registration pipeline modules."""

from .io_utils import (
    read_image,
    write_image,
    extract_volume_from_4d,
    combine_volumes_to_4d,
    read_stl_mesh,
    write_stl_mesh,
)

from .preprocessing import (
    extract_surface_from_mask,
    mask_to_surface,
    surface_to_mask,
    align_images,
)

from .registration import (
    register_pairwise_svf,
    compose_transformations,
)

from .interpolation import (
    interpolate_transformations,
    apply_transformation_to_mesh,
)

from .mesh_utils import (
    transform_mesh_points,
    mesh_to_mask,
)

__all__ = [
    # I/O
    "read_image",
    "write_image",
    "extract_volume_from_4d",
    "combine_volumes_to_4d",
    "read_stl_mesh",
    "write_stl_mesh",
    # Preprocessing
    "extract_surface_from_mask",
    "mask_to_surface",
    "surface_to_mask",
    "align_images",
    # Registration
    "register_pairwise_svf",
    "compose_transformations",
    # Interpolation
    "interpolate_transformations",
    "apply_transformation_to_mesh",
    # Mesh utilities
    "transform_mesh_points",
    "mesh_to_mask",
]