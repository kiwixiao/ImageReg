"""Deepali registration pipeline modules."""

# Import existing modules
from .io_utils import (
    read_image,
    write_image,
    extract_volume_from_4d,
    combine_volumes_to_4d,
    read_stl_mesh,
    write_stl_mesh,
)

# Skip existing registration module for now due to import issues
# from .registration import (
#     register_pairwise_svf,
#     compose_transformations,
# )

# Import new registration modules
from .rigid_registration import run_rigid_registration
from .rigid_affine_registration import run_rigid_affine_registration
from .rigid_affine_svffd_registration import run_rigid_affine_svffd_registration

__all__ = [
    # I/O
    "read_image",
    "write_image", 
    "extract_volume_from_4d",
    "combine_volumes_to_4d",
    "read_stl_mesh",
    "write_stl_mesh",
    # Registration modules
    "run_rigid_registration",
    "run_rigid_affine_registration",
    "run_rigid_affine_svffd_registration",
]