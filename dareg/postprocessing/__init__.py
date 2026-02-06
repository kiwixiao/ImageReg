"""DAREG Postprocessing Module

Provides surface mesh generation and temporal interpolation for 4D motion visualization.

Note: This module supports two import patterns:
1. From within DAREG package (e.g., from registration/motion.py) - all imports work
2. From standalone scripts (e.g., main_postprocess.py) - only STL/interpolation imports work

For standalone usage, import specific modules directly:
    from postprocessing.stl_generator import generate_stl_sequence
"""

__all__ = []

# ============================================================
# Core STL/Interpolation functions - always available
# ============================================================
try:
    from .stl_generator import (
        generate_stl_from_segmentation,
        generate_stl_sequence,
        STLGenerator,
        SurfaceMesh,
    )
    __all__.extend([
        "generate_stl_from_segmentation",
        "generate_stl_sequence",
        "STLGenerator",
        "SurfaceMesh",
    ])
except ImportError as e:
    # STL generator should always work - raise if it fails
    raise ImportError(f"Failed to import stl_generator: {e}")

try:
    from .temporal_interpolation import (
        TemporalInterpolator,
        TemporalPoint,
        get_temporal_resolution_from_nifti,
        generate_interpolated_stl_sequence,
        animate_stl_sequence,
    )
    __all__.extend([
        "TemporalInterpolator",
        "TemporalPoint",
        "get_temporal_resolution_from_nifti",
        "generate_interpolated_stl_sequence",
        "animate_stl_sequence",
    ])
except ImportError as e:
    # Temporal interpolation should work - raise if it fails
    raise ImportError(f"Failed to import temporal_interpolation: {e}")

# ============================================================
# Transform/Registration functions - require full DAREG package
# These use relative imports (..utils) that only work within package
# ============================================================
try:
    from .transformer import apply_transform, warp_image
    __all__.extend(["apply_transform", "warp_image"])
except ImportError:
    # Expected to fail when imported standalone from main_postprocess.py
    pass

try:
    from .segmentation import transform_segmentation
    __all__.extend(["transform_segmentation"])
except ImportError:
    pass

try:
    from .quality_metrics import compute_quality_metrics, compute_nmi, compute_dice
    __all__.extend(["compute_quality_metrics", "compute_nmi", "compute_dice"])
except ImportError:
    pass

# ============================================================
# MIRTK-style Transform Utilities - for STL deformation
# ============================================================
try:
    from .transform_utils import (
        build_world_to_lattice_matrix,
        apply_ffd_transform_mirtk_style,
        apply_sequential_transform_to_vertices,
        load_and_apply_alignment_transform,
        load_and_apply_longitudinal_transform,
    )
    __all__.extend([
        "build_world_to_lattice_matrix",
        "apply_ffd_transform_mirtk_style",
        "apply_sequential_transform_to_vertices",
        "load_and_apply_alignment_transform",
        "load_and_apply_longitudinal_transform",
    ])
except ImportError as e:
    # These should always work - they only use scipy and numpy
    import warnings
    warnings.warn(f"Failed to import transform_utils: {e}")
