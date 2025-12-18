"""
DAREG - Deepali Registration

Professional medical image registration library replicating MIRTK functionality
using the deepali library. Implements rigid, affine, and FFD/SVFFD registration
with MIRTK-equivalent parameters and behavior.

Key Features:
- World coordinate registration (physical mm coordinates)
- MIRTK-style foreground overlap masking (FG_Overlap)
- Multi-resolution pyramid optimization
- NMI similarity with configurable histogram bins
- Bending + diffusion energy regularization
- World coordinate regularization for anisotropic images
- 4D motion tracking (pairwise + compose + refine strategy)
"""

__version__ = "1.0.0"
__author__ = "DAREG Team"

from .config import load_config, default_config
from .data import ImagePair, load_image, save_image, Image4D, load_image_4d
from .registration import (
    RigidRegistration,
    AffineRegistration,
    FFDRegistration,
    MotionRegistration,
    run_motion_registration,
)
from .postprocessing import apply_transform, compute_quality_metrics
from .visualization import create_pdf_report

__all__ = [
    # Configuration
    "load_config",
    "default_config",
    # Data
    "ImagePair",
    "load_image",
    "save_image",
    "Image4D",
    "load_image_4d",
    # Registration
    "RigidRegistration",
    "AffineRegistration",
    "FFDRegistration",
    "MotionRegistration",
    "run_motion_registration",
    # Postprocessing
    "apply_transform",
    "compute_quality_metrics",
    # Visualization
    "create_pdf_report",
]
