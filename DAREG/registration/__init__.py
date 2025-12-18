"""DAREG Registration Module"""

from .base import BaseRegistration, RegistrationResult
from .rigid import RigidRegistration
from .affine import AffineRegistration
from .ffd import FFDRegistration
from .composer import TransformComposer, ComposedTransform, compose_transforms
from .motion import (
    MotionRegistration,
    MotionResult,
    PairwiseResult,
    run_motion_registration,
)

__all__ = [
    "BaseRegistration",
    "RegistrationResult",
    "RigidRegistration",
    "AffineRegistration",
    "FFDRegistration",
    # Transform composition
    "TransformComposer",
    "ComposedTransform",
    "compose_transforms",
    # Motion registration
    "MotionRegistration",
    "MotionResult",
    "PairwiseResult",
    "run_motion_registration",
]
