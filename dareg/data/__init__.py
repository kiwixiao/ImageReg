"""DAREG Data Module - Image loading, saving, and data structures"""

from .image_pair import ImagePair
from .loader import load_image, load_image_pair
from .saver import save_image, save_transform
from .image_4d import (
    Image4D,
    MotionSequence,
    load_image_4d,
    extract_frames_to_files,
    create_frame_pairs,
)

__all__ = [
    "ImagePair",
    "load_image",
    "load_image_pair",
    "save_image",
    "save_transform",
    # 4D image handling
    "Image4D",
    "MotionSequence",
    "load_image_4d",
    "extract_frames_to_files",
    "create_frame_pairs",
]
