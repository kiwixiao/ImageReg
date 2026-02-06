"""DAREG Visualization Module"""

from .plotting import plot_side_by_side, plot_overlay, save_visualization
from .convergence import plot_convergence, plot_multi_level_convergence
from .deformation import plot_grid_deformation, plot_displacement_field
from .pdf_report import create_pdf_report, RegistrationReport
from .alignment_overlay import (
    create_alignment_overlay_figure,
    create_alignment_overlay_3views,
    save_alignment_progression,
)
from .segmentation_viewer import (
    create_segmentation_progression_view,
    create_all_segmentation_progression_views,
    create_segmentation_overlay_progression,
)

__all__ = [
    "plot_side_by_side",
    "plot_overlay",
    "save_visualization",
    "plot_convergence",
    "plot_multi_level_convergence",
    "plot_grid_deformation",
    "plot_displacement_field",
    "create_pdf_report",
    "RegistrationReport",
    "create_alignment_overlay_figure",
    "create_alignment_overlay_3views",
    "save_alignment_progression",
    "create_segmentation_progression_view",
    "create_all_segmentation_progression_views",
    "create_segmentation_overlay_progression",
]
