"""
DAREG PDF Report Generation

Create comprehensive PDF reports for registration results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from deepali.data import Image

from .plotting import _to_numpy, _get_slice, _normalize
from ..utils.logging_config import get_logger

logger = get_logger("pdf_report")


@dataclass
class RegistrationReport:
    """Container for registration report data"""
    title: str = "DAREG Registration Report"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Images
    source: Optional[Image] = None
    target: Optional[Image] = None
    warped_rigid: Optional[Image] = None
    warped_affine: Optional[Image] = None
    warped_ffd: Optional[Image] = None

    # Metrics
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Convergence
    loss_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)


def create_pdf_report(
    report: RegistrationReport,
    output_path: Union[str, Path],
    dpi: int = 150,
):
    """
    Create comprehensive PDF report

    Args:
        report: RegistrationReport with all data
        output_path: Output PDF path
        dpi: DPI for figures
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # Title page
        _create_title_page(pdf, report)

        # Image comparison pages
        if report.source is not None and report.target is not None:
            _create_comparison_page(pdf, report, "axial")
            _create_comparison_page(pdf, report, "sagittal")
            _create_comparison_page(pdf, report, "coronal")

        # Metrics page
        if report.metrics:
            _create_metrics_page(pdf, report)

        # Convergence page
        if report.loss_history:
            _create_convergence_page(pdf, report)

        # Configuration page
        if report.config:
            _create_config_page(pdf, report)

    logger.info(f"Created PDF report: {output_path.name}")


def _create_title_page(pdf: PdfPages, report: RegistrationReport):
    """Create title page"""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    fig.text(0.5, 0.7, report.title, ha='center', va='center',
             fontsize=24, fontweight='bold')

    # Date
    fig.text(0.5, 0.6, f"Generated: {report.date}", ha='center', va='center',
             fontsize=14)

    # Summary
    summary = []
    if report.source is not None:
        src_shape = tuple(report.source.tensor().shape[-3:])
        summary.append(f"Source shape: {src_shape}")
    if report.target is not None:
        tgt_shape = tuple(report.target.tensor().shape[-3:])
        summary.append(f"Target shape: {tgt_shape}")

    stages = []
    if report.warped_rigid is not None:
        stages.append("Rigid")
    if report.warped_affine is not None:
        stages.append("Affine")
    if report.warped_ffd is not None:
        stages.append("FFD/SVFFD")

    if stages:
        summary.append(f"Stages: {' -> '.join(stages)}")

    summary_text = "\n".join(summary)
    fig.text(0.5, 0.4, summary_text, ha='center', va='center',
             fontsize=12, family='monospace')

    plt.axis('off')
    pdf.savefig(fig, dpi=150)
    plt.close()


def _create_comparison_page(pdf: PdfPages, report: RegistrationReport, view: str):
    """Create image comparison page for a specific view"""
    # Count available images
    images = [
        ("Source", report.source),
        ("Target", report.target),
    ]
    if report.warped_rigid is not None:
        images.append(("Rigid", report.warped_rigid))
    if report.warped_affine is not None:
        images.append(("Affine", report.warped_affine))
    if report.warped_ffd is not None:
        images.append(("FFD", report.warped_ffd))

    n_images = len(images)
    n_cols = min(n_images, 3)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, (name, img) in enumerate(images):
        if img is None:
            continue

        arr = _to_numpy(img.tensor())
        slice_2d = _get_slice(arr, None, view)
        slice_norm = _normalize(slice_2d)

        axes[i].imshow(slice_norm, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')

    # Hide unused axes
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"{view.capitalize()} View Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def _create_metrics_page(pdf: PdfPages, report: RegistrationReport):
    """Create metrics summary page"""
    fig = plt.figure(figsize=(8.5, 11))

    # Create table data
    stages = list(report.metrics.keys())
    if not stages:
        return

    # Get all metric names
    all_metrics = set()
    for stage_metrics in report.metrics.values():
        all_metrics.update(stage_metrics.keys())
    metric_names = sorted(all_metrics)

    # Build table
    cell_text = []
    for metric in metric_names:
        row = [metric]
        for stage in stages:
            val = report.metrics[stage].get(metric, "-")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        cell_text.append(row)

    # Create table
    ax = fig.add_subplot(111)
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric"] + stages,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(stages) + 1):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    fig.suptitle("Registration Quality Metrics", fontsize=14, fontweight='bold', y=0.95)
    pdf.savefig(fig, dpi=150)
    plt.close()


def _create_convergence_page(pdf: PdfPages, report: RegistrationReport):
    """Create convergence plots page"""
    n_stages = len(report.loss_history)
    if n_stages == 0:
        return

    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    import numpy as np

    for i, (stage, level_losses) in enumerate(report.loss_history.items()):
        ax = axes[i]

        # Plot each level
        colors = plt.cm.viridis(np.linspace(0, 1, len(level_losses)))
        cumulative = 0

        for j, (level, losses) in enumerate(sorted(level_losses.items())):
            iters = np.arange(len(losses)) + cumulative
            ax.plot(iters, losses, color=colors[j], linewidth=1.5, label=level)
            cumulative += len(losses)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{stage.capitalize()} Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Optimization Convergence", fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def _create_config_page(pdf: PdfPages, report: RegistrationReport):
    """Create configuration summary page"""
    import yaml

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Convert config to YAML string
    config_str = yaml.dump(report.config, default_flow_style=False, sort_keys=False)

    ax.text(0.05, 0.95, config_str, transform=ax.transAxes,
            fontsize=8, family='monospace', verticalalignment='top')

    fig.suptitle("Configuration", fontsize=14, fontweight='bold', y=0.98)
    pdf.savefig(fig, dpi=150)
    plt.close()
