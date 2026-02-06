"""
DAREG Convergence Plotting

Visualize optimization convergence across registration levels.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.logging_config import get_logger

logger = get_logger("convergence")


def plot_convergence(
    loss_history: List[float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    figsize: tuple = (10, 6),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot single loss convergence curve

    Args:
        loss_history: List of loss values
        output_path: Optional path to save figure
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add min annotation
    min_idx = np.argmin(loss_history)
    min_val = loss_history[min_idx]
    ax.axhline(y=min_val, color='r', linestyle='--', alpha=0.5)
    ax.annotate(f'Min: {min_val:.6f}', xy=(min_idx, min_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='red')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def plot_multi_level_convergence(
    loss_history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Multi-Level Convergence",
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot convergence across multiple pyramid levels

    Args:
        loss_history: Dictionary with level names as keys and loss lists as values
        output_path: Optional path to save figure
        title: Figure title
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color map for levels
    n_levels = len(loss_history)
    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))

    # Plot each level
    cumulative_iters = 0
    for i, (level_name, losses) in enumerate(sorted(loss_history.items())):
        iters = np.arange(len(losses)) + cumulative_iters
        ax.plot(iters, losses, color=colors[i], linewidth=2, label=level_name)

        # Add vertical line between levels
        if i > 0:
            ax.axvline(x=cumulative_iters, color='gray', linestyle=':', alpha=0.5)

        cumulative_iters += len(losses)

    ax.set_xlabel("Cumulative Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig


def plot_stage_comparison(
    stage_results: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Registration Stages",
    figsize: tuple = (10, 6),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot comparison of final loss across registration stages

    Args:
        stage_results: Dictionary with stage names and final loss values
        output_path: Optional path to save figure
        title: Figure title
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Figure object (if not saved) or None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    stages = list(stage_results.keys())
    losses = list(stage_results.values())

    # Bar plot
    bars = ax.bar(stages, losses, color='steelblue', edgecolor='black')

    # Add value labels
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("Registration Stage")
    ax.set_ylabel("Final Loss")
    ax.set_title(title)

    # Add percentage improvement
    if len(losses) > 1:
        for i in range(1, len(losses)):
            improvement = (losses[i-1] - losses[i]) / losses[i-1] * 100
            ax.annotate(f'{improvement:.1f}% better',
                       xy=(i, losses[i]),
                       xytext=(0, -20), textcoords='offset points',
                       ha='center', fontsize=9, color='green')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {Path(output_path).name}")
        return None

    return fig
