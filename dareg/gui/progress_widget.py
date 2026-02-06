"""
DAREG Progress Widget

PyQt widget showing real-time registration progress:
- Overall progress bar (based on phases, frames, levels)
- Current level progress bar
- Status labels (phase, frame, loss, ETA)
- Log viewer
- Live visualization preview

Can be embedded in main GUI or used in standalone monitor.
"""

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QProgressBar,
    QLabel, QTextEdit, QGroupBox, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont

from ..utils.progress_tracker import ProgressState


class ProgressWidget(QWidget):
    """
    Combined widget showing registration progress.

    Components:
    - Progress bars (overall + current level)
    - Status labels
    - Log viewer
    - Live visualization preview
    """

    def __init__(self, parent=None, show_visualization: bool = True):
        """
        Initialize progress widget.

        Args:
            parent: Parent widget
            show_visualization: Whether to show live preview panel
        """
        super().__init__(parent)
        self.show_visualization = show_visualization
        self._setup_ui()

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === Progress Section ===
        progress_group = QGroupBox("Registration Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)

        # Overall progress bar
        overall_layout = QHBoxLayout()
        overall_label = QLabel("Overall:")
        overall_label.setFixedWidth(60)
        overall_layout.addWidget(overall_label)

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("%p%")
        overall_layout.addWidget(self.overall_progress)

        self.overall_pct_label = QLabel("0%")
        self.overall_pct_label.setFixedWidth(50)
        overall_layout.addWidget(self.overall_pct_label)
        progress_layout.addLayout(overall_layout)

        # Current level progress bar
        level_layout = QHBoxLayout()
        level_label = QLabel("Level:")
        level_label.setFixedWidth(60)
        level_layout.addWidget(level_label)

        self.level_progress = QProgressBar()
        self.level_progress.setRange(0, 100)
        self.level_progress.setTextVisible(True)
        self.level_progress.setFormat("%p%")
        level_layout.addWidget(self.level_progress)

        self.level_info_label = QLabel("0/0")
        self.level_info_label.setFixedWidth(120)
        level_layout.addWidget(self.level_info_label)
        progress_layout.addLayout(level_layout)

        # Status info grid
        status_grid = QHBoxLayout()

        # Left column: Status, Phase, Frame
        left_col = QVBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold;")
        left_col.addWidget(self.status_label)

        self.phase_label = QLabel("Phase: -")
        left_col.addWidget(self.phase_label)

        self.frame_label = QLabel("Frame: -")
        left_col.addWidget(self.frame_label)

        status_grid.addLayout(left_col)

        # Right column: Loss, ETA, Elapsed
        right_col = QVBoxLayout()
        self.loss_label = QLabel("Loss: -")
        right_col.addWidget(self.loss_label)

        self.eta_label = QLabel("ETA: -")
        right_col.addWidget(self.eta_label)

        self.elapsed_label = QLabel("Elapsed: 0m 0s")
        right_col.addWidget(self.elapsed_label)

        status_grid.addLayout(right_col)
        progress_layout.addLayout(status_grid)

        layout.addWidget(progress_group)

        # === Log and Visualization Section ===
        if self.show_visualization:
            splitter = QSplitter(Qt.Horizontal)
            splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Log viewer
            log_group = QGroupBox("Log")
            log_layout = QVBoxLayout(log_group)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Courier", 10))
            self.log_text.setLineWrapMode(QTextEdit.NoWrap)
            log_layout.addWidget(self.log_text)
            splitter.addWidget(log_group)

            # Visualization preview
            viz_group = QGroupBox("Live Preview")
            viz_layout = QVBoxLayout(viz_group)
            self.viz_label = QLabel("No preview available")
            self.viz_label.setAlignment(Qt.AlignCenter)
            self.viz_label.setMinimumSize(300, 200)
            self.viz_label.setStyleSheet("background-color: #2a2a2a; color: #888;")
            viz_layout.addWidget(self.viz_label)
            splitter.addWidget(viz_group)

            # Set initial splitter sizes (60% log, 40% viz)
            splitter.setSizes([600, 400])
            layout.addWidget(splitter)
        else:
            # Just log viewer, no visualization
            log_group = QGroupBox("Log")
            log_layout = QVBoxLayout(log_group)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Courier", 10))
            self.log_text.setMaximumHeight(200)
            log_layout.addWidget(self.log_text)
            layout.addWidget(log_group)
            self.viz_label = None

    def update_progress(self, state: ProgressState):
        """
        Update all UI elements from ProgressState.

        Args:
            state: Current progress state
        """
        # Calculate overall progress percentage
        total_work = (
            state.total_phases *
            max(1, state.max_frames) *
            max(1, state.max_level) *
            max(1, state.max_iteration)
        )
        current_work = (
            state.phase_index * max(1, state.max_frames) * max(1, state.max_level) * max(1, state.max_iteration) +
            state.frame * max(1, state.max_level) * max(1, state.max_iteration) +
            state.level * max(1, state.max_iteration) +
            state.iteration
        )

        if total_work > 0:
            overall_pct = min(100, int(100 * current_work / total_work))
            self.overall_progress.setValue(overall_pct)
            self.overall_pct_label.setText(f"{overall_pct}%")

        # Level progress
        if state.max_iteration > 0:
            level_pct = min(100, int(100 * state.iteration / state.max_iteration))
            self.level_progress.setValue(level_pct)

        self.level_info_label.setText(
            f"L{state.level + 1}/{state.max_level}, "
            f"It {state.iteration}/{state.max_iteration}"
        )

        # Status labels
        status_color = {
            "idle": "#888",
            "running": "#4CAF50",
            "completed": "#2196F3",
            "failed": "#f44336"
        }.get(state.status, "#888")
        self.status_label.setText(f"Status: {state.status.upper()}")
        self.status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")

        self.phase_label.setText(
            f"Phase: {state.phase} ({state.phase_index + 1}/{state.total_phases})"
        )
        self.frame_label.setText(
            f"Frame: {state.frame + 1}/{state.max_frames}"
        )

        # Loss display
        if state.loss > 0:
            self.loss_label.setText(
                f"Loss: {state.loss:.6f} (NMI: {state.nmi_loss:.6f})"
            )
        else:
            self.loss_label.setText("Loss: -")

        # Timing
        elapsed_mins, elapsed_secs = divmod(int(state.elapsed_seconds), 60)
        self.elapsed_label.setText(f"Elapsed: {elapsed_mins}m {elapsed_secs}s")

        if state.estimated_remaining_seconds > 0:
            eta_mins, eta_secs = divmod(int(state.estimated_remaining_seconds), 60)
            if eta_mins > 60:
                eta_hours, eta_mins = divmod(eta_mins, 60)
                self.eta_label.setText(f"ETA: {eta_hours}h {eta_mins}m")
            else:
                self.eta_label.setText(f"ETA: {eta_mins}m {eta_secs}s")
        else:
            self.eta_label.setText("ETA: calculating...")

    def append_log(self, message: str):
        """
        Append message to log viewer.

        Args:
            message: Log message to append
        """
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear the log viewer."""
        self.log_text.clear()

    def update_visualization(self, image_path: str):
        """
        Update visualization preview from image file.

        Args:
            image_path: Path to visualization image (PNG)
        """
        if self.viz_label is None:
            return

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.viz_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.viz_label.setPixmap(scaled)
                self.viz_label.setStyleSheet("")  # Remove placeholder style
        else:
            self.viz_label.setText("Preview not available")
            self.viz_label.setStyleSheet("background-color: #2a2a2a; color: #888;")

    def reset(self):
        """Reset widget to initial state."""
        self.overall_progress.setValue(0)
        self.overall_pct_label.setText("0%")
        self.level_progress.setValue(0)
        self.level_info_label.setText("0/0")
        self.status_label.setText("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #888;")
        self.phase_label.setText("Phase: -")
        self.frame_label.setText("Frame: -")
        self.loss_label.setText("Loss: -")
        self.eta_label.setText("ETA: -")
        self.elapsed_label.setText("Elapsed: 0m 0s")
        self.clear_log()
        if self.viz_label:
            self.viz_label.clear()
            self.viz_label.setText("No preview available")
            self.viz_label.setStyleSheet("background-color: #2a2a2a; color: #888;")
