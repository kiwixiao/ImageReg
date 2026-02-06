"""
DAREG Standalone Progress Monitor

A separate window for monitoring CLI registration runs.
Polls progress.json and log files from the output directory.

Usage:
    python -m dareg.monitor --watch ./output_dir
    python -m dareg.monitor --watch ./output_dir --refresh 500

This allows monitoring registration progress from a separate terminal
while the main registration runs in the background.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QStatusBar, QMenuBar, QAction
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon

from .progress_widget import ProgressWidget
from ..utils.progress_tracker import ProgressState


class MonitorWindow(QMainWindow):
    """
    Standalone window for monitoring registration progress.

    Polls the output directory for:
    - progress.json: Current registration state
    - progress_log.txt: Log messages
    - progress_overlay.png: Visualization updates
    """

    def __init__(
        self,
        watch_dir: Path,
        refresh_ms: int = 1000,
        parent=None
    ):
        """
        Initialize monitor window.

        Args:
            watch_dir: Directory to watch for progress updates
            refresh_ms: Refresh interval in milliseconds
            parent: Parent widget
        """
        super().__init__(parent)
        self.watch_dir = watch_dir
        self.refresh_ms = refresh_ms
        self.last_log_position = 0
        self.last_viz_mtime = 0

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Build the window UI."""
        self.setWindowTitle(f"DAREG Monitor - {self.watch_dir.name}")
        self.setMinimumSize(900, 700)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        refresh_action = QAction("Refresh Now", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._poll_progress)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header with watch directory
        header = QLabel(f"Watching: {self.watch_dir}")
        header.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(header)

        # Progress widget
        self.progress_widget = ProgressWidget(show_visualization=True)
        layout.addWidget(self.progress_widget)

        # Status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._update_status("Waiting for registration to start...")

    def _setup_timer(self):
        """Set up polling timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_progress)
        self.timer.start(self.refresh_ms)

        # Initial poll
        self._poll_progress()

    def _poll_progress(self):
        """Read progress files and update UI."""
        progress_file = self.watch_dir / "progress.json"
        log_file = self.watch_dir / "progress_log.txt"
        viz_file = self.watch_dir / "progress_overlay.png"

        # Read progress state
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                state = ProgressState(**data)
                self.progress_widget.update_progress(state)

                # Update status bar
                if state.status == "running":
                    pct = self.progress_widget.overall_progress.value()
                    self._update_status(f"Running: {pct}% complete")
                elif state.status == "completed":
                    self._update_status("Registration completed!")
                    self.timer.stop()  # Stop polling after completion
                elif state.status == "failed":
                    self._update_status(f"Registration failed: {state.log_message}")
                    self.timer.stop()

            except (json.JSONDecodeError, TypeError) as e:
                # File being written, skip this poll
                pass
            except Exception as e:
                self._update_status(f"Error reading progress: {e}")
        else:
            self._update_status("Waiting for registration to start...")

        # Read new log entries
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    f.seek(self.last_log_position)
                    new_lines = f.read()
                    self.last_log_position = f.tell()

                if new_lines.strip():
                    for line in new_lines.strip().split('\n'):
                        self.progress_widget.append_log(line)
            except Exception:
                pass

        # Update visualization if changed
        if viz_file.exists():
            try:
                mtime = viz_file.stat().st_mtime
                if mtime > self.last_viz_mtime:
                    self.last_viz_mtime = mtime
                    self.progress_widget.update_visualization(str(viz_file))
            except Exception:
                pass

    def _update_status(self, message: str):
        """Update status bar message."""
        self.statusbar.showMessage(message)

    def closeEvent(self, event):
        """Handle window close."""
        self.timer.stop()
        event.accept()


def launch_monitor(watch_dir: Path, refresh_ms: int = 1000) -> int:
    """
    Launch the monitor window.

    Args:
        watch_dir: Directory to watch
        refresh_ms: Refresh interval in milliseconds

    Returns:
        Application exit code
    """
    app = QApplication(sys.argv)
    app.setApplicationName("DAREG Monitor")

    window = MonitorWindow(watch_dir, refresh_ms)
    window.show()

    return app.exec_()


def main():
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Monitor DAREG registration progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Monitor registration in ./motion_output
    python -m dareg.monitor --watch ./motion_output

    # Faster refresh (500ms instead of 1000ms)
    python -m dareg.monitor --watch ./motion_output --refresh 500

    # Watch a specific output directory
    python -m dareg.monitor -w /path/to/registration/output
        """
    )
    parser.add_argument(
        "--watch", "-w",
        required=True,
        help="Output directory to watch for progress updates"
    )
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=1000,
        help="Refresh interval in milliseconds (default: 1000)"
    )

    args = parser.parse_args()

    watch_dir = Path(args.watch)
    if not watch_dir.exists():
        print(f"Error: Directory does not exist: {watch_dir}")
        print("The directory will be created when registration starts.")
        print("You can start the monitor now and it will update once registration begins.")
        # Create directory so monitor can start
        watch_dir.mkdir(parents=True, exist_ok=True)

    sys.exit(launch_monitor(watch_dir, args.refresh))


if __name__ == "__main__":
    main()
