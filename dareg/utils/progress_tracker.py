"""
DAREG Progress Tracker

Tracks registration progress and emits updates for monitoring.
Supports both:
- In-process callbacks (for GUI in same process)
- File-based updates (for separate monitor watching output directory)

Usage:
    # In main_motion.py
    tracker = ProgressTracker(output_dir=output_path)
    tracker.start(total_phases=4, max_frames=10)

    # In registration loop
    tracker.set_phase("alignment", phase_index=0)
    tracker.set_level(level_idx, max_iterations)
    tracker.update_iteration(iter, loss, nmi_loss, reg_loss)

    # When done
    tracker.finish(success=True)
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, List
from pathlib import Path
import json
import time
from datetime import datetime
from copy import deepcopy


@dataclass
class ProgressState:
    """Current state of registration progress."""
    status: str = "idle"  # idle, running, completed, failed
    phase: str = ""
    phase_index: int = 0
    total_phases: int = 0
    level: int = 0
    max_level: int = 0
    iteration: int = 0
    max_iteration: int = 0
    loss: float = 0.0
    nmi_loss: float = 0.0
    regularization_loss: float = 0.0
    frame: int = 0
    max_frames: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    visualization_path: str = ""
    last_update: str = ""
    log_message: str = ""


class ProgressTracker:
    """
    Tracks registration progress and emits updates.

    Works in two modes:
    1. File-based: Writes progress.json to output_dir for external monitoring
    2. Callback-based: Calls registered callbacks for in-process GUI updates

    PyQt signals are optional - if PyQt is available and this is subclassed
    with QObject, signals will work. Otherwise, use callbacks.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_file_output: bool = True,
        visualization_interval: int = 20,
        log_interval: int = 10
    ):
        """
        Initialize progress tracker.

        Args:
            output_dir: Directory to write progress.json (None = no file output)
            enable_file_output: Whether to write progress.json
            visualization_interval: Generate visualization every N iterations
            log_interval: Log progress every N iterations
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.enable_file_output = enable_file_output and output_dir is not None
        self.visualization_interval = visualization_interval
        self.log_interval = log_interval

        self.state = ProgressState()
        self.start_time: Optional[float] = None
        self._iteration_times: List[float] = []

        # Callbacks for in-process updates
        self._on_progress_callbacks: List[Callable[[ProgressState], None]] = []
        self._on_log_callbacks: List[Callable[[str], None]] = []
        self._on_visualization_callbacks: List[Callable[[str], None]] = []
        self._on_finish_callbacks: List[Callable[[bool, str], None]] = []

        # Visualization generator (set externally)
        self._visualization_generator: Optional[Callable] = None

    def on_progress(self, callback: Callable[[ProgressState], None]):
        """Register callback for progress updates."""
        self._on_progress_callbacks.append(callback)

    def on_log(self, callback: Callable[[str], None]):
        """Register callback for log messages."""
        self._on_log_callbacks.append(callback)

    def on_visualization(self, callback: Callable[[str], None]):
        """Register callback for visualization updates."""
        self._on_visualization_callbacks.append(callback)

    def on_finish(self, callback: Callable[[bool, str], None]):
        """Register callback for completion."""
        self._on_finish_callbacks.append(callback)

    def set_visualization_generator(self, generator: Callable):
        """Set function to generate progress visualization."""
        self._visualization_generator = generator

    def start(self, total_phases: int, max_frames: int):
        """Called at pipeline start."""
        self.start_time = time.time()
        self.state.status = "running"
        self.state.total_phases = total_phases
        self.state.max_frames = max_frames
        self._emit_update()
        self.log(f"Pipeline started: {total_phases} phases, {max_frames} frames")

    def set_phase(self, phase: str, phase_index: int, max_level: int = 4):
        """Called when entering a new phase (alignment, pairwise, etc.)."""
        self.state.phase = phase
        self.state.phase_index = phase_index
        self.state.max_level = max_level
        self.state.level = 0
        self.state.iteration = 0
        self._emit_update()
        self.log(f"Phase {phase_index + 1}/{self.state.total_phases}: {phase}")

    def set_frame(self, frame: int):
        """Called when starting registration for a new frame."""
        self.state.frame = frame
        self._emit_update()
        self.log(f"  Frame {frame + 1}/{self.state.max_frames}")

    def set_level(self, level: int, max_iteration: int):
        """Called when entering a new pyramid level."""
        self.state.level = level
        self.state.max_iteration = max_iteration
        self.state.iteration = 0
        self._emit_update()
        self.log(f"    Level {level + 1}/{self.state.max_level}: {max_iteration} iterations")

    def update_iteration(
        self,
        iteration: int,
        loss: float,
        nmi_loss: float = 0.0,
        reg_loss: float = 0.0
    ):
        """Called each optimization iteration."""
        self.state.iteration = iteration
        self.state.loss = loss
        self.state.nmi_loss = nmi_loss
        self.state.regularization_loss = reg_loss

        # Track timing for ETA estimation
        now = time.time()
        self._iteration_times.append(now)
        if len(self._iteration_times) > 100:
            self._iteration_times.pop(0)

        self._update_timing()
        self._emit_update()

        # Log periodically
        if iteration > 0 and iteration % self.log_interval == 0:
            self.log(
                f"      Iter {iteration}: loss={loss:.6f}, "
                f"nmi={nmi_loss:.6f}, reg={reg_loss:.6f}"
            )

        # Generate visualization periodically
        if self._visualization_generator and iteration % self.visualization_interval == 0:
            self._request_visualization()

    def log(self, message: str):
        """Emit a log message."""
        self.state.log_message = message

        # Call callbacks
        for callback in self._on_log_callbacks:
            try:
                callback(message)
            except Exception:
                pass

        # Write to log file
        if self.enable_file_output and self.output_dir:
            self._append_log(message)

    def set_visualization(self, image_path: str):
        """Called when a new visualization is available."""
        self.state.visualization_path = image_path

        # Call callbacks
        for callback in self._on_visualization_callbacks:
            try:
                callback(image_path)
            except Exception:
                pass

        self._emit_update()

    def finish(self, success: bool = True, message: str = ""):
        """Called when pipeline completes."""
        self.state.status = "completed" if success else "failed"
        self._update_timing()
        self._emit_update()

        status_msg = "completed successfully" if success else f"failed: {message}"
        elapsed = self.state.elapsed_seconds
        mins, secs = divmod(int(elapsed), 60)
        self.log(f"Pipeline {status_msg} in {mins}m {secs}s")

        # Call callbacks
        for callback in self._on_finish_callbacks:
            try:
                callback(success, message)
            except Exception:
                pass

    def _update_timing(self):
        """Update elapsed and estimated remaining time."""
        if self.start_time:
            self.state.elapsed_seconds = time.time() - self.start_time

        # Estimate remaining time based on iteration rate
        if len(self._iteration_times) >= 2:
            # Calculate average time per iteration
            elapsed_for_iters = self._iteration_times[-1] - self._iteration_times[0]
            avg_iter_time = elapsed_for_iters / len(self._iteration_times)

            # Estimate remaining work
            remaining_iters = max(0, self.state.max_iteration - self.state.iteration)
            remaining_levels = max(0, self.state.max_level - self.state.level - 1)
            remaining_frames = max(0, self.state.max_frames - self.state.frame - 1)
            remaining_phases = max(0, self.state.total_phases - self.state.phase_index - 1)

            # Very rough estimate: assume each level/frame/phase has similar iteration count
            avg_iters_per_level = self.state.max_iteration if self.state.max_iteration > 0 else 100

            total_remaining_iters = (
                remaining_iters +
                remaining_levels * avg_iters_per_level +
                remaining_frames * self.state.max_level * avg_iters_per_level +
                remaining_phases * self.state.max_frames * self.state.max_level * avg_iters_per_level
            )

            self.state.estimated_remaining_seconds = total_remaining_iters * avg_iter_time

    def _emit_update(self):
        """Emit progress update via callbacks and file."""
        self.state.last_update = datetime.now().isoformat()

        # Call progress callbacks with a copy of state
        state_copy = deepcopy(self.state)
        for callback in self._on_progress_callbacks:
            try:
                callback(state_copy)
            except Exception:
                pass

        # Write to file
        if self.enable_file_output and self.output_dir:
            self._write_progress_file()

    def _write_progress_file(self):
        """Write current state to progress.json."""
        if not self.output_dir:
            return

        progress_file = self.output_dir / "progress.json"
        try:
            with open(progress_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception:
            pass  # Don't fail registration if progress file write fails

    def _append_log(self, message: str):
        """Append message to log file."""
        if not self.output_dir:
            return

        log_file = self.output_dir / "progress_log.txt"
        try:
            with open(log_file, 'a') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass

    def _request_visualization(self):
        """Request visualization generation."""
        if self._visualization_generator and self.output_dir:
            try:
                viz_path = self.output_dir / "progress_overlay.png"
                self._visualization_generator(str(viz_path))
                self.set_visualization(str(viz_path))
            except Exception:
                pass

    def get_state(self) -> ProgressState:
        """Get current progress state."""
        return deepcopy(self.state)

    def get_progress_percentage(self) -> float:
        """Get overall progress as percentage (0-100)."""
        total_work = (
            self.state.total_phases *
            self.state.max_frames *
            self.state.max_level *
            self.state.max_iteration
        )
        if total_work == 0:
            return 0.0

        current_work = (
            self.state.phase_index * self.state.max_frames * self.state.max_level * self.state.max_iteration +
            self.state.frame * self.state.max_level * self.state.max_iteration +
            self.state.level * self.state.max_iteration +
            self.state.iteration
        )

        return min(100.0, 100.0 * current_work / total_work)
