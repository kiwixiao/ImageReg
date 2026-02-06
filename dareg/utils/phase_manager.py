"""
Phase Manager for Motion Registration Pipeline

Handles phase tracking, checkpoint management, and resume functionality
for the motion registration pipeline.

Phases:
    0: FRAME_EXTRACTION - Extract 3D frames from 4D image
    1: ALIGNMENT - Register static to frame 0 (optional)
    2: PAIRWISE - Register consecutive frames (0→1, 1→2, ...)
    3: COMPOSE - Build longitudinal transforms (0→N)
    4: REFINE - Fine-tune longitudinal transforms (optional)
    5: PROPAGATE - Transform segmentation through all frames
    6: FINALIZE - Save final results and create visualizations
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto


class Phase(Enum):
    """Pipeline phases in execution order"""
    FRAME_EXTRACTION = 0
    ALIGNMENT = 1
    PAIRWISE = 2
    COMPOSE = 3
    REFINE = 4
    PROPAGATE = 5
    FINALIZE = 6

    @property
    def display_name(self) -> str:
        """Human-readable phase name"""
        names = {
            Phase.FRAME_EXTRACTION: "Frame Extraction",
            Phase.ALIGNMENT: "Static→Frame0 Alignment",
            Phase.PAIRWISE: "Pairwise Registration",
            Phase.COMPOSE: "Compose Longitudinal",
            Phase.REFINE: "Refine Longitudinal",
            Phase.PROPAGATE: "Propagate Segmentation",
            Phase.FINALIZE: "Finalize Results",
        }
        return names.get(self, self.name)


@dataclass
class PhaseStatus:
    """Status of a single phase"""
    phase: str
    status: str  # "pending", "in_progress", "completed", "skipped", "failed"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStatus:
    """Complete pipeline status"""
    pipeline_version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    config_hash: str = ""
    num_frames: int = 0
    phases: Dict[str, PhaseStatus] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class PhaseManager:
    """
    Manages pipeline phase tracking and resume functionality.

    Features:
    - Detects completed phases from output files
    - Maintains status file for tracking
    - Enables resuming from last completed phase
    - Validates phase dependencies
    """

    # Output file patterns for each phase (relative to output_dir)
    # These patterns match the actual output structure from MotionRegistration
    # Naming convention: f###_t### where f=absolute, t=relative (both 1-indexed)
    PHASE_OUTPUT_PATTERNS = {
        Phase.FRAME_EXTRACTION: [
            "frames/dynamic_extracted_f*_t*.nii.gz",
        ],
        Phase.ALIGNMENT: [
            "alignment/alignment_transform.pth",
            "alignment/aligned_static_original_resolution.nii.gz",
        ],
        Phase.PAIRWISE: [
            "pairwise/pairwise_*.pth",
        ],
        Phase.COMPOSE: [
            "transforms/longitudinal_*_composed.pth",
            "longitudinal/longitudinal_*.pth",  # Alternative location
        ],
        Phase.REFINE: [
            "transforms/longitudinal_*_refined.pth",
            "longitudinal/refined_*.pth",  # Alternative location
        ],
        Phase.PROPAGATE: [
            "segmentations/seg_f*_t*.nii.gz",
            "segmentations/segmentation_f*_t*.nii.gz",  # Alternative naming
        ],
        Phase.FINALIZE: [
            "visualizations/*.png",
            "pipeline_status.json",
        ],
    }

    # Required output counts for phases that depend on frame count
    FRAME_DEPENDENT_PHASES = {
        Phase.FRAME_EXTRACTION,
        Phase.PAIRWISE,
        Phase.COMPOSE,
        Phase.REFINE,
        Phase.PROPAGATE,
    }

    # Phase dependencies (phase: list of required prior phases)
    PHASE_DEPENDENCIES = {
        Phase.FRAME_EXTRACTION: [],
        Phase.ALIGNMENT: [Phase.FRAME_EXTRACTION],
        Phase.PAIRWISE: [Phase.FRAME_EXTRACTION],
        Phase.COMPOSE: [Phase.PAIRWISE],
        Phase.REFINE: [Phase.COMPOSE],
        Phase.PROPAGATE: [Phase.REFINE],  # Or COMPOSE if REFINE is skipped
        Phase.FINALIZE: [Phase.PROPAGATE],
    }

    STATUS_FILE = "pipeline_status.json"

    def __init__(
        self,
        output_dir: Path,
        num_frames: int = 0,
        skip_alignment: bool = False,
        skip_refinement: bool = False,
    ):
        """
        Initialize PhaseManager.

        Args:
            output_dir: Pipeline output directory
            num_frames: Number of frames in 4D image
            skip_alignment: Whether alignment phase is skipped
            skip_refinement: Whether refinement phase is skipped
        """
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        self.skip_alignment = skip_alignment
        self.skip_refinement = skip_refinement
        self.status_file = self.output_dir / self.STATUS_FILE

        # Load or create status
        self.status = self._load_status()

    def _load_status(self) -> PipelineStatus:
        """Load existing status or create new one"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct PipelineStatus from dict
                    status = PipelineStatus(
                        pipeline_version=data.get("pipeline_version", "1.0"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        config_hash=data.get("config_hash", ""),
                        num_frames=data.get("num_frames", 0),
                    )
                    # Reconstruct phase statuses
                    for phase_name, phase_data in data.get("phases", {}).items():
                        status.phases[phase_name] = PhaseStatus(
                            phase=phase_data.get("phase", phase_name),
                            status=phase_data.get("status", "pending"),
                            start_time=phase_data.get("start_time"),
                            end_time=phase_data.get("end_time"),
                            outputs=phase_data.get("outputs", []),
                            error=phase_data.get("error"),
                            metadata=phase_data.get("metadata", {}),
                        )
                    return status
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load status file, creating new: {e}")

        # Create new status
        status = PipelineStatus(num_frames=self.num_frames)
        for phase in Phase:
            status.phases[phase.name] = PhaseStatus(
                phase=phase.name,
                status="pending",
            )
        return status

    def _save_status(self) -> None:
        """Save current status to file"""
        self.status.updated_at = datetime.now().isoformat()

        # Convert to serializable dict
        data = {
            "pipeline_version": self.status.pipeline_version,
            "created_at": self.status.created_at,
            "updated_at": self.status.updated_at,
            "config_hash": self.status.config_hash,
            "num_frames": self.status.num_frames,
            "phases": {},
        }
        for phase_name, phase_status in self.status.phases.items():
            data["phases"][phase_name] = {
                "phase": phase_status.phase,
                "status": phase_status.status,
                "start_time": phase_status.start_time,
                "end_time": phase_status.end_time,
                "outputs": phase_status.outputs,
                "error": phase_status.error,
                "metadata": phase_status.metadata,
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _count_matching_files(self, pattern: str) -> int:
        """Count files matching a glob pattern"""
        return len(list(self.output_dir.glob(pattern)))

    def _get_expected_count(self, phase: Phase) -> int:
        """Get expected output count for a phase"""
        if phase == Phase.FRAME_EXTRACTION:
            return self.num_frames
        elif phase == Phase.PAIRWISE:
            return max(0, self.num_frames - 1)
        elif phase in (Phase.COMPOSE, Phase.REFINE):
            return max(0, self.num_frames - 1)
        elif phase == Phase.PROPAGATE:
            return self.num_frames
        else:
            return 1  # Single output expected

    def detect_phase_completion(self, phase: Phase) -> bool:
        """
        Detect if a phase is complete by checking output files.

        Args:
            phase: Phase to check

        Returns:
            True if phase outputs exist and are complete
        """
        patterns = self.PHASE_OUTPUT_PATTERNS.get(phase, [])
        if not patterns:
            return False

        # Check if this phase should be skipped
        if phase == Phase.ALIGNMENT and self.skip_alignment:
            return True  # Consider "complete" via skip
        if phase == Phase.REFINE and self.skip_refinement:
            return True  # Consider "complete" via skip

        # For frame-dependent phases, check count
        if phase in self.FRAME_DEPENDENT_PHASES:
            expected = self._get_expected_count(phase)
            for pattern in patterns:
                found = self._count_matching_files(pattern)
                if found >= expected:
                    return True
            return False
        else:
            # For non-frame-dependent phases, just check existence
            for pattern in patterns:
                if self._count_matching_files(pattern) > 0:
                    return True
            return False

    def get_completed_phases(self) -> Set[Phase]:
        """Get set of phases that are detected as complete"""
        completed = set()
        for phase in Phase:
            if self.detect_phase_completion(phase):
                completed.add(phase)
        return completed

    def get_resume_phase(self) -> Optional[Phase]:
        """
        Determine which phase to resume from.

        Returns:
            First incomplete phase that should be executed, or None if all complete
        """
        completed = self.get_completed_phases()

        for phase in Phase:
            # Skip phases that are optionally skipped
            if phase == Phase.ALIGNMENT and self.skip_alignment:
                continue
            if phase == Phase.REFINE and self.skip_refinement:
                continue

            if phase not in completed:
                return phase

        return None  # All phases complete

    def can_resume_from(self, phase: Phase) -> bool:
        """
        Check if we can resume from a specific phase.

        Verifies that all dependencies are satisfied.

        Args:
            phase: Phase to check

        Returns:
            True if all required prior phases are complete
        """
        dependencies = self.PHASE_DEPENDENCIES.get(phase, [])
        completed = self.get_completed_phases()

        for dep in dependencies:
            # Skip dependency check for optionally skipped phases
            if dep == Phase.ALIGNMENT and self.skip_alignment:
                continue
            if dep == Phase.REFINE and self.skip_refinement:
                # If refinement is skipped, PROPAGATE depends on COMPOSE instead
                if phase == Phase.PROPAGATE:
                    if Phase.COMPOSE not in completed:
                        return False
                    continue

            if dep not in completed:
                return False

        return True

    def start_phase(self, phase: Phase) -> None:
        """Mark a phase as started"""
        phase_status = self.status.phases.get(phase.name)
        if phase_status:
            phase_status.status = "in_progress"
            phase_status.start_time = datetime.now().isoformat()
            phase_status.end_time = None
            phase_status.error = None
            self._save_status()

    def complete_phase(
        self,
        phase: Phase,
        outputs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a phase as completed"""
        phase_status = self.status.phases.get(phase.name)
        if phase_status:
            phase_status.status = "completed"
            phase_status.end_time = datetime.now().isoformat()
            if outputs:
                phase_status.outputs = outputs
            if metadata:
                phase_status.metadata = metadata
            self._save_status()

    def skip_phase(self, phase: Phase, reason: str = "skipped by configuration") -> None:
        """Mark a phase as skipped"""
        phase_status = self.status.phases.get(phase.name)
        if phase_status:
            phase_status.status = "skipped"
            phase_status.metadata["skip_reason"] = reason
            self._save_status()

    def fail_phase(self, phase: Phase, error: str) -> None:
        """Mark a phase as failed"""
        phase_status = self.status.phases.get(phase.name)
        if phase_status:
            phase_status.status = "failed"
            phase_status.end_time = datetime.now().isoformat()
            phase_status.error = error
            self._save_status()

    def get_phase_status(self, phase: Phase) -> str:
        """Get status string for a phase"""
        phase_status = self.status.phases.get(phase.name)
        if phase_status:
            return phase_status.status
        return "unknown"

    def print_status_summary(self) -> None:
        """Print a summary of all phase statuses"""
        print("\n" + "=" * 60)
        print("PIPELINE PHASE STATUS")
        print("=" * 60)

        for phase in Phase:
            status = self.get_phase_status(phase)
            detected = self.detect_phase_completion(phase)

            # Status indicator
            if status == "completed" or detected:
                indicator = "✅"
            elif status == "skipped":
                indicator = "⏭️ "
            elif status == "in_progress":
                indicator = "🔄"
            elif status == "failed":
                indicator = "❌"
            else:
                indicator = "⏳"

            # Build display line
            phase_name = f"{phase.value}: {phase.display_name}"
            file_status = "(files detected)" if detected else ""

            print(f"  {indicator} {phase_name:<35} [{status:^12}] {file_status}")

        # Print resume info
        resume_phase = self.get_resume_phase()
        if resume_phase:
            print(f"\n  📌 Resume from: {resume_phase.display_name}")
        else:
            print(f"\n  ✅ All phases complete!")

        print("=" * 60 + "\n")

    def get_outputs_for_phase(self, phase: Phase) -> List[Path]:
        """Get list of output files for a phase"""
        outputs = []
        patterns = self.PHASE_OUTPUT_PATTERNS.get(phase, [])
        for pattern in patterns:
            outputs.extend(self.output_dir.glob(pattern))
        return outputs

    def clean_phase_outputs(self, phase: Phase, dry_run: bool = False) -> List[Path]:
        """
        Clean up all output files for a phase (for fresh restart).

        This ensures a phase starts from scratch without partial/corrupted files.

        Args:
            phase: Phase to clean
            dry_run: If True, only list files without deleting

        Returns:
            List of files that were (or would be) deleted
        """
        outputs = self.get_outputs_for_phase(phase)
        deleted = []

        for file_path in outputs:
            if file_path.exists():
                if dry_run:
                    print(f"  Would delete: {file_path}")
                else:
                    try:
                        file_path.unlink()
                        deleted.append(file_path)
                    except Exception as e:
                        print(f"  Warning: Could not delete {file_path}: {e}")

        # Reset phase status to pending
        if not dry_run and phase.name in self.status.phases:
            self.status.phases[phase.name].status = "pending"
            self.status.phases[phase.name].outputs = []
            self.status.phases[phase.name].error = None
            self._save_status()

        return deleted

    def prepare_phase_for_restart(self, phase: Phase) -> None:
        """
        Prepare a phase for fresh restart by cleaning partial outputs.

        Call this before starting a phase that was previously incomplete.

        Args:
            phase: Phase to prepare
        """
        if self.detect_phase_completion(phase):
            # Phase is complete, no cleanup needed
            return

        # Check if there are any partial outputs
        outputs = self.get_outputs_for_phase(phase)
        if outputs:
            print(f"\n🧹 Cleaning partial outputs for {phase.display_name}...")
            deleted = self.clean_phase_outputs(phase)
            if deleted:
                print(f"   Removed {len(deleted)} partial file(s)")
            print(f"   Phase will restart fresh\n")

    def update_num_frames(self, num_frames: int) -> None:
        """Update the number of frames (call after loading 4D image)"""
        self.num_frames = num_frames
        self.status.num_frames = num_frames
        self._save_status()

    def should_run_phase(self, phase: Phase) -> bool:
        """
        Determine if a phase should be run.

        Considers:
        - Skip configuration (alignment, refinement)
        - Whether phase is already complete
        - Whether resume mode is enabled

        Args:
            phase: Phase to check

        Returns:
            True if phase should be executed
        """
        # Check for skipped phases
        if phase == Phase.ALIGNMENT and self.skip_alignment:
            return False
        if phase == Phase.REFINE and self.skip_refinement:
            return False

        # Check if already complete
        if self.detect_phase_completion(phase):
            return False

        # Check dependencies
        return self.can_resume_from(phase)
