#!/usr/bin/env python3
"""
DAREG Motion Registration Pipeline

Main entry point for 4D motion tracking registration.

Workflow:
1. Load 4D dynamic image and extract frames
2. Optionally align static high-res image to frame 0
3. Perform pairwise registration between consecutive frames
4. Compose transforms to get longitudinal (frame 0 → frame N)
5. Refine longitudinal transforms
6. Propagate segmentation through all frames

Usage:
    # Basic motion tracking (4D only)
    dareg motion --image4d dynamic.nii.gz --output ./output

    # With alignment to static image
    dareg motion --image4d dynamic.nii.gz --static static.nii.gz --seg seg.nii.gz

    # Extract specific frames
    dareg motion --image4d dynamic.nii.gz --start-frame 5 --num-frames 10
"""

import sys
import os
import platform
import argparse
import json
from pathlib import Path
from datetime import datetime

# Fix macOS threading crash (condition_variable wait failed)
# This must be done BEFORE importing torch
if platform.system() == "Darwin":
    # macOS: conservative thread count to avoid condition_variable crash
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
else:
    # Linux: use 20 cores for parallel computation
    os.environ.setdefault("OMP_NUM_THREADS", "20")
    os.environ.setdefault("MKL_NUM_THREADS", "20")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")

import torch

from .data import load_image
from .data.image_4d import load_image_4d, extract_frames_to_files, Image4D
from .registration.motion import MotionRegistration, MotionResult
from .config import load_config, generate_output_name, list_available_presets
from .visualization import (
    plot_side_by_side,
    plot_convergence,
    create_pdf_report,
    RegistrationReport,
    create_all_segmentation_progression_views,
)
from .utils.logging_config import get_logger, Timer
from .utils.device import get_device
from .utils.phase_manager import PhaseManager, Phase
from .utils.progress_tracker import ProgressTracker

logger = get_logger("main_motion")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DAREG Motion Registration - 4D Motion Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Extract 3D frames from 4D dynamic image
  2. (Optional) Align static high-res to frame 0
  3. Pairwise registration: frame 0→1, 1→2, 2→3, ...
  4. Compose longitudinal: frame 0→1, 0→2, 0→3, ...
  5. Refine each longitudinal transform
  6. Propagate segmentation through all frames

Examples:
  # Basic motion tracking
  dareg motion --image4d breathing_4d.nii.gz -o ./motion_output

  # With static alignment and segmentation
  dareg motion \\
      --image4d breathing_4d.nii.gz \\
      --static highres_static.nii.gz \\
      --seg airway_seg.nii.gz \\
      --model rigid+affine+svffd \\
      -o ./motion_output

  # Using JSON config file (recommended for complex setups)
  dareg motion --json-config configs/motion_config.json

  # JSON config + CLI override (CLI takes priority)
  dareg motion -j configs/motion_config.json --model rigid+affine+svffd

  # Extract specific frame range
  dareg motion \\
      --image4d cardiac_4d.nii.gz \\
      --start-frame 0 \\
      --num-frames 20 \\
      -o ./cardiac_motion
        """,
    )

    # Input files
    parser.add_argument(
        "--image4d", "-4d",
        type=str,
        help="Path to 4D dynamic image (NIfTI). Required unless provided in JSON config.",
    )
    parser.add_argument(
        "--static", "-s",
        type=str,
        help="Path to static high-resolution image (for alignment)",
    )
    parser.add_argument(
        "--seg", "--segmentation",
        type=str,
        help="Path to segmentation mask (on static or frame 0)",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Path to binary ROI mask (NIfTI) to restrict registration region. "
             "If provided, similarity is computed only within this mask. "
             "Useful for focusing registration on specific anatomical structures.",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./motion_output",
        help="Output directory (default: ./motion_output)",
    )

    # Frame selection
    parser.add_argument(
        "--start-frame", "-t",
        type=int,
        default=0,
        help="Starting frame index in 4D (default: 0)",
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=None,
        help="Number of frames to process (default: all)",
    )

    # Registration options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="rigid+affine+ffd",
        choices=["rigid", "rigid+affine", "rigid+affine+ffd", "rigid+affine+svffd"],
        help="Registration model for alignment (default: rigid+affine+ffd)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip alignment step (if static already in frame 0 space)",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip longitudinal refinement (faster but less accurate)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device (default: auto — GPU on Linux, CPU on macOS)",
    )

    # Visualization
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization output",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Also save individual 3D frame files",
    )
    parser.add_argument(
        "--extract-frames-only",
        action="store_true",
        help="Only extract 3D frames from 4D image, skip registration",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # JSON config file
    parser.add_argument(
        "--json-config", "-j",
        type=str,
        help="Path to JSON config file (CLI args override JSON values)",
    )

    # Config preset
    parser.add_argument(
        "--config-preset", "-p",
        type=str,
        help="Name of config preset to use (e.g., 'svffd_standard', 'ffd_mirtk'). "
             "Use --list-presets to see available presets.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available configuration presets and exit",
    )

    # Resume capability
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed phase (detects existing outputs)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit (no registration)",
    )

    # GUI mode
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch interactive configuration GUI before running pipeline",
    )

    return parser.parse_args()


def load_json_config(json_path: str) -> dict:
    """Load JSON configuration file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_args_with_json(args: argparse.Namespace, json_config: dict) -> argparse.Namespace:
    """
    Merge JSON config with CLI arguments.
    CLI arguments take priority over JSON values.
    Only non-default CLI values override JSON.
    """
    # Mapping from JSON keys to argparse attribute names
    json_to_args = {
        # inputs section
        ('inputs', 'image4d'): 'image4d',
        ('inputs', 'static'): 'static',
        ('inputs', 'seg'): 'seg',
        # output section
        ('output', 'output_dir'): 'output',
        ('output', 'extract_frames'): 'extract_frames',
        ('output', 'extract_frames_only'): 'extract_frames_only',
        ('output', 'no_viz'): 'no_viz',
        # frame_selection section
        ('frame_selection', 'start_frame'): 'start_frame',
        ('frame_selection', 'num_frames'): 'num_frames',
        # registration section
        ('registration', 'model'): 'model',
        ('registration', 'config'): 'config',
        ('registration', 'skip_alignment'): 'skip_alignment',
        ('registration', 'skip_refinement'): 'skip_refinement',
        # similarity section
        ('similarity', 'roi_mask'): 'roi_mask',
        # pipeline control
        ('resume',): 'resume',
        # top-level
        ('device',): 'device',
        ('verbose',): 'verbose',
    }

    # Get default values from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image4d", type=str)
    parser.add_argument("--static", type=str)
    parser.add_argument("--seg", type=str)
    parser.add_argument("--roi-mask", type=str, default=None)
    parser.add_argument("--output", type=str, default="./motion_output")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--model", type=str, default="rigid+affine+ffd")
    parser.add_argument("--config", type=str)
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--extract-frames", action="store_true")
    parser.add_argument("--extract-frames-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    defaults = parser.parse_args([])

    # Apply JSON values where CLI argument is at default
    for json_keys, arg_name in json_to_args.items():
        # Navigate nested JSON structure
        value = json_config
        for key in json_keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break

        if value is not None:
            # Get current CLI value and default value
            cli_value = getattr(args, arg_name, None)
            default_value = getattr(defaults, arg_name, None)

            # Only apply JSON value if CLI is at default
            # For boolean flags, check if they're False (not set)
            if cli_value == default_value:
                setattr(args, arg_name, value)

    return args


def create_output_structure(output_path: Path) -> dict:
    """Create output directory structure"""
    dirs = {
        "root": output_path,
        "transforms": output_path / "transforms",
        "segmentations": output_path / "segmentations",
        "frames": output_path / "frames",
        "visualizations": output_path / "visualizations",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def create_motion_visualizations(
    result: MotionResult,
    image_4d: Image4D,
    output_dir: Path,
):
    """Create visualization outputs for motion tracking"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    logger.info("Creating motion visualizations...")

    # 1. Frame-by-frame loss plot
    if result.pairwise_transforms:
        fig, ax = plt.subplots(figsize=(10, 6))

        frame_indices = [pw.source_idx for pw in result.pairwise_transforms]
        losses = [pw.final_loss for pw in result.pairwise_transforms]

        ax.plot(frame_indices, losses, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Registration Loss')
        ax.set_title('Pairwise Registration Loss per Frame')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pairwise_losses.png", dpi=150)
        plt.close()

    # 2. Segmentation progression (if available) - Multi-slice views
    if result.segmentation_sequence and len(result.segmentation_sequence) > 1:
        logger.info("Creating multi-slice segmentation progression visualizations...")
        # Extract spacing from first segmentation for correct aspect ratio
        first_seg = result.segmentation_sequence[0]
        seg_spacing = None
        if hasattr(first_seg, 'grid'):
            grid = first_seg.grid()
            if hasattr(grid, 'spacing'):
                sp = grid.spacing()
                # Grid spacing is (D, H, W) order for deepali
                seg_spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        create_all_segmentation_progression_views(
            result.segmentation_sequence,
            output_dir,
            num_slices=8,
            colormap="viridis",
            title_prefix="Segmentation Propagation",
            spacing=seg_spacing
        )

    # 3. Motion magnitude over time
    if result.longitudinal_transforms:
        fig, ax = plt.subplots(figsize=(10, 6))

        frame_indices = []
        max_displacements = []

        for long in result.longitudinal_transforms:
            frame_indices.append(long.source_idx)

            # Estimate displacement magnitude from transform
            # This is a rough estimate - actual displacement varies spatially
            try:
                if hasattr(long.transform, 'data'):
                    params = long.transform.data()
                    if params is not None:
                        max_disp = params.abs().max().item()
                        max_displacements.append(max_disp)
                    else:
                        max_displacements.append(0)
                else:
                    max_displacements.append(0)
            except Exception:
                max_displacements.append(0)

        if max_displacements and max(max_displacements) > 0:
            ax.plot(frame_indices, max_displacements, 'g-o', linewidth=2, markersize=8)
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Max Displacement (normalized)')
            ax.set_title('Motion Magnitude Over Time')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "motion_magnitude.png", dpi=150)
            plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def _yaml_config_to_gui_dict(config_path: str) -> dict:
    """Convert YAML config file to GUI-compatible dict format.

    Maps YAML keys (io, pipeline) to GUI keys (inputs, registration, device).
    Keys that are identical (similarity, ffd, svffd, frame_selection) pass through.
    """
    from .config.config_loader import load_yaml_config

    raw = load_yaml_config(Path(config_path))
    gui_dict = {}

    # Map io → inputs + output
    if "io" in raw:
        io = raw["io"]
        inputs = {}
        if io.get("source_image"):
            inputs["image4d"] = io["source_image"]
        if io.get("target_image"):
            inputs["static"] = io["target_image"]
        if io.get("segmentation"):
            inputs["seg"] = io["segmentation"]
        if inputs:
            gui_dict["inputs"] = inputs
        if io.get("output_dir"):
            gui_dict["output"] = {"output_dir": io["output_dir"]}

    # Map pipeline → registration + device
    if "pipeline" in raw:
        pipe = raw["pipeline"]
        reg = {}
        if "model" in pipe:
            reg["model"] = pipe["model"]
        if pipe.get("skip_alignment"):
            reg["skip_alignment"] = True
        if pipe.get("skip_refinement"):
            reg["skip_refinement"] = True
        if reg:
            gui_dict["registration"] = reg
        if "device" in pipe:
            gui_dict["device"] = pipe["device"]

    # Pass-through keys (same structure in YAML and GUI)
    for key in ("similarity", "ffd", "svffd", "frame_selection",
                "rigid", "affine"):
        if key in raw:
            gui_dict[key] = raw[key]

    return gui_dict


def run_motion_pipeline(args):
    """Run the motion registration pipeline with phase tracking and resume capability"""

    # Handle --list-presets flag
    if hasattr(args, 'list_presets') and args.list_presets:
        presets = list_available_presets()
        print("\nAvailable configuration presets:")
        print("-" * 40)
        for preset in presets:
            print(f"  {preset}")
        print("-" * 40)
        print("\nUsage: dareg motion --config-preset <preset_name>")
        print("Example: dareg motion --config-preset svffd_standard --output auto")
        return 0

    # Handle --gui flag: Launch interactive configuration GUI
    if hasattr(args, 'gui') and args.gui:
        try:
            from .gui.config_gui import launch_config_gui
        except ImportError as e:
            print(f"Error: GUI requires PyQt5 or PyQt6. Install with: pip install PyQt5")
            print(f"Details: {e}")
            return 1

        # Pre-populate GUI with any provided config
        initial_config = None
        if args.json_config:
            initial_config = load_json_config(args.json_config)
        elif args.config:
            initial_config = _yaml_config_to_gui_dict(args.config)

        print("Launching configuration GUI...")
        gui_config, gui_output_path = launch_config_gui(initial_config)

        if gui_config is None:
            print("GUI cancelled by user.")
            return 0

        # Update args from GUI config
        if "inputs" in gui_config:
            if "image4d" in gui_config["inputs"]:
                args.image4d = gui_config["inputs"]["image4d"]
            if "static" in gui_config["inputs"]:
                args.static = gui_config["inputs"]["static"]
            if "seg" in gui_config["inputs"]:
                args.seg = gui_config["inputs"]["seg"]

        if "registration" in gui_config:
            if "model" in gui_config["registration"]:
                args.model = gui_config["registration"]["model"]
            if "skip_alignment" in gui_config["registration"]:
                args.skip_alignment = gui_config["registration"]["skip_alignment"]
            if "skip_refinement" in gui_config["registration"]:
                args.skip_refinement = gui_config["registration"]["skip_refinement"]

        if "device" in gui_config:
            args.device = gui_config["device"]

        if "frame_selection" in gui_config:
            if "start_frame" in gui_config["frame_selection"]:
                args.start_frame = gui_config["frame_selection"]["start_frame"]
            if "num_frames" in gui_config["frame_selection"]:
                args.num_frames = gui_config["frame_selection"]["num_frames"]

        # Use GUI output path
        if gui_output_path:
            args.output = str(gui_output_path)

        # Store GUI config as json_config for parameter passing
        json_config = gui_config
        print(f"GUI configuration loaded. Output: {args.output}")

    # Load and merge JSON config if provided (and not already from GUI)
    elif args.json_config:
        json_config = load_json_config(args.json_config)
        args = merge_args_with_json(args, json_config)
        print(f"Loaded config from: {args.json_config}")
    else:
        json_config = None

    # Detect inputs directory for project-specific config (needed for auto-naming)
    inputs_dir = None
    if args.image4d:
        inputs_dir = Path(args.image4d).parent

    # Pre-load config for auto-naming (will reload with full hierarchy later)
    preset = getattr(args, 'config_preset', None)
    reg_overrides_preview = {}
    if json_config:
        for key in ['similarity', 'rigid', 'affine', 'ffd', 'svffd', 'output']:
            if key in json_config:
                reg_overrides_preview[key] = json_config[key]

    preview_config = load_config(
        config_path=args.config,
        preset=preset,
        inputs_dir=inputs_dir,
        overrides=reg_overrides_preview if reg_overrides_preview else None
    )

    # Fall back to config values for args that weren't explicitly provided on CLI.
    # Pattern: CLI arg wins if explicitly set, otherwise use config value.
    if hasattr(preview_config, 'io'):
        io_cfg = preview_config.io
        if not args.image4d and io_cfg.source_image:
            args.image4d = io_cfg.source_image
            logger.info(f"Using source_image from config: {args.image4d}")
        if not args.static and io_cfg.target_image:
            args.static = io_cfg.target_image
            logger.info(f"Using target_image from config: {args.static}")
        if not args.seg and io_cfg.segmentation:
            args.seg = io_cfg.segmentation
            logger.info(f"Using segmentation from config: {args.seg}")
        if args.output == "./motion_output" and io_cfg.output_dir not in ("./motion_output", "./dareg_outputs"):
            args.output = io_cfg.output_dir
            logger.info(f"Using output_dir from config: {args.output}")

    if hasattr(preview_config, 'pipeline'):
        pipe_cfg = preview_config.pipeline
        if args.model == "rigid+affine+ffd" and pipe_cfg.model != "rigid+affine+ffd":
            args.model = pipe_cfg.model
            logger.info(f"Using model from config: {args.model}")
        if args.device == "auto" and pipe_cfg.device != "auto":
            args.device = pipe_cfg.device
            logger.info(f"Using device from config: {args.device}")
        if not args.skip_alignment and getattr(pipe_cfg, 'skip_alignment', False):
            args.skip_alignment = True
        if not args.skip_refinement and getattr(pipe_cfg, 'skip_refinement', False):
            args.skip_refinement = True
        if not args.verbose and getattr(pipe_cfg, 'verbose', False):
            args.verbose = True
        if not args.resume and getattr(pipe_cfg, 'resume', False):
            args.resume = True
        if not args.extract_frames and getattr(pipe_cfg, 'extract_frames', False):
            args.extract_frames = True
        if not args.no_viz and getattr(pipe_cfg, 'no_viz', False):
            args.no_viz = True

    # Validate required arguments (after JSON/GUI/config merge)
    if not args.image4d:
        raise ValueError(
            "--image4d is required (via CLI, config io.source_image, JSON config, or GUI)"
        )

    # Re-detect inputs_dir now that image4d may have been populated from config
    if inputs_dir is None and args.image4d:
        inputs_dir = Path(args.image4d).parent

    # Handle auto-naming for output directory
    output_arg = args.output
    if output_arg.lower() == "auto" or (hasattr(preview_config.output, 'auto_name') and preview_config.output.auto_name and output_arg == "./motion_output"):
        # Determine FFD model type from registration model string
        ffd_type = "svffd" if "svffd" in args.model.lower() else "ffd"
        auto_name = generate_output_name(preview_config, ffd_type)
        output_path = Path(f"./motion_output_{auto_name}")
        logger.info(f"Auto-generated output name: {output_path}")
    else:
        output_path = Path(output_arg)

    # Create output structure (needed for log file and phase manager)
    dirs = create_output_structure(output_path)

    # Initialize PhaseManager early for --status check
    # We'll update num_frames after loading the 4D image
    phase_manager = PhaseManager(
        output_dir=output_path,
        num_frames=0,  # Will be updated after loading 4D
        skip_alignment=args.skip_alignment or not args.static,
        skip_refinement=args.skip_refinement,
    )

    # Handle --status flag: show pipeline status and exit
    if hasattr(args, 'status') and args.status:
        # Try to load 4D image for accurate frame count (optional)
        try:
            image_4d = load_image_4d(
                args.image4d,
                start_frame=args.start_frame,
                num_frames=args.num_frames,
            )
            phase_manager.update_num_frames(image_4d.num_frames)
        except Exception:
            pass  # Status will show with 0 frames

        phase_manager.print_status_summary()
        return 0

    # Setup file logging to output directory
    from .utils.logging_config import setup_logging
    log_file = output_path / "pipeline_log.txt"
    setup_logging(level="DEBUG", log_file=log_file, module_name="DAREG")
    logger.info(f"Pipeline log file: {log_file}")

    # Setup device
    if args.device == "auto":
        device = get_device(None, verbose=True)
    else:
        device = get_device(args.device, verbose=True)

    # Build registration config overrides from JSON (use same config as preview)
    reg_overrides = reg_overrides_preview

    # Reuse the already-loaded config (same hierarchy)
    config = preview_config

    if preset:
        logger.info(f"Using config preset: {preset}")

    # Apply frame_selection from config if CLI args not explicitly set
    # (GUI and --json-config already handled above, this is for project YAML config)
    if hasattr(config, 'frame_selection') and config.frame_selection:
        fs = config.frame_selection
        # Only override if CLI arg is at default value
        if args.start_frame == 0 and fs.start_frame != 0:
            args.start_frame = fs.start_frame
            logger.info(f"Using start_frame from config: {args.start_frame}")
        if args.num_frames is None and fs.num_frames is not None:
            args.num_frames = fs.num_frames
            logger.info(f"Using num_frames from config: {args.num_frames}")

    # Load 4D image
    logger.info("Loading 4D dynamic image...")
    with Timer("Load 4D image"):
        image_4d = load_image_4d(
            args.image4d,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
        )

    logger.info(f"Loaded 4D image:")
    logger.info(f"  Frames: {image_4d.num_frames}")
    logger.info(f"  Frame shape: {image_4d.frame_shape}")
    logger.info(f"  Spacing: {image_4d.spacing}")

    # Update PhaseManager with actual frame count
    phase_manager.update_num_frames(image_4d.num_frames)

    # Show resume status if --resume flag is set
    resume_mode = hasattr(args, 'resume') and args.resume
    if resume_mode:
        logger.info("\n" + "=" * 70)
        logger.info("RESUME MODE: Detecting completed phases...")
        logger.info("=" * 70)
        phase_manager.print_status_summary()

        resume_phase = phase_manager.get_resume_phase()
        if resume_phase is None:
            logger.info("All phases already complete. Nothing to do.")
            return 0
        logger.info(f"Will resume from: {resume_phase.display_name}")

        # Clean up partial outputs from the resume phase (fresh restart)
        phase_manager.prepare_phase_for_restart(resume_phase)

        logger.info("=" * 70 + "\n")

    # Phase 0: Frame Extraction
    # NOTE: Frames are extracted automatically in save_results() as dynamic_extracted_frame_XXX.nii.gz
    # This phase only handles the --extract-frames-only flag for standalone extraction
    if args.extract_frames_only:
        if phase_manager.should_run_phase(Phase.FRAME_EXTRACTION):
            phase_manager.start_phase(Phase.FRAME_EXTRACTION)
            try:
                logger.info("Extracting individual frame files...")
                extract_frames_to_files(
                    args.image4d,
                    dirs["frames"],
                    prefix="dynamic_extracted_frame_",  # Use consistent naming
                    start_frame=args.start_frame,
                    num_frames=args.num_frames,
                )
                phase_manager.complete_phase(Phase.FRAME_EXTRACTION)
            except Exception as e:
                phase_manager.fail_phase(Phase.FRAME_EXTRACTION, str(e))
                raise

            logger.info("\n" + "=" * 70)
            logger.info("FRAME EXTRACTION COMPLETE (--extract-frames-only)")
            logger.info("=" * 70)
            logger.info(f"Output directory: {output_path}")
            logger.info(f"Frames extracted: {image_4d.num_frames}")
            logger.info(f"Frame files: {dirs['frames']}/dynamic_extracted_frame_*.nii.gz")
            logger.info("=" * 70)
            return 0
        else:
            logger.info("Frame extraction already complete (resume mode). Exiting.")
            return 0

    # Load static image (optional)
    static_image = None
    if args.static:
        logger.info(f"Loading static image: {args.static}")
        static_image, _, _ = load_image(args.static)  # load_image returns (Image, sitk, metadata)

    # Load segmentation (optional)
    segmentation = None
    if args.seg:
        logger.info(f"Loading segmentation: {args.seg}")
        segmentation, _, _ = load_image(args.seg)  # load_image returns (Image, sitk, metadata)

    # Load ROI mask (optional) - for restricting registration computation region
    # Can be provided via CLI (--roi-mask) or config file (similarity.roi_mask)
    roi_mask_tensor = None
    roi_mask_path = getattr(args, 'roi_mask', None)
    if roi_mask_path is None and config and hasattr(config, 'similarity'):
        roi_mask_path = getattr(config.similarity, 'roi_mask', None)
    if roi_mask_path:
        logger.info(f"Loading ROI mask: {roi_mask_path}")
        roi_mask_image, _, _ = load_image(roi_mask_path)
        roi_mask_tensor = roi_mask_image.tensor().to(device)
        logger.info(f"  ROI mask loaded: {roi_mask_tensor.sum().item():.0f} non-zero voxels")

    # Create motion registration pipeline
    motion_reg = MotionRegistration(
        device=str(device),
        config=config,
        registration_model=args.model,
        roi_mask=roi_mask_tensor,
    )

    # Run full pipeline (with phase tracking)
    logger.info("\n" + "=" * 70)
    logger.info("STARTING MOTION REGISTRATION PIPELINE")
    if resume_mode:
        logger.info("(Resume mode: skipping completed phases)")
    logger.info("=" * 70)

    # Create progress tracker for monitoring
    progress_tracker = ProgressTracker(
        output_dir=output_path,
        enable_file_output=True  # Enables progress.json for external monitoring
    )

    with Timer("Total motion pipeline"):
        result = motion_reg.run_full_pipeline(
            image_4d=image_4d,
            static_image=static_image,
            segmentation=segmentation,
            skip_alignment=args.skip_alignment,
            skip_refinement=args.skip_refinement,
            output_dir=output_path,  # Enable incremental saving
            progress_tracker=progress_tracker,
        )

    # Mark remaining phases as complete
    phase_manager.complete_phase(Phase.PAIRWISE)
    phase_manager.complete_phase(Phase.COMPOSE)
    if not args.skip_refinement:
        phase_manager.complete_phase(Phase.REFINE)
    else:
        phase_manager.skip_phase(Phase.REFINE, "skipped by --skip-refinement")
    if segmentation is not None:
        phase_manager.complete_phase(Phase.PROPAGATE)

    # Save results
    logger.info("Saving results...")
    motion_reg.save_results(result, output_path, image_4d, static_image, segmentation)

    # Create visualizations
    if not args.no_viz:
        create_motion_visualizations(result, image_4d, dirs["visualizations"])

    # Mark finalize phase complete
    phase_manager.complete_phase(Phase.FINALIZE)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MOTION REGISTRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Frames processed: {image_4d.num_frames}")
    logger.info(f"Pairwise registrations: {len(result.pairwise_transforms)}")
    logger.info(f"Longitudinal transforms: {len(result.longitudinal_transforms)}")
    if result.segmentation_sequence:
        logger.info(f"Segmentations generated: {len(result.segmentation_sequence)}")
    logger.info(f"Total time: {result.total_time:.1f}s")
    logger.info("=" * 70)

    # Show final phase status
    phase_manager.print_status_summary()

    return 0


def main():
    """Main entry point"""
    args = parse_args()

    try:
        return run_motion_pipeline(args)
    except Exception as e:
        logger.error(f"Motion registration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
