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
    python -m DAREG.main_motion --image4d dynamic.nii.gz --output ./output

    # With alignment to static image
    python -m DAREG.main_motion --image4d dynamic.nii.gz --static static.nii.gz --seg seg.nii.gz

    # Extract specific frames
    python -m DAREG.main_motion --image4d dynamic.nii.gz --start-frame 5 --num-frames 10
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add deepali to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deepali" / "src"))

import torch

from .data import load_image
from .data.image_4d import load_image_4d, extract_frames_to_files, Image4D
from .registration.motion import MotionRegistration, MotionResult
from .config import load_config
from .visualization import (
    plot_side_by_side,
    plot_convergence,
    create_pdf_report,
    RegistrationReport,
)
from .utils.logging_config import get_logger, Timer
from .utils.device import get_device

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
  python -m DAREG.main_motion --image4d breathing_4d.nii.gz -o ./motion_output

  # With static alignment and segmentation
  python -m DAREG.main_motion \\
      --image4d breathing_4d.nii.gz \\
      --static highres_static.nii.gz \\
      --seg airway_seg.nii.gz \\
      --model rigid+affine+svffd \\
      -o ./motion_output

  # Extract specific frame range
  python -m DAREG.main_motion \\
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
        required=True,
        help="Path to 4D dynamic image (NIfTI)",
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
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device (default: cpu)",
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

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


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

    # 2. Segmentation progression (if available)
    if result.segmentation_sequence and len(result.segmentation_sequence) > 1:
        num_segs = min(len(result.segmentation_sequence), 8)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        indices = np.linspace(0, len(result.segmentation_sequence) - 1, num_segs, dtype=int)

        for i, idx in enumerate(indices):
            seg = result.segmentation_sequence[idx]
            seg_np = seg.tensor().squeeze().cpu().numpy()

            # Get middle slice
            mid_slice = seg_np.shape[0] // 2
            slice_2d = seg_np[mid_slice, :, :]

            axes[i].imshow(slice_2d, cmap='viridis')
            axes[i].set_title(f'Frame {idx}')
            axes[i].axis('off')

        # Hide unused axes
        for i in range(num_segs, len(axes)):
            axes[i].axis('off')

        fig.suptitle('Segmentation Propagation Through Frames', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "segmentation_progression.png", dpi=150)
        plt.close()

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


def run_motion_pipeline(args):
    """Run the motion registration pipeline"""

    # Create output structure first (needed for log file)
    output_path = Path(args.output)
    dirs = create_output_structure(output_path)

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

    # Load configuration
    config = load_config(args.config) if args.config else load_config()

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

    # Optionally extract frames to files
    if args.extract_frames:
        logger.info("Extracting individual frame files...")
        extract_frames_to_files(
            args.image4d,
            dirs["frames"],
            prefix="frame_",
            start_frame=args.start_frame,
            num_frames=args.num_frames,
        )

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

    # Create motion registration pipeline
    motion_reg = MotionRegistration(
        device=str(device),
        config=config,
        registration_model=args.model,
    )

    # Run full pipeline
    logger.info("\n" + "=" * 70)
    logger.info("STARTING MOTION REGISTRATION PIPELINE")
    logger.info("=" * 70)

    with Timer("Total motion pipeline"):
        result = motion_reg.run_full_pipeline(
            image_4d=image_4d,
            static_image=static_image,
            segmentation=segmentation,
            skip_alignment=args.skip_alignment,
            skip_refinement=args.skip_refinement,
            output_dir=output_path,  # Enable incremental saving
        )

    # Save results
    logger.info("Saving results...")
    motion_reg.save_results(result, output_path, image_4d)

    # Create visualizations
    if not args.no_viz:
        create_motion_visualizations(result, image_4d, dirs["visualizations"])

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
