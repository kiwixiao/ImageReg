#!/usr/bin/env python3
"""
DAREG - Deepali Registration Pipeline

Main entry point for medical image registration using deepali.
Supports rigid, affine, and FFD/SVFFD registration stages.

Usage:
    dareg register --source source.nii.gz --target target.nii.gz --output ./output
    dareg register --config config.yaml
"""

import sys
import os
import platform
import argparse
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

from .config import load_config, RegistrationConfig
from .data import ImagePair, load_image, save_image
from .preprocessing import (
    normalize_intensity,
    match_histograms,
    create_common_grid,
    compute_bounding_box,
    create_pyramid,
    PyramidLevel,
)
from .registration import RigidRegistration, AffineRegistration, FFDRegistration
from .postprocessing import (
    apply_transform,
    warp_image,
    transform_segmentation,
    compute_quality_metrics,
)
from .visualization import (
    plot_side_by_side,
    plot_overlay,
    plot_convergence,
    plot_multi_level_convergence,
    plot_grid_deformation,
    plot_displacement_field,
    create_pdf_report,
    RegistrationReport,
)
from .utils.logging_config import get_logger, Timer
from .utils.device import DeviceManager

logger = get_logger("main")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DAREG - Medical Image Registration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic registration with default config
  dareg register --source moving.nii.gz --target fixed.nii.gz

  # Full pipeline with all stages
  dareg register --source moving.nii.gz --target fixed.nii.gz --method rigid+affine+svffd

  # Use custom config file
  dareg register --config registration_config.yaml
        """,
    )

    # Input/Output
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Path to source (moving) image",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Path to target (fixed) image",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        help="Optional path to segmentation to transform",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./dareg_output",
        help="Output directory (default: ./dareg_output)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file",
    )

    # Registration method
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="rigid+affine+ffd",
        choices=["rigid", "rigid+affine", "rigid+affine+ffd", "rigid+affine+svffd"],
        help="Registration method (default: rigid+affine+ffd)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device (default: auto)",
    )

    # Visualization
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization output",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF report generation",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def create_output_dirs(output_path: Path) -> dict:
    """Create output directory structure"""
    dirs = {
        "root": output_path,
        "transforms": output_path / "transforms",
        "intermediate": output_path / "intermediate_results",
        "final": output_path / "final_results",
        "debug": output_path / "debug_analysis",
        "segmentation": output_path / "segmentation_results",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def run_pipeline(args):
    """Run the registration pipeline"""

    # Setup device
    device_mgr = DeviceManager()
    if args.device == "auto":
        device = device_mgr.get_best_device()
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()  # Default config

    # Create output directories
    output_path = Path(args.output)
    dirs = create_output_dirs(output_path)

    # Initialize report
    report = RegistrationReport(
        title="DAREG Registration Report",
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=config.__dict__ if hasattr(config, "__dict__") else {},
    )

    # Load images
    logger.info("Loading images...")

    with Timer("Image loading"):
        source_path = args.source
        target_path = args.target

        # If using config, get paths from config
        if args.config and hasattr(config, "source_image"):
            config_dir = Path(args.config).parent
            source_path = source_path or str(config_dir / config.source_image)
            target_path = target_path or str(config_dir / config.target_image)

        if not source_path or not target_path:
            raise ValueError("Source and target images must be specified")

        source = load_image(source_path)
        target = load_image(target_path)

        logger.info(f"Source: {source.tensor().shape}, spacing: {source.grid().spacing()}")
        logger.info(f"Target: {target.tensor().shape}, spacing: {target.grid().spacing()}")

    # Store in report
    report.source = source
    report.target = target

    # Preprocessing
    logger.info("Preprocessing...")

    with Timer("Preprocessing"):
        # Normalize intensities
        normalizer = ImageNormalizer()
        source_norm = normalizer.normalize(source)
        target_norm = normalizer.normalize(target)

        # Create common grid
        grid_mgr = GridManager()
        common_grid = grid_mgr.create_common_grid(source_norm, target_norm)

        logger.info(f"Common grid: {common_grid.shape}")

    # Create ImagePair for tracking
    image_pair = ImagePair(
        source_original=source,
        target_original=target,
        source_normalized=source_norm,
        target_normalized=target_norm,
        common_grid=common_grid,
    )

    # Parse method stages
    stages = args.method.split("+")
    logger.info(f"Registration stages: {stages}")

    # Current warped source (updated after each stage)
    current_source = source_norm
    transforms = []

    # Stage 1: Rigid registration
    if "rigid" in stages:
        logger.info("=" * 60)
        logger.info("Stage 1: RIGID REGISTRATION")
        logger.info("=" * 60)

        with Timer("Rigid registration"):
            rigid_reg = RigidRegistration(device=str(device), config=config)
            rigid_result = rigid_reg.register(current_source, target_norm, common_grid)

            current_source = rigid_result.warped_source
            transforms.append(("rigid", rigid_result.transform))

            report.warped_rigid = rigid_result.warped_source
            report.metrics["rigid"] = rigid_result.metrics
            report.loss_history["rigid"] = rigid_result.loss_history

            # Save rigid transform
            torch.save(rigid_result.transform.state_dict(), dirs["transforms"] / "rigid_transform.pth")

            logger.info(f"Rigid final loss: {rigid_result.final_loss:.6f}")

    # Stage 2: Affine registration
    if "affine" in stages:
        logger.info("=" * 60)
        logger.info("Stage 2: AFFINE REGISTRATION")
        logger.info("=" * 60)

        with Timer("Affine registration"):
            affine_reg = AffineRegistration(device=str(device), config=config)

            # Initialize from rigid if available
            init_transform = transforms[-1][1] if transforms else None
            affine_result = affine_reg.register(
                current_source, target_norm, common_grid,
                initial_transform=init_transform
            )

            current_source = affine_result.warped_source
            transforms.append(("affine", affine_result.transform))

            report.warped_affine = affine_result.warped_source
            report.metrics["affine"] = affine_result.metrics
            report.loss_history["affine"] = affine_result.loss_history

            # Save affine transform
            torch.save(affine_result.transform.state_dict(), dirs["transforms"] / "affine_transform.pth")

            logger.info(f"Affine final loss: {affine_result.final_loss:.6f}")

    # Stage 3: FFD/SVFFD registration
    if "ffd" in stages or "svffd" in stages:
        ffd_type = "svffd" if "svffd" in stages else "ffd"

        logger.info("=" * 60)
        logger.info(f"Stage 3: {ffd_type.upper()} REGISTRATION")
        logger.info("=" * 60)

        with Timer(f"{ffd_type.upper()} registration"):
            ffd_reg = FFDRegistration(
                device=str(device),
                config=config,
                use_svffd=(ffd_type == "svffd"),
            )

            # Initialize from previous transform
            init_transform = transforms[-1][1] if transforms else None
            ffd_result = ffd_reg.register(
                current_source, target_norm, common_grid,
                initial_transform=init_transform
            )

            current_source = ffd_result.warped_source
            transforms.append((ffd_type, ffd_result.transform))

            report.warped_ffd = ffd_result.warped_source
            report.metrics[ffd_type] = ffd_result.metrics
            report.loss_history[ffd_type] = ffd_result.loss_history

            # Save FFD transform
            torch.save(ffd_result.transform.state_dict(), dirs["transforms"] / f"{ffd_type}_transform.pth")

            logger.info(f"{ffd_type.upper()} final loss: {ffd_result.final_loss:.6f}")

    # Postprocessing
    logger.info("=" * 60)
    logger.info("POSTPROCESSING")
    logger.info("=" * 60)

    with Timer("Postprocessing"):
        # Compose all transforms
        img_transformer = ImageTransformer(device=str(device))

        # Apply transforms to original resolution source
        final_warped = source
        for name, transform in transforms:
            final_warped = img_transformer.apply_transform(final_warped, transform, common_grid)

        # Save final results
        save_image(final_warped, dirs["final"] / "source_warped_final.nii.gz", source)
        save_image(target, dirs["final"] / "target_reference.nii.gz", target)
        save_image(source, dirs["final"] / "source_reference.nii.gz", source)

        logger.info(f"Saved final results to {dirs['final']}")

    # Transform segmentation if provided
    if args.segmentation:
        logger.info("Transforming segmentation...")

        with Timer("Segmentation transform"):
            seg = load_image(args.segmentation)
            seg_transformer = SegmentationTransformer(device=str(device))

            seg_warped = seg
            for name, transform in transforms:
                seg_warped = seg_transformer.transform_segmentation(seg_warped, transform, common_grid)

            save_image(seg_warped, dirs["segmentation"] / "segmentation_warped.nii.gz", seg)
            logger.info(f"Saved warped segmentation to {dirs['segmentation']}")

    # Compute quality metrics
    logger.info("Computing quality metrics...")
    quality = QualityMetrics(device=str(device))
    final_metrics = quality.compute_all(current_source, target_norm)
    logger.info(f"Final metrics: {final_metrics}")

    # Visualization
    if not args.no_viz:
        logger.info("Creating visualizations...")

        with Timer("Visualization"):
            # Side-by-side comparison
            plot_side_by_side(
                source_norm,
                target_norm,
                current_source,
                output_path=dirs["debug"] / "registration_comparison.png",
                title="Registration Comparison",
            )

            # Overlay
            plot_overlay(
                current_source,
                target_norm,
                output_path=dirs["debug"] / "registration_overlay.png",
                title="Final Overlay (Red=Warped, Green=Target)",
            )

            # Convergence plots
            for stage_name, stage_losses in report.loss_history.items():
                if isinstance(stage_losses, dict):
                    plot_multi_level_convergence(
                        stage_losses,
                        output_path=dirs["debug"] / f"{stage_name}_convergence.png",
                        title=f"{stage_name.capitalize()} Convergence",
                    )
                elif isinstance(stage_losses, list):
                    plot_convergence(
                        stage_losses,
                        output_path=dirs["debug"] / f"{stage_name}_convergence.png",
                        title=f"{stage_name.capitalize()} Convergence",
                    )

            # FFD-specific visualizations
            if "ffd" in stages or "svffd" in stages:
                ffd_name = "svffd" if "svffd" in stages else "ffd"
                ffd_transform = transforms[-1][1]

                plot_grid_deformation(
                    ffd_transform,
                    common_grid,
                    output_path=dirs["debug"] / f"{ffd_name}_grid_deformation.png",
                    title=f"{ffd_name.upper()} Grid Deformation",
                )

            logger.info(f"Saved visualizations to {dirs['debug']}")

    # PDF Report
    if not args.no_pdf:
        logger.info("Generating PDF report...")

        with Timer("PDF report"):
            pdf_path = output_path / "registration_report.pdf"
            create_pdf_report(report, pdf_path)
            logger.info(f"Saved PDF report: {pdf_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("REGISTRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Stages completed: {' -> '.join(stages)}")
    logger.info(f"Final NMI: {final_metrics.get('nmi', 'N/A')}")

    return 0


def main():
    """Main entry point"""
    args = parse_args()

    try:
        return run_pipeline(args)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
