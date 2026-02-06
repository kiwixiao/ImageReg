# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image registration system replicating MIRTK (Medical Image Registration ToolKit) functionality using Python and the deepali library. Designed for 4D dynamic MRI motion tracking with segmentation propagation.

**Primary Goal**: Register 4D dynamic MRI to high-resolution static images and propagate segmentations through all time frames.

## Common Development Commands

```bash
# Primary Pipeline - DAREG Motion Registration (with YAML config)
python -m DAREG.main_motion \
    --image4d inputs_OSAMRI016/OSAMRI016_2601_Dynamic.nii \
    --static inputs_OSAMRI016/OSAMRI016_2501_static.nii \
    --seg inputs_OSAMRI016/OSAMRI016_2501_airway_seg.nii.gz \
    --config inputs_OSAMRI016/registration_config.yaml \
    --output ./motion_output \
    --device cpu --verbose

# With multi-threading (14 cores)
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 python -m DAREG.main_motion ...

# Subset of frames (e.g., frames 0-3)
python -m DAREG.main_motion ... --start-frame 0 --num-frames 4

# Available registration models:
# rigid+affine+ffd   - Standard B-spline FFD (faster)
# rigid+affine+svffd - Diffeomorphic SVFFD (topology-preserving)
```

## Core Architecture

### DAREG Pipeline (`DAREG/`)

The primary motion registration pipeline with modular architecture:

```
DAREG/
├── main_motion.py          # Entry point for motion registration
├── main_postprocess.py     # Post-processing (STL generation, etc.)
├── main.py                 # Generic registration entry point
├── registration/
│   ├── base.py             # BaseRegistration with NMI, foreground masking
│   ├── rigid.py            # 6-DOF rigid registration (CG optimizer)
│   ├── affine.py           # 12-DOF affine registration (CG optimizer)
│   ├── ffd.py              # B-spline FFD with MIRTK-style regularization
│   ├── svffd.py            # Diffeomorphic SVFFD (velocity field + ExpFlow)
│   ├── optimizers.py       # ConjugateGradientOptimizer (MIRTK-equivalent)
│   ├── motion.py           # Motion pipeline orchestrator
│   └── composer.py         # Transform composition (SequentialTransform)
├── preprocessing/
│   ├── normalizer.py       # normalize_intensity, match_histograms
│   ├── pyramid.py          # create_pyramid, PyramidLevel
│   └── grid_manager.py     # create_common_grid, compute_bounding_box
├── postprocessing/
│   ├── segmentation.py          # transform_segmentation, ApproximateInverseTransform
│   ├── transformer.py           # apply_transform, warp_image
│   ├── quality_metrics.py       # compute_quality_metrics, compute_nmi, compute_dice
│   ├── stl_generator.py         # STLGenerator, marching_cubes + smoothing
│   └── temporal_interpolation.py # TemporalInterpolator for B-spline motion
├── data/
│   ├── image_4d.py         # Image4D, load_image_4d (uses nibabel, not SITK)
│   ├── image_pair.py       # ImagePair container
│   ├── loader.py           # load_image, load_image_pair
│   └── saver.py            # save_image, save_transform
├── visualization/
│   ├── alignment_overlay.py   # Alignment progression visualization
│   ├── convergence.py         # plot_convergence, plot_multi_level_convergence
│   ├── deformation.py         # plot_grid_deformation, plot_displacement_field
│   ├── pdf_report.py          # create_pdf_report, RegistrationReport
│   ├── plotting.py            # plot_side_by_side, plot_overlay
│   └── segmentation_viewer.py # Multi-slice sagittal/axial/coronal views
├── config/
│   └── config_loader.py    # Configuration loading
├── utils/
│   ├── logging_config.py   # Logging setup
│   ├── device.py           # Device management (CPU/GPU)
│   └── phase_manager.py    # Pipeline phase tracking
└── docs/
    ├── MOTION_REGISTRATION_ARCHITECTURE.md
    ├── MIRTK_EQUIVALENCE.md
    ├── ENHANCEMENT_PLAN.md     # Planned/completed improvements tracker
    └── DEEPALI_INTEGRATION.md
```

### Motion Registration Flow

```
1. ALIGNMENT (Static → Frame 0)
   Rigid(6DOF) → Affine(12DOF) → FFD/SVFFD
   Output: aligned_static_original_resolution.nii.gz

2. PAIRWISE (Frame N → Frame N-1)
   FFD registration between consecutive frames
   Output: pairwise/pairwise_N_M.pth

3. COMPOSE LONGITUDINAL
   Chain: T_01 ∘ T_12 ∘ T_23 = T_03
   Output: longitudinal/longitudinal_0_to_N.pth

4. PROPAGATE SEGMENTATION
   Apply INVERSE transforms (Frame 0 → Frame N)
   Uses nearest neighbor interpolation
   Output: segmentations/seg_frame_NNN.nii.gz
```

### Reference Implementations

- **`deepali/`**: Local deepali library source - always verify API usage here first
- **`MIRTK/`**: MIRTK C++ source code for algorithm reference
- **`mirtk_binary_reg_pipeline_demo/`**: Working MIRTK binary pipeline (ground truth for output comparison)

## Key Technical Concepts

### 4D/5D NIfTI Handling

Dynamic MRI often has 5D shape `(X, Y, Z, 1, T)` with singleton dim:
```python
# WRONG: SimpleITK reads 5D as 3D
sitk.ReadImage('Dynamic.nii').GetSize()  # (192, 192, 12) - INCORRECT

# CORRECT: Use nibabel
nib.load('Dynamic.nii').shape  # (192, 192, 12, 1, 90) - CORRECT
```

DAREG's `image_4d.py` handles this automatically. Also accepts a directory of sorted 3D NIfTI files as input.

### Foreground Masking (MIRTK FG_Overlap)

MIRTK computes similarity only where both images have valid foreground:
```python
# 1. Normalize to [0,1], threshold for foreground
# 2. Skip zero-gradient voxels (MIRTK line 195)
# 3. Mask = source_fg AND target_fg AND has_gradient
overlap_mask = self._compute_foreground_overlap_mask(source, target, threshold=0.01)
```

This prevents artifacts at image boundaries and in background regions.

### FFD Inverse for Segmentation Propagation

FFD lacks analytical inverse. DAREG uses Newton-Raphson approximation (`ApproximateInverseTransform`):
```python
# Fixed-point iteration: x_{n+1} = y - u(x_n)
# Inherits from NonRigidTransform for SequentialTransform compatibility
```

SVFFD has analytical inverse (negate velocity field).

### Tensor Dimension Conventions

```python
# deepali requires 5D for registration: [N, C, D, H, W]
if tensor.dim() == 3:
    tensor = tensor.unsqueeze(0).unsqueeze(0)

# Segmentations: ALWAYS use nearest neighbor interpolation
spatial.ImageTransformer(transform, sampling="nearest")
```

### Device Limitations

MPS (Apple Silicon) doesn't support `grid_sampler_3d`. Always use CPU:
```bash
--device cpu
```

## Configuration

### YAML Config (registration_config.yaml)
```yaml
# Subject-specific config (place in input folder for auto-loading)
similarity:
  metric: nmi
  num_bins: 64           # MIRTK uses adaptive 16-64 bins; 64 matches MIRTK max
  foreground_threshold: 0.01

rigid:
  pyramid_levels: 4
  iterations_per_level: [100, 100, 100, 100]
  learning_rates_per_level: [0.01, 0.01, 0.01, 0.01]

affine:
  pyramid_levels: 4
  iterations_per_level: [100, 100, 100, 100]
  learning_rates_per_level: [0.005, 0.005, 0.005, 0.005]

ffd:
  control_point_spacing: 4
  pyramid_levels: 4
  iterations_per_level: [100, 100, 100, 100]
  regularization:
    bending_weight: 0.0005    # MIRTK: 0.001
    diffusion_weight: 0.00025  # MIRTK: 0.0005
```

## Output Structure

```
motion_output/
├── frames/                       # Extracted 3D frames
│   └── dynamic_extracted_frame_NNN.nii.gz
├── alignment/                    # Static→Frame0 registration
│   ├── source_after_*.nii.gz     # Intermediate results
│   ├── aligned_static_original_resolution.nii.gz
│   └── aligned_segmentation.nii.gz
├── pairwise/                     # Frame-to-frame transforms (.pth)
├── longitudinal/                 # Composed Frame N → Frame 0 (.pth)
├── segmentations/                # Propagated segmentations (high-res)
│   └── seg_frame_NNN.nii.gz
├── segmentations_frame_grid/     # Segmentations at frame resolution
├── final_results/                # Static moved to each frame
│   └── static_image_moved_to_frame_NNN.nii.gz
├── stl_surfaces/                 # Optional STL meshes
├── visualizations/
│   ├── segmentation_progression_sagittal.png
│   ├── segmentation_progression_axial.png
│   └── segmentation_progression_coronal.png
└── pipeline_log.txt
```

## Development Guidelines

### API Reference Priority

When using deepali, always check `deepali/src/` source first:
- `deepali.spatial`: Transforms, ImageTransformer
- `deepali.losses.functional`: nmi_loss, bending_loss, diffusion_loss
- `deepali.data.Image`: Medical image class with grid handling
- `deepali.core.Grid`: Coordinate system representation

### Transform Inheritance

Custom transforms must inherit from `SpatialTransform` (or subclass like `NonRigidTransform`) to work with `SequentialTransform`:
```python
class ApproximateInverseTransform(NonRigidTransform):
    def update(self):
        # Register buffer 'u' for displacement field
        self.register_buffer('u', inverse_disp)
```

### World Coordinate Registration

All registration operates in physical mm coordinates, not voxel indices:
- Resolution independence across different image spacings
- Proper handling of anisotropic voxels
- ITK-SNAP compatible overlays

## MIRTK Equivalence Status

**Status: 100% Logical Equivalence Achieved** (verified Feb 2026)

| Component | MIRTK | DAREG | Status |
|-----------|-------|-------|--------|
| Optimizer | Conjugate Gradient (Polak-Ribière) | ConjugateGradientOptimizer | ✅ |
| NMI Similarity | Adaptive 16-64 bins, FG_Overlap masking | Configurable bins (default 64), FG_Overlap | ✅ |
| Bending Energy | 0.001 weight | Configurable (default 0.0005) | ✅ |
| Laplacian Energy | 0.0005 weight | `_compute_laplacian_loss()` | ✅ |
| FFD Inverse | Newton-Raphson | `ApproximateInverseTransform` | ✅ |
| SVFFD Smoothing | Gaussian velocity smoothing | `_smooth_velocity_field()` | ✅ |
| Pipeline Direction | Frame[j] → Frame[j-1] | Same convention | ✅ |

See `planning/MIRTK_DEEPALI_EQUIVALENCE_PLAN.md` for detailed verification.

## Dependencies

- **PyTorch**: Optimization and tensor operations
- **nibabel**: 4D/5D NIfTI loading (not SimpleITK for dynamic images)
- **deepali**: Local installation in `deepali/src/` (auto-added to path)
- **SimpleITK**: 3D image I/O and coordinate system handling
- **matplotlib**: Visualization generation
- **scikit-image**: marching_cubes for STL generation (optional)
- **trimesh** or **numpy-stl**: STL I/O (optional, for STL export)

## Reference Documents

- `DAREG/docs/MOTION_REGISTRATION_ARCHITECTURE.md` - Full pipeline architecture
- `DAREG/docs/MIRTK_EQUIVALENCE.md` - MIRTK to DAREG command mapping
- `DAREG/docs/ENHANCEMENT_PLAN.md` - Feature tracker (STL, temporal interpolation, etc.)
- `DAREG/docs/DEEPALI_INTEGRATION.md` - Deepali library integration notes
