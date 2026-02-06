# DAREG

Medical image registration system replicating [MIRTK](https://mirtk.github.io/) functionality using Python and the [deepali](https://github.com/BioMedIA/deepali) library. Designed for 4D dynamic MRI motion tracking with segmentation propagation.

## Installation

```bash
# From local checkout
pip install .

# Editable mode (for development)
pip install -e .

# With optional STL/video support
pip install -e ".[all]"
```

## Quick Start

### 4D Motion Tracking

```bash
dareg motion \
    --image4d dynamic.nii \
    --static static.nii \
    --seg airway_seg.nii.gz \
    --output ./motion_output \
    --device cpu --verbose
```

### Single-Pair Registration

```bash
dareg register \
    --source moving.nii.gz \
    --target fixed.nii.gz \
    --method rigid+affine+ffd \
    --output ./output
```

### Post-Processing (STL, Video)

```bash
dareg postprocess \
    --output_dir ./motion_output \
    --stl --video
```

### CLI Help

```bash
dareg --help
dareg motion --help
dareg register --help
dareg postprocess --help
```

## Configuration

DAREG supports YAML configuration files for registration parameters:

```bash
dareg motion \
    --image4d dynamic.nii \
    --config registration_config.yaml \
    --output ./output
```

See `dareg/configs/` for preset configurations (FFD, SVFFD variants).

## Built With

- **[hf-deepali](https://github.com/BioMedIA/deepali)** (Apache-2.0, BioMedIA) - Differentiable medical image registration library providing spatial transforms, similarity metrics, and regularization. DAREG builds its registration pipeline on top of deepali's core components.
- **PyTorch** - Optimization and tensor operations
- **nibabel** - 4D/5D NIfTI loading
- **SimpleITK** - 3D image I/O and coordinate system handling

## Documentation

- `dareg/docs/MOTION_REGISTRATION_ARCHITECTURE.md` - Full pipeline architecture
- `dareg/docs/MIRTK_EQUIVALENCE.md` - MIRTK to DAREG command mapping
- `dareg/docs/ENHANCEMENT_PLAN.md` - Feature tracker

## License

Apache-2.0
