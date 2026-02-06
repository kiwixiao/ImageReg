# DAREG Documentation

## Overview

DAREG (Deepali-based Registration) is a Python implementation of medical image registration that replicates MIRTK functionality using the deepali library.

## Documentation Files

| Document | Description |
|----------|-------------|
| [MOTION_REGISTRATION_ARCHITECTURE.md](MOTION_REGISTRATION_ARCHITECTURE.md) | Full pipeline architecture, 4D/5D handling, workflow |
| [MIRTK_EQUIVALENCE.md](MIRTK_EQUIVALENCE.md) | MIRTK to DAREG command/config mapping |
| [DEEPALI_INTEGRATION.md](DEEPALI_INTEGRATION.md) | Deepali library usage, common pitfalls |

## Quick Reference

### Run Motion Registration

```bash
dareg motion \
    --image4d dynamic.nii.gz \
    --static static.nii.gz \
    --seg segmentation.nii.gz \
    --model rigid+affine+ffd \
    --output ./output \
    --device cpu
```

### Key Lessons Learned

1. **Use nibabel for 5D NIfTI** - SimpleITK doesn't handle 5D correctly
2. **Use deepali built-ins** - `Image.read()`, `image.sample()` handle coordinates properly
3. **Foreground masking is critical** - MIRTK FG_Overlap prevents boundary artifacts
4. **MPS doesn't support grid_sampler_3d** - Use CPU for 3D medical imaging
5. **Sequential pairwise is stable** - Better than direct long-range for motion tracking

### Image Format Notes

```
5D NIfTI: (X, Y, Z, 1, T) - 90 time frames with singleton dim
         Shape (192, 192, 12, 1, 90)
         └─ nibabel handles correctly
         └─ SimpleITK reads as 3D (WRONG!)
```

### Foreground Masking (MIRTK Equivalent)

```python
# MIRTK FG_Overlap mode:
# 1. Normalize intensities to [0,1]
# 2. Foreground = intensity > threshold
# 3. Skip zero-gradient voxels (MIRTK line 195)
# 4. Mask = source_fg AND target_fg AND has_gradient
```
