# MIRTK Equivalence Guide

## Purpose

This document maps MIRTK commands and concepts to their DAREG Python equivalents.

## Command Mapping

### Image Extraction

**MIRTK:**
```bash
mirtk extract-image-volume dynamic.nii frame_%03d.nii.gz -t 0 -n 10
```

**DAREG:**
```python
from DAREG.data.image_4d import load_image_4d, extract_frames_to_files

# Load and extract frames
image_4d = load_image_4d("dynamic.nii", start_frame=0, num_frames=10)

# Or save to files
extract_frames_to_files("dynamic.nii", output_dir, prefix="frame_", start_frame=0, num_frames=10)
```

### Registration

**MIRTK:**
```bash
mirtk register source.nii target.nii -model Rigid+Affine+FFD -dofout transform.dof
```

**DAREG:**
```python
from DAREG.registration.rigid import RigidRegistration
from DAREG.registration.affine import AffineRegistration
from DAREG.registration.ffd import FFDRegistration

# Sequential registration
rigid_result = RigidRegistration(device="cpu").register(source, target)
affine_result = AffineRegistration(device="cpu").register(rigid_result.warped_source, target)
ffd_result = FFDRegistration(device="cpu").register(affine_result.warped_source, target)
```

### Transform Composition

**MIRTK:**
```bash
mirtk compose-dofs dof1.dof dof2.dof dof3.dof -output composed.dof
```

**DAREG:**
```python
from deepali.spatial import SequentialTransform

composed = SequentialTransform(transform1, transform2, transform3)
```

### Image Transformation

**MIRTK:**
```bash
mirtk transform-image source.nii output.nii -dofin transform.dof -interp Linear
mirtk transform-image seg.nii output_seg.nii -dofin transform.dof -interp NN
```

**DAREG:**
```python
from deepali.spatial import ImageTransformer

# Linear interpolation for images
transformer = ImageTransformer(transform)
warped = transformer(source_tensor)

# Nearest neighbor for segmentations
transformer_nn = ImageTransformer(transform, mode='nearest')
warped_seg = transformer_nn(seg_tensor)
```

## Configuration Mapping

### MIRTK register.cfg

```
# MIRTK Configuration
Transformation model   = FFD
Control point spacing  = 4
Energy function        = NMI(I(1), I(2:end) o T) + 0.001 BE(T) + 0.0005 LE(T)
No. of bins            = 64
No. of levels          = 4
Maximum no. of iterations = 100
```

### DAREG Equivalent

```yaml
# DAREG Configuration
ffd:
  control_point_spacing: 4  # mm
  bending_energy_weight: 0.001  # BE(T)
  linear_energy_weight: 0.0005  # LE(T)

similarity:
  metric: nmi
  num_bins: 64

pyramid:
  levels: 4

optimization:
  max_iterations_per_level: 100
```

## Energy Function Equivalence

### MIRTK Energy

```
E = Similarity + λ_BE * BE(T) + λ_LE * LE(T)

Where:
- Similarity = -NMI (Normalized Mutual Information)
- BE(T) = Bending Energy (2nd order derivatives, smoothness)
- LE(T) = Linear Energy (1st order derivatives, diffusion)
```

### DAREG Energy

```python
# In ffd_registration.py
similarity_loss = self._compute_nmi_loss(warped_source, target, num_bins=64, mask=overlap_mask)
bending_energy = L.bending_loss(params, mode="bspline", stride=ffd.stride)
linear_energy = L.diffusion_loss(params, stride=ffd.stride)

total_loss = similarity_loss + 0.001 * bending_energy + 0.0005 * linear_energy
```

## Foreground Handling

### MIRTK FG_Overlap Mode

MIRTK's default similarity computation uses `FG_Overlap`:
- Only compute similarity where BOTH images have foreground
- Skip voxels with zero gradient (flat regions)

**MIRTK Code** (`ImageSimilarity.h`):
```cpp
// FG_Overlap: Compute over the intersection of foregrounds
if (source_foreground(x,y,z) && target_foreground(x,y,z)) {
    // compute similarity
}
```

**MIRTK Code** (`FreeFormTransformation3D.cc` line 195):
```cpp
// Skip zero-gradient voxels
if (*gx == .0 && *gy == .0 && *gz == .0) continue;
```

### DAREG FG_Overlap Equivalent

```python
def _compute_foreground_overlap_mask(self, source, target, threshold=0.01):
    # Normalize to [0,1]
    source_norm = (source - source.min()) / (source.max() - source.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # Foreground detection
    source_fg = source_norm > threshold
    target_fg = target_norm > threshold

    # Gradient magnitude filtering (MIRTK line 195)
    source_grad = self._compute_gradient_magnitude(source_norm)
    target_grad = self._compute_gradient_magnitude(target_norm)

    # Final mask: intersection of foregrounds AND non-zero gradients
    overlap_mask = (source_fg & target_fg).float()
    overlap_mask *= (source_grad > 1e-5).float()
    overlap_mask *= (target_grad > 1e-5).float()

    return overlap_mask
```

## Multi-Resolution (Pyramid) Equivalence

### MIRTK Pyramid Levels

```
Level 4: Resolution Y=12, Z=12 (coarsest, 1/16)
Level 3: Resolution Y=6, Z=6 (1/8)
Level 2: Resolution Y=3, Z=3 (1/4)
Level 1: Resolution Y=0, Z=0 (finest, full resolution)
```

### DAREG Pyramid

```python
# Using deepali's built-in pyramid
source_pyramid = source_image.pyramid(levels=4)
target_pyramid = target_image.pyramid(levels=4)

# Coarse to fine optimization
for level in range(3, -1, -1):  # 3, 2, 1, 0
    source_level = source_pyramid[level]
    target_level = target_pyramid[level]
    # Optimize at this level...
```

## Transform Model Equivalence

| MIRTK Model | DAREG Class | DOF |
|-------------|-------------|-----|
| Rigid | `RigidTransform` | 6 (3 rotation + 3 translation) |
| Affine | `AffineTransform` | 12 |
| FFD | `FreeFormDeformation` | N (B-spline control points) |
| SVFFD | `StationaryVelocityFreeFormDeformation` | N (diffeomorphic) |

## Key Differences

### 1. Coordinate System

- **MIRTK**: Uses RAS (Right-Anterior-Superior) convention
- **Deepali**: Uses same convention, but tensor order is [C, D, H, W]

### 2. Transform Storage

- **MIRTK**: `.dof` files (custom binary format)
- **DAREG**: `.pth` files (PyTorch state dict)

### 3. Optimization

- **MIRTK**: Conjugate gradient, LBFGS
- **DAREG**: Adam, LBFGS (PyTorch optimizers)

### 4. GPU Support

- **MIRTK**: CPU only (unless CUDA build)
- **DAREG**: CPU, CUDA, MPS (Apple Silicon, limited)

## Validation

To verify DAREG produces MIRTK-equivalent results:

1. **Visual Comparison**: Overlay results in ITK-SNAP
2. **Metric Comparison**: Compare NMI values before/after registration
3. **Transform Comparison**: Apply both transforms to same test grid
4. **Jacobian Check**: Verify diffeomorphism (det(J) > 0 everywhere)

```python
# Quick validation
from DAREG.postprocessing import compute_quality_metrics

metrics = compute_quality_metrics(warped_source, target)
print(f"NMI: {metrics['nmi']:.4f}")
print(f"NCC: {metrics['ncc']:.4f}")
```
