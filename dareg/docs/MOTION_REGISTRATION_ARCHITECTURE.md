# DAREG Motion Registration Architecture

## Overview

DAREG (Deepali-based Registration) is a Python implementation of medical image registration that replicates MIRTK (Medical Image Registration ToolKit) functionality using the deepali library.

## 4D/5D Image Handling

### Image Formats

Medical dynamic imaging (breathing, cardiac) often produces multi-dimensional data:

| Format | Shape | Description |
|--------|-------|-------------|
| 3D | (X, Y, Z) | Single volume |
| 4D | (X, Y, Z, T) | Standard 4D time series |
| 5D | (X, Y, Z, 1, T) | Common in dynamic MRI (singleton dim 3) |

### Example Test Data

```
static.nii         - 3D static high-res (158, 352, 352), spacing ~0.6mm
dynamic.nii        - 5D dynamic (192, 192, 12, 1, 90) = 90 time frames
airway_seg.nii.gz  - Segmentation mask on static image
```

The Dynamic image appears as 3D to SimpleITK because SITK doesn't handle 5D well.
**Solution**: Use nibabel which correctly reads the NIfTI header `dim` field.

```python
# nibabel correctly reads 5D
import nibabel as nib
img = nib.load('Dynamic.nii')
print(img.shape)  # (192, 192, 12, 1, 90)

# SimpleITK incorrectly reads as 3D
import SimpleITK as sitk
img = sitk.ReadImage('Dynamic.nii')
print(img.GetSize())  # (192, 192, 12) - WRONG!
```

### DAREG's Solution: `image_4d.py`

The `load_image_4d()` function handles both formats:

```python
# 5D with singleton dim 3: [X, Y, Z, 1, T] -> squeeze -> [X, Y, Z, T]
if data.shape[3] == 1:
    data = data.squeeze(axis=3)

# Transpose to torch convention: [X, Y, Z, T] -> [T, Z, Y, X]
data = np.transpose(data, (3, 2, 1, 0))
```

## Motion Registration Pipeline

### MIRTK Workflow (Reference)

```bash
# 1. Extract frames from 4D
mirtk extract-image-volume dynamic.nii frame_%03d.nii.gz -t 0 -n 90

# 2. Align static to frame 0
mirtk register static.nii frame_000.nii.gz -model Rigid+Affine+FFD -dofout align.dof

# 3. Pairwise registration (frame N -> frame N-1)
mirtk register frame_001.nii.gz frame_000.nii.gz -model FFD -dofout pair_0_1.dof

# 4. Compose transforms for longitudinal (frame 0 -> frame N)
mirtk compose-dofs pair_0_1.dof pair_1_2.dof ... -output long_0_N.dof

# 5. Transform segmentation through all frames
mirtk transform-image seg.nii.gz seg_frameN.nii.gz -dofin long_0_N.dof -interp NN
```

### DAREG Workflow (Python Implementation)

```
main_motion.py
    │
    ├── 1. Load 4D Image (image_4d.py)
    │       └── load_image_4d() - handles 4D/5D NIfTI
    │
    ├── 2. Static Alignment (motion.py: register_alignment)
    │       ├── Rigid Registration
    │       ├── Affine Registration
    │       └── FFD/SVFFD Registration
    │
    ├── 3. Pairwise Registration (motion.py: register_pairwise)
    │       └── Frame N+1 → Frame N (sequential)
    │
    ├── 4. Compose Longitudinal (motion.py: compose_longitudinal)
    │       └── Chain: T_01 ∘ T_12 ∘ T_23 ... = T_0N
    │
    ├── 5. Refine Longitudinal (motion.py: refine_longitudinal)
    │       └── Direct registration frame 0 → frame N
    │
    └── 6. Propagate Segmentation (motion.py: propagate_segmentation)
            └── Transform seg through each T_0N
```

## Registration Modules

### Base Registration (`registration/base.py`)

All registration methods inherit from `BaseRegistration`:

```python
class BaseRegistration(ABC):
    def __init__(self, device, config):
        self.device = device
        self.config = config

    @abstractmethod
    def register(self, source: Image, target: Image) -> RegistrationResult:
        pass

    def _compute_nmi_loss(self, source, target, num_bins=64, mask=None):
        """MIRTK-style NMI with foreground masking"""
        pass

    def _compute_foreground_overlap_mask(self, source, target, threshold=0.01):
        """MIRTK FG_Overlap equivalent"""
        pass
```

### Foreground Masking (CRITICAL)

MIRTK uses `FG_Overlap` mode which computes similarity ONLY where both images have valid foreground data.

**MIRTK Code Reference** (`FreeFormTransformation3D.cc` line 195):
```cpp
// Skip voxels with zero gradient
if (*gx == .0 && *gy == .0 && *gz == .0) continue;
```

**DAREG Implementation**:
```python
def _compute_foreground_overlap_mask(self, source, target, threshold=0.01):
    # 1. Normalize intensities to [0,1]
    source_norm = (source - source.min()) / (source.max() - source.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # 2. Foreground detection
    source_fg = source_norm > threshold
    target_fg = target_norm > threshold

    # 3. MIRTK line 195: Skip zero-gradient voxels
    source_grad = self._compute_gradient_magnitude(source_norm)
    target_grad = self._compute_gradient_magnitude(target_norm)
    source_grad_mask = source_grad > 1e-5
    target_grad_mask = target_grad > 1e-5

    # 4. INTERSECTION of foregrounds AND gradient masks
    overlap_mask = source_fg & target_fg
    final_mask = overlap_mask.float() * source_grad_mask.float() * target_grad_mask.float()

    return final_mask
```

**Why This Matters**:
- Prevents computing gradients in background/air regions
- Avoids unrealistic deformation at image boundaries
- Critical for anisotropic images where background dominates

## Deepali Integration

### Key Deepali Components Used

| Component | Purpose |
|-----------|---------|
| `Image.read()` | Load NIfTI with proper coordinate handling |
| `image.sample(grid)` | Resample to new grid (orientation-aware) |
| `Grid` | Represents image coordinate system |
| `ImageTransformer` | Apply transforms with proper gradient flow |
| `RigidTransform` | 6 DOF transformation |
| `AffineTransform` | 12 DOF transformation |
| `FreeFormDeformation` | B-spline FFD (direct displacement) |
| `StationaryVelocityFreeFormDeformation` | SVFFD (diffeomorphic) |

### Critical Pattern: Use Deepali Built-ins

**WRONG** (previous bug - causes orientation issues):
```python
# Manual loading with SimpleITK
sitk_img = sitk.ReadImage(path)
array = sitk.GetArrayFromImage(sitk_img)
tensor = torch.from_numpy(array)
# Missing: direction, proper spacing handling
```

**CORRECT** (use deepali built-ins):
```python
# Deepali handles all coordinate system details
image = Image.read(path)
resampled = image.sample(target_grid)  # Proper resampling
```

## Transform Composition

### Sequential Pairwise Strategy

For motion tracking, consecutive frame registration is more stable than direct long-range registration:

```
Frame 0 ← Frame 1 ← Frame 2 ← Frame 3 ...
    T_01      T_12      T_23

Longitudinal: T_03 = T_01 ∘ T_12 ∘ T_23
```

### Composition in DAREG

```python
def compose_longitudinal(self, pairwise_transforms):
    """Compose pairwise transforms into longitudinal transforms"""
    longitudinal = []

    for target_idx in range(1, num_frames):
        # Compose T_01 ∘ T_12 ∘ ... ∘ T_(N-1)N
        composed = SequentialTransform(
            *[pw.transform for pw in pairwise_transforms[:target_idx]]
        )
        longitudinal.append(composed)

    return longitudinal
```

## Device Handling

### MPS (Apple Silicon) Limitations

PyTorch MPS doesn't support `grid_sampler_3d` operator:
```
NotImplementedError: The operator 'aten::grid_sampler_3d' is not currently
implemented for the MPS device.
```

**Solution**: Force CPU for 3D medical image registration:
```bash
dareg motion --device cpu ...
```

Or set environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Configuration

### Registration Config Structure

```yaml
rigid:
  pyramid_levels: 3
  iterations_per_level: [100, 100, 100]
  learning_rates: [1e-2, 1e-2, 1e-2]

affine:
  pyramid_levels: 3
  iterations_per_level: [100, 100, 100]
  learning_rates: [1e-3, 1e-3, 1e-3]

ffd:
  control_point_spacing: 4  # mm (MIRTK default)
  bending_energy_weight: 0.001
  linear_energy_weight: 0.0005
  pyramid_levels: 4
  iterations_per_level: 100

similarity:
  metric: nmi
  num_bins: 64
  foreground_threshold: 0.01
```

## Output Structure

```
dareg_test_ffd/
├── transforms/
│   ├── alignment_rigid.pth
│   ├── alignment_affine.pth
│   ├── alignment_ffd.pth
│   ├── pairwise_0_1.pth
│   ├── pairwise_1_2.pth
│   └── longitudinal_0_N.pth
├── segmentations/
│   ├── seg_frame_000.nii.gz
│   ├── seg_frame_001.nii.gz
│   └── ...
├── frames/
│   ├── frame_000.nii.gz
│   └── ...
└── visualizations/
    ├── pairwise_losses.png
    ├── segmentation_progression.png
    └── motion_magnitude.png
```

## Key Learnings

1. **Always use nibabel for 5D NIfTI** - SimpleITK doesn't handle it correctly
2. **Use deepali built-ins** - `Image.read()`, `image.sample()` handle coordinate systems properly
3. **Foreground masking is CRITICAL** - MIRTK FG_Overlap prevents artifacts at boundaries
4. **Gradient filtering (MIRTK line 195)** - Skip zero-gradient voxels for stable registration
5. **MPS doesn't support grid_sampler_3d** - Use CPU for 3D medical imaging
6. **Sequential pairwise is more stable** than direct long-range registration for motion tracking
