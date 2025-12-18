# Deepali Equivalents for MIRTK Pipeline

## Overview

This document maps each MIRTK motion registration step to its deepali equivalent.
Use alongside `MIRTK_PIPELINE_STEPS.md` for complete pipeline reproduction.

---

## Quick Reference Table

| MIRTK Step | MIRTK Command | Deepali Equivalent |
|------------|---------------|-------------------|
| Frame Extraction | `mirtk extract-image-volume` | `Image.read()` + nibabel slicing |
| Image Loading | `mirtk info` | `deepali.data.Image.read()` |
| Resampling | `mirtk resample-image` | `image.sample(target_grid)` |
| FFD Transform | `Transformation model = FFD` | `deepali.spatial.FreeFormDeformation` |
| SVFFD Transform | `Transformation model = SVFFD` | `deepali.spatial.StationaryVelocityFreeFormDeformation` |
| NMI Similarity | `NMI(I(1), I(2:end) o T)` | `deepali.losses.NMI` |
| Bending Energy | `BE(T)` | `deepali.losses.BendingEnergy` |
| Linear Energy | `LE(T)` (Diffusion) | `deepali.losses.Diffusion` |
| Transform Composition | `mirtk compose-dofs` | `deepali.spatial.SequentialTransform` |
| Apply Transform | `mirtk transform-image` | `deepali.spatial.ImageTransformer` |
| Jacobian Check | `mirtk evaluate-jacobian` | `deepali.core.flow.jacobian_det()` |

---

## Step 1: Frame Extraction

### MIRTK
```bash
mirtk extract-image-volume dynamic_4d.nii.gz -t 0 -n 10 frame_%03d.nii.gz
```

### Deepali + nibabel
```python
import nibabel as nib
from deepali.data import Image

def extract_frames(path_4d, output_dir, start_frame=0, num_frames=None):
    """
    Extract 3D frames from 4D/5D NIfTI with EXACT affine preservation.

    CRITICAL: Use nibabel, NOT SimpleITK!
    SimpleITK loses affine information for 5D NIfTI files.
    """
    # Load 4D/5D image with nibabel (preserves affine!)
    nii = nib.load(str(path_4d))
    data = nii.get_fdata()
    affine = nii.affine  # MUST preserve this exactly
    header = nii.header

    # Handle 5D: shape (X, Y, Z, 1, T) -> squeeze singleton
    if data.ndim == 5 and data.shape[3] == 1:
        data = data.squeeze(axis=3)  # Now (X, Y, Z, T)

    # Determine frame range
    n_frames = data.shape[-1]
    if num_frames is None:
        num_frames = n_frames - start_frame

    frames = []
    for t in range(start_frame, start_frame + num_frames):
        frame_data = data[..., t]  # Extract 3D volume

        # Save with EXACT same affine
        frame_nii = nib.Nifti1Image(frame_data, affine, header)
        output_path = output_dir / f"frame_{t:03d}.nii.gz"
        nib.save(frame_nii, str(output_path))

        # Load as deepali Image for registration
        frame_img = Image.read(str(output_path))
        frames.append(frame_img)

    return frames
```

**Key Points:**
- Use `nibabel.load()` NOT `SimpleITK.ReadImage()` for 5D NIfTI
- Preserve exact affine matrix when saving extracted frames
- 5D shape `(192, 192, 12, 1, 90)` has singleton dimension at index 3

---

## Step 2: Image Loading & Grid Management

### MIRTK
```bash
mirtk info image.nii.gz  # Get spacing, origin, direction
```

### Deepali
```python
from deepali.data import Image
from deepali.core import Grid

# Load image - handles all coordinate system details
image = Image.read("brain.nii.gz")

# Access properties
tensor = image.tensor()      # [C, D, H, W] torch.Tensor
grid = image.grid()          # Grid object with coordinate system
spacing = image.spacing()    # (dD, dH, dW) in mm

# Grid properties (CRITICAL for registration)
grid.shape      # (D, H, W) - tensor dimensions
grid.size()     # Physical size in mm
grid.spacing()  # Voxel spacing in mm
grid.origin()   # World coordinate origin
grid.direction()  # 3x3 direction cosine matrix

# Coordinate conversions
world_coords = grid.world_coords()  # Physical mm coordinates
cube_coords = grid.cube_coords()    # Normalized [-1, 1] coordinates
```

**Key Points:**
- `Image.read()` parses NIfTI headers correctly including direction matrix
- Grid class handles all coordinate system conversions internally
- Use `image.sample(target_grid)` for resampling

---

## Step 3: Multi-Resolution Pyramid

### MIRTK Configuration
```ini
No. of levels = 4

[ level 4 ]    # Coarsest (1/16 resolution)
Resolution in Y = 12
Resolution in Z = 12

[ level 3 ]    # 1/8 resolution
Resolution in Y = 6
Resolution in Z = 6

[ level 2 ]    # 1/4 resolution
Resolution in Y = 3
Resolution in Z = 3

[ level 1 ]    # Finest (full resolution)
Resolution in Y = 0
Resolution in Z = 0
```

### Deepali
```python
# Create image pyramids (built-in method)
source_pyramid = source_image.pyramid(levels=4)
target_pyramid = target_image.pyramid(levels=4)

# Access levels (0 = finest, 3 = coarsest for 4 levels)
# Level indices: higher number = coarser
for level in range(3, -1, -1):  # Coarse to fine: 3, 2, 1, 0
    source_level = source_pyramid[level]
    target_level = target_pyramid[level]

    # Each level is a deepali Image with appropriate grid
    print(f"Level {level}: shape={source_level.grid().shape}")

    # Update transform for this level (subdivides control points)
    transform.grid_(target_level.grid())
```

**Key Points:**
- `image.pyramid(levels=4)` creates Gaussian pyramid automatically
- Levels are indexed 0 (finest) to levels-1 (coarsest)
- Use `transform.grid_()` to subdivide B-spline control points at finer levels

---

## Step 4: FFD Transformation

### MIRTK Configuration
```ini
Transformation model  = FFD
Control point spacing = 4      # 4mm between control points
```

### Deepali (Standard FFD)
```python
from deepali.spatial import FreeFormDeformation

# Create FFD transform
# stride = control_point_spacing / voxel_spacing (in voxels)
ffd = FreeFormDeformation(
    grid=target_grid,
    stride=(4, 4, 4),  # Approximate 4mm spacing
    groups=1,
)

# FFD produces direct displacement field
ffd.update()
displacement = ffd.u  # [3, D, H, W] displacement field

# Get flow for warping
flow = ffd.flow(target_grid)
warped_image = flow.warp_image(source_image)
```

### Deepali (SVFFD - Diffeomorphic)
```python
from deepali.spatial import StationaryVelocityFreeFormDeformation

# Create SVFFD transform (guarantees diffeomorphism)
svffd = StationaryVelocityFreeFormDeformation(
    grid=target_grid,
    stride=(4, 4, 4),
)

# SVFFD stores velocity field, integrates to get displacement
svffd.update()
velocity = svffd.v       # Velocity field (B-spline coefficients)
displacement = svffd.u   # Displacement after exponential integration

# Integration uses scaling-and-squaring algorithm
# exp(v) is computed via repeated composition: v/2^n composed n times
```

**Key Points:**
- FFD = direct displacement (faster, not guaranteed diffeomorphic)
- SVFFD = velocity field + exponential (slower, diffeomorphic)
- `stride` parameter = number of voxels between control points
- Call `transform.update()` before accessing displacement field

---

## Step 5: Similarity Metric (NMI)

### MIRTK
```ini
Energy function = NMI(I(1), I(2:end) o T) + ...
No. of bins = 64
```

### Deepali
```python
from deepali.losses import NMI
from deepali.losses import functional as L

# Method 1: Using loss class
nmi_loss = NMI(
    num_bins=64,        # MIRTK default: 64 bins
    sample_ratio=1.0,   # Use all voxels
    normalized=True,    # Use NMI (not MI)
)
loss = nmi_loss(warped_source, target)

# Method 2: Using functional interface
loss = L.nmi_loss(
    warped_source,      # [N, C, D, H, W]
    target,             # [N, C, D, H, W]
    num_bins=64,
    mask=overlap_mask,  # Optional: FG_Overlap equivalent
)

# NMI Implementation Details:
# - Uses Parzen window estimation with Gaussian kernel
# - Bins: 64 (MIRTK default)
# - Returns NEGATIVE NMI (for minimization)
# - Higher NMI = better alignment, so we minimize -NMI
```

**MIRTK FG_Overlap Equivalent:**
```python
def compute_foreground_overlap_mask(source, target, threshold=0.01):
    """
    MIRTK FG_Overlap mode: compute similarity only where BOTH images
    have foreground data and non-zero gradient.

    Reference: MIRTK ImageSimilarity.h:87, FreeFormTransformation3D.cc:195
    """
    # Normalize to [0, 1]
    source_norm = (source - source.min()) / (source.max() - source.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # Foreground = intensity above threshold
    source_fg = source_norm > threshold
    target_fg = target_norm > threshold

    # Overlap mask (MIRTK FG_Overlap mode)
    overlap = source_fg & target_fg

    # Optional: filter zero-gradient voxels (MIRTK line 195)
    # if (*gx == .0 && *gy == .0 && *gz == .0) continue;
    source_grad = compute_gradient_magnitude(source_norm)
    has_gradient = source_grad > 1e-6

    return (overlap & has_gradient).float()

# Use mask in NMI computation
loss = L.nmi_loss(warped_source, target, mask=overlap_mask)
```

---

## Step 6: Regularization (Bending + Linear Energy)

### MIRTK
```ini
Energy function = NMI(...) + 0.001 BE(T) + 0.0005 LE(T)
```
- `BE(T)` = Bending Energy (2nd order derivatives, smoothness)
- `LE(T)` = Linear Energy (1st order derivatives, diffusion)

### Deepali
```python
from deepali.losses import BendingEnergy, Diffusion
from deepali.losses import functional as L

# Get transform parameters (control point coefficients)
params = ffd.data()  # [N, C, D, H, W] B-spline coefficients

# Method 1: Using loss classes
bending_loss = BendingEnergy(mode="bspline", stride=ffd.stride)
diffusion_loss = Diffusion(stride=ffd.stride)

be = bending_loss(params)
le = diffusion_loss(params)

# Method 2: Using functional interface
be = L.bending_loss(params, mode="bspline", stride=ffd.stride)
le = L.diffusion_loss(params, stride=ffd.stride)

# Total energy (MIRTK equivalent)
total_loss = nmi_loss + 0.001 * be + 0.0005 * le
```

**Implementation Details:**
```python
# Bending Energy (BE): Sum of squared 2nd order derivatives
# BE = sum_i sum_jk (d²u_i / dx_j dx_k)²
# Penalizes high curvature, promotes smooth deformations

# Linear/Diffusion Energy (LE): Sum of squared 1st order derivatives
# LE = 0.5 * sum_i sum_j (du_i / dx_j)²
# Penalizes large gradients, promotes small local deformations

# Both support 'spacing' parameter for anisotropic voxels
be = L.bending_loss(params, mode="bspline", stride=stride, spacing=(dz, dy, dx))
le = L.diffusion_loss(params, stride=stride, spacing=(dz, dy, dx))
```

---

## Step 7: Optimization

### MIRTK
```ini
Maximum no. of iterations = 100
Strict step length range = No
Maximum streak of rejected steps = 1
```

### Deepali (PyTorch Optimizers)
```python
import torch.optim as optim

# Option 1: LBFGS (MIRTK-like, quasi-Newton)
optimizer = optim.LBFGS(
    transform.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=10,
    line_search_fn='strong_wolfe'
)

# Option 2: Adam (reliable gradient descent)
optimizer = optim.Adam(transform.parameters(), lr=0.01)

# Optimization loop (Adam style)
for iteration in range(100):
    optimizer.zero_grad()

    # Forward: warp source
    warped_source = transformer(source_batch)

    # Compute loss
    similarity = L.nmi_loss(warped_source, target_batch, num_bins=64)
    bending = L.bending_loss(transform.data(), mode="bspline", stride=transform.stride)
    diffusion = L.diffusion_loss(transform.data(), stride=transform.stride)

    total_loss = similarity + 0.001 * bending + 0.0005 * diffusion

    # Backward and update
    total_loss.backward()
    optimizer.step()

# LBFGS style (closure required)
def closure():
    optimizer.zero_grad()
    warped = transformer(source_batch)
    loss = compute_total_loss(warped, target_batch)
    loss.backward()
    return loss

for iteration in range(5):  # Fewer outer iterations for LBFGS
    optimizer.step(closure)
```

---

## Step 8: Image Warping (Apply Transform)

### MIRTK
```bash
# For images (linear interpolation)
mirtk transform-image source.nii output.nii -dofin transform.dof -interp Linear

# For segmentations (nearest neighbor)
mirtk transform-image seg.nii output_seg.nii -dofin transform.dof -interp NN
```

### Deepali
```python
from deepali.spatial import ImageTransformer

# Create transformer (wraps transform for gradient flow)
transformer = ImageTransformer(transform, target_grid)

# Warp image (linear interpolation - default)
warped_image = transformer(source_batch)  # Gradients flow through!

# For segmentation (nearest neighbor)
transformer_nn = ImageTransformer(transform, target_grid, mode='nearest')
warped_seg = transformer_nn(seg_batch)

# Alternative: Using flow field directly
flow = transform.flow(target_grid, device=device)
warped = flow.warp_image(source_image)  # Returns ImageBatch

# CRITICAL: For gradient-tracked training, use ImageTransformer
# flow.warp_image().tensor() returns DETACHED tensor (no gradients!)
```

**Key Points:**
- `ImageTransformer` maintains gradient flow for backpropagation
- `mode='nearest'` for segmentations preserves discrete labels
- Call `transform.update()` before computing flow

---

## Step 9: Transform Composition

### MIRTK
```bash
mirtk compose-dofs ffd_0_1.dof ffd_1_2.dof ffd_0_2.dof
```

### Deepali
```python
from deepali.spatial import SequentialTransform

# Compose multiple transforms
composed = SequentialTransform(transform_0_1, transform_1_2, transform_2_3)

# Apply composed transform
transformer = ImageTransformer(composed, target_grid)
warped = transformer(source_batch)

# For sequential motion tracking
transforms = []
for t in range(1, n_frames):
    # Register frame[t-1] → frame[t]
    pairwise_transform = register_pair(frames[t-1], frames[t])

    if t == 1:
        longitudinal = pairwise_transform
    else:
        # Compose: T_0→t = T_0→(t-1) ∘ T_(t-1)→t
        longitudinal = SequentialTransform(transforms[-1], pairwise_transform)

    transforms.append(longitudinal)
```

---

## Step 10: Quality Validation (Jacobian)

### MIRTK
```bash
mirtk evaluate-jacobian transform.dof jacobian.nii.gz
```

### Deepali
```python
from deepali.core.flow import jacobian_det

# Compute Jacobian determinant
transform.update()
displacement = transform.u  # [N, 3, D, H, W] or [3, D, H, W]

# Jacobian determinant of deformation field
# J = det(I + grad(u)) where I is identity
jac_det = jacobian_det(displacement)

# Check for folding (negative Jacobian = topology violation)
min_jac = jac_det.min().item()
max_jac = jac_det.max().item()
folding_percentage = (jac_det < 0).float().mean().item() * 100

print(f"Jacobian: min={min_jac:.4f}, max={max_jac:.4f}")
print(f"Folding: {folding_percentage:.2f}% of voxels")

# Diffeomorphic check
is_diffeomorphic = min_jac > 0
```

**Key Points:**
- Jacobian determinant < 0 indicates topology violation (folding)
- SVFFD guarantees positive Jacobian (diffeomorphic)
- Regular FFD may have negative Jacobian if regularization is too weak

---

## Complete Pipeline Example

```python
"""
DAREG Motion Registration Pipeline
Reproduces MIRTK workflow using deepali
"""

import torch
import nibabel as nib
from pathlib import Path
from deepali.data import Image
from deepali.core import Grid
from deepali.spatial import (
    StationaryVelocityFreeFormDeformation,
    SequentialTransform,
    ImageTransformer,
)
from deepali.losses import functional as L

def motion_registration_pipeline(
    dynamic_4d_path: str,
    seg_path: str,
    output_dir: str,
    num_levels: int = 4,
    control_spacing: float = 4.0,  # mm
    be_weight: float = 0.001,
    le_weight: float = 0.0005,
    max_iterations: int = 100,
):
    """
    MIRTK-equivalent motion registration pipeline.

    Registers all frames to frame 0 using sequential pairwise registration
    with SVFFD (diffeomorphic) and NMI similarity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Step 1: Frame Extraction (nibabel)
    # =========================================
    print("Step 1: Extracting frames...")
    nii = nib.load(dynamic_4d_path)
    data = nii.get_fdata()
    affine = nii.affine

    if data.ndim == 5 and data.shape[3] == 1:
        data = data.squeeze(axis=3)

    n_frames = data.shape[-1]
    frames = []

    for t in range(n_frames):
        frame_path = output_dir / f"frame_{t:03d}.nii.gz"
        frame_nii = nib.Nifti1Image(data[..., t].astype(np.float32), affine)
        nib.save(frame_nii, str(frame_path))
        frames.append(Image.read(str(frame_path)))

    # =========================================
    # Step 2: Sequential Pairwise Registration
    # =========================================
    print("Step 2: Sequential pairwise registration...")

    device = torch.device("cpu")  # MPS doesn't support grid_sample_3d
    pairwise_transforms = []

    for t in range(1, n_frames):
        print(f"  Registering frame {t-1} -> {t}")

        source = frames[t]      # Moving image
        target = frames[t - 1]  # Fixed image

        # Create pyramids
        source_pyr = source.pyramid(levels=num_levels)
        target_pyr = target.pyramid(levels=num_levels)

        # Create SVFFD at coarsest level
        coarsest_grid = target_pyr[num_levels - 1].grid()
        stride = int(round(control_spacing / coarsest_grid.spacing()[0].item()))
        stride = max(1, stride)

        transform = StationaryVelocityFreeFormDeformation(
            grid=coarsest_grid,
            stride=(stride, stride, stride),
        ).to(device)

        # Multi-resolution optimization (coarse to fine)
        for level in range(num_levels - 1, -1, -1):
            source_level = source_pyr[level].batch().tensor().to(device)
            target_level = target_pyr[level].batch().tensor().to(device)

            # Subdivide control points for finer level
            transform.grid_(target_pyr[level].grid())

            optimizer = torch.optim.Adam(transform.parameters(), lr=0.01)
            transformer = ImageTransformer(transform)

            for iteration in range(max_iterations):
                optimizer.zero_grad()

                warped = transformer(source_level)

                nmi = L.nmi_loss(warped, target_level, num_bins=64)
                be = L.bending_loss(transform.data(), mode="bspline", stride=transform.stride)
                le = L.diffusion_loss(transform.data(), stride=transform.stride)

                loss = nmi + be_weight * be + le_weight * le
                loss.backward()
                optimizer.step()

        pairwise_transforms.append(transform)

    # =========================================
    # Step 3: Transform Composition
    # =========================================
    print("Step 3: Composing transforms...")

    longitudinal_transforms = []
    for t in range(1, n_frames):
        if t == 1:
            T_0_t = pairwise_transforms[0]
        else:
            T_0_t = SequentialTransform(
                longitudinal_transforms[-1],
                pairwise_transforms[t - 1]
            )
        longitudinal_transforms.append(T_0_t)

    # =========================================
    # Step 4: Apply to Segmentation
    # =========================================
    print("Step 4: Transforming segmentation...")

    seg = Image.read(seg_path)
    seg_batch = seg.batch().tensor().to(device)

    for t, transform in enumerate(longitudinal_transforms, start=1):
        transformer_nn = ImageTransformer(transform, frames[t].grid(), mode='nearest')
        warped_seg = transformer_nn(seg_batch)

        # Save with nibabel
        warped_np = warped_seg.squeeze().cpu().numpy()
        output_nii = nib.Nifti1Image(warped_np, affine)
        nib.save(output_nii, str(output_dir / f"seg_frame_{t:03d}.nii.gz"))

    print("Pipeline complete!")
    return longitudinal_transforms
```

---

## Summary: MIRTK → Deepali Mapping

| MIRTK Concept | Deepali Equivalent |
|---------------|-------------------|
| `mirtk extract-image-volume` | nibabel slicing + `Image.read()` |
| `Transformation model = FFD` | `FreeFormDeformation` |
| `Transformation model = SVFFD` | `StationaryVelocityFreeFormDeformation` |
| `Control point spacing = 4` | `stride` parameter (voxels) |
| `NMI(I(1), I(2:end) o T)` | `L.nmi_loss()` with 64 bins |
| `BE(T)` Bending Energy | `L.bending_loss(mode="bspline")` |
| `LE(T)` Linear Energy | `L.diffusion_loss()` |
| `No. of levels = 4` | `image.pyramid(levels=4)` |
| Multi-resolution | `transform.grid_()` for subdivision |
| `mirtk compose-dofs` | `SequentialTransform` |
| `mirtk transform-image -interp Linear` | `ImageTransformer(mode='bilinear')` |
| `mirtk transform-image -interp NN` | `ImageTransformer(mode='nearest')` |
| `FG_Overlap` mode | `L.nmi_loss(..., mask=overlap_mask)` |
| Jacobian check | `deepali.core.flow.jacobian_det()` |

---

## Critical Implementation Notes

1. **Use nibabel for 5D NIfTI** - SimpleITK mishandles 5D files
2. **Use `Image.read()` for loading** - Handles coordinate systems correctly
3. **Use `ImageTransformer` for training** - Maintains gradient flow
4. **Call `transform.update()` before accessing displacement** - Computes B-spline interpolation
5. **Use `mode='nearest'` for segmentations** - Preserves discrete labels
6. **SVFFD is diffeomorphic** - Guarantees topology preservation
7. **Normalize regularization by control point count** - Consistent weights across levels
8. **FG_Overlap is critical** - Prevents boundary artifacts

