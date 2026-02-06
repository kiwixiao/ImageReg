# Deepali Integration Guide

## Overview

DAREG uses the deepali library (developed by BioMedIA) for core registration functionality.
Deepali provides PyTorch-based medical image registration with proper coordinate system handling.

## Critical Lesson: Use Deepali Built-ins

### The Bug (Previous Implementation)

The original DAREG loader manually implemented image loading:

```python
# BAD: Manual loading loses coordinate system information
def load_image_BROKEN(path):
    sitk_img = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(sitk_img)  # [Z, Y, X]
    tensor = torch.from_numpy(array).float()

    # Manual grid creation - PRONE TO ERRORS
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    # Missing: direction matrix handling!

    grid = Grid(size=..., spacing=..., origin=...)
    return Image(data=tensor, grid=grid)
```

**Problems**:
1. SimpleITK's `GetArrayFromImage` returns [Z, Y, X] but deepali expects specific conventions
2. Direction matrix not properly handled
3. Resampling done manually with potential orientation errors
4. Aspect ratio distortion in visualizations

### The Fix (Use Deepali Built-ins)

```python
# GOOD: Use deepali's built-in methods
def load_image(path):
    # Image.read() handles all coordinate system details
    image = Image.read(str(path))
    return image

def resample_to_grid(image, target_grid):
    # image.sample() handles interpolation with correct orientation
    return image.sample(target_grid)
```

**Why This Works**:
1. `Image.read()` properly parses NIfTI headers including direction matrix
2. `image.sample()` uses proper coordinate transformations
3. Grid class manages coordinate system conversions internally
4. Consistent with deepali's registration pipeline

## Key Deepali Classes

### Image

```python
from deepali.data import Image

# Load from file
image = Image.read("brain.nii.gz")

# Access data
tensor = image.tensor()      # [C, D, H, W] torch.Tensor
grid = image.grid()          # Grid object
spacing = image.spacing()    # (dD, dH, dW) in mm

# Resample to new grid
resampled = image.sample(target_grid)

# Create pyramid for multi-resolution
pyramid = image.pyramid(levels=4)
```

### Grid

```python
from deepali.core import Grid

# Grid represents the coordinate system of an image
grid = Grid(
    size=(W, H, D),           # Size in (X, Y, Z) order
    spacing=(dx, dy, dz),     # Spacing in mm
    origin=(ox, oy, oz),      # Origin in mm
    direction=(d11, d12, ...),  # 3x3 direction cosine matrix
)

# Grid properties
grid.shape      # Returns (D, H, W) - tensor dimension order
grid.size()     # Returns physical size in mm
grid.spacing()  # Returns spacing tensor
grid.origin()   # Returns origin tensor

# Coordinate conversions
world_coords = grid.world_coords()  # Physical coordinates
cube_coords = grid.cube_coords()    # Normalized [-1, 1] coordinates
```

### Transforms

```python
from deepali.spatial import (
    RigidTransform,
    AffineTransform,
    FreeFormDeformation,
    StationaryVelocityFreeFormDeformation,
    SequentialTransform,
    ImageTransformer,
)

# Create transforms
rigid = RigidTransform(grid)
affine = AffineTransform(grid)
ffd = FreeFormDeformation(grid, stride=(4, 4, 4))
svffd = StationaryVelocityFreeFormDeformation(grid)

# Compose transforms
composed = SequentialTransform(rigid, affine, ffd)

# Apply to image
transformer = ImageTransformer(transform)
warped = transformer(source_tensor)
```

### Losses

```python
from deepali.losses import functional as L

# Similarity losses (lower = better alignment)
nmi = L.nmi_loss(source, target, num_bins=64, mask=mask)
ncc = L.ncc_loss(source, target)
mse = L.mse_loss(source, target)

# Regularization losses (smoothness)
bending = L.bending_loss(params, mode="bspline", stride=stride)
diffusion = L.diffusion_loss(params, stride=stride)
```

## Tensor Dimension Conventions

### Deepali Expects

| Type | Shape | Description |
|------|-------|-------------|
| 3D Image | [C, D, H, W] | Channel, Depth, Height, Width |
| 3D Batch | [N, C, D, H, W] | Batch, Channel, Depth, Height, Width |
| Grid coords | [D, H, W, 3] | Coordinates at each voxel |
| Displacement | [3, D, H, W] or [D, H, W, 3] | Per-voxel displacement |

### Common Conversions

```python
# Ensure 5D for loss functions
if tensor.dim() == 3:
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [D,H,W] -> [1,1,D,H,W]
elif tensor.dim() == 4:
    tensor = tensor.unsqueeze(0)  # [C,D,H,W] -> [1,C,D,H,W]

# Remove batch/channel for saving
if tensor.dim() == 5:
    tensor = tensor.squeeze(0).squeeze(0)  # [1,1,D,H,W] -> [D,H,W]
```

## ImageTransformer Pattern

The correct pattern for warping images with gradient flow:

```python
from deepali.spatial import ImageTransformer

# Create transformer (wraps transform + handles grid_sample)
transformer = ImageTransformer(transform, target_grid)

# Forward pass with gradient tracking
optimizer.zero_grad()
warped_source = transformer(source_batch)  # Gradients flow through!

# Compute loss
loss = L.nmi_loss(warped_source, target_batch)
loss.backward()  # Gradients flow back to transform parameters
optimizer.step()
```

**Why ImageTransformer?**
1. Properly calls `transform.update()` to compute displacement field
2. Handles coordinate system conversions
3. Uses `torch.nn.functional.grid_sample` with correct settings
4. Maintains gradient flow for backpropagation

## Multi-Resolution Registration

### Pyramid Creation

```python
# Create image pyramids (coarse to fine)
source_pyramid = source_image.pyramid(levels=4)
target_pyramid = target_image.pyramid(levels=4)

# Level 3 = coarsest (1/8 resolution)
# Level 0 = finest (full resolution)
for level in range(3, -1, -1):
    source_level = source_pyramid[level]
    target_level = target_pyramid[level]

    # Update transform grid for this level
    transform.grid_(target_level.grid())

    # Optimize at this level
    ...
```

### Control Point Subdivision

For FFD, control points are subdivided at finer levels:

```python
# At coarser level
ffd = FreeFormDeformation(coarse_grid, stride=stride)

# Move to finer level - subdivide control points
ffd.grid_(fine_grid)  # Internally subdivides using B-spline rules
```

## Flow Field and Displacement

### Getting Displacement Field

```python
# For FFD
transform.update()  # Compute displacement field
u = transform.u  # Displacement field [3, D, H, W]

# For SVFFD (velocity field + exponential)
v = transform.v  # Velocity field
u = transform.u  # Displacement after exponential integration
```

### Warping with Flow Field

```python
from deepali.data import FlowFields

# Get flow field at specific grid
flow = transform.flow(target_grid)

# Warp image using flow
warped = flow.warp_image(source_image)
```

## Common Pitfalls

### 1. Detached Tensors

```python
# WRONG: .tensor() may detach from computation graph
warped = flow.warp_image(source).tensor()  # No gradients!

# RIGHT: Use ImageTransformer for training
warped = transformer(source_batch)  # Gradients preserved
```

### 2. Dimension Mismatch

```python
# WRONG: Mixing tensor dimensions
loss = L.nmi_loss(source_3d, target_5d)  # Error!

# RIGHT: Ensure consistent dimensions
source_5d = source.unsqueeze(0).unsqueeze(0) if source.dim() == 3 else source
target_5d = target.unsqueeze(0).unsqueeze(0) if target.dim() == 3 else target
loss = L.nmi_loss(source_5d, target_5d)
```

### 3. Grid Coordinate Order

```python
# Deepali Grid uses (X, Y, Z) for size/spacing
# But Grid.shape returns (D, H, W) for tensor compatibility

grid = Grid(size=(W, H, D), spacing=(dx, dy, dz))
print(grid.shape)  # Returns (D, H, W) - NOT (W, H, D)!
```

### 4. Interpolation Mode for Segmentation

```python
# WRONG: Linear interpolation for segmentation (blurs labels)
transformer = ImageTransformer(transform)
warped_seg = transformer(seg_tensor)

# RIGHT: Nearest neighbor for segmentation
transformer_nn = ImageTransformer(transform, mode='nearest')
warped_seg = transformer_nn(seg_tensor)
```

## Integration Checklist

When integrating deepali into DAREG:

- [ ] Use `Image.read()` instead of manual SimpleITK loading
- [ ] Use `image.sample()` instead of manual resampling
- [ ] Use `ImageTransformer` for gradient-tracked warping
- [ ] Ensure tensors are 5D [N,C,D,H,W] for loss functions
- [ ] Use `mode='nearest'` for segmentation transformation
- [ ] Call `transform.update()` before accessing displacement field
- [ ] Use `.pyramid()` for multi-resolution registration
- [ ] Use `transform.grid_()` to update transform at new resolution level
