# MIRTK to Deepali Cross-Reference Manual

**Purpose**: Comprehensive guide for achieving MIRTK-equivalent medical image registration using the deepali library.

**MIRTK** = Gold standard reference implementation (C++)
**Deepali** = PyTorch-based implementation used by DAREG

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Coordinate Systems](#2-coordinate-systems)
3. [Grid and Domain Representation](#3-grid-and-domain-representation)
4. [Transform Types](#4-transform-types)
5. [B-Spline FFD Implementation](#5-b-spline-ffd-implementation)
6. [Similarity Metrics](#6-similarity-metrics)
7. [Regularization Terms](#7-regularization-terms)
8. [Optimization](#8-optimization)
9. [Transform Composition](#9-transform-composition)
10. [Image Warping](#10-image-warping)
11. [Transform I/O](#11-transform-io)
12. [Module-by-Module Mapping](#12-module-by-module-mapping)
13. [Code Examples](#13-code-examples)

---

## 1. Architecture Overview

### MIRTK Architecture
```
MIRTK/
├── Modules/Transformation/     # Transform classes (Rigid, Affine, FFD)
├── Modules/Registration/       # Registration pipeline, similarity metrics
├── Modules/Numerics/          # Optimizers (CG, LBFGS)
├── Modules/Image/             # Image I/O and processing
└── Applications/              # Command-line tools (register, transform-image)
```

### Deepali Architecture
```
deepali/
├── core/          # Grid, coordinates, tensor utilities
├── spatial/       # Transform classes (all types)
├── losses/        # Similarity metrics, regularization
├── data/          # Image/FlowField with metadata
├── modules/       # Stateless PyTorch modules
└── networks/      # Neural network architectures
```

### Equivalence Summary

| Aspect | MIRTK | Deepali |
|--------|-------|---------|
| Language | C++ | Python/PyTorch |
| Differentiation | Manual gradients | Automatic (autograd) |
| GPU Support | Limited (OpenMP) | Full CUDA support |
| Coordinate System | World (mm) | Normalized cube + World |
| B-Spline Order | Cubic (order 3) | Cubic (order 3) |

---

## 2. Coordinate Systems

### MIRTK Coordinate Spaces

1. **World Space**: Physical coordinates in mm
   - Origin at image corner or center (configurable)
   - Respects image spacing and orientation

2. **Lattice Space**: Control point indices (0 to N-1)
   - Used internally for B-spline evaluation
   - `WorldToLattice()` / `LatticeToWorld()` conversions

3. **Image Space**: Voxel indices
   - Integer coordinates (i, j, k)

### Deepali Coordinate Spaces

1. **World Space**: Physical coordinates in mm
   - Same as MIRTK
   - Accessed via `Grid.cube_to_world()` / `Grid.world_to_cube()`

2. **Cube Space**: Normalized [-1, 1]^D
   - Primary space for transform operations
   - `align_corners=True`: extrema at corner voxel centers
   - `align_corners=False`: extrema at grid boundaries

3. **Grid Space**: Voxel indices
   - Accessed via `Grid.coords()`

### Coordinate Conversion Equivalence

| Operation | MIRTK | Deepali |
|-----------|-------|---------|
| World → Internal | `WorldToLattice(x,y,z)` | `grid.world_to_cube(points)` |
| Internal → World | `LatticeToWorld(x,y,z)` | `grid.cube_to_world(points)` |
| Get grid coords | `ImageToWorld(i,j,k)` | `grid.coords(axes=Axes.WORLD)` |

### Key Insight: Why Both Work

```
MIRTK Lattice:     [0, 1, 2, ..., N-1]  →  B-spline basis at integer indices
Deepali Cube:      [-1, ..., 0, ..., 1]  →  B-spline basis at normalized coords

Both map input to a space where B-spline basis functions are evaluated.
The mathematical result is identical - just different parameterization.
```

---

## 3. Grid and Domain Representation

### MIRTK Grid (from DOF file header)

```cpp
// BSplineFreeFormTransformation3D attributes
int    _x, _y, _z;           // Control point counts
double _dx, _dy, _dz;        // Control point spacing (mm)
double _xorigin, _yorigin, _zorigin;  // Grid origin (mm)
double _xaxis[3], _yaxis[3], _zaxis[3];  // Orientation vectors
double _torigin, _dt;        // Temporal (4D only)
```

### Deepali Grid

```python
from deepali.core import Grid

grid = Grid(
    size=(X, Y, Z),           # Grid dimensions (NOTE: X,Y,Z order!)
    spacing=(dx, dy, dz),     # Voxel spacing in mm
    origin=(ox, oy, oz),      # Grid origin in mm
    direction=[[...], [...], [...]],  # 3x3 orientation matrix
    align_corners=True,       # Corner alignment mode
)

# Key methods
grid.size()       # Returns (X, Y, Z) - NOT (D, H, W)!
grid.spacing()    # Returns (dx, dy, dz)
grid.origin()     # Returns (ox, oy, oz)
grid.direction()  # Returns 3x3 matrix
grid.shape        # Returns (Z, Y, X) for tensor indexing
```

### Critical Convention Difference

```python
# MIRTK convention: Often stores as (x, y, z)
# Deepali Grid methods: Return (X, Y, Z) order
# Deepali tensor shape: Uses (D, H, W) = (Z, Y, X) order

# When saving domain info:
domain = {
    'size': tuple(grid.size().tolist()),      # (X, Y, Z)
    'spacing': tuple(grid.spacing().tolist()), # (dx, dy, dz)
    'origin': tuple(grid.origin().tolist()),   # (ox, oy, oz)
}

# When creating Grid from saved domain:
shape_dhw = (size_xyz[2], size_xyz[1], size_xyz[0])  # Convert to (Z, Y, X)
```

---

## 4. Transform Types

### Linear Transforms

| Transform | MIRTK Class | Deepali Class | DOF |
|-----------|-------------|---------------|-----|
| Rigid | `RigidTransformation` | `RigidTransform` | 6 |
| Similarity | `SimilarityTransformation` | (compose Scale+Rigid) | 7 |
| Affine | `AffineTransformation` | `AffineTransform` | 12 |

### Non-Rigid Transforms

| Transform | MIRTK Class | Deepali Class | Notes |
|-----------|-------------|---------------|-------|
| FFD | `BSplineFreeFormTransformation3D` | `FreeFormDeformation` | Cubic B-spline |
| SVFFD | `BSplineFreeFormTransformationSV` | `StationaryVelocityFreeFormDeformation` | Diffeomorphic |

### Parameter Representation

**Rigid (6 DOF)**:
```python
# MIRTK: Rotation (rx, ry, rz) + Translation (tx, ty, tz)
# Deepali: Same - EulerRotation + Translation

rigid = RigidTransform(grid)
# Internal: _transforms.rotation.params (3), _transforms.translation.params (3)
```

**Affine (12 DOF)**:
```python
# MIRTK: 4x4 homogeneous matrix
# Deepali: Rotation (3) + Scaling (3) + Shearing (3) + Translation (3)

affine = AffineTransform(grid)
# Internal: rotation, scaling, shearing, translation sub-transforms
```

**FFD**:
```python
# MIRTK: Control point displacements in world mm
# Deepali: Control point displacements in normalized space

ffd = FreeFormDeformation(grid, stride=4)
# params shape: [1, 3, D_cp, H_cp, W_cp]
# where *_cp = control point grid dimensions
```

---

## 5. B-Spline FFD Implementation

### MIRTK LocalTransform Algorithm

```cpp
// BSplineFreeFormTransformation3D::LocalTransform()
void LocalTransform(double &x, double &y, double &z) const {
    double dx = x, dy = y, dz = z;

    // Step 1: World → Lattice coordinates
    this->WorldToLattice(dx, dy, dz);

    // Step 2: Evaluate B-spline displacement
    this->Evaluate(dx, dy, dz);

    // Step 3: Add displacement to original world coords
    x += dx;
    y += dy;
    z += dz;
}

// WorldToLattice builds matrix: inv(T0) * inv(S) * inv(R) * inv(T)
// T: Translate by -origin
// R: Rotate by direction matrix transpose
// S: Scale by 1/spacing
// T0: Translate by (N-1)/2 to center
```

### Deepali FFD Algorithm

```python
# FreeFormDeformation.forward()
def forward(self, points, grid=False):
    # points: normalized cube coords [-1, 1]

    # Step 1: Evaluate B-spline displacement at normalized coords
    # Uses transposed convolution for efficient evaluation
    displacement = self._evaluate_bspline(points)

    # Step 2: Add displacement (in normalized space)
    return points + displacement

# Full world-to-world transform:
points_norm = grid.world_to_cube(world_points)
result_norm = ffd.forward(points_norm)
result_world = grid.cube_to_world(result_norm)
```

### B-Spline Basis Functions (Identical in Both)

```
Cubic B-spline basis (order 3):
B0(t) = (1-t)³ / 6
B1(t) = (3t³ - 6t² + 4) / 6
B2(t) = (-3t³ + 3t² + 3t + 1) / 6
B3(t) = t³ / 6

At lattice points (t=0): weights = [1/6, 2/3, 1/6, 0]
```

### Control Point Spacing

```python
# MIRTK: Direct spacing in mm
control_point_spacing = 5.0  # mm

# Deepali: Stride relative to image grid
stride = 4  # Control points every 4 voxels
# Effective spacing = stride * image_spacing
```

### Equivalence Verification

| Aspect | MIRTK | Deepali | Match |
|--------|-------|---------|-------|
| B-spline order | Cubic (3) | Cubic (3) | ✅ |
| Basis functions | Standard cubic | Standard cubic | ✅ |
| Boundary handling | Extrapolation mode | Border/constant | ✅ |
| Displacement storage | World mm | Normalized cube | ✅ (converted) |

---

## 6. Similarity Metrics

### Normalized Mutual Information (NMI)

**MIRTK Implementation** (`NormalizedMutualImageInformation.cc`):
```cpp
// Joint histogram with B-spline Parzen windows
// Entropy: H = -sum(p * log(p))
// NMI = (H(I) + H(T)) / H(I,T)
// Loss = 2 - NMI (minimized)

// Key features:
// - 64 or 256 bins (configurable)
// - Cubic B-spline Parzen window estimation
// - Foreground masking via FG_Overlap
// - Gradient computed analytically
```

**Deepali Implementation** (`losses/functional.py`):
```python
def nmi_loss(input, target, mask=None, num_bins=64, ...):
    """
    Normalized Mutual Information loss.

    Returns: 2 - (H(I) + H(T)) / H(I,T)
    Range: [0, 2] where 0 = perfect alignment

    Uses Gaussian Parzen window density estimation.
    """
    # Build joint histogram with Parzen windows
    # Compute marginal and joint entropies
    # Return normalized form
```

### Equivalence

| Aspect | MIRTK | Deepali | Match |
|--------|-------|---------|-------|
| Histogram bins | 64/256 | Configurable | ✅ |
| Parzen window | Cubic B-spline | Gaussian | ~✅ |
| Normalization | (H_I + H_T) / H_IT | Same | ✅ |
| Loss form | 2 - NMI | Same | ✅ |
| Foreground mask | FG_Overlap | Manual mask | ✅ |

### Foreground Masking (FG_Overlap)

```python
# MIRTK: Only computes similarity where both images have foreground
# DAREG equivalent:
def compute_foreground_overlap_mask(source, target, threshold=0.01):
    source_fg = source > threshold
    target_fg = target > threshold
    # Also exclude zero-gradient regions
    has_gradient = compute_gradient_magnitude(source) > 0
    return source_fg & target_fg & has_gradient
```

---

## 7. Regularization Terms

### Bending Energy

**MIRTK** (`BSplineFreeFormTransformation3D::BendingEnergy()`):
```cpp
// Sum of squared second derivatives
// E_bend = ∫∫∫ (∂²u/∂x²)² + (∂²u/∂y²)² + (∂²u/∂z²)²
//              + 2(∂²u/∂x∂y)² + 2(∂²u/∂y∂z)² + 2(∂²u/∂x∂z)² dxdydz

// Default weight: 0.001
```

**Deepali** (`losses/functional.py`):
```python
def bending_loss(u, mode='sobel', sigma=None, spacing=None):
    """
    Bending energy = sum of squared 2nd derivatives.
    Penalizes non-smooth deformations.
    """
    # Computes ∂²u/∂x², ∂²u/∂y², ∂²u/∂z² and mixed partials
    # Returns mean squared value
```

### Diffusion/Laplacian Energy

**MIRTK**:
```cpp
// Laplacian regularization
// E_laplacian = ∫∫∫ ||∇u||² dxdydz
// Default weight: 0.0005
```

**Deepali**:
```python
def diffusion_loss(u, ...):
    """
    Diffusion energy = 0.5 * sum(||∇u||²)
    Penalizes gradient magnitude.
    """
    return grad_loss(u, p=2, q=1) * 0.5
```

### Weight Equivalence

| Regularizer | MIRTK Default | DAREG Equivalent |
|-------------|---------------|------------------|
| Bending | 0.001 | 0.0005 - 0.001 |
| Laplacian/Diffusion | 0.0005 | 0.00025 - 0.0005 |

---

## 8. Optimization

### Conjugate Gradient Descent

**MIRTK** (`ConjugateGradientDescent.cc`):
```cpp
// Polak-Ribière formula
// β = max(0, (g_new · (g_new - g_old)) / (g_old · g_old))
// d_new = -g_new + β * d_old

// Features:
// - Adaptive step size with line search
// - Convergence on rejected streak
// - Gradient normalization option
```

**DAREG** (`registration/optimizers.py`):
```python
class ConjugateGradientOptimizer:
    """MIRTK-equivalent CG optimizer with Polak-Ribière."""

    def step(self, closure):
        # Compute gradient
        # Apply Polak-Ribière formula
        # Line search with backtracking
        # Accept/reject based on loss improvement
```

### Equivalence

| Aspect | MIRTK | DAREG | Match |
|--------|-------|-------|-------|
| Formula | Polak-Ribière | Polak-Ribière | ✅ |
| Line search | Adaptive | Adaptive | ✅ |
| Convergence | Rejected streak | Rejected streak | ✅ |
| Reset condition | β < 0 | β < 0 | ✅ |

---

## 9. Transform Composition

### MIRTK Composition

```cpp
// Multi-level FFD (MFFD)
// T_total = T_global ∘ T_local1 ∘ T_local2 ∘ ...
// Stored as separate DOF files, composed at runtime

// For velocity fields (SVFFD):
// Composition via BCH formula or explicit composition
```

### Deepali Composition

```python
from deepali.spatial import SequentialTransform

# Compose transforms
pipeline = SequentialTransform(rigid, affine, ffd)

# Forward application: T_n ∘ ... ∘ T_1
# For linear: matrix multiplication
# For non-rigid: iterative point transformation

# Access composed displacement
displacement_field = pipeline.disp(grid)
```

### Composition Equivalence

| Operation | MIRTK | Deepali |
|-----------|-------|---------|
| Linear compose | Matrix multiply | Matrix multiply |
| FFD compose | Add displacements* | Iterative transform |
| Save composed | `compose-dofs` | `save_transform()` |

*Note: MIRTK adds displacement fields for efficiency in some cases.

---

## 10. Image Warping

### MIRTK Image Transformation

```bash
# Command line
mirtk transform-image source.nii target.nii -dofin transform.dof.gz -target ref.nii

# Internally:
# 1. For each target voxel position
# 2. Apply transform to get source position
# 3. Interpolate source image at that position
```

### Deepali Image Transformation

```python
from deepali.spatial import ImageTransformer

transformer = ImageTransformer(
    transform=composed_transform,
    target=target_grid,      # Output grid
    source=source_grid,      # Input image grid
    sampling="linear",       # Interpolation mode
    padding="border",        # Edge handling
)

warped_image = transformer(source_image)
```

### Interpolation Modes

| Mode | MIRTK | Deepali | Use Case |
|------|-------|---------|----------|
| Linear | Default | `sampling="linear"` | Images |
| Nearest | `-interp NN` | `sampling="nearest"` | Segmentations |
| B-spline | `-interp BSpline` | `sampling="bspline"` | Smooth images |

---

## 11. Transform I/O

### MIRTK DOF File Format

```
Binary format (BSpline FFD 3D v4):
├── Magic number (int)
├── Transform type ID (int)
├── Grid dimensions (_x, _y, _z)
├── Grid spacing (_dx, _dy, _dz)
├── Grid origin
├── Orientation vectors (3x3)
├── Temporal info (if 4D)
├── Control point displacements [3 * _x * _y * _z]
└── Control point status flags
```

### DAREG Transform Save Format

```python
# From composer.py save_transform()
save_dict = {
    'state_dict': transform.state_dict(),
    'type': 'FreeFormDeformation',
    'domain': {
        'size': (X, Y, Z),
        'spacing': (dx, dy, dz),
        'origin': (ox, oy, oz),
        'direction': [[...], [...], [...]],
    },
    'stride': 4,
    'metadata': {...},

    # For SequentialTransform with FFD sub-transforms:
    'ffd_domains': {
        '2': {  # Index of FFD in sequence
            'size': ...,
            'spacing': ...,
            'origin': ...,
            'direction': ...,
            'stride': ...,
        }
    }
}
```

### Reconstruction for Application

```python
# Load and reconstruct FFD
checkpoint = torch.load(transform_path)
domain = checkpoint['domain']

# Create Grid (note coordinate order conversion)
grid = Grid(
    shape=(domain['size'][2], domain['size'][1], domain['size'][0]),
    spacing=(domain['spacing'][2], domain['spacing'][1], domain['spacing'][0]),
    origin=domain['origin'],
    direction=domain['direction'],
    align_corners=True,
)

# Reconstruct FFD
ffd = FreeFormDeformation(grid, stride=checkpoint['stride'])
ffd.load_state_dict(checkpoint['state_dict'])
```

---

## 12. Module-by-Module Mapping

### Complete Cross-Reference Table

| Function | MIRTK | Deepali | DAREG File |
|----------|-------|---------|------------|
| **Coordinate Conversion** |
| World → Internal | `WorldToLattice()` | `Grid.world_to_cube()` | core/grid.py |
| Internal → World | `LatticeToWorld()` | `Grid.cube_to_world()` | core/grid.py |
| **Transforms** |
| Rigid 6-DOF | `RigidTransformation` | `RigidTransform` | spatial/linear.py |
| Affine 12-DOF | `AffineTransformation` | `AffineTransform` | spatial/linear.py |
| B-Spline FFD | `BSplineFreeFormTransformation3D` | `FreeFormDeformation` | spatial/bspline.py |
| SVFFD | `BSplineFreeFormTransformationSV` | `StationaryVelocityFreeFormDeformation` | spatial/bspline.py |
| Composition | `MultiLevelTransformation` | `SequentialTransform` | spatial/composite.py |
| **Transform Application** |
| Apply to point | `LocalTransform()` | `transform.forward()` | spatial/base.py |
| Apply to image | `Resample()` | `ImageTransformer()` | spatial/transformer.py |
| Get displacement | `Displacement()` | `transform.disp()` | spatial/base.py |
| **Similarity Metrics** |
| NMI | `NormalizedMutualImageInformation` | `nmi_loss()` | losses/functional.py |
| SSD | `SumOfSquaredIntensityDifferences` | `ssd_loss()` | losses/functional.py |
| NCC | `NormalizedCrossCorrelation` | `ncc_loss()` | losses/functional.py |
| **Regularization** |
| Bending Energy | `BendingEnergy()` | `bending_loss()` | losses/functional.py |
| Diffusion | `LaplacianEnergy()` | `diffusion_loss()` | losses/functional.py |
| **Optimization** |
| CG Descent | `ConjugateGradientDescent` | `ConjugateGradientOptimizer` | registration/optimizers.py |
| Line Search | Built-in | Built-in | registration/optimizers.py |
| **I/O** |
| Save transform | `WriteDOF()` | `composer.save_transform()` | registration/composer.py |
| Load transform | `ReadDOF()` | `composer.load_transform()` | registration/composer.py |
| **Pyramid** |
| Multi-level | `GenericRegistrationFilter` | `create_pyramid()` | preprocessing/pyramid.py |
| Gaussian blur | `GaussianBlurring` | `torch.nn.functional.gaussian_blur` | Built-in |

---

## 13. Code Examples

### Example 1: Basic FFD Registration (Equivalent Pipeline)

**MIRTK Command**:
```bash
mirtk register target.nii source.nii -model FFD -ds 5 -sim NMI -be 0.001
```

**DAREG/Deepali Equivalent**:
```python
from deepali.core import Grid
from deepali.spatial import FreeFormDeformation
from deepali.losses.functional import nmi_loss, bending_loss

# Create grid from target image
grid = Grid.from_file("target.nii")

# Create FFD with 5mm control point spacing
# stride = spacing_mm / voxel_spacing
stride = int(5.0 / grid.spacing()[0])
ffd = FreeFormDeformation(grid, stride=stride)

# Optimization loop
optimizer = torch.optim.Adam(ffd.parameters(), lr=0.01)

for iteration in range(100):
    optimizer.zero_grad()

    # Warp source to target space
    warped = warp_image(source, ffd, target_grid)

    # Compute losses
    sim_loss = nmi_loss(warped, target, num_bins=64)
    reg_loss = bending_loss(ffd.disp(grid)) * 0.001

    total_loss = sim_loss + reg_loss
    total_loss.backward()
    optimizer.step()
```

### Example 2: Apply Transform to STL Vertices

**MIRTK Command**:
```bash
mirtk transform-points input.vtk output.vtk -dofin transform.dof.gz
```

**DAREG/Deepali Equivalent**:
```python
from deepali.core import Grid
from deepali.spatial import FreeFormDeformation

# Load transform with domain info
checkpoint = torch.load("transform.pth")
domain = checkpoint['domain']

# Reconstruct Grid
grid = Grid(
    shape=(domain['size'][2], domain['size'][1], domain['size'][0]),
    spacing=(domain['spacing'][2], domain['spacing'][1], domain['spacing'][0]),
    origin=domain['origin'],
    direction=domain['direction'],
    align_corners=True,
)

# Reconstruct FFD
ffd = FreeFormDeformation(grid, stride=checkpoint['stride'])
ffd.load_state_dict(checkpoint['state_dict'])

# Transform vertices
vertices = torch.from_numpy(stl_vertices)  # [N, 3] in world coords
points = vertices.unsqueeze(0)              # [1, N, 3]

with torch.no_grad():
    points_norm = grid.world_to_cube(points)
    transformed_norm = ffd.forward(points_norm, grid=False)
    transformed_world = grid.cube_to_world(transformed_norm)

output_vertices = transformed_world.squeeze(0).numpy()
```

### Example 3: Segmentation Propagation with Inverse

**MIRTK Approach**:
```bash
# Forward transform computed during registration
# Apply inverse to propagate segmentation
mirtk transform-image seg.nii warped_seg.nii -dofin transform.dof.gz -invert -interp NN
```

**DAREG/Deepali Approach (FFD)**:
```python
# FFD has no analytical inverse - use Newton-Raphson approximation
from DAREG.postprocessing.segmentation import ApproximateInverseTransform

# Create inverse transform
inverse = ApproximateInverseTransform(ffd, grid, iterations=10)

# Warp segmentation with nearest neighbor
transformer = ImageTransformer(
    transform=inverse,
    target=moving_frame_grid,
    source=static_grid,
    sampling="nearest",  # Critical for labels!
)

propagated_seg = transformer(static_segmentation)
```

**DAREG/Deepali Approach (SVFFD - has analytical inverse)**:
```python
# SVFFD inverse = negate velocity scale
inverse_svffd = svffd.inverse()

transformer = ImageTransformer(
    transform=inverse_svffd,
    sampling="nearest",
)

propagated_seg = transformer(static_segmentation)
```

---

## Summary: Key Points for MIRTK-Equivalent Results

### 1. Coordinate Handling
- Always use `grid.world_to_cube()` / `grid.cube_to_world()` for coordinate conversion
- Never directly sample FFD params - use `ffd.forward()`

### 2. Domain Info Preservation
- Save complete domain info with transforms (size, spacing, origin, direction)
- For SequentialTransform, save each FFD's domain separately in `ffd_domains`

### 3. B-Spline Evaluation
- Let deepali handle B-spline interpolation internally
- Control point params ≠ dense displacement field

### 4. Registration Pipeline
- Rigid → Affine → FFD sequence matches MIRTK multi-level approach
- Use same regularization weights for equivalent smoothness

### 5. Segmentation Propagation
- FFD: Use `ApproximateInverseTransform` (Newton-Raphson)
- SVFFD: Use `.inverse()` method (analytical)
- Always use nearest-neighbor interpolation for labels

---

## References

- MIRTK Source: `/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/MIRTK/`
- Deepali Source: `/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src/deepali/`
- DAREG Implementation: `/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/DAREG/`

**Document Version**: 1.0
**Last Updated**: February 2026
