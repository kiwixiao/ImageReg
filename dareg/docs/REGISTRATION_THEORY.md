# Registration Theory: Mathematical Foundations of DAREG

This document explains the mathematical theory behind all regularization, similarity, and constraint terms used in DAREG's FFD and SVFFD registration pipelines.

---

## 1. Registration as Optimization

Image registration finds a spatial transformation **T** that aligns a **source** (moving) image to a **target** (fixed) image by minimizing a cost function:

```
T* = argmin_T  L_similarity(source ∘ T, target) + Σ w_i · L_reg_i(T)
```

where:
- **L_similarity**: Measures image dissimilarity (lower = better alignment)
- **L_reg_i**: Regularization terms that constrain the transformation
- **w_i**: Weights balancing similarity against regularization

### Why regularization is needed

Registration is **ill-posed**: infinitely many deformations can improve image similarity. Without constraints, the optimizer can produce:
- Non-smooth, physically implausible deformations
- Folding (where the transformation maps two distinct points to the same location)
- Noise amplification in low-signal regions

Regularization enforces **smoothness** and **physical plausibility**, converting the ill-posed problem into a well-posed one.

---

## 2. Similarity: Normalized Mutual Information (NMI)

DAREG uses NMI as the default similarity metric. NMI is information-theoretic and works across imaging modalities because it measures statistical dependence rather than intensity correspondence.

### Joint Histogram via Parzen Window Estimation

Given source image **X** and target image **Y**, a joint histogram is constructed using Parzen window (kernel density) estimation. Each voxel pair (x, y) contributes a soft vote to the 2D histogram bins via Gaussian kernels:

```
p(i, j) = (1/N) Σ_k  G(x_k - b_i; σ) · G(y_k - b_j; σ)
```

where **b_i**, **b_j** are bin centers, **G** is a Gaussian kernel, and **σ** is derived from bin width:

```
σ = bin_width / (2 · sqrt(2 · ln(2)))
```

This yields a differentiable approximation to the discrete histogram, enabling gradient-based optimization.

### Marginal and Joint Entropy

From the joint distribution **p(i, j)**, marginals are computed by summation:

```
p_X(i) = Σ_j p(i, j)        (marginal of source)
p_Y(j) = Σ_i p(i, j)        (marginal of target)
```

Entropies:

```
H(X) = -Σ_i  p_X(i) · log(p_X(i))
H(Y) = -Σ_j  p_Y(j) · log(p_Y(j))
H(X,Y) = -Σ_i Σ_j  p(i,j) · log(p(i,j))
```

### NMI Formula

```
NMI = 2 - (H(X) + H(Y)) / H(X,Y)
```

This is the **MIRTK convention** where NMI is formulated as a loss (lower = better alignment):
- Perfect alignment: H(X,Y) is minimal → NMI approaches 0
- No alignment: H(X,Y) is maximal → NMI approaches 2

**Code**: `deepali/src/deepali/losses/functional.py` (`mi_loss` with `normalized=True`)

### Why NMI Works for Multimodal Images

Unlike MSE or cross-correlation, NMI makes no assumption about the functional relationship between intensities. It only requires that aligned images have a more predictable (lower entropy) joint distribution. This makes it robust for:
- Different MRI sequences (T1 vs T2)
- Dynamic MRI where contrast changes over time
- CT to MRI registration

### Foreground Overlap Masking (FG_Overlap)

MIRTK computes similarity **only where both images have valid foreground signal**. DAREG replicates this with a three-part mask:

1. **Intensity thresholding**: After normalizing both images to [0, 1], voxels below `threshold` (default 0.01) are classified as background
2. **Gradient masking**: Voxels with gradient magnitude below 1e-5 are excluded (zero-gradient regions carry no registration information)
3. **Intersection**: The final mask is the AND of both foreground masks and both gradient masks

```
mask = (source > threshold) AND (target > threshold) AND (|∇source| > ε) AND (|∇target| > ε)
```

This prevents artifacts at image boundaries and avoids wasting optimization effort on background regions.

**Code**: `DAREG/registration/base.py` (`_compute_foreground_overlap_mask`)

### ROI Mask

An optional **ROI mask** can further restrict the similarity computation to a region of interest. When provided, it is intersected with the foreground overlap mask:

```
final_mask = fg_overlap_mask AND roi_mask
```

---

## 3. FFD: Free-Form Deformation

FFD parameterizes the deformation as a cubic B-spline interpolation over a regular lattice of **control points**.

### Control Point Lattice

A 3D grid of control points is placed over the image domain with uniform spacing (default: 4 mm). The displacement at any voxel is computed by cubic B-spline interpolation from the surrounding 4×4×4 neighborhood of control points:

```
u(x) = Σ_l Σ_m Σ_n  B_l(s) · B_m(t) · B_n(v) · φ_{i+l, j+m, k+n}
```

where **B** are cubic B-spline basis functions, **(s, t, v)** are local coordinates within the control point cell, and **φ** are the control point displacements (the optimization parameters).

### Multi-Resolution Pyramid

Registration proceeds coarse-to-fine across multiple resolution levels (default: 4 levels). At each level:
- Both images are downsampled
- The control point grid density matches the image resolution
- Optimization runs for a fixed number of iterations

Coarse levels capture large-scale motion; fine levels refine local detail.

### Optimizer: Conjugate Gradient (Polak-Ribiere)

FFD uses a **Conjugate Gradient** optimizer with Polak-Ribiere update and adaptive line search, matching MIRTK's optimization strategy:

```
d_k = -g_k + β_k · d_{k-1}
β_k = max(0, (g_k · (g_k - g_{k-1})) / (g_{k-1} · g_{k-1}))    (Polak-Ribiere)
```

Parameters:
- `epsilon`: 1e-4 (convergence threshold)
- `delta`: 1e-12 (numerical stability)
- `max_rejected_streak`: 1 (reset after 1 rejected step)

**Code**: `DAREG/registration/optimizers.py` (`ConjugateGradientOptimizer`)

---

## 4. Regularization Terms

All regularization terms operate on the displacement field (FFD) or velocity field (SVFFD) and penalize spatial derivatives to enforce smoothness.

### World Coordinate Scaling

Medical images often have **anisotropic voxel spacing** (e.g., 1×1×3 mm). Regularization computed in voxel coordinates would disproportionately penalize deformation along the high-resolution axes. DAREG converts to world (mm) coordinates:

```
scale = extent_mm / 2.0     (per dimension, converting normalized [-1,1] to mm)
u_world = u_normalized × scale
```

The scale tensor has shape `[1, 3, 1, 1, 1]` for broadcasting across the displacement field.

**Code**: `DAREG/registration/ffd.py` (`_compute_world_scale`)

### Finite Differences

All spatial derivatives are computed using **central finite differences**:

```
∂u/∂x ≈ (u[i+1] - u[i-1]) / 2        (1st order)
∂²u/∂x² ≈ u[i+1] - 2·u[i] + u[i-1]   (2nd order)
```

Boundary voxels (where the stencil extends outside the domain) are handled by zero-padding or replication.

### 4a. Bending Energy (BE)

**Physical meaning**: Penalizes **curvature** of the deformation field. Analogous to the energy of bending a thin elastic plate — produces smooth, low-curvature deformations.

**Formula**:

```
BE = mean( Σ_{i,j}  w_{ij} · (∂²u/∂x_i ∂x_j)² )
```

where the sum runs over all pairs of spatial dimensions (including mixed derivatives), and:

```
w_{ij} = 1  if i = j   (unmixed: ∂²u/∂x², ∂²u/∂y², ∂²u/∂z²)
w_{ij} = 2  if i ≠ j   (mixed:   ∂²u/∂x∂y, ∂²u/∂x∂z, ∂²u/∂y∂z)
```

The factor of 2 for mixed derivatives accounts for the symmetry of the Hessian (∂²u/∂x∂y = ∂²u/∂y∂x).

In 3D with displacement components (u₁, u₂, u₃), the full expansion is:

```
BE = mean( Σ_c [ (∂²u_c/∂x²)² + (∂²u_c/∂y²)² + (∂²u_c/∂z²)²
                + 2·(∂²u_c/∂x∂y)² + 2·(∂²u_c/∂x∂z)² + 2·(∂²u_c/∂y∂z)² ] )
```

**Derivative order**: 2nd
**MIRTK default weight**: 0.001 (FFD), 0.0005 (SVFFD)
**Code**: `deepali/src/deepali/losses/functional.py` (`bending_loss`)

### 4b. Diffusion Energy (DE)

**Physical meaning**: Penalizes **spatial variation** of the displacement field. Encourages locally constant (uniform) displacement — the deformation field should not change rapidly from voxel to voxel.

**Formula**:

```
DE = 0.5 · mean( Σ_c Σ_i (∂u_c/∂x_i)² )
```

Expanded in 3D:

```
DE = 0.5 · mean( Σ_c [ (∂u_c/∂x)² + (∂u_c/∂y)² + (∂u_c/∂z)² ] )
```

This is the **Dirichlet energy** of the displacement field, the most common smoothness prior in variational registration.

**Derivative order**: 1st (softer than bending energy)
**MIRTK default weight**: 0.0005 (FFD), 0.00025 (SVFFD)
**Code**: `deepali/src/deepali/losses/functional.py` (`diffusion_loss`)

### 4c. Laplacian Energy (LE)

**Physical meaning**: Encourages **harmonic deformation** (∇²u ≈ 0). A harmonic displacement field satisfies Laplace's equation and represents the smoothest possible deformation consistent with boundary conditions.

**Formula**:

```
LE = mean( Σ_c (∇²u_c)² )
```

where the Laplacian of each component is:

```
∇²u_c = ∂²u_c/∂x² + ∂²u_c/∂y² + ∂²u_c/∂z²
```

**Key difference from Bending Energy**: Laplacian energy **sums the second derivatives first, then squares** the sum. Bending energy **squares each second derivative, then sums**. This distinction matters:

```
BE ∝ Σ_{i,j} (∂²u/∂x_i∂x_j)²    → penalizes each derivative independently
LE ∝ (Σ_i ∂²u/∂x_i²)²            → allows cancellation between dimensions
```

Laplacian energy permits deformations where positive curvature in one dimension is offset by negative curvature in another (as long as the net Laplacian is zero). Bending energy penalizes all curvature regardless.

**Derivative order**: 2nd
**MIRTK default weight**: 0.0005
**Code**: `DAREG/registration/ffd.py` (`_compute_laplacian_loss`)

### 4d. Comparison Table: BE vs DE vs LE

| Property | Bending Energy (BE) | Diffusion Energy (DE) | Laplacian Energy (LE) |
|----------|--------------------|-----------------------|----------------------|
| **Penalizes** | Curvature (all 2nd derivatives) | Spatial variation (1st derivatives) | Non-harmonic deformation |
| **Derivative order** | 2nd | 1st | 2nd |
| **Formula** | Σ (∂²u/∂xᵢ∂xⱼ)² | 0.5 · Σ (∂u/∂xᵢ)² | Σ (∇²u)² |
| **Sums then squares?** | No (squares then sums) | No (squares then sums) | Yes (sums then squares) |
| **Permits cancellation?** | No | N/A | Yes (across dimensions) |
| **Stiffness** | High (rigid-like) | Moderate (fluid-like) | Moderate (harmonic) |
| **Physical analogy** | Thin plate bending | Heat diffusion | Electrostatic potential |
| **FFD default weight** | 0.001 | 0.0005 | 0.0005 |

---

## 5. SVFFD: Stationary Velocity Free-Form Deformation

### Core Idea

Instead of directly optimizing a displacement field **u**, SVFFD optimizes a **stationary velocity field** **v** on a B-spline control point lattice. The displacement is obtained by integrating the velocity field:

```
u = exp(v)
```

This is the exponential map from Lie group theory: the velocity field generates a one-parameter group of diffeomorphisms.

### Exponential Map via Scaling-and-Squaring

The exponential map is computed numerically using the **scaling-and-squaring** algorithm:

```
1. Scale:    u₀ = v / 2^N        (N = integration steps, default 5)
2. Square:   u_{k+1} = u_k + u_k ∘ (id + u_k)    for k = 0, ..., N-1
3. Result:   u = u_N ≈ exp(v)
```

Each squaring step composes the displacement with itself, doubling the effective integration time. After N steps, the result approximates the flow of velocity field **v** for unit time.

**Default integration steps**: 5 (giving 2⁵ = 32 sub-steps)
**Code**: `deepali/src/deepali/core/flow.py` (`expv`)

### Why SVFFD Guarantees Diffeomorphism

The exponential of a smooth velocity field is always a **diffeomorphism** (smooth, invertible, with smooth inverse). This means:
- **det(J) > 0 everywhere**: No folding or topology violations
- The transformation preserves the topological structure of anatomical regions
- Physically plausible: tissue cannot pass through itself

In practice, the discrete approximation may have small violations at very large deformations, but SVFFD is far more robust than unconstrained FFD.

### Analytical Inverse

A key advantage of SVFFD: the inverse transformation is obtained by simply **negating the velocity field**:

```
T⁻¹ = exp(-v)
```

No iterative inversion is needed (unlike FFD, which requires Newton-Raphson approximation). This is particularly valuable for segmentation propagation, where inverse transforms are applied to warp labels from the reference frame to each target frame.

### Optimizer: Adam

SVFFD uses **Adam** rather than Conjugate Gradient because:
- Velocity fields have more complex loss landscapes than direct displacement fields
- Adam's per-parameter adaptive learning rates handle the varying gradient scales across spatial dimensions
- Momentum helps navigate the non-linear relationship between velocity and displacement

Convergence is monitored with patience-based early stopping (`convergence_delta`: 1e-6, `convergence_patience`: 20 iterations).

**Code**: `DAREG/registration/svffd.py` (line 241)

---

## 6. SVFFD-Specific Regularization

### 6a. Velocity Field Bending + Diffusion

The same bending and diffusion energy formulas from Section 4 are applied, but to the **velocity field v** rather than the displacement field u:

```
BE(v) = mean( Σ_{i,j} w_{ij} · (∂²v/∂x_i ∂x_j)² )
DE(v) = 0.5 · mean( Σ_c Σ_i (∂v_c/∂x_i)² )
```

Default weights are **lower than FFD** because the exponential map itself provides implicit regularization (smooth v → smooth exp(v)):

| Term | FFD Weight | SVFFD Weight |
|------|-----------|-------------|
| Bending | 0.001 | 0.0005 |
| Diffusion | 0.0005 | 0.00025 |

### 6b. Velocity Field Gaussian Smoothing

An **explicit smoothing** operation applied directly to the velocity field during optimization (not as a loss term).

**Formula**:

```
v_smoothed = G_σ * v
```

where G_σ is a separable 3D Gaussian kernel with standard deviation σ (specified in mm).

**Implementation details**:
- σ is converted from mm to voxels per dimension: `σ_voxel = σ_mm / spacing_mm`
- Smoothing is skipped for dimensions where σ_voxel < 0.5
- Kernel size: `max(3, min(4σ | 1, 15))` (odd-sized, capped at 15)
- Applied as separable 1D convolutions along each axis
- Padding mode: replicate (to avoid boundary artifacts)

**When to use**: For noisy velocity fields or when large-scale coherent deformations are expected. Disabled by default (`velocity_smoothing_sigma`: 0.0 mm).

**Explicit vs implicit smoothing**: This is explicit smoothing (directly modifies the field), contrasted with implicit smoothing via loss terms (bending/diffusion energy) that guide the optimizer toward smooth solutions through gradient descent.

**Code**: `DAREG/registration/svffd.py` (`_smooth_velocity_field`)

### 6c. Laplacian on Velocity Field

Same formula as FFD Laplacian (Section 4c) but applied to the velocity field:

```
LE(v) = mean( Σ_c (∇²v_c)² )
```

**Default**: Disabled (weight = 0.0). Enable for additional harmonic smoothness constraint on the velocity field.

**Code**: `DAREG/registration/svffd.py` (`_compute_laplacian_loss`)

### 6d. Jacobian Determinant Penalty

Penalizes **volume compression and folding** using a log-barrier formulation.

**Formula**:

```
JAC = mean( (max(0, -log(det(J))))² )
```

where **J** is the Jacobian matrix of the transformation (identity + displacement gradient), and det(J) is clamped to a minimum of ε = 1e-6.

**Behavior by region**:

| det(J) value | Physical meaning | log(det(J)) | Penalty |
|-------------|-----------------|-------------|---------|
| < 0 | Folding (topology violation) | undefined (clamped) | Very high |
| 0 < det(J) << 1 | Severe compression | Large negative | High |
| det(J) ≈ 1 | Volume-preserving | ≈ 0 | Zero |
| det(J) > 1 | Expansion | Positive | Zero |

**Why log-barrier**: The logarithm provides a smooth gradient landscape that becomes increasingly steep as det(J) approaches zero, naturally guiding the optimizer away from folding without a hard constraint. Only compression/folding is penalized (expansion is free).

**Default**: Disabled (weight = 0.0). SVFFD already promotes diffeomorphism through its velocity-field parameterization, so this penalty is a supplementary safeguard for aggressive deformations.

**Code**: `DAREG/registration/svffd.py` (`_compute_jacobian_loss`)

---

## 7. Gradient Processing (MIRTK-style)

After computing gradients via backpropagation, DAREG applies four MIRTK-style post-processing steps before the optimizer updates parameters.

### 7a. Gradient Normalization (Preconditioning)

```
σ = 0.5 · max(||g||)
g_normalized = g / (||g|| + σ)
```

where ||g|| is the per-control-point gradient norm. This prevents any single control point from dominating the update and equalizes the effective step size across the lattice. The factor σ ensures numerical stability when gradients are small.

### 7b. Boundary Constraint

Gradients within `support_radius` (default: 2) voxels of the control point lattice boundary are zeroed:

```
g[:, :, :2, :, :] = 0     (and symmetrically for all faces)
```

This enforces zero displacement at the domain boundary, preventing edge artifacts.

### 7c. Small Gradient Thresholding

Gradients with magnitude below 1e-8 are zeroed:

```
g[||g|| < 1e-8] = 0
```

This eliminates numerical noise from near-zero gradients that contribute nothing meaningful to the optimization.

### 7d. Support Region Constraint

Control points whose B-spline support region lies predominantly **outside the foreground overlap mask** are frozen (gradients zeroed). The foreground overlap mask is downsampled to control point resolution using average pooling, and control points where the average mask value falls below `support_region_threshold` (default: 0.3) do not update.

This ensures that control points in background or partially-visible regions do not introduce spurious deformations driven by noise rather than anatomy.

**Code**: `DAREG/registration/ffd.py` (`_process_gradients`, `_apply_support_region_constraint`)

---

## 8. FFD vs SVFFD Comparison

| Property | FFD | SVFFD |
|----------|-----|-------|
| **Parameterization** | Displacement field u | Velocity field v |
| **Deformation** | u (direct) | u = exp(v) (integrated) |
| **Invertibility** | Approximate (Newton-Raphson) | Analytical: exp(-v) |
| **Topology preservation** | Not guaranteed (det(J) can be ≤ 0) | Guaranteed (diffeomorphic) |
| **Optimizer** | Conjugate Gradient (Polak-Ribiere) | Adam |
| **Speed** | Faster (no integration step) | Slower (scaling-and-squaring) |
| **Default bending weight** | 0.001 | 0.0005 |
| **Default diffusion weight** | 0.0005 | 0.00025 |
| **Laplacian energy** | Enabled (0.0005) | Optional (disabled) |
| **Jacobian penalty** | N/A | Optional (disabled) |
| **Velocity smoothing** | N/A | Optional (disabled) |
| **Best for** | Small-to-moderate deformations | Large deformations, segmentation propagation |

---

## 9. Total Loss Functions

### FFD Total Loss

```
L_FFD = NMI + w_be · BE(u) + w_de · DE(u) + w_le · LE(u)
```

### SVFFD Total Loss

```
L_SVFFD = NMI + w_be · BE(v) + w_de · DE(v) + w_le · LE(v) + w_jac · JAC(u)
```

Note: Bending, diffusion, and Laplacian are computed on the **velocity field v** (before exponentiation). The Jacobian penalty is computed on the resulting **displacement field u** (after exponentiation), since it measures properties of the actual deformation.

### Default Weight Table

| Term | Symbol | FFD Default | SVFFD Default | MIRTK Equivalent |
|------|--------|------------|--------------|-------------------|
| NMI (similarity) | — | 1.0 (implicit) | 1.0 (implicit) | SIM[Image similarity] |
| Bending energy | w_be | 0.001 | 0.0005 | BE(T) = 0.001 |
| Diffusion energy | w_de | 0.0005 | 0.00025 | — |
| Laplacian energy | w_le | 0.0005 | 0.0 (disabled) | LE(T) = 0.0005 |
| Jacobian penalty | w_jac | N/A | 0.0 (disabled) | — |

### MIRTK Reference Formula

MIRTK's standard energy function for FFD registration:

```
E = NMI + 0.001 · BE(T) + 0.0005 · LE(T)
```

DAREG extends this with additional diffusion energy and (for SVFFD) Jacobian and velocity smoothing terms.

---

## 10. References

### Code Locations

| Concept | File | Function/Method |
|---------|------|----------------|
| NMI loss | `deepali/src/deepali/losses/functional.py` | `mi_loss` (with `normalized=True`) |
| Bending energy | `deepali/src/deepali/losses/functional.py` | `bending_loss` |
| Diffusion energy | `deepali/src/deepali/losses/functional.py` | `diffusion_loss` |
| Laplacian energy (FFD) | `DAREG/registration/ffd.py` | `_compute_laplacian_loss` |
| Laplacian energy (SVFFD) | `DAREG/registration/svffd.py` | `_compute_laplacian_loss` |
| Jacobian penalty | `DAREG/registration/svffd.py` | `_compute_jacobian_loss` |
| Velocity smoothing | `DAREG/registration/svffd.py` | `_smooth_velocity_field` |
| FG_Overlap masking | `DAREG/registration/base.py` | `_compute_foreground_overlap_mask` |
| World coordinate scaling | `DAREG/registration/ffd.py` | `_compute_world_scale` |
| Gradient processing | `DAREG/registration/ffd.py` | `_process_gradients` |
| CG optimizer | `DAREG/registration/optimizers.py` | `ConjugateGradientOptimizer` |
| Scaling-and-squaring | `deepali/src/deepali/core/flow.py` | `expv` |
| Regularization (FFD) | `DAREG/registration/ffd.py` | `_compute_regularization` |
| Total loss (SVFFD) | `DAREG/registration/svffd.py` | closure in `register()` |
