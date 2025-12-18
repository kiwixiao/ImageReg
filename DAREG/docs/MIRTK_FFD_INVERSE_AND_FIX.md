# MIRTK FFD Inverse Transform Handling and DAREG Fix

## Overview

This document explains how MIRTK handles inverse transforms for FFD (Free-Form Deformation) and what bug was fixed in DAREG's segmentation propagation.

---

## 1. The Problem: Segmentation Propagation Requires Inverse Transform

In motion registration pipelines:
- **Registration direction**: We register `frame_i → frame_0` to align all frames to a reference
- **Segmentation propagation**: We need to move segmentation from `frame_0 → frame_i` (opposite direction)

This means we need the **inverse** of the registration transform.

---

## 2. How MIRTK Handles FFD Inverse

### 2.1 FFD Has No Closed-Form Inverse

Unlike rigid/affine transforms (which have matrix inverse), FFD (B-spline Free-Form Deformation) has **no analytical inverse**. The forward transform is:

```
y = x + u(x)  where u(x) is the displacement field
```

To find the inverse (given y, find x), we need to solve:
```
x + u(x) = y  →  find x such that T(x) = y
```

This is a nonlinear equation with no closed-form solution.

### 2.2 MIRTK's Newton-Raphson Solution

MIRTK uses **Newton-Raphson iteration** to approximate the inverse. Found in:
- `MIRTK/Modules/Transformation/src/TransformationInverse.cc`

#### Algorithm

```cpp
// Newton-Raphson iteration to find x such that T(x) = target_point
bool multivariate_newton_raphson_iterate(F f, vector_type &x, int digits, uintmax_t max_iter=100) {
    do {
        // f0 = residual = T(x) - target_point
        // f1 = Jacobian matrix of T at current x
        std::tie(f0, f1) = f(x);

        // Invert Jacobian
        if (!invert_matrix(f1, f1_inv)) return false;

        // Newton step: x_new = x_old - J^(-1) * f(x_old)
        d_x = prod(f1_inv, f0);
        x -= d_x;

        // Check convergence
    } while (iter++ < max_iter && !converged);

    return true;
}
```

#### Key Parameters
- **Precision**: 6 digits (ldexp(1.0, 1-6))
- **Max iterations**: 100
- **Convergence**: When `|d_x| < |x * factor|`

### 2.3 MIRTK InverseDisplacement Functions

```cpp
// For general transforms
bool EvaluateInverse(const Transformation *T,
                     double &x, double &y, double &z, double t, double t0);

// For local (FFD) component
bool EvaluateLocalInverse(const Transformation *T,
                          double &x, double &y, double &z, double t, double t0);

// For global (affine) component
bool EvaluateGlobalInverse(const Transformation *T,
                           double &x, double &y, double &z, double t, double t0);
```

### 2.4 When MIRTK Uses Inverse

In `transform-image.cc`:
```cpp
if (inv[i]) {
    // Use Newton-Raphson to compute inverse displacement
    nsingular += dof[i]->InverseDisplacement(*local, source_attr, target_attr);
} else {
    // Use forward displacement directly
    dof[i]->Displacement(*local, source_attr, target_attr);
}
```

---

## 3. Deepali/DAREG Transform Types

### 3.1 FreeFormDeformation (FFD)
- Direct displacement field: `y = x + u(x)`
- **NO inverse() method** - raises `NotImplementedError`
- Would need Newton-Raphson implementation (like MIRTK)

### 3.2 StationaryVelocityFreeFormDeformation (SVFFD)
- Velocity field integrated via scaling-and-squaring: `y = exp(v)(x)`
- **HAS inverse() method** - simply negate velocity field: `x = exp(-v)(y)`
- Diffeomorphic (topology-preserving)

```python
# SVFFD inverse (from deepali/spatial/bspline.py)
def inverse(self, link=False, update_buffers=False):
    inv = shallow_copy(self)
    inv.exp = self.exp.inverse()  # Negates the scaling factor
    return inv
```

---

## 4. The Bug in DAREG

### 4.1 Location
`DAREG/registration/motion.py` in two functions:
- `_propagate_segmentation()` (lines 652-662)
- `_propagate_segmentation_incremental()` (lines 1188-1195)

### 4.2 Buggy Code

```python
# BUGGY CODE - silently uses WRONG direction!
if hasattr(transform, 'inverse'):
    try:
        inverse_transform = transform.inverse()
    except Exception:
        logger.warning(f"Could not compute inverse for frame {composed.source_idx}")
        inverse_transform = transform  # BUG: Forward transform, not inverse!
else:
    inverse_transform = transform  # BUG: Forward transform, not inverse!
```

### 4.3 Why This Was Wrong

1. `hasattr(transform, 'inverse')` returns `True` for FFD (base class has the method)
2. But calling `transform.inverse()` raises `NotImplementedError` for FFD
3. The `except Exception` catches this and falls back to **forward transform**
4. Result: Segmentation warped in **wrong direction** (~15 voxel / 11-18mm offset)

---

## 5. The Fix Applied

### 5.1 New Code

```python
# FIXED CODE - raises clear error instead of using wrong direction
# Compute inverse transform for segmentation propagation
# Direction: We have transforms that map frame_i -> frame_0
# For segmentation, we need to map frame_0 -> frame_i (inverse direction)
#
# CRITICAL: Do NOT fall back to forward transform - that would warp
# the segmentation in the WRONG direction (approximately 15 voxel offset)
#
# SVFFD has analytical inverse (negated velocity field)
# FFD does NOT have inverse - MIRTK uses Newton-Raphson approximation
try:
    inverse_transform = transform.inverse()
except NotImplementedError as e:
    # FFD (non-diffeomorphic) doesn't support inverse
    # This is a critical error - cannot propagate segmentation correctly
    error_msg = (
        f"Cannot compute inverse transform for frame {composed.source_idx}. "
        f"Transform type '{type(transform).__name__}' does not support inverse(). "
        f"For segmentation propagation, use SVFFD (model='svffd') which has "
        f"analytical inverse, or implement Newton-Raphson approximation for FFD."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg) from e
except Exception as e:
    # Unexpected error during inverse computation
    error_msg = (
        f"Error computing inverse transform for frame {composed.source_idx}: {e}. "
        f"Segmentation propagation requires a valid inverse transform."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg) from e
```

### 5.2 Behavior After Fix

| Transform Type | Behavior |
|---------------|----------|
| **SVFFD** (default) | Works correctly - has analytical inverse |
| **FFD** | Raises clear error with guidance |

---

## 6. Recommendations

### 6.1 For Motion Registration with Segmentation Propagation
**Always use SVFFD** (`model='svffd'`) which is the default in DAREG:

```python
from DAREG.registration import FFDRegistration

# SVFFD (default) - has analytical inverse
ffd_reg = FFDRegistration(model='svffd')  # Recommended

# FFD - will fail during segmentation propagation
ffd_reg = FFDRegistration(model='ffd')  # Will raise error
```

### 6.2 Future Enhancement: Newton-Raphson for FFD
If FFD support is needed, implement Newton-Raphson inverse like MIRTK:

```python
def newton_raphson_inverse(transform, target_point, max_iter=100, tol=1e-6):
    """
    Compute inverse of FFD transform at a point using Newton-Raphson.

    Given y = T(x), find x such that T(x) = y
    """
    x = target_point.clone()  # Initial guess

    for _ in range(max_iter):
        # Compute T(x) and Jacobian J(x)
        Tx = transform(x)
        J = transform.jacobian(x)

        # Residual: f(x) = T(x) - y
        residual = Tx - target_point

        # Newton step: x_new = x - J^(-1) * f(x)
        J_inv = torch.linalg.inv(J)
        delta = J_inv @ residual
        x = x - delta

        if delta.norm() < tol:
            break

    return x
```

---

## 7. Summary

| Aspect | MIRTK | DAREG (before fix) | DAREG (after fix) |
|--------|-------|-------------------|-------------------|
| FFD inverse | Newton-Raphson iteration | Silent fallback to forward (BUG) | Clear error with guidance |
| SVFFD inverse | N/A (uses FFD) | Works correctly | Works correctly |
| Segmentation accuracy | Correct | ~15 voxel offset with FFD | Correct with SVFFD |

---

## References

- MIRTK Source: `MIRTK/Modules/Transformation/src/TransformationInverse.cc`
- Deepali SVFFD: `deepali/src/deepali/spatial/bspline.py`
- DAREG Fix: `DAREG/registration/motion.py` lines 652-681, 1188-1217
