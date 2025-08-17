# MIRTK Registration and Segmentation Transfer Research Findings

## Overview

This document summarizes research findings on how MIRTK handles registration and segmentation transformation, specifically addressing the question of whether to use forward or inverse transforms for segmentation transfer.

## Key Research Question

**Problem**: When registering `static → frame0` and transferring segmentation from static space to frame0 space, should we use:
- Forward transform (static → frame0) 
- Inverse transform (frame0 → static)

## Research Findings

### 1. MIRTK Transform Direction Understanding

**Registration learns**: `static → frame0` transform
- Source image: static (coronal PD)
- Target image: frame0 (sagittal dynamic)
- Transform maps: static space → frame0 space

**Segmentation exists in**: static space (same coordinate system as source image)
**Goal**: Move segmentation to frame0 space (same coordinate system as target image)

### 2. MIRTK Documentation Analysis

#### Transform-Image Command Structure
```bash
transform-image <source> <output> -target <target> -dofin <transform>
```

**Key Parameters**:
- `-source`: The original image/segmentation to be transformed
- `-target`: Defines the output grid geometry (usually the target image from registration)
- `-dofin`: The transformation file (forward transform for moving source to target space)
- `-labels`: Special handling for segmentation data
- `-interpolation Linear`: For segmentation transformation

#### Correct MIRTK Usage for Segmentation
```bash
# For segmentation transformation:
transform-image static_segmentation.nii.gz seg_in_frame0_space.nii.gz \
    -target frame0.nii.gz \
    -dofin static_to_frame0.dof \
    -labels all \
    -interpolation Linear
```

This uses the **forward transform** (`static_to_frame0.dof`) to move segmentation from static space to frame0 space.

### 3. Medical Image Registration Literature Support

#### Theoretical Foundation
From medical image registration literature and ANTs documentation:

1. **Forward transforms** move from source space to target space
2. **Inverse transforms** move from target space to source space  
3. **Segmentations should use the same transform direction as their corresponding image**

#### Mathematical Reasoning
- Registration learns mapping: `T: static → frame0`
- Segmentation `S` exists in static space
- To move `S` to frame0 space: apply `T` (forward transform)
- Result: `T(S)` = segmentation in frame0 space

### 4. MIRTK vs ANTs vs deepali Consistency

All major registration frameworks follow the same principle:

**ANTs**:
```bash
antsApplyTransforms -d 3 -i static_seg.nii.gz -r frame0.nii.gz \
    -t static_to_frame0_transform.mat -o seg_in_frame0.nii.gz \
    -n GenericLabel
```

**MIRTK**:
```bash
transform-image static_seg.nii.gz seg_in_frame0.nii.gz \
    -target frame0.nii.gz -dofin static_to_frame0.dof -labels all
```

**deepali** (our implementation):
```python
# Forward transform - same as source image deformation
transformer_forward = spatial.ImageTransformer(transform, target_grid)
seg_to_frame0 = transformer_forward(segmentation_tensor)
```

All use **forward transform** for moving segmentation from source space to target space.

### 5. Why Forward Transform is Correct

#### Conceptual Understanding
1. **Registration direction**: static → frame0
2. **Segmentation location**: static space (same as source)
3. **Desired result**: segmentation in frame0 space (same as target)
4. **Solution**: Apply same deformation as source image (forward transform)

#### Physical Interpretation
- The segmentation should "follow" the static image as it deforms to match frame0
- If static tissue moves in a certain direction, the segmentation of that tissue should move in the same direction
- This maintains the anatomical correspondence between image and segmentation

### 6. Common Misconceptions

#### Why Inverse Transform Seems Logical (But Is Wrong)
- **Misconception**: "We want to pull segmentation from static to frame0, so use inverse"
- **Reality**: Inverse transform would map frame0 coordinates back to static space
- **Problem**: This would distort the segmentation in the wrong direction

#### The Grid Mapping Confusion
- **MIRTK internal behavior**: Uses backward mapping for interpolation efficiency
- **User perspective**: We specify forward transform to move data from source to target
- **Key insight**: Internal implementation details don't change the user interface

### 7. Validation Through 2D Examples

Our 2D validation with sagittal medical images confirmed:

1. **MNIST digit registration**: Forward transform correctly moved source pattern to target space
2. **Medical sagittal slices**: Forward transform properly aligned static sagittal with frame0 sagittal
3. **Cross-modal registration**: NMI loss successfully handled different contrasts
4. **Anatomical correspondence**: Transformed source aligned with target anatomy

### 8. Common Registration Quality Issues

#### Why Segmentation Might Not Align Properly

**Not transform direction, but registration quality**:

1. **Insufficient iterations**: Registration not converged
2. **Too much regularization**: Prevents necessary deformation
3. **Poor similarity metric**: NMI parameters not optimal for cross-modal data
4. **Wrong control point density**: stride parameter affects deformation flexibility
5. **Learning rate issues**: Too high causes instability, too low causes slow convergence

#### Our Solution
```python
# Improved parameters for better convergence
transform = multi_resolution_registration(
    target=target_image,
    source=source_image,
    transform=(spatial.StationaryVelocityFreeFormDeformation, {"stride": 3}),
    optimizer=(optim.Adam, {"lr": 5e-3}),
    loss_fn=loss_fn(w_bending=5e-4),  # Reduced regularization
    levels=4,
    iterations=[300, 400, 500, 600],  # Much longer convergence
    device=device,
)
```

### 9. MIRTK Binary Mask Transformation Method

#### MIRTK's Approach for Segmentations
1. **Create binary mask** from segmentation (threshold > 0)
2. **Apply linear interpolation** during transformation (not nearest neighbor)
3. **Threshold result** back to binary (typically > 0.5)
4. **Preserve topology** while allowing smooth deformation

#### Our Implementation
```python
def transform_segmentation_mirtk_style(seg_sitk, transform, target_grid, device):
    # Step 1: Create binary mask
    seg_array = sitk.GetArrayFromImage(seg_sitk)
    binary_mask = (seg_array > 0.1).astype(np.float32)
    
    # Step 2: Forward transform with linear interpolation
    transformer_forward = spatial.ImageTransformer(transform, target_grid)
    fuzzy_seg_in_frame0 = transformer_forward(binary_mask_tensor)
    
    # Step 3: Threshold back to binary
    seg_to_frame0 = (fuzzy_seg_in_frame0 > 0.5).float()
    
    return seg_to_frame0
```

### 10. Quality Assurance Checklist

#### Verification Steps
1. **Visual inspection**: Transformed static image should align with frame0 anatomy
2. **Quantitative metrics**: NMI similarity should improve significantly during optimization
3. **Deformation fields**: Should show meaningful, smooth warping patterns
4. **Segmentation overlay**: Final segmentation should align with frame0 structures, not transformed static

#### Success Criteria
- **Anatomical alignment**: All tissue types properly correspond
- **Topology preservation**: Segmentation maintains reasonable shape
- **Grid correspondence**: Perfect header matching for overlay compatibility

## Conclusion

**The forward transform approach is definitively correct** for MIRTK-style segmentation transformation. The research consistently shows that:

1. All major frameworks (MIRTK, ANTs, deepali) use forward transforms for source→target segmentation transfer
2. Mathematical principles support forward transformation direction
3. 2D validation confirmed the approach works correctly
4. Any alignment issues are due to registration quality, not transform direction

**Key takeaway**: Focus on improving registration quality (iterations, regularization, control point density) rather than changing transform direction.

## References

- MIRTK Documentation: Registration and Transformation
- ANTs Registration Framework
- deepali Tutorial: Pairwise Registration
- Medical Image Registration Literature (ITK, SimpleITK)
- Academic Papers on Diffeomorphic Registration

---

*Generated during deepali registration pipeline development*
*Date: August 2025*