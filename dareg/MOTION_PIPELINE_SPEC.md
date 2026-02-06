# DAREG Motion Registration Pipeline Specification

## Overview

This document specifies the complete motion registration pipeline, including expected inputs, outputs, array dimensions, and transform directions at each step.

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAREG MOTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS:                                                                    │
│  ├── 4D Dynamic Image: [X, Y, Z, T] e.g., (192, 192, 12, 26)               │
│  ├── Static 3D Image:  [X, Y, Z]    e.g., (256, 256, 160) high-res         │
│  └── Segmentation:     [X, Y, Z]    e.g., (256, 256, 160) on static        │
│                                                                             │
│  STEP 0: Extract Frames                                                     │
│  ├── Input:  4D image [X,Y,Z,T]                                            │
│  ├── Output: List of 3D frames [X,Y,Z] × T                                 │
│  └── Files:  frames/frame_000.nii.gz, frame_001.nii.gz, ...                │
│                                                                             │
│  STEP 1: ALIGNMENT (Static → Frame 0)                                       │
│  ├── Purpose: Bring static image into frame 0 coordinate space             │
│  ├── Direction: Static → Frame 0 (forward transform)                       │
│  │                                                                          │
│  │   ┌─────────┐    Rigid     ┌─────────┐   Affine    ┌─────────┐   FFD    │
│  │   │ Static  │ ──────────▶ │ Rigid   │ ──────────▶ │ Affine  │ ──────▶ │
│  │   │ Image   │   6 DOF     │ Aligned │   12 DOF    │ Aligned │  B-spline│
│  │   └─────────┘             └─────────┘             └─────────┘          │
│  │                                                                          │
│  ├── Array Dimensions:                                                      │
│  │   - Static input:     [C, D, H, W] or [D, H, W]                         │
│  │   - Common grid:      Matches frame 0 grid shape                        │
│  │   - Resampled static: [1, D_common, H_common, W_common]                 │
│  │   - Output warped:    [1, D_common, H_common, W_common]                 │
│  │                                                                          │
│  └── Files:                                                                 │
│      ├── alignment/source_after_rigid_common.nii.gz                        │
│      ├── alignment/source_after_affine_common.nii.gz                       │
│      ├── alignment/source_after_ffd_common.nii.gz                          │
│      ├── alignment/aligned_static_original_resolution.nii.gz               │
│      ├── alignment/aligned_segmentation.nii.gz                             │
│      └── alignment/alignment_transform.pth                                 │
│                                                                             │
│  STEP 2: PAIRWISE REGISTRATION (Consecutive Frames)                         │
│  ├── Purpose: Track motion between adjacent frames                          │
│  ├── Direction: Frame N → Frame N-1 (maps later frame to earlier)          │
│  │                                                                          │
│  │   Frame 0 ◀── T_01 ── Frame 1                                           │
│  │   Frame 1 ◀── T_12 ── Frame 2                                           │
│  │   Frame 2 ◀── T_23 ── Frame 3                                           │
│  │   ...                                                                    │
│  │                                                                          │
│  ├── Array Dimensions:                                                      │
│  │   - Source frame:  [1, 1, D, H, W] (5D for registration)                │
│  │   - Target frame:  [1, 1, D, H, W]                                      │
│  │   - Transform:     Maps source coords → target coords                   │
│  │                                                                          │
│  └── Files:                                                                 │
│      ├── pairwise/pairwise_0_1.pth  (Frame 1 → Frame 0)                    │
│      ├── pairwise/pairwise_1_2.pth  (Frame 2 → Frame 1)                    │
│      └── ...                                                                │
│                                                                             │
│  STEP 3: COMPOSE LONGITUDINAL TRANSFORMS                                    │
│  ├── Purpose: Create direct Frame N → Frame 0 transforms                   │
│  ├── Composition:                                                           │
│  │   - T_0→1 = T_01 (direct)                                               │
│  │   - T_0→2 = T_01 ∘ T_12                                                 │
│  │   - T_0→3 = T_01 ∘ T_12 ∘ T_23                                          │
│  │                                                                          │
│  └── Files:                                                                 │
│      ├── longitudinal/longitudinal_0_to_1_composed.pth                     │
│      ├── longitudinal/longitudinal_0_to_2_composed.pth                     │
│      └── ...                                                                │
│                                                                             │
│  STEP 4: REFINE LONGITUDINAL TRANSFORMS (Optional)                          │
│  ├── Purpose: Fine-tune composed transforms via direct registration        │
│  ├── Method: Use composed as initialization, run 1-level FFD               │
│  │                                                                          │
│  └── Files:                                                                 │
│      ├── longitudinal/longitudinal_0_to_1_refined.pth                      │
│      ├── longitudinal/longitudinal_0_to_2_refined.pth                      │
│      └── ...                                                                │
│                                                                             │
│  STEP 5: PROPAGATE SEGMENTATION                                             │
│  ├── Purpose: Move segmentation from Frame 0 to each Frame N               │
│  ├── Direction: Frame 0 → Frame N (INVERSE of longitudinal)                │
│  │                                                                          │
│  │   CRITICAL: We have T: Frame N → Frame 0                                │
│  │             We need T⁻¹: Frame 0 → Frame N for segmentation             │
│  │                                                                          │
│  │   For SVFFD: Analytical inverse (negate velocity field)                 │
│  │   For FFD:   Newton-Raphson approximation (MIRTK-style)                 │
│  │              x_{n+1} = y - u(x_n), iterate until convergence            │
│  │                                                                          │
│  ├── Interpolation: NEAREST NEIGHBOR (preserves discrete labels)           │
│  │                                                                          │
│  └── Files:                                                                 │
│      ├── segmentations/seg_frame_000.nii.gz (original)                     │
│      ├── segmentations/seg_frame_001.nii.gz (propagated, high-res)         │
│      ├── segmentations/seg_frame_002.nii.gz                                │
│      ├── segmentations_frame_grid/seg_frame_001_frame_grid.nii.gz          │
│      └── ...                                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Array Dimension Conventions

### deepali Library Conventions
```
Image tensor:     [C, D, H, W]     or [D, H, W]      (3D spatial)
Batch tensor:     [N, C, D, H, W]                    (5D for operations)
Grid coordinates: [D, H, W, 3]     or [N, D, H, W, 3] (last dim = xyz)
Transform params: [N, C, D, H, W]  for FFD control points
```

### NIfTI File Conventions
```
NIfTI storage:    [X, Y, Z]        (RAS orientation)
Torch/deepali:    [D, H, W]        (internal, may differ)
Transpose needed: np.transpose(tensor, (2, 1, 0)) for saving
```

### Critical Dimension Handling Points

1. **Before registration**: Ensure 5D tensor `[N, C, D, H, W]`
   ```python
   if tensor.dim() == 3:
       tensor = tensor.unsqueeze(0).unsqueeze(0)  # [D,H,W] → [1,1,D,H,W]
   elif tensor.dim() == 4:
       tensor = tensor.unsqueeze(0)               # [C,D,H,W] → [1,C,D,H,W]
   ```

2. **After warping**: Remove batch dimension for saving
   ```python
   warped = warped.squeeze(0)  # [1,C,D,H,W] → [C,D,H,W]
   ```

3. **For NIfTI saving**: Transpose to [X,Y,Z]
   ```python
   nifti_array = np.transpose(tensor.numpy(), (2, 1, 0))
   ```

---

## Transform Direction Reference

| Step | Transform Name | Direction | Usage |
|------|---------------|-----------|-------|
| 1 | alignment_transform | Static → Frame 0 | Align static to dynamic |
| 2 | pairwise_N_M | Frame M → Frame N | Track consecutive motion |
| 3 | longitudinal_0_to_N | Frame N → Frame 0 | Composed pairwise |
| 4 | longitudinal_refined | Frame N → Frame 0 | Fine-tuned |
| 5 | inverse_longitudinal | Frame 0 → Frame N | For segmentation propagation |

### Transform Application Rules

**Forward Transform (registration result)**:
- Maps SOURCE coordinates to TARGET coordinates
- Applied to SOURCE image to produce warped image in TARGET space

**Inverse Transform (for segmentation)**:
- Maps TARGET coordinates to SOURCE coordinates
- Applied to get where each TARGET voxel came from in SOURCE

---

## Expected Outputs Structure

```
motion_output_test/
├── frames/
│   ├── frame_000.nii.gz          # Extracted 3D frame
│   ├── frame_001.nii.gz
│   ├── frame_002.nii.gz
│   └── frame_003.nii.gz
│
├── alignment/
│   ├── source_after_rigid_common.nii.gz     # Intermediate
│   ├── source_after_affine_common.nii.gz    # Intermediate
│   ├── source_after_ffd_common.nii.gz       # Intermediate (if FFD used)
│   ├── aligned_static_original_resolution.nii.gz  # Final aligned
│   ├── aligned_segmentation.nii.gz          # Seg in frame 0 space
│   └── alignment_transform.pth              # Composed transform
│
├── pairwise/
│   ├── pairwise_0_1.pth          # Frame 1 → Frame 0
│   ├── pairwise_1_2.pth          # Frame 2 → Frame 1
│   └── pairwise_2_3.pth          # Frame 3 → Frame 2
│
├── longitudinal/
│   ├── longitudinal_0_to_1_*.pth  # Direct Frame 1 → Frame 0
│   ├── longitudinal_0_to_2_*.pth  # Direct Frame 2 → Frame 0
│   └── longitudinal_0_to_3_*.pth  # Direct Frame 3 → Frame 0
│
├── segmentations/
│   ├── seg_frame_000.nii.gz      # Original (frame 0)
│   ├── seg_frame_001.nii.gz      # Propagated (high-res)
│   ├── seg_frame_002.nii.gz
│   └── seg_frame_003.nii.gz
│
├── segmentations_frame_grid/
│   ├── seg_frame_001_frame_grid.nii.gz  # For ITK-SNAP overlay
│   ├── seg_frame_002_frame_grid.nii.gz
│   └── seg_frame_003_frame_grid.nii.gz
│
├── visualizations/
│   └── alignment_progression.png
│
└── pipeline_log.txt              # Detailed execution log
```

---

## Verification Checklist

### Step 1: Alignment
- [ ] Common grid shape matches frame 0 grid
- [ ] Resampled images have same shape as common grid
- [ ] Each stage output (rigid/affine/ffd) saved incrementally
- [ ] Segmentation uses nearest neighbor interpolation

### Step 2: Pairwise
- [ ] Number of pairwise = num_frames - 1
- [ ] Each pairwise maps Frame N → Frame N-1
- [ ] Transform saved immediately after each pair

### Step 3: Compose
- [ ] Number of longitudinal = num_frames - 1
- [ ] First longitudinal = first pairwise (no composition needed)
- [ ] Composition order: T_01 ∘ T_12 ∘ T_23 (left to right application)

### Step 4: Refine
- [ ] Skip Frame 1 (already optimal from pairwise)
- [ ] Uses composed as initialization
- [ ] Only 1 pyramid level for fine-tuning

### Step 5: Propagate Segmentation
- [ ] Uses INVERSE transform (Frame 0 → Frame N)
- [ ] FFD uses Newton-Raphson approximation for inverse
- [ ] SVFFD uses analytical inverse
- [ ] Nearest neighbor interpolation preserves labels
- [ ] Both high-res and frame-grid versions saved

---

## Common Issues and Debugging

### Issue: Segmentation offset/misalignment
**Cause**: Using forward transform instead of inverse
**Check**: Verify `transform.inverse()` is called before `transform_segmentation()`

### Issue: FFD inverse not supported
**Cause**: FreeFormDeformation doesn't have analytical inverse
**Fix**: Use `approximate_ffd_inverse()` with Newton-Raphson iteration

### Issue: Array dimension mismatch
**Cause**: Inconsistent tensor dimensions between operations
**Check**: Log tensor shapes at each step, ensure 5D for registration

### Issue: NIfTI orientation wrong
**Cause**: Missing transpose or wrong affine
**Check**: Verify LPS→RAS conversion and proper transpose

---

## Logging Requirements

The pipeline should log the following at each step:

1. **Input dimensions**: Shape, spacing, origin of all input images
2. **Grid information**: Common grid shape, spacing
3. **Transform parameters**: Number of DOF, control point spacing
4. **Iteration progress**: Loss values, convergence status
5. **Output dimensions**: Shape of warped images
6. **Timing**: Duration of each step
7. **File saves**: Path and description of each saved file
