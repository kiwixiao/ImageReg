# MIRTK Motion Registration Pipeline - Step by Step

## Overview

This document details the exact step-by-step workflow that MIRTK uses for motion registration of 4D medical images. The pipeline registers all frames to frame 0 using sequential pairwise registration with FFD (Free-Form Deformation).

## Pipeline Architecture

```
4D Image (N frames)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Frame Extraction                                  │
│  mirtk extract-image-volume → frame_0, frame_1, ... frame_N │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Sequential Pairwise Registration                  │
│  frame_0→frame_1, frame_1→frame_2, ... frame_(N-1)→frame_N  │
│  Each pair: mirtk register with FFD model                   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Transform Composition                             │
│  Compose: T_0→2 = T_0→1 ∘ T_1→2                             │
│  Compose: T_0→3 = T_0→2 ∘ T_2→3                             │
│  ...                                                        │
│  mirtk compose-dofs                                         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Refinement (Optional)                             │
│  Fine-level optimization directly from frame_0 to frame_j   │
│  Using composed transform as initialization                 │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Apply Transforms                                  │
│  Transform segmentation from frame_0 to all other frames    │
│  mirtk transform-points / transform-image                   │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Steps

### Step 1: Frame Extraction

**MIRTK Command:**
```bash
mirtk extract-image-volume dynamic_4d.nii.gz -t <start_frame> -n <num_frames> frame_{n}.nii.gz
```

**Purpose:** Extract 3D frames from 4D image volume.

**Parameters:**
- `-t`: Starting time point (0-indexed)
- `-n`: Number of frames to extract

**Output:** Individual 3D NIfTI files with same affine/header as original 4D.

**Critical:** The extracted frames MUST preserve the original affine matrix exactly.

---

### Step 2: FFD Registration Configuration

**MIRTK Configuration (register.cfg):**
```ini
[ transformation model ]
Transformation model  = FFD
Control point spacing = 4      # 4mm between control points

[ objective function ]
Energy function = NMI(I(1), I(2:end) o T) + 0.001 BE(T) + 0.0005 LE(T)
No. of bins = 64               # NMI histogram bins

[ optimization ]
No. of levels                    = 4    # Multi-resolution pyramid levels
Maximum no. of iterations        = 100  # Per level
Strict step length range         = No
Maximum streak of rejected steps = 1

[ level 1 ]    # Finest level (full resolution)
Resolution in X = 0
Resolution in Y = 0
Resolution in Z = 0

[ level 2 ]
Resolution in X = 0
Resolution in Y = 3
Resolution in Z = 3

[ level 3 ]
Resolution in X = 0
Resolution in Y = 6
Resolution in Z = 6

[ level 4 ]    # Coarsest level
Resolution in X = 0
Resolution in Y = 12
Resolution in Z = 12
```

**Energy Function Breakdown:**
| Term | Description | Default Weight |
|------|-------------|----------------|
| `NMI(I(1), I(2:end) o T)` | Normalized Mutual Information similarity | 1.0 (implicit) |
| `BE(T)` | Bending Energy (2nd order derivatives) | 0.001 |
| `LE(T)` | Linear Energy (1st order derivatives, diffusion) | 0.0005 |

---

### Step 3: Sequential Pairwise Registration

**MIRTK Command:**
```bash
mirtk register target.nii source.nii \
    -parin register.cfg \
    -dofin Id \
    -dofout transform.dof.gz
```

**Registration Order:**
```
Frame 0 (target) ← Frame 1 (source)  → ffd_0_1.dof.gz
Frame 1 (target) ← Frame 2 (source)  → ffd_1_2.dof.gz
Frame 2 (target) ← Frame 3 (source)  → ffd_2_3.dof.gz
...
Frame (N-1) (target) ← Frame N (source) → ffd_(N-1)_N.dof.gz
```

**Why Sequential Pairwise?**
1. Adjacent frames have smallest motion → stable registration
2. Multi-resolution optimization converges reliably
3. Avoids large deformation instability

---

### Step 4: Transform Composition

**MIRTK Command:**
```bash
mirtk compose-dofs ffd_0_1.dof.gz ffd_1_2.dof.gz ffd_0_2.dof.gz
```

**Composition Chain:**
```
ffd_0_1  ∘  ffd_1_2  =  ffd_0_2
ffd_0_2  ∘  ffd_2_3  =  ffd_0_3
ffd_0_3  ∘  ffd_3_4  =  ffd_0_4
...
```

**Mathematical Representation:**
```
T_0→j = T_0→(j-1) ∘ T_(j-1)→j
where ∘ denotes function composition
```

---

### Step 5: Refinement Pass

**MIRTK Command:**
```bash
mirtk register frame_0.nii frame_j.nii \
    -parin register.cfg \
    -dofin ffd_0_j.dof.gz \
    -dofout ffd_0_j.dof.gz \
    -levels 1
```

**Purpose:** Fine-tune composed transform directly on frame pairs.

**Key Settings:**
- `-dofin`: Use composed transform as initialization
- `-levels 1`: Only finest resolution level
- Result overwrites the composed transform

---

### Step 6: Apply Transforms

**For STL/Surface Points:**
```bash
mirtk transform-points seg_0.stl seg_j.stl -dofin ffd_0_j.dof.gz
```

**For Segmentation Images:**
```bash
mirtk transform-image seg_0.nii.gz seg_j.nii.gz \
    -dofin ffd_0_j.dof.gz \
    -interp NN  # Nearest Neighbor for discrete labels
```

**For Medical Images:**
```bash
mirtk transform-image img_0.nii.gz img_j.nii.gz \
    -dofin ffd_0_j.dof.gz \
    -interp Linear  # Linear for continuous intensities
```

---

## Multi-Resolution Pyramid Details

MIRTK uses coarse-to-fine optimization:

| Level | Resolution Setting | Actual Size Factor | Control Points |
|-------|-------------------|-------------------|----------------|
| 4 (coarsest) | Y=12, Z=12 | ~1/16 | Sparse |
| 3 | Y=6, Z=6 | ~1/8 | Medium |
| 2 | Y=3, Z=3 | ~1/4 | Dense |
| 1 (finest) | Y=0, Z=0 | Full | Full density |

**Note:** Resolution=0 means full resolution (no downsampling).

---

## Foreground Handling (FG_Overlap Mode)

MIRTK computes NMI only in overlapping foreground regions:

1. **Foreground Detection:** Intensity > threshold
2. **Overlap Mask:** source_fg AND target_fg
3. **Gradient Filter:** Skip voxels with zero image gradient
4. **Final Mask:** overlap AND has_gradient

This prevents:
- Unrealistic deformation at image boundaries
- Registration artifacts from background regions
- "Twisted" boundary effects

---

## Transform File Format (.dof.gz)

MIRTK stores transforms in proprietary `.dof.gz` format:
- Compressed binary format
- Contains control point grid + displacement coefficients
- Grid information: origin, spacing, size
- B-spline coefficients for each control point

---

## Summary: MIRTK Motion Tracking Workflow

```python
# Pseudocode representation
for frame_j in range(1, N):
    if frame_j == 1:
        # First pair: direct registration
        T_0_1 = mirtk_register(frame_0, frame_1, model="FFD")
        T_0_j = T_0_1
    else:
        # Step 1: Sequential pairwise
        T_jm1_j = mirtk_register(frame[j-1], frame[j], model="FFD")

        # Step 2: Compose with previous
        T_0_j = compose(T_0_jm1, T_jm1_j)

        # Step 3: Refine at finest level
        T_0_j = refine(T_0_j, frame_0, frame_j, levels=1)

    # Step 4: Apply to segmentation
    seg_j = transform(seg_0, T_0_j, interp="NN")
```

## Key Parameters Summary

| Parameter | MIRTK Default | Purpose |
|-----------|---------------|---------|
| Transformation model | FFD | B-spline Free-Form Deformation |
| Control point spacing | 4mm | Distance between B-spline knots |
| NMI bins | 64 | Histogram resolution for NMI |
| Bending energy weight | 0.001 | Smoothness regularization (2nd order) |
| Linear energy weight | 0.0005 | Gradient regularization (1st order) |
| Pyramid levels | 4 | Multi-resolution optimization |
| Max iterations/level | 100 | Optimization budget per level |
