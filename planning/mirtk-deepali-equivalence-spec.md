# MIRTK to Deepali Equivalence Specification

## Project Goal

Achieve 100% logical equivalence between the MIRTK (C++ based) medical image registration pipeline and DAREG (Python/deepali-based) implementation. The MIRTK pipeline is the proven working version - DAREG must produce identical quality outputs.

## Background

### Current State
- **MIRTK**: Working C++ pipeline in `mirtk_binary_reg_pipeline_demo/` (ground truth)
- **deepali**: Python library with FFD registration capabilities (local source in `deepali/`)
- **DAREG**: Python implementation attempting to replicate MIRTK (in `DAREG/`)

### Why This Matters
- Eliminate C++ dependency for easier deployment and maintenance
- Enable pure Python medical image registration workflow
- Maintain production-quality outputs matching MIRTK

## Scope: What Needs 100% Equivalence

### 1. Registration Algorithms
- **Rigid Registration** (6 DOF): Translation (3) + Rotation (3)
- **Affine Registration** (12 DOF): + Scaling (3) + Shearing (6)
- **FFD Registration**: Free-Form Deformation using B-splines
- **SVFFD** (optional): Stationary Velocity FFD for diffeomorphic registration

### 2. Similarity Metrics
- **NMI** (Normalized Mutual Information): Primary metric for multi-modal registration
- **SSD** (Sum of Squared Differences): For same-modality registration
- Foreground overlap masking (MIRTK's FG_Overlap behavior)

### 3. Regularization Terms
- **Bending Energy**: Smoothness constraint on displacement field
- **Diffusion Regularization**: Alternative smoothness term
- **Volume Preservation**: Jacobian determinant constraints

### 4. Transform Handling
- **Composition**: Chain transforms (T_01 ∘ T_12 = T_02)
- **Inversion**: Especially FFD inverse (no analytical solution)
- **Interpolation**: Linear for images, nearest-neighbor for segmentations

### 5. Multi-Resolution Strategy
- **Pyramid Levels**: Coarse-to-fine registration
- **Control Point Spacing**: FFD grid resolution at each level
- **Iteration Counts**: Per-level optimization

### 6. Optimization
- **Conjugate Gradient**: MIRTK's primary optimizer
- **L-BFGS**: Alternative optimizer
- **Line Search**: Step size determination
- **Convergence Criteria**: Early stopping conditions

### 7. Image Handling
- **4D/5D NIfTI**: Correct parsing (MIRTK handles this internally)
- **World Coordinates**: All registration in physical mm space
- **Resampling**: Image transformation with proper interpolation

## Success Criteria

1. **Numerical Equivalence**: For same inputs, DAREG and MIRTK produce transforms within acceptable tolerance
2. **Quality Metrics**: NMI, DICE scores comparable between implementations
3. **Visual Equivalence**: Registered images and segmentations visually match
4. **Performance**: Acceptable runtime (doesn't need to match C++ speed)

## Research Requirements

### MIRTK Source Analysis
- Trace function calls in `MIRTK/` source code
- Document exact algorithms for each registration step
- Identify default parameter values
- Map CLI options to internal behavior

### deepali Source Analysis
- Catalog available modules in `deepali/src/`
- Document API for each relevant class
- Identify gaps where MIRTK features aren't available
- Note any behavioral differences

### Current DAREG Analysis
- Audit existing implementation against MIRTK
- Identify what's correctly implemented
- Find bugs or deviations from MIRTK behavior
- List missing features

## Deliverables

1. **Gap Analysis Document**: MIRTK function → deepali equivalent → DAREG status
2. **Implementation Plan**: Step-by-step fixes to achieve equivalence
3. **Test Protocol**: How to verify equivalence (input/output comparisons)
4. **Updated DAREG Code**: Fixes and new implementations as needed

## Reference Files

- `MIRTK/` - MIRTK C++ source code
- `deepali/src/` - deepali Python library source
- `DAREG/` - Current Python implementation
- `mirtk_binary_reg_pipeline_demo/` - Working MIRTK pipeline outputs
- `inputs_testing/` - Test dataset
