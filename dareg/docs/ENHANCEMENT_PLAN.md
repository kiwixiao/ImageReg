# DAREG Enhancement Plan

## Overview
This document tracks planned enhancements for the DAREG motion registration pipeline.
The core registration functionality is working correctly - these are output/visualization improvements.

---

## Task 1: Clean Up Output Folder Structure

### Problem
- `segmentation_results/` folder is empty and not needed
- `final_results/` should contain static image moved to ALL frames at original resolution
- Need static_frame_000.nii.gz, static_frame_001.nii.gz, etc.

### Solution
1. **Remove** `segmentation_results/` folder creation entirely
2. **Fix** `final_results/` to save static image to all frames at original resolution:
   - `static_frame_000.nii.gz` (static moved to frame 0, original resolution)
   - `static_frame_001.nii.gz` (static moved to frame 1, original resolution)
   - `static_frame_002.nii.gz`, etc.
   - Keep frame0_reference.nii.gz for overlay verification
3. **Keep** `alignment/` folder for common-grid intermediate results (useful for debugging)
4. **Keep** `segmentations/` folder (propagated segs at original resolution)
5. **Keep** `segmentations_frame_grid/` folder (segs resampled to 4D frame resolution)

### Implementation
- Added `_save_static_to_all_frames()` method that:
  - Composes alignment_transform with inverse of longitudinal transforms
  - Applies composed transform using `apply_transform_preserve_resolution()`
  - Saves to final_results/ with original static image resolution

### Files Modified
- `DAREG/registration/motion.py` - added new function and updated save_results()

---

## Task 2: Improve Segmentation Progression Visualization

### Problem
Current `segmentation_progression.png` shows simple single-slice view, hard to assess 3D changes.

### Solution
Create multi-slice visualization for each anatomical plane:
- **Layout**: 8 slices per row (spread from mid-slice to both sides)
- **Rows**: One row per frame (frame 0, 1, 2, ...)
- **Files**: Separate PNG for each view:
  - `segmentation_progression_sagittal.png`
  - `segmentation_progression_axial.png`
  - `segmentation_progression_coronal.png`

### Slice Selection Strategy
```
For N=8 slices from dimension D:
  mid = D // 2
  offsets = [-3, -2, -1, 0, 1, 2, 3, 4] * (D // 8)
  slices = [mid + offset for offset in offsets]
```

### Files to Create/Modify
- `DAREG/visualization/segmentation_viewer.py` (new file)
- Update visualization pipeline to use new viewer

---

## Task 3: STL Surface Generation from Segmentation

### Problem
Need 3D surface meshes from binary segmentation masks for visualization and analysis.

### Solution
1. **Add marching cubes** surface extraction using `skimage.measure.marching_cubes`
2. **Apply smoothing** (optional Laplacian smoothing)
3. **Save as STL** format using `numpy-stl` or `trimesh`
4. **Handle coordinate system** - ensure STL is in world coordinates (mm)

### Output Structure
```
stl_surfaces/
├── seg_frame_000.stl
├── seg_frame_001.stl
├── seg_frame_002.stl
└── ...
```

### Dependencies to Add
- `scikit-image` (marching_cubes)
- `numpy-stl` or `trimesh` (STL I/O)

### Files to Create
- `DAREG/postprocessing/stl_generator.py`

---

## Task 4: Temporal Interpolation for STL Surfaces

### Problem
Need to interpolate surfaces between discrete time frames to create smooth 4D animation.

### Solution

#### Step 1: Extract Temporal Information from 4D Image
- Read temporal resolution from NIfTI header (`pixdim[4]` = TR or frame duration)
- Calculate actual time points for each frame
- Store timing metadata with transforms

#### Step 2: Implement Surface Interpolation
Two approaches (in order of complexity):

**Approach A: Displacement Field Interpolation (Recommended)**
- Interpolate the FFD displacement field at arbitrary time t
- Apply interpolated displacement to base mesh (frame 0)
- Produces smooth motion between frames
- Preserves topology

**Approach B: Mesh Vertex Interpolation**
- Requires mesh correspondence (same topology across frames)
- Linear/spline interpolation of vertex positions
- Simpler but may have artifacts at boundaries

#### Step 3: Generate Interpolated STLs
```python
def interpolate_surface(t: float, frame_times: List[float],
                       transforms: List[Transform]) -> STL:
    """
    Interpolate surface at arbitrary time t.

    Args:
        t: Target time in seconds
        frame_times: [0.0, 0.5, 1.0, ...] actual times per frame
        transforms: FFD transforms for each frame

    Returns:
        Interpolated STL surface
    """
```

### MIRTK Reference
MIRTK uses temporal B-spline interpolation of velocity fields:
- Continuous temporal parameterization
- Smooth velocity field evolution
- Diffeomorphic at all time points

### Output Structure
```
stl_interpolated/
├── surface_t0.000.stl
├── surface_t0.050.stl
├── surface_t0.100.stl
└── ... (at user-specified temporal resolution)
```

### Files to Create
- `DAREG/postprocessing/temporal_interpolation.py`
- `DAREG/postprocessing/stl_animator.py`

---

## Implementation Order

1. **Task 1**: Clean up output folders (quick fix)
2. **Task 2**: Multi-slice segmentation visualization (medium)
3. **Task 3**: STL generation (medium)
4. **Task 4**: Temporal interpolation (complex)

---

## Status Tracking

| Task | Status | Notes |
|------|--------|-------|
| Task 1: Output cleanup | Complete | Saves `static_image_moved_to_frame_XXX.nii.gz` at original resolution |
| Task 2: Multi-slice viz | Complete | Created segmentation_viewer.py with sagittal/axial/coronal views |
| Task 3: STL generation | Complete | Created stl_generator.py with marching cubes + smoothing |
| Task 4: Temporal interp | Complete | Created temporal_interpolation.py with B-spline interpolation |
| Task 5: Naming conventions | Complete | `static_image_moved_to_frame_XXX.nii.gz`, `dynamic_extracted_frame_XXX.nii.gz` |
| Task 6: Directory input | Complete | `load_image_4d()` accepts directory of 3D NIfTI files |

---

## Output File Naming Convention

- **Extracted frames**: `dynamic_extracted_frame_000.nii.gz`, `dynamic_extracted_frame_001.nii.gz`, etc.
- **Static to frames**: `static_image_moved_to_frame_000.nii.gz`, `static_image_moved_to_frame_001.nii.gz`, etc.

## Input Flexibility

`load_image_4d()` now accepts either:
- **4D NIfTI file**: Standard 4D or 5D with singleton dimension
- **Directory of 3D files**: Sorted naturally by filename (e.g., `frame_000.nii.gz`, `frame_001.nii.gz`)

Both produce identical `Image4D` container - pipeline unchanged.

---

## Notes

- Do NOT modify core registration code (rigid/affine/ffd modules)
- All changes are additive (new functions) or output-related
- Test after each task to ensure no regression
