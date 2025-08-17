# How ITK-SNAP Handles Coordinate System Differences

## Problem Statement

Medical images often come with different coordinate systems, orientations, and physical spacing. When users load multiple images in ITK-SNAP, they expect to see them aligned for comparison, even when the images have vastly different coordinate systems.

## The Challenge: OSAMRI007 Dataset Example

Our investigation used two cardiac MRI images:

### Static Image
- **Size**: 256×256×160 voxels  
- **Spacing**: (0.78125, 0.78125, 0.8) mm
- **Origin**: (-94.9, -91.2, 63.8) mm
- **Direction Matrix**:
  ```
  [ 0.99796  0.06241 -0.01320]
  [-0.01008  0.35856  0.93345]
  [ 0.06299 -0.93142  0.35846]
  ```

### Frame0 Image  
- **Size**: 144×144×12 voxels
- **Spacing**: (1.5, 1.5, 3.0) mm  
- **Origin**: (21.1, -135.9, 62.6) mm
- **Direction Matrix**:
  ```
  [-0.01320  0.06241 -0.99796]
  [ 0.93345  0.35856  0.01008]
  [ 0.35846 -0.93142 -0.06299]
  ```

### Key Differences
- **Direction matrix difference**: 1.01 (maximum element difference)
- **Different orientations**: Completely different axis alignments
- **Different spacing**: Factor of ~2x difference in X/Y, ~4x in Z
- **Different coverage**: Static covers full heart, Frame0 is thin slice stack

## ITK-SNAP's Solution: Automatic Resampling

### What ITK-SNAP Does

1. **Loads images in native coordinate systems**: Each image retains its original coordinate system information
2. **Analyzes coordinate system compatibility**: Checks if images can be displayed together
3. **Automatically resamples to common space**: When displaying multiple images, ITK-SNAP resamples them to a common coordinate system
4. **Uses reference image**: Typically uses the first loaded image as the reference coordinate system
5. **Transparent to user**: This happens automatically without user intervention

### The ITK Resampling Process

```python
# ITK-SNAP's approach (simplified)
def resample_to_common_space(moving_image, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)  # Use reference coordinate system
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # Identity transform
    
    return resampler.Execute(moving_image)
```

### Results After Resampling

After ITK resampling:
- **Image centers**: Only 2.0mm apart (vs 124mm before)
- **Same coordinate system**: Both images now share identical:
  - Size: (256, 256, 160)
  - Spacing: (0.78125, 0.78125, 0.8) mm
  - Origin: (-94.9, -91.2, 63.8) mm  
  - Direction matrix: Same as reference
- **Aligned display**: Sagittal slices at same physical coordinates (X = 4.9mm)

## Why This Works

### Physical Space Alignment
- Images are already well-aligned in **physical coordinates**
- The issue was **coordinate system representation**, not actual spatial alignment
- ITK resampling maintains physical relationships while unifying coordinate systems

### Key Insight
ITK-SNAP doesn't try to register or align images - it simply puts them in the same coordinate system for visualization. The images were already physically aligned; they just had different ways of describing the same physical space.

## Implementation Details

### Core ITK Functions Used

1. **`sitk.ResampleImageFilter()`**: Main resampling engine
2. **`SetReferenceImage()`**: Defines target coordinate system
3. **`TransformIndexToPhysicalPoint()`**: Converts voxel indices to physical coordinates
4. **`GetDirection()`, `GetOrigin()`, `GetSpacing()`**: Access coordinate system parameters

### Automatic Detection

ITK-SNAP automatically detects when images need resampling by:
- Comparing direction matrices
- Checking spacing compatibility  
- Analyzing origin relationships
- Assessing size differences

## Benefits of ITK-SNAP's Approach

1. **Robust**: Works with any image orientation or coordinate system
2. **Automatic**: No user intervention required
3. **Preserves data**: Uses appropriate interpolation to maintain image quality
4. **Fast**: Leverages optimized ITK resampling algorithms
5. **Transparent**: Users see aligned images without knowing about coordinate differences

## Comparison with Other Approaches

### Our Previous Attempts
- **Custom coordinate transforms**: Tried to manually handle lattice-to-world matrices
- **MIRTK coordinate system**: Attempted to replicate MIRTK's approach
- **Manual header interpretation**: Tried various coordinate conventions

### Why They Failed
- **Over-complicated**: Added unnecessary coordinate transformations
- **Missed the point**: ITK-SNAP doesn't interpret coordinates differently, it resamples
- **Fought the system**: Worked against ITK's built-in coordinate handling

## Lessons Learned

1. **Trust ITK**: ITK's coordinate system handling is robust and well-tested
2. **Resampling > Custom transforms**: Use ITK's resampling instead of custom coordinate math
3. **Physical space matters**: Images aligned in physical space just need unified coordinate systems
4. **Automatic detection**: Let ITK detect and handle coordinate system differences
5. **KISS principle**: Simple solutions (resampling) often work better than complex ones

## Conclusion

ITK-SNAP's "magic" for handling coordinate differences is actually quite simple: **automatic resampling to a common coordinate system**. This approach:

- Maintains physical accuracy
- Handles any coordinate system combination
- Requires minimal code
- Matches user expectations
- Provides robust foundation for further analysis

The key insight is that coordinate system differences are a **representation problem**, not an **alignment problem**. ITK-SNAP solves this by standardizing the representation while preserving the physical relationships.