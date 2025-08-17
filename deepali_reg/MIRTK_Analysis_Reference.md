# MIRTK Registration Analysis - Complete Reference

## Overview

This document provides a comprehensive analysis of how MIRTK (Medical Image Registration ToolKit) handles image registration, based on examination of the actual source code. This analysis was conducted to understand and implement similar functionality in our deepali registration pipeline.

## 1. MIRTK Architecture & Core Principles

### World Coordinate System Approach
- **Primary Philosophy**: MIRTK works primarily in world/physical coordinate space
- **NIfTI Integration**: Uses NIfTI header information (sform/qform matrices) directly
- **Coordinate Transformation Pipeline**:
  ```
  Source Voxel (i,j,k) → Source World (x,y,z) → Target World (x',y',z') → Target Voxel (i',j',k')
  ```

### Automatic Orientation Handling
- **No Manual Axis Detection**: Eliminates need for manual slice axis determination
- **Header Matrix Reliance**: Uses existing orientation matrices directly
- **Robust to Complex Orientations**: Handles arbitrary image orientations automatically

## 2. Similarity Calculation Mechanism

### Normalized Mutual Information (NMI)
- **Default Similarity Measure**: NMI is the primary similarity metric
- **Histogram-Based Approach**: Uses joint and marginal histograms
- **Parzen Windowing**: Optional cubic B-spline Parzen window smoothing
- **Formula**: `NMI = (H(Target) + H(Source)) / H(Target,Source)`

### Implementation Details
```cpp
// From NormalizedMutualImageInformation.cc
if (_This->IsForeground(idx)) {
    // Only calculate similarity for foreground voxels
    const double target_value = ValToRange(_LogMarginalXHistogram, *tgt);
    const double source_value = ValToRange(_LogMarginalYHistogram, *src);
    // ... NMI calculation
}
```

### Gradient Calculation
- **Chain Rule Application**: Uses chain rule for transformation parameter gradients
- **Fixed Precision**: Rounds derivative values to avoid floating point differences
- **Symmetric Registration**: Ensures inverse consistent results in SVFFD registration

## 3. Foreground Region Handling - Key Innovation

### ForegroundRegion Enum System
```cpp
enum ForegroundRegion {
    FG_Domain,   ///< Evaluate similarity for all voxels in image domain
    FG_Mask,     ///< Evaluate similarity with explicit mask
    FG_Target,   ///< Evaluate similarity for foreground of untransformed image
    FG_Overlap,  ///< Evaluate similarity for intersection of foregrounds ⭐
    FG_Union     ///< Evaluate similarity for union of foregrounds
};
```

### IsForeground() Logic - Core Implementation
```cpp
inline bool ImageSimilarity::IsForeground(int idx) const {
    if (_Foreground == FG_Domain || !_Mask || _Mask->Get(idx)) {
        switch (_Foreground) {
            case FG_Domain: case FG_Mask:
                return true;
            case FG_Target:
                // Complex logic for transformed vs untransformed images
                return _Source->IsForeground(idx) || _Target->IsForeground(idx);
            case FG_Overlap:  // ⭐ CRITICAL FOR REGISTRATION
                return _Source->IsForeground(idx) && _Target->IsForeground(idx);
            case FG_Union:
                return _Source->IsForeground(idx) || _Target->IsForeground(idx);
        };
    }
    return false;
}
```

### Automatic Mutual Region Focus
- **FG_Overlap Mode**: Default for most registrations - focuses on intersection of foregrounds
- **Prevents Random Registration**: Ensures similarity only calculated on mutual anatomical regions
- **Background Detection**: Automatic detection based on padding values and intensity thresholds

## 4. Energy Function & Optimization

### Energy Function Formulation
```
Energy = SIM[Image dissimilarity](I(1), I(2:end) ∘ T) + 
         regularization_terms
```

Where:
- `SIM` = Image similarity measure (NMI, NCC, SSD, etc.)
- `I(1)` = Target image
- `I(2:end)` = Source image(s)
- `T` = Transformation being optimized
- `regularization_terms` = Bending energy, linear energy, etc.

### Regularization Terms
- **Bending Energy**: `BE[Bending energy](T)`
- **Linear Energy**: `LE[Linear energy](T)`
- **Topology Preservation**: `TP[Topology preservation](T)`
- **Volume Preservation**: `VP[Volume preservation](T)`
- **LogJac Penalty**: `LogJac[LogJac penalty](T)`

### Domain Control via Masks
```bash
-mask <file>    # Reference mask which defines domain within which to 
                # evaluate the energy function (i.e. image similarity)
```

## 5. Resolution Handling

### Multi-Resolution Framework
- **Pyramid Approach**: Multiple resolution levels with configurable parameters
- **Progressive Refinement**: Coarse to fine optimization strategy
- **Control Point Spacing**: Configurable spacing at each level

### Handling Different Image Resolutions
- **World Coordinate Mapping**: Handles resolution differences through physical coordinate system
- **Registration Domain**: Typically uses target image resolution as registration grid
- **Source Interpolation**: Source image interpolated to registration domain during optimization
- **Normal Practice**: High-resolution source → coarse target is standard and handled automatically

**Example**: Our case with high-res static (352×352×158) → coarse Frame0 (192×192×12) is perfectly normal for MIRTK.

## 6. MIRTK Command-Line Usage

### Basic Registration Command
```bash
mirtk register target.nii source.nii -dofout transform.dof [options]
```

### Key Registration Options
```bash
-model <name>           # Transformation model (Rigid+Affine+FFD)
-mask <file>           # Reference mask for registration domain
-par <name> <value>    # Direct parameter specification
-parin <file>          # Read parameters from configuration file
```

### Similarity Measure Options
```bash
# From evaluate-similarity command
-metric <sim>          # NMI, NCC, SSD, etc.
-bins <int>           # Number of histogram bins
-parzen               # Enable Parzen windowing
-padding <value>      # Background/padding value
```

## 7. Source Code File Structure

### Key Files Analyzed
- `Applications/src/register.cc` - Main registration application
- `Applications/src/evaluate-similarity.cc` - Similarity evaluation tool
- `Modules/Registration/src/NormalizedMutualImageInformation.cc` - NMI implementation
- `Modules/Registration/src/ImageSimilarity.cc` - Base similarity class
- `Modules/Registration/include/mirtk/ImageSimilarity.h` - IsForeground implementation

### Important Classes
- `ImageSimilarity` - Base class for all similarity measures
- `NormalizedMutualImageInformation` - NMI implementation
- `GenericRegistrationFilter` - Main registration framework
- `RegisteredImage` - Handles image transformations and interpolation

## 8. Comparison with Our Deepali Approach

### What We're Doing Right ✅
- **NMI Similarity**: Using normalized mutual information
- **Physical Coordinates**: World coordinate system approach
- **Multi-Resolution**: Pyramid optimization strategy
- **Mutual FOV Masking**: Similar to MIRTK's FG_Overlap mode
- **Dimension Preservation**: Maintaining original image resolutions
- **Resolution Handling**: Properly handling different image resolutions

### Areas for Improvement 🔧
- **Automatic Background Detection**: We manually create masks vs MIRTK's automatic detection
- **ForegroundRegion System**: No enum system for different foreground modes
- **IsForeground Logic**: No automatic foreground intersection logic
- **Domain Handling**: Could be more sophisticated like MIRTK's approach

## 9. Implementation Insights for Deepali

### MIRTK-Style Foreground Detection
```python
def detect_foreground_mask(self, image_tensor, padding_value=0.0):
    """MIRTK-style foreground detection"""
    # Method 1: Padding value based (MIRTK default)
    foreground_mask = image_tensor != padding_value
    
    # Method 2: Intensity threshold based
    # foreground_mask = image_tensor > intensity_threshold
    
    return foreground_mask.float()

def create_registration_mask(self, target_tensor, source_tensor, mode="overlap"):
    """MIRTK-style registration mask creation"""
    if mode == "overlap":
        # FG_Overlap: intersection of foregrounds
        target_fg = self.detect_foreground_mask(target_tensor)
        source_fg = self.detect_foreground_mask(source_tensor) 
        mask = target_fg * source_fg  # Element-wise AND
    elif mode == "union":
        # FG_Union: union of foregrounds
        target_fg = self.detect_foreground_mask(target_tensor)
        source_fg = self.detect_foreground_mask(source_tensor)
        mask = torch.clamp(target_fg + source_fg, 0, 1)  # Element-wise OR
    # ... other modes
    
    return mask
```

### Enhanced Loss Function
```python
def mirtk_style_loss(warped_source, target, transform, foreground_handler):
    # Create MIRTK-style foreground mask
    registration_mask = foreground_handler.create_registration_mask(
        target, warped_source, mode="overlap"
    )
    
    # Apply mask to both images
    warped_source_masked = warped_source * registration_mask
    target_masked = target * registration_mask
    
    # Calculate NMI only on masked regions
    nmi_loss = NMI(bins=64)
    similarity = nmi_loss(warped_source_masked, target_masked)
    
    return similarity
```

## 10. Key Takeaways

### MIRTK's Core Strengths
1. **Automatic Foreground Detection**: No manual region specification needed
2. **Mutual Region Focus**: Automatic focus on overlapping anatomical content
3. **Resolution Agnostic**: Handles different image resolutions seamlessly
4. **World Coordinate System**: Eliminates orientation handling issues
5. **Robust Energy Function**: Well-designed combination of similarity and regularization

### Why Our Approach Works
- We follow MIRTK's fundamental principles correctly
- Physical coordinate usage matches MIRTK's philosophy
- Multi-resolution and NMI usage align with MIRTK defaults
- Manual mutual FOV masking achieves similar results to FG_Overlap

### Critical Insight
**MIRTK automatically prevents "random force registration" through sophisticated foreground handling that focuses optimization only on mutual anatomical regions.** This is the key to anatomically meaningful registration results.

## 11. Future Enhancements

### Immediate Improvements
1. Implement automatic background detection based on padding values
2. Add ForegroundRegion enum system with configurable modes
3. Create IsForeground-style checking in loss functions  
4. Make foreground handling mode user-configurable

### Advanced Features
1. Add support for explicit mask files like MIRTK's `-mask` option
2. Implement more sophisticated background detection algorithms
3. Add support for different similarity measures beyond NMI
4. Create MIRTK-compatible parameter file system

---

*This analysis was conducted by examining the MIRTK source code directly to understand the implementation details and design principles behind this robust medical image registration toolkit.*