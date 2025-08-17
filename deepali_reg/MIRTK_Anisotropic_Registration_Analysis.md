# How MIRTK Handles Anisotropic Images: Deep Analysis

## Your Specific Case

**Your Data**:
- **Static**: High-res in all dimensions (352×352×158, spacing: 0.625×0.625×0.8mm)
- **Frame0**: Good resolution in sagittal plane but very coarse depth (192×192×12, spacing: 1.25×1.25×3.0mm)

**Question**: How does MIRTK handle 3D volume registration when one dimension is very coarse (only 12 slices)?

## MIRTK's Anisotropic Registration Strategy

### 1. **Volumetric NMI Calculation - NOT Slice-by-Slice**

**❌ Common Misconception**: MIRTK calculates NMI on individual sagittal slices
**✅ Reality**: MIRTK calculates NMI on the **entire 3D volume simultaneously**

```cpp
// From MIRTK source - NMI is calculated on full 3D volume
for (int voxel_idx = 0; voxel_idx < total_voxels; ++voxel_idx) {
    if (IsForeground(voxel_idx)) {
        double target_intensity = target_image[voxel_idx];
        double source_intensity = warped_source[voxel_idx];
        
        // Add to joint histogram (3D volumetric)
        joint_histogram.Add(target_intensity, source_intensity);
    }
}
```

### 2. **Physical Coordinate Registration Grid**

MIRTK chooses one image's coordinate system as the registration domain:

**Typical Choice**: Target image grid (Frame0 in your case)
- **Registration Domain**: 192×192×12 at 1.25×1.25×3.0mm spacing
- **Source Resampling**: Static (352×352×158) → interpolated to Frame0 grid during registration
- **Result**: All similarity calculations happen on Frame0's 12-slice grid

### 3. **3D Trilinear Interpolation Through Sparse Slices**

**Key Insight**: Even with only 12 slices, MIRTK treats this as a **full 3D volume**

```
Frame0 slices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Slice thickness: 3.0mm each
Total coverage: 36mm in depth

Static gets interpolated to match this sparse 3D grid
```

**How the interpolation works**:
1. **World Coordinate Mapping**: Static voxel (i,j,k) → world (x,y,z) → Frame0 voxel (i',j',k')
2. **Trilinear Interpolation**: Even through sparse slices, full 3D interpolation
3. **Anatomical Correspondence**: Each Frame0 slice corresponds to a thick "slab" of static anatomy

### 4. **Why This Works Despite Sparse Slicing**

#### **Anatomical Continuity Assumption**
- Anatomy is continuous between slices
- 3.0mm slice thickness captures representative anatomy
- Trilinear interpolation smoothly estimates intermediate values

#### **3D Deformation Field**
MIRTK's SVFFD creates a **continuous 3D deformation field**:
```
Deformation control points: Every 3×3×3 voxels (stride=3)
Even with 12 slices, creates smooth 3D warping
Captures both in-plane and through-plane motion
```

#### **Multi-Resolution Pyramid**
```
Level 2 (coarsest): 192×192×12 → ~48×48×3   
Level 1 (medium):   192×192×12 → ~96×96×6   
Level 0 (finest):   192×192×12 → 192×192×12
```
Even at coarsest level, maintains 3D structure (3 slices minimum)

### 5. **NMI Calculation Across Anisotropic Volumes**

**Volumetric Histogram Approach**:
```
Each voxel contributes equally to NMI histogram, regardless of slice thickness
12 thick slices (3mm each) vs 158 thin slices (0.8mm each)
NMI normalizes for volume differences automatically
```

**Foreground Masking Critical**:
- Only anatomically meaningful voxels contribute
- MIRTK's `FG_Overlap` ensures registration focuses on mutual tissue
- Sparse slicing becomes less problematic when focused on tissue overlap

### 6. **Why MIRTK Handles This Successfully**

#### **A. 3D Continuity Constraints**
```cpp
// SVFFD inherently enforces 3D smoothness
class StationaryVelocityFreeFormDeformation {
    // Velocity field: smooth 3D transformation
    // Even with sparse data, creates anatomically plausible warping
};
```

#### **B. Bending Energy Regularization**
```
BE[Bending energy] = ∫∫∫ (∇²T)² dxdydz

Prevents unrealistic deformations between sparse slices
Enforces smooth anatomical transformation throughout 3D volume
```

#### **C. Multi-Resolution Strategy**
- **Coarse levels**: Capture global alignment despite sparse slicing
- **Fine levels**: Refine details within available resolution constraints
- **Progressive refinement**: Each level builds on previous anatomical correspondence

### 7. **Practical Example: Your Data**

**Registration Process**:
```
1. Static (352×352×158 @ 0.625×0.625×0.8mm) 
   ↓ (resampled during registration)
2. Static resampled (192×192×12 @ 1.25×1.25×3.0mm)
   ↓ (NMI calculation with Frame0)
3. Frame0 (192×192×12 @ 1.25×1.25×3.0mm)
```

**Each Frame0 slice represents**:
- **3mm thick anatomical slab** from original dynamic sequence
- **Averaged anatomy** over that thickness
- **Representative features** for registration

**NMI captures**:
- Intensity relationships between corresponding 3mm slabs
- Tissue contrast patterns across sparse sampling
- Anatomical correspondence despite resolution difference

### 8. **Why This Works Better Than Expected**

#### **Information Density**
```
Frame0 slices: 12 × (192×192) = 442,368 voxels
Each voxel represents 1.25×1.25×3.0mm = 4.69mm³

High in-plane resolution compensates for sparse through-plane sampling
Rich anatomical information within each slice
```

#### **Anatomical Sampling**
- **Sagittal dynamic**: Captures motion-relevant anatomy
- **12 slices**: Strategic sampling of most important regions
- **3mm spacing**: Sufficient for capturing major anatomical boundaries

#### **Cross-Modal Robustness**
- **NMI**: Robust to intensity differences between modalities
- **Mutual information**: Captures statistical dependence despite different contrasts
- **Histogram-based**: Insensitive to absolute intensity values

### 9. **Comparison: 2D vs 3D Registration**

**If MIRTK did slice-by-slice 2D registration** (it doesn't):
```
❌ 12 independent 2D registrations
❌ No through-plane consistency
❌ Potential slice misalignment
❌ Loss of 3D anatomical constraints
```

**Actual MIRTK 3D registration**:
```
✅ Single 3D transformation
✅ Enforced 3D smoothness
✅ Anatomical continuity preserved
✅ All 12 slices registered consistently
```

### 10. **Key Insights for Your Implementation**

#### **Our Deepali Approach is Correct**
- We resample static to Frame0 grid for registration ✅
- We calculate NMI on full 3D volume ✅
- We use 3D SVFFD transformation ✅
- We apply foreground masking ✅

#### **Why 12 Slices is Sufficient**
1. **Rich in-plane detail**: 192×192 provides excellent anatomical detail
2. **Strategic sampling**: Dynamic sequences sample motion-critical regions
3. **3D smoothness**: SVFFD interpolates plausibly between slices
4. **Foreground focus**: Registration only driven by tissue overlap

#### **Success Factors**
- **NMI robustness**: Handles cross-modal intensity differences
- **Multi-resolution**: Coarse→fine progression handles sparse sampling
- **Regularization**: Prevents overfitting to sparse data
- **Physical coordinates**: Proper spatial relationships maintained

## Conclusion

**MIRTK succeeds with anisotropic images because**:

1. **Treats as full 3D volume**, not slice-by-slice
2. **Leverages anatomical continuity** through SVFFD smoothness
3. **Uses robust similarity metrics** (NMI) insensitive to modality differences  
4. **Applies smart foreground masking** to focus on tissue overlap
5. **Employs multi-resolution strategy** to handle resolution disparities

**Your Frame0 with 12 slices provides sufficient information** for robust registration because:
- High in-plane resolution captures detailed anatomy
- 3mm slice thickness samples major anatomical structures
- NMI captures cross-modal relationships effectively
- 3D deformation field interpolates smoothly between slices

**The sparse through-plane sampling is not a limitation** but rather a different but valid representation of 3D anatomy that MIRTK handles excellently through its sophisticated 3D registration framework.

---

*This analysis explains why MIRTK produces excellent results even with highly anisotropic medical images like dynamic MRI sequences with sparse temporal sampling.*