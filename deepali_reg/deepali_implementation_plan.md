# Deepali MIRTK Implementation Plan

## Overview

This document provides a comprehensive plan for implementing EXACT MIRTK registration pipeline using deepali, addressing fundamental issues with coordinate systems and resolution handling that were causing incorrect results.

## 🚨 CRITICAL INSIGHT FROM MIRTK SOURCE CODE

**KEY DISCOVERY**: MIRTK creates transforms at **FULL RESOLUTION DOMAIN** from the start. Multi-resolution optimization uses different **IMAGE resolutions** but the **SAME transform domain**.

### ❌ **Previous Wrong Understanding**
- Transforms created at pyramid level resolution
- Grid adaptation needed between levels
- Transform parameters change size between levels

### ✅ **Correct MIRTK Approach**
```cpp
// From MIRTK GenericRegistrationFilter.cc line 3381
struct ImageAttributes domain = _RegistrationDomain;  // FULL resolution domain
ffd = new FFD(domain);                                // Transform at FULL domain

for (level = coarse_to_fine) {
    images_at_level = pyramid[level];                 // Multi-resolution IMAGES
    optimize(ffd, images_at_level);                   // SAME transform, different images
}
```

**No grid adaptation in MIRTK - transforms are ALWAYS at full resolution domain!**

## ✅ Deepali Diffeomorphic Capabilities Confirmed

**Yes, deepali has full diffeomorphic support!**

1. **`StationaryVelocityFreeFormDeformation` (SVFFD)** - Exactly what MIRTK uses
2. **Built-in inverse computation** - `transform.inverse()` method
3. **Scaling and squaring** - `ExpFlow` module with configurable steps
4. **Physical coordinates** - All transforms work in physical space (mm)

---

# 📋 Comprehensive MIRTK Implementation Plan

## 🎯 PHASE 1: FULL RESOLUTION DOMAIN SETUP

### **A1. Image Loading & Analysis**
```python
# Load original images with full metadata
static_sitk = sitk.ReadImage("static.nii")           # 352×352×158 @ 0.625mm
frame0_sitk = sitk.ReadImage("frame0.nii")           # 192×192×12 @ coarser
seg_sitk = sitk.ReadImage("segmentation.nii")        # 352×352×158 @ 0.625mm

# Extract physical properties
static_spacing = static_sitk.GetSpacing()            # (0.625, 0.625, 0.625)
static_origin = static_sitk.GetOrigin()              # Physical origin (x,y,z)
static_direction = static_sitk.GetDirection()        # 3x3 orientation matrix

frame0_spacing = frame0_sitk.GetSpacing()            # Coarser spacing
frame0_origin = frame0_sitk.GetOrigin()              
frame0_direction = frame0_sitk.GetDirection()        
```

### **A2. FULL RESOLUTION DOMAIN CREATION**
```python
# CRITICAL: Create FULL RESOLUTION domain covering entire field of view
# This is what MIRTK calls _RegistrationDomain

# Use FINEST spacing from both images
full_domain_spacing = [
    min(static_spacing[i], frame0_spacing[i]) for i in range(3)
]

# Calculate physical bounding box union
def get_physical_bounds(image_sitk):
    # Convert corner points to physical coordinates
    # Return min_bounds, max_bounds

static_min, static_max = get_physical_bounds(static_sitk)
frame0_min, frame0_max = get_physical_bounds(frame0_sitk)

# Union bounds at finest resolution
union_min = np.minimum(static_min, frame0_min)
union_max = np.maximum(static_max, frame0_max)
union_size_voxels = np.ceil((union_max - union_min) / full_domain_spacing).astype(int)

# Create FULL RESOLUTION DOMAIN
full_resolution_grid = Grid(
    size=union_size_voxels,
    spacing=full_domain_spacing, 
    origin=union_min,
    direction=frame0_direction  # Use target orientation
)
```

### **A3. Registration Image Preparation**
```python
# Prepare images for registration at TARGET resolution (Frame0)
# This is SEPARATE from transform domain

target_resampled = frame0_sitk  # Already at target resolution
source_resampled = resample_to_grid(static_sitk, frame0_grid)
```

## 🎯 PHASE 2: IMAGE PREPARATION

### **B1. Resample for Registration**
```python
# Resample both images to common registration grid
# This is ONLY for registration - original resolution preserved for output

def resample_to_registration_grid(image_sitk, registration_grid):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(registration_grid.size)
    resampler.SetOutputSpacing(registration_grid.spacing)
    resampler.SetOutputOrigin(registration_grid.origin)
    resampler.SetOutputDirection(registration_grid.direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1.0)  # MIRTK padding
    return resampler.Execute(image_sitk)

# Resample for registration
target_resampled = frame0_sitk  # Already at target resolution
source_resampled = resample_to_registration_grid(static_sitk, registration_grid)
```

### **B2. Convert to deepali Images**
```python
# Save resampled images temporarily
sitk.WriteImage(target_resampled, "/tmp/target_reg.nii.gz")
sitk.WriteImage(source_resampled, "/tmp/source_reg.nii.gz")

# Load with deepali (preserves physical coordinates)
target_image_deepali = Image.read("/tmp/target_reg.nii.gz", device=device)
source_image_deepali = Image.read("/tmp/source_reg.nii.gz", device=device)

# Registration grid for deepali transforms
deepali_registration_grid = target_image_deepali.grid()
```

## 🎯 PHASE 3: MULTI-RESOLUTION REGISTRATION

### **C1. Pyramid Construction**
```python
# Build MIRTK-style pyramids (4 levels)
target_pyramid = target_image_deepali.pyramid(levels=4)
source_pyramid = source_image_deepali.pyramid(levels=4)

# Level mapping: MIRTK Level 4,3,2,1 = deepali index 0,1,2,3
print("Pyramid levels:")
for i in range(4):
    mirtk_level = 4 - i
    print(f"  MIRTK Level {mirtk_level}: {target_pyramid[i].shape}")
```

### **C2. Transform Initialization at FULL RESOLUTION**
```python
# CRITICAL: All transforms use SAME FULL RESOLUTION GRID
# NO grid adaptation needed - this is the MIRTK way!

rigid_transform = spatial.RigidTransform(full_resolution_grid).to(device)
affine_transform = spatial.AffineTransform(full_resolution_grid).to(device)

# SVFFD with MIRTK parameters at FULL RESOLUTION
control_point_spacing_mm = 2.5  # MIRTK default
voxel_spacing = min(full_resolution_grid.spacing())  # Finest spacing
stride = max(1, int(control_point_spacing_mm / voxel_spacing))

svffd_transform = spatial.StationaryVelocityFreeFormDeformation(
    full_resolution_grid,  # FULL RESOLUTION DOMAIN
    stride=stride,
    steps=64,  # MIRTK default scaling and squaring steps
    scale=1.0
).to(device)

print("🎯 ALL TRANSFORMS CREATED AT FULL RESOLUTION - NO ADAPTATION NEEDED!")
```

### **C3. Sequential Registration Stages**
```python
# STAGE 1: RIGID (6 DOF) - FULL RESOLUTION TRANSFORM
def run_rigid_stage():
    rigid_transform.train()
    for level_idx in range(4):  # Levels 4,3,2,1
        target_level = target_pyramid[level_idx]
        source_level = source_pyramid[level_idx]
        
        # KEY: NO grid update - transform stays at FULL RESOLUTION
        # Only images change resolution between levels
        
        # Create transformer with FULL RESOLUTION transform
        transformer = spatial.ImageTransformer(rigid_transform)
        
        print(f"Transform grid: {rigid_transform.grid().size()} (FULL RES)")
        print(f"Image resolution: {target_level.shape}")
        
        # Optimize with MIRTK parameters
        optimizer = Adam(rigid_transform.parameters(), lr=1e-3)
        optimize_level_with_full_transform(rigid_transform, target_level, source_level, optimizer)
    
    return rigid_transform.eval()

# STAGE 2: AFFINE (12 DOF) - initialized with rigid result
def run_affine_stage(rigid_result):
    # Initialize affine with rigid transformation
    affine_transform = initialize_affine_from_rigid(rigid_result)
    affine_transform.train()
    
    for level_idx in range(4):
        target_level = target_pyramid[level_idx]
        source_level = source_pyramid[level_idx]
        
        affine_transform.grid_(target_level.grid())
        
        optimizer = Adam(affine_transform.parameters(), lr=8e-4)
        optimize_level(affine_transform, target_level, source_level, optimizer)
    
    return affine_transform.eval()

# STAGE 3: SVFFD (hundreds of DOF) - initialized with affine result
def run_svffd_stage(affine_result):
    # Initialize SVFFD in coordinate system established by affine
    composed_linear = compose_linear_transforms(rigid_transform, affine_transform)
    
    svffd_transform.train()
    
    for level_idx in range(4):
        target_level = target_pyramid[level_idx]
        source_level = source_pyramid[level_idx]
        
        # Create new SVFFD for this level to avoid grid constraints
        level_grid = target_level.grid()
        level_spacing = min(level_grid.spacing())
        level_stride = max(1, int(2.5 / level_spacing))
        
        level_svffd = spatial.StationaryVelocityFreeFormDeformation(
            level_grid, stride=level_stride, steps=64
        ).to(device).train()
        
        optimizer = Adam(level_svffd.parameters(), lr=5e-3)
        loss_fn = create_mirtk_loss_function(bending_weight=1e-3)
        
        optimize_level(level_svffd, target_level, source_level, optimizer, loss_fn)
        svffd_transform = level_svffd  # Update to final level result
    
    return svffd_transform.eval()
```

## 🎯 PHASE 4: TRANSFORM COMPOSITION

### **D1. Final Transform Chain**
```python
# Compose all transforms in registration coordinate system
final_transform = compose_transforms([
    rigid_transform,
    affine_transform, 
    svffd_transform
])

# Compute inverse for bidirectional warping
inverse_transform = final_transform.inverse()
```

### **D2. Transform Verification**
```python
# Verify transforms work in registration space
test_point = torch.tensor([0., 0., 0.], device=device)  # Origin
forward_point = final_transform(test_point)
reconstructed_point = inverse_transform(forward_point)
print(f"Round-trip error: {torch.norm(test_point - reconstructed_point)}")
```

## 🎯 PHASE 5: OUTPUT GENERATION 

### **E1. Forward Warping (Static → Frame0 coordinate system)**
```python
# Apply to ORIGINAL STATIC at ORIGINAL STATIC RESOLUTION
def warp_static_to_frame0_space():
    # Load original static
    static_original = Image.read("static_original.nii", device=device)
    
    # Transform: static coordinates → frame0 coordinate system
    # Resolution: PRESERVED (352×352×158 @ 0.625mm)
    # Space: Frame0 coordinate system
    
    # Update transform grid to static's original grid
    transform_for_static = adapt_transform_to_grid(final_transform, static_original.grid())
    
    # Apply transformation 
    warped_static = spatial.ImageTransformer(transform_for_static)(static_original.tensor())
    
    # Create output with static resolution, frame0 coordinate system
    output_static = create_image_with_mixed_properties(
        data=warped_static,
        source_resolution=static_original,  # Keep 352×352×158 @ 0.625mm
        target_coordinate_system=frame0_sitk  # Frame0 origin/direction
    )
    
    return output_static

# Apply to segmentation (same as static)
warped_segmentation = warp_static_to_frame0_space()  # Same process
```

### **E2. Inverse Warping (Frame0 → Static coordinate system)**
```python
def warp_frame0_to_static_space():
    # Load original frame0
    frame0_original = Image.read("frame0_original.nii", device=device)
    
    # Transform: frame0 coordinates → static coordinate system  
    # Resolution: PRESERVED (192×192×12 @ frame0 spacing)
    # Space: Static coordinate system
    
    # Update inverse transform grid to frame0's original grid
    inverse_for_frame0 = adapt_transform_to_grid(inverse_transform, frame0_original.grid())
    
    # Apply inverse transformation
    warped_frame0 = spatial.ImageTransformer(inverse_for_frame0)(frame0_original.tensor())
    
    # Create output with frame0 resolution, static coordinate system
    output_frame0 = create_image_with_mixed_properties(
        data=warped_frame0,
        source_resolution=frame0_original,  # Keep 192×192×12 
        target_coordinate_system=static_sitk  # Static origin/direction  
    )
    
    return output_frame0
```

### **E3. Final Output Files**
```python
# Save with proper headers
save_with_proper_headers(warped_static, 
                        "static_warped_to_frame0_space.nii.gz",
                        reference_resolution=static_sitk,
                        reference_coordinates=frame0_sitk)

save_with_proper_headers(warped_segmentation,
                        "segmentation_warped_to_frame0_space.nii.gz", 
                        reference_resolution=static_sitk,
                        reference_coordinates=frame0_sitk)

save_with_proper_headers(warped_frame0,
                        "frame0_warped_to_static_space.nii.gz",
                        reference_resolution=frame0_sitk, 
                        reference_coordinates=static_sitk)
```

## 🎯 PHASE 6: VALIDATION

### **F1. Output Verification**
```python
# Verify output properties
static_warped = sitk.ReadImage("static_warped_to_frame0_space.nii.gz")
print(f"Static warped size: {static_warped.GetSize()}")      # Should be (352,352,158)
print(f"Static warped spacing: {static_warped.GetSpacing()}")  # Should be (0.625,0.625,0.625)
print(f"Static warped origin: {static_warped.GetOrigin()}")    # Should be Frame0 origin

frame0_warped = sitk.ReadImage("frame0_warped_to_static_space.nii.gz")  
print(f"Frame0 warped size: {frame0_warped.GetSize()}")      # Should be (192,192,12)
print(f"Frame0 warped spacing: {frame0_warped.GetSpacing()}")  # Should be Frame0 spacing
print(f"Frame0 warped origin: {frame0_warped.GetOrigin()}")    # Should be Static origin
```

---

## 🔑 Key Implementation Details

### **NO GRID ADAPTATION NEEDED!**
```python
# MIRTK INSIGHT: No grid adaptation function needed!
# Transforms are created at full resolution from start

def apply_full_resolution_transform(transform, image_tensor):
    """Apply full resolution transform to any image"""
    # Transform is ALREADY at full resolution
    # Works with any image resolution - no adaptation needed
    transformer = spatial.ImageTransformer(transform)
    return transformer(image_tensor)
```

### **Mixed Properties Image Creation**
```python
def create_image_with_mixed_properties(data, source_resolution, target_coordinate_system):
    """Create image with source resolution but target coordinate system"""
    # Use source for: size, spacing (resolution properties)
    # Use target for: origin, direction (coordinate system properties)
    
    output = sitk.GetImageFromArray(data.cpu().numpy())
    output.SetSpacing(source_resolution.GetSpacing())
    output.SetOrigin(target_coordinate_system.GetOrigin())  
    output.SetDirection(target_coordinate_system.GetDirection())
    
    return output
```

### **MIRTK Loss Function**
```python
def create_mirtk_loss_function(bending_weight=1e-3):
    """Create MIRTK-style loss function"""
    
    def loss_fn(warped_source, target, transform):
        # MIRTK-style static foreground mask
        target_fg = target != -1.0  # Padding value
        source_fg = warped_source != -1.0
        mask = target_fg.float() * source_fg.float()
        
        # Apply mask
        warped_masked = warped_source * mask
        target_masked = target * mask
        
        # NMI similarity (MIRTK default)
        nmi_loss = NMI(bins=64)
        similarity = nmi_loss(warped_masked, target_masked)
        
        total_loss = similarity
        
        # Add bending energy for SVFFD only
        if hasattr(transform, 'v'):  # SVFFD has velocity field
            v = transform.v
            bending_energy = L.bending_loss(v)
            total_loss = similarity + bending_weight * bending_energy
        
        return total_loss
    
    return loss_fn
```

---

## 🚨 Critical Differences from Previous Attempts

### **1. COORDINATE SYSTEM HANDLING**
- **Previous**: Mixed voxel and physical coordinates
- **Correct**: Pure physical coordinates (mm) throughout

### **2. RESOLUTION PRESERVATION**
- **Previous**: Output at registration resolution
- **Correct**: Output preserves source resolution, changes coordinate system only

### **3. GRID CONSISTENCY**
- **Previous**: Each transform had different grid references
- **Correct**: All transforms use unified registration grid, adapted for application

### **4. TRANSFORM COMPOSITION**
- **Previous**: Broken sequential application with grid mismatches
- **Correct**: Proper grid adaptation and sequential application

### **5. OUTPUT GENERATION**
- **Previous**: Simple resampling without coordinate system consideration
- **Correct**: Mixed properties (source resolution + target coordinates)

---

## 📊 Expected Results

### **Correct Output Properties**
1. **`static_warped_to_frame0_space.nii.gz`**: 
   - Size: 352×352×158 
   - Spacing: 0.625×0.625×0.625mm
   - Coordinate system: Frame0 (origin, direction)

2. **`segmentation_warped_to_frame0_space.nii.gz`**: 
   - Size: 352×352×158 
   - Spacing: 0.625×0.625×0.625mm
   - Coordinate system: Frame0 (origin, direction)

3. **`frame0_warped_to_static_space.nii.gz`**: 
   - Size: 192×192×12 
   - Spacing: Frame0 spacing
   - Coordinate system: Static (origin, direction)

### **Validation Criteria**
- **Resolution preserved**: Source image dimensions and spacing unchanged
- **Coordinate transformation**: Proper spatial alignment in target coordinate system
- **Anatomical correspondence**: Structures align between warped and target images
- **Smooth deformations**: No artifacts or discontinuities
- **Inverse consistency**: Round-trip transformations minimize error

---

## 🎯 Implementation Success Factors

1. **Sequential DOF progression**: 6 → 12 → hundreds
2. **Multi-resolution per stage**: 4 levels each
3. **Proper initialization**: Each stage builds on previous
4. **MIRTK parameters**: Exact energy function and weights
5. **Coarse-to-fine strategy**: At every stage
6. **Physical coordinate consistency**: Throughout pipeline
7. **Resolution preservation**: Source properties maintained
8. **Diffeomorphic transforms**: SVFFD with scaling and squaring

This plan ensures **EXACT MIRTK compliance** while preserving source resolutions and properly handling coordinate system transformations.