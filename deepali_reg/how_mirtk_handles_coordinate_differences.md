# How MIRTK Handles Coordinate System Differences

## Overview

MIRTK (Medical Image Registration ToolKit) provides a sophisticated approach to handling coordinate system differences between medical images. Unlike ITK-SNAP's resampling approach, MIRTK uses a **transformation-based coordinate system** that can handle arbitrary orientations and coordinate systems through mathematical transformations.

## MIRTK's Coordinate System Philosophy

### Core Concept: Transformation Matrices

MIRTK handles coordinate differences through a **lattice-to-world transformation system**:

1. **Lattice Coordinates**: Voxel-based coordinate system (i, j, k)
2. **World Coordinates**: Physical space coordinates (x, y, z) in millimeters
3. **Transformation Matrices**: Mathematical mappings between coordinate systems

### Key Components

#### 1. BaseImage Class Coordinate Methods

```cpp
// Core coordinate transformation methods
ImageToWorld(i, j, k, &x, &y, &z)  // Convert voxel indices to physical coordinates
WorldToImage(x, y, z, &i, &j, &k)  // Convert physical coordinates to voxel indices
```

#### 2. Transformation Matrix Stack

MIRTK maintains multiple transformation levels:
- **Image-to-World Matrix**: Basic spatial transformation from voxel lattice to physical space
- **Affine World Transformation**: Additional affine transformation applied after image-to-world
- **Registration Transformations**: Computed transformations between different images

#### 3. Orientation Handling

```cpp
// Orientation support
GetOrientation()           // Get image orientation codes (L2R, P2A, I2S)
PutOrientation()          // Set image orientation
GetAxis(axis)             // Get axis orientation relative to patient
```

## How MIRTK Handles Our OSAMRI007 Case

### The MIRTK Approach

Instead of resampling images to a common grid (like ITK-SNAP), MIRTK:

1. **Preserves Original Coordinate Systems**: Each image keeps its native coordinate system
2. **Computes Transformation Mappings**: Creates mathematical mappings between coordinate systems
3. **Uses On-Demand Coordinate Conversion**: Converts coordinates as needed during processing
4. **Applies Composed Transformations**: Combines multiple transformations for complex mappings

### Initial Alignment Strategy

MIRTK's registration process handles coordinate differences through:

#### 1. Center of Mass Alignment (init-dof)

```bash
# MIRTK's automatic initialization
mirtk init-dof -center static.nii.gz frame0.nii.gz initial.dof.gz
```

This command:
- Computes center of mass for each image in their respective coordinate systems
- Creates a transformation that aligns centers of mass in world coordinates
- Handles orientation differences through transformation composition

#### 2. Coordinate System Mapping

MIRTK's `init-dof` can create transformations that:
- Map between different lattice orientations
- Handle spacing differences
- Account for origin offsets
- Preserve physical relationships

### The MIRTK Transformation Pipeline

#### Step 1: Individual Image Coordinate Systems

```
Static Image Lattice → Static World Coordinates
Frame0 Image Lattice → Frame0 World Coordinates
```

#### Step 2: Inter-Image Transformation

```
Static World ← Transform ← Frame0 World
```

#### Step 3: Composed Transformation

```
Static Lattice ← Static World ← Transform ← Frame0 World ← Frame0 Lattice
```

## MIRTK vs ITK-SNAP: Different Philosophies

### ITK-SNAP Approach
- **Resampling Strategy**: Put all images in same coordinate system
- **Common Grid**: Create unified voxel grid for comparison
- **Data Duplication**: Resampled images require additional memory
- **Fixed Resolution**: All images share same spacing/resolution

### MIRTK Approach  
- **Transformation Strategy**: Keep original coordinate systems, map between them
- **Preserve Original Data**: No resampling of source images
- **Memory Efficient**: Original images unchanged
- **Flexible Resolution**: Each image maintains native resolution

## MIRTK's Coordinate Transformation Implementation

### 1. Lattice-to-World Matrix Construction

Based on our analysis of MIRTK source code, the transformation follows:

```cpp
// MIRTK's transformation composition
Matrix lattice_to_world = A * T * R * S * T0;

where:
T0 = Translation to center lattice at origin
S  = Scaling by voxel spacing  
R  = Rotation (direction matrix)
T  = Translation to world origin
A  = Additional affine transformation
```

### 2. Multi-Resolution Handling

MIRTK handles different resolutions through:
- **Coordinate space registration**: Optimization happens in transformed coordinate spaces
- **On-demand sampling**: Sample images at arbitrary coordinates during registration
- **Gradient computation**: Compute gradients in appropriate coordinate systems

### 3. Registration Process

```cpp
// Simplified MIRTK registration workflow
for each pyramid level:
    for each iteration:
        // Sample moving image at transformed coordinates
        moving_value = SampleImageAtWorldCoordinate(moving_image, world_x, world_y, world_z)
        
        // Compute similarity metric in common coordinate space
        similarity = ComputeSimilarity(static_value, moving_value)
        
        // Update transformation parameters
        UpdateTransformation(similarity_gradient)
```

## Practical Implications

### Advantages of MIRTK's Approach

1. **Preservation of Original Data**: No interpolation artifacts from resampling
2. **Flexible Coordinate Systems**: Can handle any coordinate system combination
3. **Memory Efficiency**: No duplicate image data
4. **Numerical Precision**: Exact coordinate transformations
5. **Multi-Resolution Support**: Native support for different image resolutions

### Complexity Trade-offs

1. **Mathematical Complexity**: Requires understanding of transformation composition
2. **Implementation Complexity**: More complex than simple resampling
3. **Debugging Difficulty**: Coordinate bugs can be harder to trace
4. **Performance Considerations**: On-demand coordinate conversion overhead

## Why Both Approaches Work

### Physical Space Alignment
Both MIRTK and ITK-SNAP succeed because:
- **Images are already physically aligned**: The OSAMRI007 images have 2mm center separation
- **Different representations**: The issue is coordinate system representation, not spatial alignment
- **Mathematical equivalence**: Both approaches achieve the same physical alignment through different means

### MIRTK's Robustness
MIRTK handles coordinate differences robustly because:
1. **Automatic center-of-mass initialization**: Aligns image centers automatically
2. **Flexible transformation composition**: Can handle arbitrary coordinate system differences
3. **Multi-level optimization**: Coarse-to-fine registration handles large coordinate differences
4. **Robust similarity metrics**: NMI and other metrics work across coordinate systems

## Conclusion

MIRTK's approach to coordinate system differences is fundamentally different from ITK-SNAP's:

- **ITK-SNAP**: "Make everything the same coordinate system"
- **MIRTK**: "Keep everything different, but map between them mathematically"

Both approaches work for the OSAMRI007 dataset because the images are physically well-aligned. The choice between approaches depends on:

- **Use case requirements**: Visualization vs. registration
- **Memory constraints**: Resampling vs. transformation overhead  
- **Processing pipeline**: Unified grids vs. flexible transformations
- **Accuracy needs**: Interpolation artifacts vs. mathematical precision

MIRTK's transformation-based approach provides maximum flexibility for handling diverse medical imaging coordinate systems while preserving the original data integrity.