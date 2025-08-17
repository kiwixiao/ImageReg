# MIRTK's On-the-Fly Coordinate Transformation Method

## Core Philosophy

MIRTK uses a **"never resample during registration"** approach. Instead of projecting images to a common coordinate system, MIRTK performs all coordinate transformations **on-the-fly** during the registration process.

## The MIRTK Registration Process

### 1. Preserve Original Coordinate Systems

```python
# Each image keeps its native coordinate system
static_image:
  - coordinate_system: (origin, spacing, direction)
  - data: original voxel values
  - NO resampling

moving_image:
  - coordinate_system: (origin, spacing, direction) 
  - data: original voxel values
  - NO resampling
```

### 2. On-the-Fly Coordinate Transformation

```python
def mirtk_registration_step(static_image, moving_image, current_transform):
    """Single registration iteration - MIRTK style"""
    
    similarity = 0
    
    # Loop through static image voxels
    for static_voxel_coord in static_image.domain:
        
        # Step 1: Convert static voxel to world coordinates
        world_point = static_image.voxel_to_world(static_voxel_coord)
        
        # Step 2: Apply current transformation
        transformed_world_point = current_transform.apply(world_point)
        
        # Step 3: Sample moving image at transformed world point
        # Moving image uses ITS coordinate system for world→voxel conversion
        moving_value = moving_image.sample_at_world_point(transformed_world_point)
        
        # Step 4: Compute similarity metric
        static_value = static_image.get_voxel_value(static_voxel_coord)
        similarity += compute_metric(static_value, moving_value)
    
    # Step 5: Update transformation parameters
    gradient = compute_gradient(similarity)
    current_transform.update_parameters(gradient)
    
    return current_transform
```

### 3. Key Coordinate System Operations

#### Static Image: Voxel → World
```python
def static_voxel_to_world(voxel_coord):
    """Convert static image voxel to world coordinates"""
    return static_origin + static_direction @ (voxel_coord * static_spacing)
```

#### Moving Image: World → Voxel → Sample
```python
def sample_moving_at_world_point(world_point):
    """Sample moving image at world coordinate"""
    # Convert world point to moving image voxel coordinates
    voxel_coord = moving_direction_inv @ ((world_point - moving_origin) / moving_spacing)
    
    # Interpolate moving image at (possibly fractional) voxel coordinate
    return interpolate(moving_image_data, voxel_coord)
```

## World Coordinates: The Common Language

### What Are World Coordinates?

**World coordinates** are the physical space coordinates (in millimeters) that represent actual anatomical locations. They serve as the "common language" between different image coordinate systems.

```python
# Example: Heart apex location
heart_apex_world = [10.5, -45.2, 120.8]  # mm in physical space

# This same anatomical point has different voxel coordinates:
static_voxel = static_image.world_to_voxel(heart_apex_world)   # [45, 123, 89]
moving_voxel = moving_image.world_to_voxel(heart_apex_world)   # [78, 34, 12]

# But both refer to the SAME anatomical location!
```

### Why World Coordinates Work

1. **Physically meaningful**: Represent actual anatomy positions
2. **Coordinate system independent**: Same world point regardless of image's coordinate system
3. **Transformation target**: Transformations operate in world space
4. **Common reference frame**: All images can convert to/from world coordinates

## The MIRTK Registration Algorithm

### Initialization
```python
def initialize_mirtk_registration(static_image, moving_image):
    """Initialize MIRTK-style registration"""
    
    # 1. Keep original coordinate systems
    static_coords = extract_coordinate_system(static_image)
    moving_coords = extract_coordinate_system(moving_image)
    
    # 2. Initialize transformation (e.g., center of mass alignment)
    static_center_world = static_coords.image_center_to_world()
    moving_center_world = moving_coords.image_center_to_world()
    
    initial_translation = static_center_world - moving_center_world
    transform = RigidTransform(translation=initial_translation)
    
    return transform, static_coords, moving_coords
```

### Registration Loop
```python
def mirtk_registration_loop(static_image, moving_image, transform):
    """Main MIRTK registration loop"""
    
    for iteration in range(max_iterations):
        
        # Multi-resolution pyramid level
        static_level = get_pyramid_level(static_image, current_level)
        moving_level = get_pyramid_level(moving_image, current_level)
        
        # Registration step with on-the-fly coordinate transformation
        transform = registration_step(static_level, moving_level, transform)
        
        # Check convergence
        if converged(transform):
            break
    
    return transform
```

### Core Registration Step
```python
def registration_step(static_image, moving_image, transform):
    """Single registration step - no resampling"""
    
    gradient = initialize_gradient(transform.num_parameters)
    similarity = 0
    
    # Sample static image domain
    for static_voxel in sample_image_domain(static_image):
        
        # Static voxel → world coordinates
        world_point = static_image.coordinate_system.voxel_to_world(static_voxel)
        
        # Apply current transformation
        transformed_point = transform.apply(world_point)
        
        # Sample moving image at transformed world point
        moving_value = moving_image.sample_at_world_coordinate(transformed_point)
        static_value = static_image.get_voxel_value(static_voxel)
        
        # Compute metric and gradient
        metric_value, metric_gradient = compute_nmi_with_gradient(
            static_value, moving_value, transform.parameters
        )
        
        similarity += metric_value
        gradient += metric_gradient
    
    # Update transformation parameters
    transform.update_parameters(gradient, learning_rate)
    
    return transform
```

## Advantages of On-the-Fly Method

### 1. No Data Loss
- **Original images never modified**: Preserve exact voxel values
- **No interpolation artifacts during optimization**: More accurate gradients
- **Maintains image quality**: No degradation from resampling

### 2. Memory Efficiency
- **No intermediate images**: Original images only
- **No common coordinate system copies**: Saves memory
- **Dynamic sampling**: Only compute what's needed

### 3. Computational Efficiency
- **No upfront resampling cost**: Start registration immediately
- **Cache-friendly**: Access patterns optimized for static image
- **Parallel-friendly**: Independent voxel processing

### 4. Maximum Flexibility
- **Any coordinate system combination**: No preprocessing required
- **Preserves native resolution**: Each image keeps optimal sampling
- **Composable transformations**: Can chain multiple registrations

## After Registration: Applying Results

### The Transformation Output
```python
# MIRTK registration produces a transformation
final_transform = mirtk_register(static_image, moving_image)

# This transformation maps:
# moving_world_coordinates → static_world_coordinates
```

### Applying the Transformation
```python
def apply_transformation(moving_image, transform, reference_image):
    """Apply transformation to create aligned image"""
    
    # Create output image in reference coordinate system
    output_image = create_image_like(reference_image)
    
    # For each voxel in output image
    for output_voxel in output_image.domain:
        
        # Convert output voxel to world coordinates
        world_point = reference_image.voxel_to_world(output_voxel)
        
        # Apply INVERSE transformation to find source point
        source_world_point = transform.apply_inverse(world_point)
        
        # Sample moving image at source point
        output_value = moving_image.sample_at_world_point(source_world_point)
        output_image.set_voxel_value(output_voxel, output_value)
    
    return output_image
```

## World Coordinate Alignment

### Key Insight: Images Already Aligned in World Space

For our OSAMRI007 dataset:
```python
static_center_world = [10.3, 3.4, -0.1]   # mm
moving_center_world = [8.5, 3.8, -0.4]    # mm
distance = 2.0 mm  # Already very close!
```

**This means:**
1. **Images are already aligned** in world coordinates (heart in same location)
2. **Registration refinement needed**: Small adjustments for precise alignment
3. **Coordinate system differences**: Just different ways to measure same anatomy
4. **MIRTK handles this naturally**: Uses world coordinates as common reference

## Implementation Strategy for deepali

### Core Components Needed

1. **Coordinate System Handler**
   ```python
   class CoordinateSystem:
       def voxel_to_world(self, voxel_coord)
       def world_to_voxel(self, world_coord)
       def sample_at_world_point(self, world_point)
   ```

2. **On-the-Fly Transformer**
   ```python
   class OnTheFlyTransformer:
       def transform_and_sample(self, world_point, current_transform)
       def compute_gradient(self, static_value, moving_value)
   ```

3. **Registration Engine**
   ```python
   class MIRTKStyleRegistration:
       def register_without_resampling(self, static, moving)
       def optimize_transform_parameters(self, similarity_gradient)
   ```

### Benefits for deepali Implementation

1. **Memory efficiency**: Critical for large medical images
2. **Accuracy**: No interpolation artifacts during optimization
3. **Flexibility**: Handle any coordinate system automatically
4. **Speed**: No preprocessing resampling overhead
5. **Robustness**: Works with OSAMRI007 and any other dataset

## Conclusion

MIRTK's on-the-fly coordinate transformation is elegant because it:

- **Respects original data**: Never modifies source images
- **Uses world coordinates as universal language**: Common reference frame
- **Performs transformations dynamically**: No intermediate coordinate systems
- **Achieves maximum accuracy**: No data degradation during optimization
- **Handles coordinate differences naturally**: Mathematical transformations only

This approach explains why MIRTK works so well with diverse medical imaging datasets - it treats coordinate system differences as what they are: different ways to measure the same physical space.