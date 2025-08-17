# MIRTK World Coordinate Registration with deepali

This implementation replicates MIRTK rigid registration using deepali, working in physical world coordinates to find optimal image alignment while preserving original resolutions.

## Overview

The goal is straightforward: register two medical images in their physical world coordinates, find the transformation matrix, and save both directions with their original resolutions intact.

### Key Principles

1. **Physical World Registration**: Work in world coordinates, not voxel coordinates
2. **Preserve Original Resolutions**: Registration = moving images, NOT changing resolution
3. **Bidirectional Results**: Save both directions with appropriate coordinate systems
4. **ITK-SNAP Verification**: Results should overlay perfectly in ITK-SNAP

## Implementation Details

### Core Algorithm

```
1. Load images preserving original coordinate systems
2. Create common registration grid (using static as reference)
3. Run multi-resolution rigid registration (6-DOF: 3 translation + 3 rotation)
4. Save bidirectional results:
   - Static moved to align with Frame0 (keeps static resolution)
   - Frame0 moved to align with Static (keeps frame0 resolution)
```

### Mathematical Foundation

- **Rigid Transform**: 6 degrees of freedom (3 translation + 3 rotation parameters)
- **Multi-resolution**: Pyramid levels [4, 3, 2] for coarse-to-fine optimization
- **Loss Function**: Normalized Mutual Information (NMI) with MIRTK-style masking
- **Coordinate Systems**: Physical world coordinates as universal language

## Usage

### Basic Usage

```bash
python mirtk_world_registration.py
```

### Expected Input

- `osamri007_static.nii.gz`: Static reference image (256×256×160 @ 0.78×0.78×0.8mm)
- `osamri007_frame0.nii.gz`: Moving target image (144×144×12 @ 1.5×1.5×3mm)

### Output Files

```
mirtk_world_registration/
├── static_moved_to_frame0_alignment.nii.gz     # Static aligned with frame0 (keeps static resolution)
├── frame0_moved_to_static_alignment.nii.gz     # Frame0 aligned with static (keeps frame0 resolution)  
├── static_reference.nii.gz                     # Original static for comparison
└── frame0_reference.nii.gz                     # Original frame0 for comparison
```

## Verification in ITK-SNAP

### Option A: Static moved to Frame0 space
1. Load main image: `frame0_reference.nii.gz`
2. Add overlay: `static_moved_to_frame0_alignment.nii.gz`
3. **Result**: Perfect overlay showing static image aligned with frame0

### Option B: Frame0 moved to Static space  
1. Load main image: `static_reference.nii.gz`
2. Add overlay: `frame0_moved_to_static_alignment.nii.gz`
3. **Result**: Perfect overlay showing frame0 image aligned with static

## Technical Implementation

### Registration Pipeline

1. **Image Loading**
   - Preserve original coordinate systems
   - Create registration grid using static as reference
   - Convert to deepali format for optimization

2. **Multi-Resolution Optimization**
   - Pyramid levels: 4 (1/16), 3 (1/8), 2 (1/4) resolution
   - MIRTK-style NMI loss with foreground masking
   - Adam optimizer with level-specific learning rates

3. **Bidirectional Saving**
   - Direction A: Apply inverse transform to static → frame0 alignment
   - Direction B: Apply forward transform to frame0 → static alignment
   - Each maintains its original coordinate system and resolution

### Interpolation Methods

**Critical Implementation Detail**: Different data types require different interpolation methods during transformation.

#### 🖼️ **Medical Images (Static, Frame0)**
```python
# Uses LINEAR interpolation (default)
inverse_transformer = spatial.ImageTransformer(inverse_transform)
warped_image_tensor = inverse_transformer(image_tensor)
```

**Why Linear Interpolation for Images:**
- **Grayscale values** can be smoothly interpolated (0.5 between 0 and 1 is valid)
- **Preserves image quality** and anatomical continuity  
- **Creates realistic intermediate intensity values**
- **Result**: Natural-looking moved images with smooth transitions

#### 🏷️ **Segmentations (Labels)**
```python
# Uses NEAREST NEIGHBOR interpolation 
seg_transformer = spatial.ImageTransformer(inverse_transform, mode='nearest')
warped_seg_tensor = seg_transformer(segmentation_tensor)
```

**Why Nearest Neighbor for Segmentations:**
- **Label values** are discrete categories (0=background, 1=airway, etc.)
- **Intermediate values** (e.g., 0.5 between labels 0 and 1) are meaningless
- **Linear interpolation** creates artifacts and holes in segmentations
- **Result**: Clean segmentation boundaries without interpolation artifacts

#### ⚠️ **Common Mistake**
```python
# WRONG - Creates holes in segmentation
warped_seg = linear_transformer(segmentation)  # ❌ Mini holes appear

# CORRECT - Preserves discrete labels  
warped_seg = nearest_transformer(segmentation)  # ✅ Clean boundaries
```

**Key Insight**: The moved images look continuous and natural **because** we use linear interpolation for them - this is the correct approach for medical image registration.

### Key Advantages

- ✅ **Exact MIRTK Replication**: Follows MIRTK's world coordinate approach
- ✅ **Resolution Preservation**: Each result keeps original voxel spacing
- ✅ **ITK-SNAP Compatible**: Perfect overlay verification
- ✅ **Robust Implementation**: Handles different image orientations and sizes
- ✅ **Fast Convergence**: Well-aligned images converge quickly
- ✅ **Proper Interpolation**: Linear for images, nearest neighbor for segmentations

## Dependencies

```python
import torch
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# deepali imports
import deepali.spatial as spatial
from deepali.core import Grid, functional as U
from deepali.data import Image
from deepali.losses import NMI
```

## File Structure

```
deepali_reg/
├── mirtk_world_registration.py          # Main implementation
├── world_coordinate_alignment_check.py  # Alignment verification tool
├── README.md                            # This documentation
└── mirtk_world_registration/            # Output directory
```

## Registration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pyramid Levels | [4, 3, 2] | Multi-resolution levels (coarse to fine) |
| Iterations | {4: 5, 3: 10, 2: 15} | Iterations per level |
| Learning Rates | {4: 1e-3, 3: 8e-4, 2: 5e-4} | Adam optimizer rates |
| NMI Bins | 64 | Normalized Mutual Information bins |
| Transform Type | Rigid (6-DOF) | Translation + Rotation only |

## Expected Results

### Before Registration
- Images are already well-aligned in world coordinates (0.0mm center distance)
- Different resolutions: Static (0.78mm) vs Frame0 (1.5-3mm)
- Different fields of view and orientations

### After Registration  
- **Static moved**: 256×256×160 image aligned with frame0, keeps 0.78mm resolution
- **Frame0 moved**: 144×144×12 image aligned with static, keeps 1.5×3mm resolution
- Perfect overlay in ITK-SNAP for both directions
- Minimal transformation due to excellent initial alignment

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure SimpleITK arrays use correct (Z, Y, X) ordering
2. **Coordinate Systems**: Verify images load with proper world coordinates  
3. **Memory Usage**: Use CPU device for large images to avoid GPU memory issues
4. **Convergence**: Well-aligned images should converge quickly with minimal loss changes

### Debug Tips

- Check initial alignment with `world_coordinate_alignment_check.py`
- Verify image sizes and spacing match expected values
- Monitor loss convergence - should be rapid for well-aligned images
- Test overlay in ITK-SNAP immediately after registration

## References

- **MIRTK**: Medical Image Registration ToolKit
- **deepali**: Deep Learning for Medical Image Analysis  
- **SimpleITK**: Simplified interface to ITK
- **ITK-SNAP**: Interactive medical image segmentation and visualization

---

*This implementation exactly replicates MIRTK's rigid registration approach using deepali, providing robust world coordinate alignment with original resolution preservation.*