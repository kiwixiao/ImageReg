# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image registration system that replicates MIRTK (Medical Image Registration ToolKit) functionality using Python and the deepali library. The system focuses on registering medical images in physical world coordinates while preserving original image resolutions.

## Core Architecture

### Main Pipeline Entry Point
- **`deepali_reg/mirtk_world_registration.py`**: Main entry point with configuration-driven registration
- Command: `python deepali_reg/mirtk_world_registration.py --inputs inputs_OSAMRI016`
- Supports multiple registration types through YAML configuration

### Registration Modules (deepali_reg/modules/)
- **`rigid_registration.py`**: 6-DOF rigid registration (translation + rotation)
- **`rigid_affine_registration.py`**: Sequential rigid→affine registration (6+12 DOF)
- **`io_utils.py`**: Image I/O utilities
- **`registration.py`**: Base registration utilities

### Configuration System
Each input folder (e.g., `inputs_OSAMRI016/`) contains:
- **`registration_config.yaml`**: Main configuration file
- Registration type options: `"rigid"`, `"rigid_affine"`, `"rigid_affine_svffd"`
- Frame extraction settings, device selection, output paths

### Working Pipeline vs Reference Implementation
- **`regTemplate_mirtk/`**: Proven working MIRTK binary pipeline (reference standard)
- **`MIRTK/`**: Source code of MIRTK binary for technical reference
- **`deepali_reg/`**: Python implementation using deepali library

## Key Technical Concepts

### World Coordinate Registration
All registration operates in physical world coordinates (millimeters), not voxel indices. This ensures:
- Resolution independence
- Proper handling of different image spacing
- ITK-SNAP compatible overlays

### Bidirectional Result Saving
Each registration produces two outputs:
1. **Static→Frame0 alignment**: Static image moved to frame0 space (keeps static resolution)
2. **Frame0→Static alignment**: Frame0 image moved to static space (keeps frame0 resolution)

### Multi-Resolution Optimization
- Pyramid levels: [4, 3, 2] representing 1/16, 1/8, 1/4 resolution
- Coarse-to-fine optimization with level-specific learning rates
- NMI (Normalized Mutual Information) loss with MIRTK-style foreground masking

### Critical Implementation Details

#### Tensor Dimension Handling
The deepali library requires specific tensor dimensions for different operations:
```python
# NMI loss requires 5D tensors [N,C,D,H,W]
if warped_source.dim() == 3:
    warped_source = warped_source.unsqueeze(0).unsqueeze(0)
elif warped_source.dim() == 4:
    warped_source = warped_source.unsqueeze(0)
```

#### Interpolation Methods
- **Medical images**: Linear interpolation for smooth grayscale transitions
- **Segmentations**: Nearest neighbor interpolation to preserve discrete labels
- Use `.detach()` before `.cpu().numpy()` for tensors with gradients

#### Coordinate System Consistency
Registration transforms are learned on resampled grids, so final transformations must be applied to appropriately gridded images to avoid dimension mismatches.

## Common Development Commands

### Basic Registration
```bash
# Rigid registration
python deepali_reg/mirtk_world_registration.py --inputs inputs_OSAMRI016

# Change registration type by editing inputs_OSAMRI016/registration_config.yaml:
# registration:
#   type: "rigid"           # or "rigid_affine" or "rigid_affine_svffd"
```

### Testing Different Registration Types
1. Edit `inputs_OSAMRI016/registration_config.yaml`
2. Change `registration.type` field
3. Run main pipeline command
4. Compare results in `outputs_OSAMRI016/`

### Output Verification
Results are designed for ITK-SNAP verification:
```
Option A: Load frame0_reference.nii.gz, overlay static_moved_to_frame0_alignment.nii.gz
Option B: Load static_reference.nii.gz, overlay frame0_moved_to_static_alignment.nii.gz
```

## File Structure Patterns

### Input Structure
```
inputs_OSAMRI016/
├── registration_config.yaml        # Main configuration
├── OSAMRI016_2501_static.nii      # Static reference image
├── OSAMRI016_2601_Dynamic.nii     # Dynamic sequence (frame extraction)
└── OSAMRI016_2501_airway_seg.nii.gz # Segmentation (optional)
```

### Output Structure
```
outputs_OSAMRI016/
├── extracted_frames/               # Frame extraction results
├── static_moved_to_frame0_alignment.nii.gz
├── frame0_moved_to_static_alignment.nii.gz
├── static_reference.nii.gz
├── frame0_reference.nii.gz
├── registration_visualization_*.png
└── *_transform.{pth,json,txt,tfm}  # Multiple transform formats
```

## Important Dependencies

### Core Libraries
- **PyTorch**: Optimization and tensor operations
- **SimpleITK**: Medical image I/O and coordinate system handling
- **deepali**: Deep learning image registration library
- **matplotlib**: Visualization generation

### Path Dependencies
The deepali library requires specific path configuration:
```python
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')
```

## Development Notes

### Module System
The codebase uses a modular architecture where each registration type is implemented as a separate module. This allows easy extension and testing of different registration approaches.

### Configuration-Driven Design
All parameters are externalized to YAML configuration files, making it easy to experiment with different settings without code changes.

### Resolution Preservation
A key design principle is that registration finds transformations but preserves original image resolutions in the final outputs. This differs from traditional approaches that resample everything to a common grid.

### MIRTK Compatibility
The implementation aims for exact replication of MIRTK behavior, including:
- Multi-resolution pyramid schemes
- NMI loss computation with foreground masking
- Transform parameter storage formats
- World coordinate registration approach

## Reference Materials

- **`MIRTK_Analysis_Reference.md`**: MIRTK background analysis
- **`MIRTK_Research_Findings.md`**: Technical findings and implementation notes
- **`deepali_reg/README.md`**: Detailed technical documentation
- **`Pairwise Image Registration — deepali.pdf`**: Academic reference for deepali library

## Testing Strategy

Always verify that changes don't break the rigid-only mode by:
1. Setting `registration.type: "rigid"` in config
2. Running the pipeline
3. Confirming identical results to previous working versions
4. Testing new registration types only after confirming rigid mode stability
- @deepali is the source code, it has all the reference, examples and documents. when writing code, please reference the source code first to make sure the api or fucntion call is correct
- run registration, always pass the method argument