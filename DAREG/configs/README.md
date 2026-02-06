# DAREG Configuration Presets

This directory contains configuration presets for different registration scenarios.

## Configuration Hierarchy

1. **CLI `--json-config` path** (highest priority)
2. **Project inputs folder**: `inputs_*/registration_config.yaml`
3. **Package defaults**: `DAREG/configs/default.yaml` (lowest priority)

## Available Presets

### `default.yaml`
Default configuration combining MIRTK-equivalent FFD and standard SVFFD settings.
Use as fallback when no project-specific config exists.

### `ffd_mirtk.yaml`
MIRTK-equivalent FFD configuration matching MIRTK register command behavior:
- Conjugate Gradient optimizer with Polak-Ribiere formula
- Adaptive line search: rise=1.1, drop=0.5, max_rejected=1
- Bending: 0.001, Diffusion: 0.0005
- Control point spacing: 4mm
- 64 histogram bins for NMI

**Use when**: Replicating MIRTK FFD results exactly.

### `svffd_standard.yaml`
Conservative SVFFD starting point:
- Lower learning rate (0.005) than FFD
- Integration steps: 5
- Minimal regularization for general use
- No velocity smoothing by default

**Use when**: General diffeomorphic registration without specific constraints.

### `svffd_high_precision.yaml`
High-precision diffeomorphic registration:
- More iterations (150, 150, 120, 100)
- Jacobian penalty: 0.005 for topology preservation
- Velocity smoothing: 1.0mm sigma
- Integration steps: 7

**Use when**: Topology preservation is critical, or fine anatomical detail matters.

### `svffd_motion_tracking.yaml`
Optimized for respiratory/airway motion tracking:
- Lighter smoothing (0.5mm) for fine motion capture
- Balanced regularization (bending: 0.0008)
- Fewer iterations at fine levels for efficiency
- Integration steps: 5

**Use when**: 4D dynamic MRI motion tracking, airway deformation analysis.

## Parameter Reference

### FFD Parameters
| Parameter | Description | MIRTK Default |
|-----------|-------------|---------------|
| `control_point_spacing` | B-spline grid spacing (mm) | 4 |
| `pyramid_levels` | Multi-resolution levels | 4 |
| `iterations_per_level` | Max iterations per level | [100, 100, 100, 100] |
| `bending_weight` | B-spline bending energy | 0.001 |
| `diffusion_weight` | Displacement diffusion | 0.0005 |

### SVFFD-Specific Parameters
| Parameter | Description | Standard |
|-----------|-------------|----------|
| `integration_steps` | Scaling-and-squaring steps | 5 |
| `velocity_smoothing_sigma` | Gaussian blur on velocity (mm) | 0.0 |
| `laplacian_weight` | ∇²v regularization | 0.0 |
| `jacobian_penalty` | Topology preservation weight | 0.0 |

### Optimizer Parameters (FFD only)
| Parameter | Description | MIRTK Default |
|-----------|-------------|---------------|
| `max_rejected_streak` | Stop after N rejections | 1 |
| `step_rise` | Step increase on accept | 1.1 |
| `step_drop` | Step decrease on reject | 0.5 |
| `epsilon` | Min relative function change | 1e-4 |
| `delta` | Min DoF change threshold | 1e-12 |

## Auto-Naming Output Folders

When `output.auto_name: true`, output folders are named using the template:
```
{model}_cp{spacing}_bend{bending}_steps{steps}
```

Examples:
- `ffd_mirtk_cp4_bend0.001`
- `svffd_std_cp4_bend0.0005_steps5`
- `svffd_hprec_cp4_bend0.001_steps7`

## Usage Examples

```bash
# Use specific preset
python -m DAREG.main_motion --config-preset svffd_motion_tracking --output auto

# Override preset with CLI args
python -m DAREG.main_motion --config-preset svffd_standard --ffd-spacing 8

# Use project-specific config (auto-detected in inputs folder)
python -m DAREG.main_motion --json-config inputs_testing/registration_config.yaml
```
