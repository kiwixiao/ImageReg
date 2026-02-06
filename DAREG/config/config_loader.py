"""
DAREG Configuration Loader

Handles loading, validation, and merging of configuration files.
Supports YAML configuration with preset system and config hierarchy.

Config Hierarchy (highest to lowest priority):
1. CLI overrides (--json-config, individual args)
2. Config preset (--config-preset svffd_standard)
3. Project config (inputs_folder/registration_config.yaml)
4. Package defaults (DAREG/configs/default.yaml)
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy

from ..utils.logging_config import get_logger

logger = get_logger("config")

# Paths to config directories
DAREG_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = DAREG_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"


@dataclass
class ConvergenceConfig:
    """Convergence criteria configuration"""
    min_delta: float = 1e-6
    patience: int = 20


@dataclass
class SimilarityConfig:
    """Similarity measure configuration"""
    metric: str = "nmi"
    num_bins: int = 64
    foreground_threshold: float = 0.01
    roi_mask: Optional[str] = None
    # Sampling options for NMI speedup (use one or neither for full computation)
    num_samples: Optional[int] = None      # Fixed number of voxels to sample (e.g., 10000)
    sample_ratio: Optional[float] = None   # Ratio of voxels to sample (e.g., 0.1 = 10%)


@dataclass
class RigidConfig:
    """Rigid registration configuration"""
    pyramid_levels: int = 4
    iterations_per_level: List[int] = field(default_factory=lambda: [100, 100, 100, 100])
    learning_rates_per_level: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01])
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


@dataclass
class AffineConfig:
    """Affine registration configuration"""
    pyramid_levels: int = 4
    iterations_per_level: List[int] = field(default_factory=lambda: [100, 100, 100, 100])
    learning_rates_per_level: List[float] = field(default_factory=lambda: [0.005, 0.005, 0.005, 0.005])
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


@dataclass
class FFDRegularizationConfig:
    """FFD regularization configuration"""
    bending_weight: float = 0.001
    diffusion_weight: float = 0.0005


@dataclass
class FFDOptimizerConfig:
    """FFD optimizer configuration (MIRTK Conjugate Gradient)"""
    max_rejected_streak: int = 1
    step_rise: float = 1.1
    step_drop: float = 0.5
    epsilon: float = 1e-4
    delta: float = 1e-12


@dataclass
class FFDConfig:
    """FFD registration configuration (MIRTK-equivalent)"""
    control_point_spacing: int = 4
    pyramid_levels: int = 4
    iterations_per_level: List[int] = field(default_factory=lambda: [100, 100, 100, 100])
    learning_rates_per_level: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01])
    regularization: FFDRegularizationConfig = field(default_factory=FFDRegularizationConfig)
    optimizer: FFDOptimizerConfig = field(default_factory=FFDOptimizerConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


@dataclass
class SVFFDRegularizationConfig:
    """SVFFD regularization configuration (includes velocity-specific params)"""
    bending_weight: float = 0.0005
    diffusion_weight: float = 0.00025
    velocity_smoothing_sigma: float = 0.0
    laplacian_weight: float = 0.0
    jacobian_penalty: float = 0.0


@dataclass
class SVFFDConfig:
    """SVFFD registration configuration (diffeomorphic with velocity field)"""
    control_point_spacing: int = 4
    pyramid_levels: int = 4
    iterations_per_level: List[int] = field(default_factory=lambda: [100, 100, 100, 100])
    learning_rates_per_level: List[float] = field(default_factory=lambda: [0.005, 0.005, 0.005, 0.005])
    integration_steps: int = 5
    regularization: SVFFDRegularizationConfig = field(default_factory=SVFFDRegularizationConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


@dataclass
class IOConfig:
    """Input/Output configuration"""
    source_image: Optional[str] = None
    target_image: Optional[str] = None
    segmentation: Optional[str] = None
    output_dir: str = "./dareg_outputs"


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    stages: List[str] = field(default_factory=lambda: ["rigid", "affine", "ffd"])
    device: str = "auto"
    model: str = "rigid+affine+ffd"
    skip_alignment: bool = False
    skip_refinement: bool = False
    resume: bool = False
    verbose: bool = False
    extract_frames: bool = False
    no_viz: bool = False


@dataclass
class OutputSettingsConfig:
    """Output settings for auto-naming and saving"""
    auto_name: bool = False
    name_template: str = "{model}_cp{spacing}_bend{bending}"
    save_intermediate: bool = True
    save_transforms: bool = True
    create_visualizations: bool = True
    create_pdf_report: bool = True


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    dpi: int = 150
    figsize: List[int] = field(default_factory=lambda: [18, 12])
    colormap: str = "gray"


@dataclass
class FrameSelectionConfig:
    """Frame selection for 4D images"""
    start_frame: int = 0
    num_frames: Optional[int] = None  # None = all frames


@dataclass
class RegistrationConfig:
    """Complete registration configuration"""
    io: IOConfig = field(default_factory=IOConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    rigid: RigidConfig = field(default_factory=RigidConfig)
    affine: AffineConfig = field(default_factory=AffineConfig)
    ffd: FFDConfig = field(default_factory=FFDConfig)
    svffd: SVFFDConfig = field(default_factory=SVFFDConfig)
    output: OutputSettingsConfig = field(default_factory=OutputSettingsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    frame_selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _coerce_type(value: Any, target_type: type) -> Any:
    """Coerce value to target type (handles YAML string parsing issues)"""
    if value is None:
        return value

    # Get the origin type for generics like Optional, List, etc.
    origin = getattr(target_type, '__origin__', None)
    if origin is not None:
        # Handle Optional[X] (which is Union[X, None])
        args = getattr(target_type, '__args__', ())
        if type(None) in args:
            # It's Optional, try non-None types
            for arg in args:
                if arg is not type(None):
                    return _coerce_type(value, arg)
        return value

    # Handle basic types
    if target_type == float and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif target_type == int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    elif target_type == bool and isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')

    return value


def _dict_to_dataclass(data: Dict, cls: type) -> Any:
    """
    Convert dictionary to dataclass instance

    Args:
        data: Dictionary with configuration values
        cls: Dataclass type

    Returns:
        Dataclass instance
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_values = {}
    for field_name, field_info in cls.__dataclass_fields__.items():
        if field_name in data:
            value = data[field_name]
            # Recursively convert nested dataclasses
            if hasattr(field_info.type, "__dataclass_fields__") and isinstance(value, dict):
                value = _dict_to_dataclass(value, field_info.type)
            else:
                # Coerce primitive types (handles YAML string parsing issues)
                value = _coerce_type(value, field_info.type)
            field_values[field_name] = value

    return cls(**field_values)


def load_yaml_config(path: Path) -> Dict:
    """Load a YAML configuration file"""
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_default_config() -> Dict:
    """Load the default configuration from YAML file"""
    if DEFAULT_CONFIG_PATH.exists():
        return load_yaml_config(DEFAULT_CONFIG_PATH)
    else:
        logger.warning(f"Default config not found at {DEFAULT_CONFIG_PATH}")
        return {}


def load_preset(preset_name: str) -> Dict:
    """
    Load a configuration preset from DAREG/configs/

    Args:
        preset_name: Name of preset (e.g., 'svffd_standard', 'ffd_mirtk')
                     Can include or omit .yaml extension

    Returns:
        Configuration dictionary from preset file

    Raises:
        FileNotFoundError: If preset file doesn't exist
    """
    # Handle preset name with or without extension
    if not preset_name.endswith('.yaml'):
        preset_name = f"{preset_name}.yaml"

    preset_path = CONFIGS_DIR / preset_name

    if not preset_path.exists():
        available = [f.stem for f in CONFIGS_DIR.glob("*.yaml") if f.stem != "README"]
        raise FileNotFoundError(
            f"Config preset '{preset_name}' not found in {CONFIGS_DIR}. "
            f"Available presets: {available}"
        )

    logger.info(f"Loading config preset: {preset_name}")
    return load_yaml_config(preset_path)


def find_project_config(inputs_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Find project-specific config in inputs directory

    Args:
        inputs_dir: Path to inputs directory (e.g., inputs_testing)

    Returns:
        Path to registration_config.yaml if found, None otherwise
    """
    if inputs_dir is None:
        return None

    inputs_path = Path(inputs_dir)
    config_path = inputs_path / "registration_config.yaml"

    if config_path.exists():
        return config_path

    # Also check for .yml extension
    config_path_yml = inputs_path / "registration_config.yml"
    if config_path_yml.exists():
        return config_path_yml

    return None


def default_config() -> RegistrationConfig:
    """
    Get the default registration configuration

    Returns:
        RegistrationConfig with default values
    """
    config_dict = load_default_config()
    return _dict_to_dataclass(config_dict, RegistrationConfig)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    preset: Optional[str] = None,
    inputs_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict] = None
) -> RegistrationConfig:
    """
    Load registration configuration with hierarchy support

    Config Priority (highest to lowest):
    1. overrides dict (from CLI args)
    2. config_path (--config or --json-config)
    3. preset (--config-preset)
    4. project config (inputs_dir/registration_config.yaml)
    5. package defaults (DAREG/configs/default.yaml)

    Args:
        config_path: Optional path to user configuration YAML
        preset: Optional preset name (e.g., 'svffd_standard')
        inputs_dir: Optional inputs directory to search for project config
        overrides: Optional dictionary of override values

    Returns:
        RegistrationConfig instance
    """
    # Start with package defaults
    config_dict = load_default_config()
    logger.debug(f"Loaded package defaults from {DEFAULT_CONFIG_PATH}")

    # Layer 1: Project config (inputs_dir/registration_config.yaml)
    project_config_path = find_project_config(inputs_dir)
    if project_config_path:
        logger.info(f"Loading project config from: {project_config_path}")
        project_config = load_yaml_config(project_config_path)
        config_dict = _deep_merge(config_dict, project_config)

    # Layer 2: Preset config
    if preset:
        preset_config = load_preset(preset)
        config_dict = _deep_merge(config_dict, preset_config)

    # Layer 3: User config file
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            logger.info(f"Loading user config from: {config_path}")
            user_config = load_yaml_config(config_path)
            config_dict = _deep_merge(config_dict, user_config)
        else:
            logger.warning(f"Config file not found: {config_path}")

    # Layer 4: CLI overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Convert to dataclass
    config = _dict_to_dataclass(config_dict, RegistrationConfig)

    # Validate required fields
    _validate_config(config)

    return config


def generate_output_name(config: RegistrationConfig, model: str = "ffd") -> str:
    """
    Generate output folder name from config settings

    Uses template from config.output.name_template with parameter substitution.

    Args:
        config: Registration configuration
        model: Registration model ('ffd' or 'svffd')

    Returns:
        Generated output folder name
    """
    # Get the appropriate config section
    if model.lower() == "svffd":
        model_config = config.svffd
        spacing = model_config.control_point_spacing
        bending = model_config.regularization.bending_weight
        steps = model_config.integration_steps
        lr = model_config.learning_rates_per_level[0] if model_config.learning_rates_per_level else 0.005
    else:
        model_config = config.ffd
        spacing = model_config.control_point_spacing
        bending = model_config.regularization.bending_weight
        steps = 0  # FFD doesn't have integration steps
        lr = model_config.learning_rates_per_level[0] if model_config.learning_rates_per_level else 0.01

    # Get template
    template = config.output.name_template if hasattr(config.output, 'name_template') else "{model}_cp{spacing}_bend{bending}"

    # Substitute parameters
    name = template.format(
        model=model.lower(),
        spacing=spacing,
        bending=bending,
        steps=steps,
        lr=lr,
    )

    return name


def _validate_config(config: RegistrationConfig):
    """
    Validate configuration

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check similarity metric
    valid_metrics = ["nmi", "ncc", "ssd"]
    if config.similarity.metric.lower() not in valid_metrics:
        raise ValueError(f"Invalid similarity metric: {config.similarity.metric}. Must be one of {valid_metrics}")

    # Check pipeline stages
    valid_stages = ["rigid", "affine", "ffd", "svffd"]
    for stage in config.pipeline.stages:
        if stage.lower() not in valid_stages:
            raise ValueError(f"Invalid pipeline stage: {stage}. Must be one of {valid_stages}")

    # Validate FFD config
    if config.ffd.control_point_spacing < 1:
        raise ValueError(f"FFD control_point_spacing must be >= 1, got {config.ffd.control_point_spacing}")

    # Validate SVFFD config
    if config.svffd.integration_steps < 1:
        raise ValueError(f"SVFFD integration_steps must be >= 1, got {config.svffd.integration_steps}")

    if config.svffd.regularization.velocity_smoothing_sigma < 0:
        raise ValueError(f"velocity_smoothing_sigma must be >= 0, got {config.svffd.regularization.velocity_smoothing_sigma}")

    logger.debug("Configuration validated successfully")


def list_available_presets() -> List[str]:
    """
    List available configuration presets

    Returns:
        List of preset names (without .yaml extension)
    """
    if not CONFIGS_DIR.exists():
        return []

    presets = []
    for f in CONFIGS_DIR.glob("*.yaml"):
        if f.stem not in ["README", "default"]:
            presets.append(f.stem)

    return sorted(presets)
