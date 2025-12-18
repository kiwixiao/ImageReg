"""
DAREG Configuration Loader

Handles loading, validation, and merging of configuration files.
Supports YAML configuration with default fallbacks.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy

from ..utils.logging_config import get_logger

logger = get_logger("config")

# Path to default config
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


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
class SVFFDSpecificConfig:
    """SVFFD-specific configuration"""
    integration_steps: int = 5
    world_coord_regularization: bool = True


@dataclass
class RegularizationConfig:
    """Regularization configuration"""
    bending_weight: float = 0.001
    diffusion_weight: float = 0.0005


@dataclass
class FFDConfig:
    """FFD/SVFFD registration configuration"""
    model: str = "ffd"  # ffd or svffd
    control_point_spacing: int = 4
    pyramid_levels: int = 4
    iterations_per_level: List[int] = field(default_factory=lambda: [100, 100, 100, 100])
    learning_rates_per_level: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01])
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    svffd: SVFFDSpecificConfig = field(default_factory=SVFFDSpecificConfig)
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


@dataclass
class OutputConfig:
    """Output settings"""
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
class RegistrationConfig:
    """Complete registration configuration"""
    io: IOConfig = field(default_factory=IOConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    rigid: RigidConfig = field(default_factory=RigidConfig)
    affine: AffineConfig = field(default_factory=AffineConfig)
    ffd: FFDConfig = field(default_factory=FFDConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


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
            field_values[field_name] = value

    return cls(**field_values)


def load_default_config() -> Dict:
    """Load the default configuration from YAML file"""
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Default config not found at {DEFAULT_CONFIG_PATH}")
        return {}


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
    overrides: Optional[Dict] = None
) -> RegistrationConfig:
    """
    Load registration configuration

    Args:
        config_path: Optional path to user configuration YAML
        overrides: Optional dictionary of override values

    Returns:
        RegistrationConfig instance
    """
    # Start with defaults
    config_dict = load_default_config()

    # Merge user config if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
            config_dict = _deep_merge(config_dict, user_config)
        else:
            logger.warning(f"Config file not found: {config_path}")

    # Apply overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Convert to dataclass
    config = _dict_to_dataclass(config_dict, RegistrationConfig)

    # Validate required fields
    _validate_config(config)

    return config


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

    # Check FFD model
    valid_models = ["ffd", "svffd"]
    if config.ffd.model.lower() not in valid_models:
        raise ValueError(f"Invalid FFD model: {config.ffd.model}. Must be one of {valid_models}")

    logger.debug("Configuration validated successfully")
