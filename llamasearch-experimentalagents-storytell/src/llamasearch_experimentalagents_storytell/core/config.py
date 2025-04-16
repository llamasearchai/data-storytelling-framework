"""
Configuration module for Data Storytelling Framework.

This module provides robust configuration management, supporting:
- Default values
- JSON config files
- Environment variable overrides
- .env file loading
- Type hints and docstrings
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # General
    "project_name": "Data Storytelling Framework",
    "version": "1.0.0",
    # Paths
    "data_directory": "data",
    "output_directory": "output",
    "visualization_directory": "visualizations",
    # MLX analyzer
    "mlx.min_data_points": 10,
    "mlx.confidence_threshold": 0.7,
    "mlx.rolling_window_size": 3,
    # Narrative
    "narrative.max_findings": 5,
    "narrative.max_recommendations": 3,
    "narrative.temperature": 0.7,
    "narrative.model": "gpt-4o",
    # Visualization
    "visualization.default_theme": "light",
    "visualization.default_format": "html",
    "visualization.chart_height": 600,
    "visualization.chart_width": 800,
    "visualization.color_palette": "viridis",
    # Analysis
    "analysis.min_segment_size": 100,
    "analysis.significance_level": 0.05,
    "analysis.highlight_threshold": 0.1
}

_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, .env, and environment variables.
    Args:
        config_path: Path to configuration JSON file (optional)
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    config = DEFAULT_CONFIG.copy()
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
    # Override with environment variables
    for key in config:
        env_key = key.upper().replace('.', '_')
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    # Create necessary directories
    for key in config:
        if key.endswith('_directory'):
            os.makedirs(config[key], exist_ok=True)
    _CONFIG = config
    return config

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    if _CONFIG is None:
        return load_config()
    return _CONFIG

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    Args:
        updates: Dictionary of configuration updates
    Returns:
        Updated configuration dictionary
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    _CONFIG.update(updates)
    return _CONFIG

def save_config(config_path: str) -> None:
    """
    Save current configuration to file.
    Args:
        config_path: Path to save configuration JSON file
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    try:
        with open(config_path, 'w') as f:
            json.dump(_CONFIG, f, indent=2)
    except IOError as e:
        print(f"Error: Could not save config file: {e}")

def get_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value, checking environment variables first.
    Args:
        key: Configuration key
        default: Default value if key not found
    Returns:
        Configuration value
    """
    config = get_config()
    env_key = key.upper().replace('.', '_')
    if env_key in os.environ:
        return os.environ[env_key]
    return config.get(key, default)

def get_env_value(key: str, default: Any = None) -> Any:
    """
    Get a value from environment variables or .env, with fallback to default.
    Args:
        key: Environment variable key
        default: Default value if not found
    Returns:
        Value from environment or default
    """
    return os.environ.get(key, default) 