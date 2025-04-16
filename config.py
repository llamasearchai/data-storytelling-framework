"""
Configuration module for Data Storytelling Agent with MLX.

This module provides configuration management for the entire system,
allowing for customization of behavior across all modules.
"""

import os
import json
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    "project_name": "Data Storytelling Agent with MLX",
    "version": "1.0.0",
    
    # Paths
    "data_directory": "data",
    "output_directory": "output",
    "visualization_directory": "visualizations",
    
    # MLX analyzer settings
    "mlx.min_data_points": 10,
    "mlx.confidence_threshold": 0.7,
    "mlx.rolling_window_size": 3,
    
    # Narrative generation settings
    "narrative.max_findings": 5,
    "narrative.max_recommendations": 3,
    "narrative.temperature": 0.7,
    "narrative.model": "gpt-4o",
    
    # Visualization settings
    "visualization.default_theme": "light",
    "visualization.default_format": "html",
    "visualization.chart_height": 600,
    "visualization.chart_width": 800,
    "visualization.color_palette": "viridis",
    
    # Analysis settings
    "analysis.min_segment_size": 100,
    "analysis.significance_level": 0.05,
    "analysis.highlight_threshold": 0.1
}

# Global config object
_CONFIG = None

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration JSON file (optional)
        
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Create necessary directories
    for key in config:
        if key.endswith('_directory'):
            os.makedirs(config[key], exist_ok=True)
    
    # Store for global access
    _CONFIG = config
    
    return config

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    
    # Load default if not already loaded
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
    
    # Load default if not already loaded
    if _CONFIG is None:
        _CONFIG = load_config()
    
    # Apply updates
    _CONFIG.update(updates)
    
    return _CONFIG

def save_config(config_path: str) -> None:
    """
    Save current configuration to file.
    
    Args:
        config_path: Path to save configuration JSON file
    """
    global _CONFIG
    
    # Load default if not already loaded
    if _CONFIG is None:
        _CONFIG = load_config()
    
    # Save to file
    try:
        with open(config_path, 'w') as f:
            json.dump(_CONFIG, f, indent=2)
    except IOError as e:
        print(f"Error: Could not save config file: {e}")

def get_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    config = get_config()
    return config.get(key, default) 