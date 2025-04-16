"""
LlamaSearch ExperimentalAgents: Data Storytelling framework with MLX integration.

This package provides a powerful AI-driven data storytelling framework
that combines Apple's MLX for computation, OpenAI's large language models
for narrative generation, and interactive visualizations for compelling
data stories.
"""

__version__ = "0.1.0"

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# Core components
from .core.engine import StorytellingEngine
from .core.config import get_config, get_value, load_config, update_config
from .core.narrative import NarrativeSummary
from .core.types import AnalysisResults

# Agent components  
from .agents.storyteller import DataStorytellerAgent

# Set up default config directory in user's home
default_config_dir = os.path.join(os.path.expanduser("~"), ".llamasearch", "experimentalagents", "storytell")
os.makedirs(default_config_dir, exist_ok=True)

# Public API
__all__ = [
    "StorytellingEngine",
    "DataStorytellerAgent",
    "NarrativeSummary",
    "AnalysisResults",
    "get_config",
    "get_value",
    "load_config",
    "update_config",
] 