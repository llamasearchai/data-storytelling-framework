"""
Core engine module for the Data Storytelling Framework.

This module provides the StorytellingEngine class which integrates all components
of the framework and serves as the main interface for users.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd

from .config import get_config, get_value, update_config
from .narrative import NarrativeSummary
from .visualization import VisualizationEngine

# Try to import the best available agent
try:
    from ..agents.storyteller import DataStorytellerAgent
    _has_agent = True
except ImportError:
    _has_agent = False
    DataStorytellerAgent = None

# Fallback: simple mock agent
class MockAgent:
    def analyze(self, data: Union[str, Dict[str, Any]]) -> NarrativeSummary:
        return NarrativeSummary(
            title="Mock Narrative",
            summary="This is a mock narrative summary.",
            insights=[],
            key_metrics={},
            recommendations=["This is a mock recommendation."],
            narrative_text="Mock narrative text.",
            creation_timestamp=datetime.now().isoformat(),
            metadata={"mock": True}
        )

class StorytellingEngine:
    """
    Main engine for the Data Storytelling Framework.
    Integrates config, agent, visualization, and pipeline orchestration.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        visualization_theme: Optional[str] = None,
        llm_model: Optional[str] = None,
        memory: Any = None
    ):
        """
        Initialize the storytelling engine.
        Args:
            config_path: Path to a custom configuration file (optional)
            openai_api_key: OpenAI API key (optional, uses env/config if not provided)
            visualization_theme: Theme for visualizations (optional)
            llm_model: LLM model to use for storytelling (optional)
            memory: Optional memory/logging object
        """
        # Load configuration
        self.config = get_config() if not config_path else get_config(config_path)
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        # Set up visualization engine
        self.visualization_engine = VisualizationEngine(
            theme=visualization_theme or get_value("visualization.default_theme", "light"),
            memory=memory
        )
        # Set up agent
        if _has_agent:
            self.agent = DataStorytellerAgent(
                model=llm_model or get_value("narrative.model", "gpt-4o"),
                memory=memory
            )
        else:
            self.agent = MockAgent()
        self.memory = memory

    def load_data(self, data: Union[str, Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Load data from a file path, dict, or DataFrame.
        Args:
            data: Path to file (CSV, JSON, Excel), dict, or DataFrame
        Returns:
            Data as a dictionary
        """
        if isinstance(data, dict):
            return data
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
        if isinstance(data, str):
            ext = os.path.splitext(data)[-1].lower()
            if ext in [".json"]:
                with open(data, "r") as f:
                    return json.load(f)
            elif ext in [".csv"]:
                return pd.read_csv(data).to_dict(orient="records")
            elif ext in [".xlsx", ".xls"]:
                return pd.read_excel(data).to_dict(orient="records")
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        raise ValueError("Unsupported data type for loading.")

    def generate_narrative(self, data: Union[str, Dict[str, Any], pd.DataFrame], context: Optional[str] = None, title: Optional[str] = None) -> NarrativeSummary:
        """
        Generate a narrative summary from data.
        Args:
            data: Data to analyze (file path, dict, or DataFrame)
            context: Optional context or prompt to guide the narrative
            title: Optional title for the narrative
        Returns:
            NarrativeSummary object
        """
        loaded_data = self.load_data(data)
        # The agent may accept context/title as kwargs if supported
        if hasattr(self.agent, 'analyze'):
            try:
                return self.agent.analyze(loaded_data, context=context, title=title)
            except TypeError:
                # Fallback for agents that don't accept context/title
                return self.agent.analyze(loaded_data)
        raise RuntimeError("No valid agent available for narrative generation.")

    def generate_visualizations(self, data: Union[str, Dict[str, Any], pd.DataFrame], narrative: NarrativeSummary, output_dir: Optional[str] = None, output_format: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate visualizations for the data and narrative.
        Args:
            data: Data to visualize (file path, dict, or DataFrame)
            narrative: NarrativeSummary object
            output_dir: Directory to save visualizations (optional)
            output_format: Format for visualizations (optional)
        Returns:
            List of visualization metadata dicts
        """
        loaded_data = self.load_data(data)
        if output_dir:
            self.visualization_engine.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        if output_format:
            update_config({"visualization.default_format": output_format})
        # The visualization engine should return a list of dicts with at least 'title' and 'file_path'
        return self.visualization_engine.visualize_all(loaded_data, narrative)

    def run_pipeline(
        self,
        data: Union[str, Dict[str, Any], pd.DataFrame],
        output_dir: Optional[str] = None,
        output_format: Optional[str] = None,
        save_narrative: bool = True,
        narrative_format: str = "json",
        context: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete data storytelling pipeline: load data, generate narrative, create visualizations, save results.
        Args:
            data: Data to analyze (file path, dict, or DataFrame)
            output_dir: Directory to save results (optional)
            output_format: Format for visualizations (optional)
            save_narrative: Whether to save the narrative (default: True)
            narrative_format: Format to save the narrative (default: json)
            context: Optional context or prompt for the narrative
            title: Optional title for the narrative
        Returns:
            Dictionary with narrative, visualizations, and file paths
        """
        if output_dir is None:
            output_dir = get_value("output_directory", "output")
        os.makedirs(output_dir, exist_ok=True)
        # Step 1: Load data
        loaded_data = self.load_data(data)
        # Step 2: Generate narrative
        narrative = self.generate_narrative(loaded_data, context=context, title=title)
        # Step 3: Generate visualizations
        visualizations = self.generate_visualizations(loaded_data, narrative, output_dir=output_dir, output_format=output_format)
        # Step 4: Save narrative
        narrative_path = None
        if save_narrative:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            narrative_path = os.path.join(output_dir, f"narrative_{timestamp}.{narrative_format}")
            narrative.save(narrative_path, format=narrative_format)
        # Step 5: Optionally generate dashboard (if dashboard generator is available)
        dashboard_path = None
        try:
            from .dashboard import DashboardGenerator
            dashboard_generator = DashboardGenerator()
            dashboard_path = os.path.join(output_dir, "dashboard.html")
            dashboard_generator.generate_dashboard(narrative, visualizations, dashboard_path)
        except Exception as e:
            dashboard_path = None
        # Step 6: Logging
        if self.memory and hasattr(self.memory, 'log'):
            self.memory.log(
                level="info",
                component="engine",
                message="Completed data storytelling pipeline",
                metadata={
                    "narrative_path": narrative_path,
                    "dashboard_path": dashboard_path,
                    "visualizations": [v.get('file_path') for v in visualizations]
                }
            )
        return {
            "narrative": narrative,
            "visualizations": visualizations,
            "narrative_path": narrative_path,
            "dashboard_path": dashboard_path
        }

    def export_memory_dashboard(self, output_dir: Optional[str] = None) -> str:
        """
        Export a Datasette configuration for exploring the memory database.
        
        Args:
            output_dir: Directory to save the Datasette configuration to
            
        Returns:
            Path to the Datasette configuration file
        """
        return self.memory.export_datasette(output_dir) 