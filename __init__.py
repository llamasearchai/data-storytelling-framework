"""
Data Storytelling Agent with MLX

This package provides a data storytelling agent that converts analytical findings
into compelling narratives with visualizations for business stakeholders.
"""

__version__ = "1.0.0"

# Import main components
from typing import Dict, Any, Optional, List, Union
import os
import json
from pathlib import Path

# Try to import core components
try:
    from config import (
        load_config,
        get_config,
        update_config,
        save_config,
        get_value
    )
    _has_config = True
except ImportError:
    _has_config = False

try:
    from visualization import (
        VisualizationEngine,
        save_visualization
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

try:
    from mlx_analyzer import (
        calculate_percentage_change,
        find_max_impact_segment,
        calculate_difference_from_mean,
        calculate_contribution_percentage,
        calculate_rolling_average,
        calculate_trend_significance,
        filter_data_by_confidence
    )
    _has_mlx_analyzer = True
except ImportError:
    _has_mlx_analyzer = False

try:
    from main_py import (
        DataStorytellerAgent,
        NarrativeSummary
    )
    _has_agent = True
except ImportError:
    _has_agent = False

# Create a unified API class
class StorytellingEngine:
    """Main interface for the Data Storytelling Agent with MLX system."""
    
    def __init__(self, config_path: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the Storytelling Engine.
        
        Args:
            config_path: Path to a JSON configuration file (optional)
            openai_api_key: OpenAI API key (optional, uses environment variable if not provided)
        """
        # Set API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        # Load configuration
        if _has_config:
            self.config = load_config(config_path)
        else:
            self.config = {}
            
        # Initialize components
        self.visualization_available = _has_visualization
        self.mlx_available = _has_mlx_analyzer
        self.agent_available = _has_agent
        
        # Set up agent if available
        if self.agent_available:
            from openai import OpenAI
            self.client = OpenAI()
            self.agent = DataStorytellerAgent(self.client)
            
        # Set up visualization engine if available
        if self.visualization_available:
            self.visualization_engine = VisualizationEngine()
            
        # Data storage
        self.analysis_results = None
        self.narrative = None
        
    def analyze(self, analysis_file: Union[str, Dict[str, Any]]) -> 'StorytellingEngine':
        """
        Analyze data and generate a narrative summary.
        
        Args:
            analysis_file: Path to JSON file or dictionary with analysis results
            
        Returns:
            Self for method chaining
        """
        if not self.agent_available:
            raise ImportError("Agent module not available")
            
        # If analysis_file is a dictionary, save it to a temporary file
        if isinstance(analysis_file, dict):
            temp_file = os.path.join(
                get_value("data_directory", "data"),
                f"temp_analysis_{hash(str(analysis_file))}.json"
            )
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            with open(temp_file, 'w') as f:
                json.dump(analysis_file, f, indent=2)
                
            analysis_file = temp_file
            
        # Run the agent
        self.narrative = self.agent.run(analysis_file, output_visualization=self.visualization_available)
        
        # Store results for later use
        if hasattr(self.agent, 'analysis_results'):
            self.analysis_results = self.agent.analysis_results
            
        return self
        
    def get_narrative(self) -> Optional[NarrativeSummary]:
        """
        Get the generated narrative summary.
        
        Returns:
            NarrativeSummary object if available
        """
        return self.narrative
        
    def print_narrative(self) -> None:
        """Print the narrative summary to the console."""
        if not self.narrative:
            print("No narrative available. Run analyze() first.")
            return
            
        print("\n" + "="*80)
        print(f"# {self.narrative.headline}\n")
        print("## Key Findings")
        for finding in self.narrative.findings:
            print(f"- {finding}")
        print("\n## Recommendations")
        for recommendation in self.narrative.recommendations:
            print(f"- {recommendation}")
        print("="*80 + "\n")
        
    def visualize(self, output_format: str = None) -> Optional[str]:
        """
        Generate visualizations for the analysis results and narrative.
        
        Args:
            output_format: Output format for visualizations (html, png, jpg, pdf)
            
        Returns:
            Path to the generated dashboard file
        """
        if not self.visualization_available:
            raise ImportError("Visualization module not available")
            
        if not self.analysis_results or not self.narrative:
            raise ValueError("No analysis results or narrative available. Run analyze() first.")
            
        # Update format in config if provided
        if output_format:
            update_config({"visualization.default_format": output_format})
            
        # Generate dashboard
        dashboard = self.visualization_engine.generate_dashboard(
            self.analysis_results,
            self.narrative
        )
        
        visualization_dir = get_value("visualization_directory", "visualizations")
        print(f"Visualizations generated in {visualization_dir} directory")
        
        # Return path to the dashboard file
        return os.path.join(visualization_dir, f"dashboard.{output_format or get_value('visualization.default_format', 'html')}")
        
    def calculate_metric(
        self,
        metric_type: str,
        values: List[float],
        labels: Optional[List[str]] = None,
        baseline: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate a secondary metric to enhance the narrative.
        
        Args:
            metric_type: Type of metric to calculate 
                         ('percentage_change', 'max_impact', 'diff_from_mean', 
                          'contribution_percentage', 'trend_significance')
            values: List of numerical values to analyze
            labels: Optional list of labels corresponding to values
            baseline: Optional baseline value for comparison
            
        Returns:
            Dictionary containing the calculated metric and associated information
        """
        if not self.mlx_available:
            raise ImportError("MLX analyzer module not available")
            
        import mlx.core as mx
        values_mx = mx.array(values)
        
        if metric_type == "percentage_change" and baseline is not None:
            result = calculate_percentage_change(values_mx, baseline)
            return {
                "metric_type": "percentage_change",
                "result": float(result),
                "description": f"Percentage change: {float(result):.2f}%"
            }
            
        elif metric_type == "max_impact" and labels is not None:
            max_index, max_value = find_max_impact_segment(values_mx)
            return {
                "metric_type": "max_impact",
                "segment": labels[int(max_index)],
                "value": float(max_value),
                "description": f"Highest impact segment: {labels[int(max_index)]} with value {float(max_value):.2f}"
            }
            
        elif metric_type == "diff_from_mean":
            differences = calculate_difference_from_mean(values_mx)
            
            if labels is not None:
                diff_dict = {labels[i]: float(diff) for i, diff in enumerate(differences)}
                return {
                    "metric_type": "diff_from_mean",
                    "differences": diff_dict,
                    "description": "Differences from mean calculated for each segment"
                }
            else:
                return {
                    "metric_type": "diff_from_mean",
                    "differences": [float(d) for d in differences],
                    "description": "Differences from mean calculated"
                }
                
        elif metric_type == "contribution_percentage":
            contributions = calculate_contribution_percentage(values_mx)
            
            if labels is not None:
                contrib_dict = {labels[i]: float(c) for i, c in enumerate(contributions)}
                return {
                    "metric_type": "contribution_percentage",
                    "contributions": contrib_dict,
                    "description": "Percentage contributions calculated for each segment"
                }
            else:
                return {
                    "metric_type": "contribution_percentage",
                    "contributions": [float(c) for c in contributions],
                    "description": "Percentage contributions calculated"
                }
                
        elif metric_type == "trend_significance":
            significance = calculate_trend_significance(values_mx)
            return {
                "metric_type": "trend_significance",
                "significance": significance,
                "description": f"Trend significance: {significance:.2f}"
            }
            
        else:
            raise ValueError(f"Unsupported metric type: {metric_type} or missing required parameters")
    
    def save(self, output_file: str) -> 'StorytellingEngine':
        """
        Save analysis results and narrative to a JSON file.
        
        Args:
            output_file: Path to save the file
            
        Returns:
            Self for method chaining
        """
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Format narrative for saving
        narrative_dict = None
        if self.narrative:
            narrative_dict = {
                "headline": self.narrative.headline,
                "findings": self.narrative.findings,
                "recommendations": self.narrative.recommendations
            }
        
        data = {
            "version": __version__,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "analysis_results": self.analysis_results,
            "narrative": narrative_dict
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return self
    
    @classmethod
    def load(cls, input_file: str) -> 'StorytellingEngine':
        """
        Load previously saved analysis results and narrative.
        
        Args:
            input_file: Path to the saved file
            
        Returns:
            StorytellingEngine instance with loaded data
        """
        import json
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        instance = cls()
        instance.analysis_results = data.get("analysis_results")
        
        # Load narrative if available
        narrative_dict = data.get("narrative")
        if narrative_dict and _has_agent:
            instance.narrative = NarrativeSummary(
                headline=narrative_dict.get("headline", ""),
                findings=narrative_dict.get("findings", []),
                recommendations=narrative_dict.get("recommendations", [])
            )
        
        return instance

# Define what's exposed in the public API
__all__ = [
    # Main class
    'StorytellingEngine',
    
    # Core components (if available)
    'DataStorytellerAgent' if _has_agent else None,
    'NarrativeSummary' if _has_agent else None,
    'VisualizationEngine' if _has_visualization else None,
    
    # Configuration functions (if available)
    'load_config' if _has_config else None,
    'get_config' if _has_config else None,
    'update_config' if _has_config else None,
    'save_config' if _has_config else None,
    'get_value' if _has_config else None,
    
    # MLX functions (if available)
    'calculate_percentage_change' if _has_mlx_analyzer else None,
    'find_max_impact_segment' if _has_mlx_analyzer else None,
    'calculate_difference_from_mean' if _has_mlx_analyzer else None,
    'calculate_contribution_percentage' if _has_mlx_analyzer else None,
    'calculate_rolling_average' if _has_mlx_analyzer else None,
    'calculate_trend_significance' if _has_mlx_analyzer else None,
    'filter_data_by_confidence' if _has_mlx_analyzer else None,
]

# Remove None values from __all__
__all__ = [item for item in __all__ if item is not None] 