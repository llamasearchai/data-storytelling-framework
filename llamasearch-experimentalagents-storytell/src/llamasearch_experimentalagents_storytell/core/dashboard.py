"""
Dashboard generator for the Data Storytelling Framework.

This module provides functionality to generate interactive dashboards
from narrative summaries and visualizations.
"""

import os
import jinja2
from typing import List, Dict, Any, Optional
from pathlib import Path

from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary


class DashboardGenerator:
    """
    Generate interactive dashboards that combine narrative summaries and visualizations.
    
    This class provides methods to render HTML dashboards from narrative summaries
    and visualization data, using Jinja2 templates.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the dashboard generator.
        
        Args:
            template_dir: Optional directory containing the templates.
                          If None, the default template directory will be used.
        """
        if template_dir is None:
            # Use the default template directory within the package
            module_dir = os.path.dirname(os.path.dirname(__file__))
            template_dir = os.path.join(module_dir, 'templates')
        
        self.template_dir = template_dir
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_dashboard(self, 
                           narrative: NarrativeSummary, 
                           visualizations: List[Dict[str, Any]],
                           output_file: str) -> str:
        """
        Generate an HTML dashboard from a narrative summary and visualizations.
        
        Args:
            narrative: The narrative summary containing insights, metrics, etc.
            visualizations: A list of visualization dictionaries, each with at least
                           'title' and 'file_path' keys.
            output_file: Path where the HTML dashboard should be saved.
        
        Returns:
            The path to the generated HTML file.
        """
        template = self.env.get_template('dashboard.html')
        
        # Render the template with the narrative and visualizations
        html_content = template.render(
            narrative=narrative,
            visualizations=visualizations
        )
        
        # Ensure the output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the rendered HTML to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def generate_dashboard_from_paths(self, 
                                      narrative_path: str, 
                                      viz_paths: List[Dict[str, Any]],
                                      output_file: str) -> str:
        """
        Generate a dashboard from paths to narrative and visualization files.
        
        Args:
            narrative_path: Path to the JSON file containing the narrative summary.
            viz_paths: List of dictionaries with visualization metadata, each with at least
                      'title' and 'file_path' keys.
            output_file: Path where the HTML dashboard should be saved.
        
        Returns:
            The path to the generated HTML file.
        """
        # Load the narrative from the file
        narrative = NarrativeSummary.load(narrative_path)
        
        return self.generate_dashboard(narrative, viz_paths, output_file) 