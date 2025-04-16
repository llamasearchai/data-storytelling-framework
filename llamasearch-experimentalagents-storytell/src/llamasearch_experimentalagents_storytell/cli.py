#!/usr/bin/env python3
"""
Command-line interface for the Llamasearch Data Storytelling Framework.

This module provides command-line functionality for generating data stories,
visualizations, and dashboards from data sources.
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.dashboard import DashboardGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Llamasearch Data Storytelling Framework CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to input data file (CSV, JSON, Excel)")
    
    parser.add_argument("--output-dir", "-o", type=str, default="./output",
                        help="Directory to store the generated outputs")
    
    parser.add_argument("--api-key", "-k", type=str,
                        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)")
    
    parser.add_argument("--model", "-m", type=str, default="gpt-4-1106-preview",
                        help="OpenAI model to use for storytelling")
    
    parser.add_argument("--title", "-t", type=str,
                        help="Optional title for the data story")
    
    parser.add_argument("--context", "-c", type=str,
                        help="Optional context or prompt to guide the data storytelling")
    
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip the visualization generation")
    
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip dashboard generation")
    
    parser.add_argument("--narrative-only", action="store_true",
                        help="Generate only the narrative summary as JSON")
    
    return parser.parse_args()


def create_output_directory(output_dir: str) -> None:
    """Create the output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Check if the input data file exists
    if not os.path.exists(args.data):
        print(f"Error: Input data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    
    # Create the output directory
    create_output_directory(args.output_dir)
    
    # Determine API key (command line takes precedence)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.",
              file=sys.stderr)
        sys.exit(1)
    
    # Initialize the storytelling engine
    engine = StorytellingEngine(openai_api_key=api_key, model=args.model)
    
    # Load the data
    print(f"Loading data from {args.data}...")
    try:
        data = engine.load_data(args.data)
    except Exception as e:
        print(f"Error loading data: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Generate the narrative
    print("Generating narrative...")
    context = args.context or "Analyze this dataset and create a data story with key insights."
    narrative = engine.generate_narrative(
        data=data,
        context=context,
        title=args.title
    )
    
    # Save the narrative to JSON
    narrative_path = os.path.join(args.output_dir, "narrative.json")
    narrative.save(narrative_path, format="json")
    print(f"Narrative saved to {narrative_path}")
    
    # Exit early if narrative-only flag is set
    if args.narrative_only:
        print("Narrative-only mode enabled. Exiting.")
        return
    
    # Generate visualizations if not skipped
    visualizations = []
    if not args.skip_visualizations:
        print("Generating visualizations...")
        try:
            visualizations = engine.generate_visualizations(
                data=data,
                narrative=narrative,
                output_dir=args.output_dir
            )
            print(f"Generated {len(visualizations)} visualizations")
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}", file=sys.stderr)
            # Continue with the process even if visualization fails
    
    # Generate dashboard if not skipped
    if not args.skip_dashboard:
        print("Generating dashboard...")
        try:
            dashboard_generator = DashboardGenerator()
            dashboard_path = os.path.join(args.output_dir, "dashboard.html")
            dashboard_generator.generate_dashboard(
                narrative=narrative,
                visualizations=visualizations,
                output_file=dashboard_path
            )
            print(f"Dashboard saved to {dashboard_path}")
        except Exception as e:
            print(f"Error generating dashboard: {str(e)}", file=sys.stderr)
    
    print("Data storytelling complete!")


if __name__ == "__main__":
    main() 