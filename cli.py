#!/usr/bin/env python3
"""
Command-line interface for the Data Storytelling Agent.

This module provides a command-line interface for analyzing data,
generating narratives, and creating visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import from package
try:
    from config import load_config, get_config, update_config, get_value
    from main_py import DataStorytellerAgent, NarrativeSummary
    from visualization import VisualizationEngine
    from mlx_analyzer import calculate_percentage_change, find_max_impact_segment
    _imports_succeeded = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    _imports_succeeded = False
    
try:
    from .__init__ import StorytellingEngine
    _package_import = True
except (ImportError, ValueError):
    try:
        from __init__ import StorytellingEngine
        _package_import = True
    except ImportError:
        _package_import = False


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Data Storytelling Agent with MLX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration JSON file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze data and generate narrative"
    )
    analyze_parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file with analysis results"
    )
    analyze_parser.add_argument(
        "--output",
        type=str,
        help="Path to save narrative results (JSON)"
    )
    analyze_parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation"
    )
    analyze_parser.add_argument(
        "--model",
        type=str,
        help="OpenAI model to use for narrative generation"
    )
    
    # visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", 
        help="Generate visualizations from saved results"
    )
    visualize_parser.add_argument(
        "input_file",
        type=str,
        help="Path to saved results JSON file"
    )
    visualize_parser.add_argument(
        "--format",
        type=str,
        choices=["html", "png", "jpg", "pdf", "svg"],
        default="html",
        help="Output format for visualizations"
    )
    visualize_parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark", "corporate", "minimal"],
        help="Visual theme for charts"
    )
    
    # calculate command
    calculate_parser = subparsers.add_parser(
        "calculate", 
        help="Calculate a specific metric from data"
    )
    calculate_parser.add_argument(
        "metric_type",
        type=str,
        choices=["percentage_change", "max_impact", "diff_from_mean", 
                 "contribution_percentage", "trend_significance"],
        help="Type of metric to calculate"
    )
    calculate_parser.add_argument(
        "--values",
        type=str,
        required=True,
        help="Comma-separated list of values to analyze"
    )
    calculate_parser.add_argument(
        "--labels",
        type=str,
        help="Comma-separated list of labels corresponding to values"
    )
    calculate_parser.add_argument(
        "--baseline",
        type=float,
        help="Baseline value for comparison (required for percentage_change)"
    )
    
    # demo command
    demo_parser = subparsers.add_parser(
        "demo", 
        help="Run a demo with sample data"
    )
    demo_parser.add_argument(
        "--output",
        type=str,
        help="Path to save demo results (JSON)"
    )
    
    return parser


def run_analyze(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the analyze command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Results dictionary
    """
    if not _imports_succeeded:
        print("Error: Required modules not available")
        sys.exit(1)
        
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
        
    # Update config with command-line arguments
    if args.model:
        update_config({"narrative.model": args.model})
    
    # Initialize the Engine
    if _package_import:
        engine = StorytellingEngine()
    else:
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI()
        
        # Initialize agent and visualization engine
        agent = DataStorytellerAgent(client)
        viz_engine = VisualizationEngine() if not args.no_visualize else None
        
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
        
    try:
        # Run analysis using the engine if available
        if _package_import:
            engine.analyze(args.input_file)
            narrative = engine.get_narrative()
            
            # Print narrative to console
            engine.print_narrative()
            
            # Generate visualizations
            if not args.no_visualize:
                engine.visualize()
                
            # Save results if requested
            if args.output:
                engine.save(args.output)
                print(f"Results saved to {args.output}")
                
            return {
                "narrative": narrative,
                "analysis_results": engine.analysis_results
            }
        else:
            # Run the agent directly
            narrative = agent.run(args.input_file, output_visualization=not args.no_visualize)
            
            # Print narrative to console
            print("\n" + "="*80)
            print(f"# {narrative.headline}\n")
            print("## Key Findings")
            for finding in narrative.findings:
                print(f"- {finding}")
            print("\n## Recommendations")
            for recommendation in narrative.recommendations:
                print(f"- {recommendation}")
            print("="*80 + "\n")
            
            # Save results if requested
            if args.output:
                # Format narrative for saving
                narrative_dict = {
                    "headline": narrative.headline,
                    "findings": narrative.findings,
                    "recommendations": narrative.recommendations
                }
                
                data = {
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "analysis_results": agent.analysis_results if hasattr(agent, 'analysis_results') else None,
                    "narrative": narrative_dict
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Results saved to {args.output}")
            
            return {
                "narrative": narrative,
                "analysis_results": agent.analysis_results if hasattr(agent, 'analysis_results') else None
            }
            
    except Exception as e:
        print(f"Error running analysis: {e}")
        sys.exit(1)


def run_visualize(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the visualize command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Results dictionary
    """
    if not _imports_succeeded:
        print("Error: Required modules not available")
        sys.exit(1)
        
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    try:
        # Load saved results
        with open(args.input_file, 'r') as f:
            data = json.load(f)
            
        analysis_results = data.get("analysis_results")
        narrative_dict = data.get("narrative")
        
        if not analysis_results or not narrative_dict:
            print("Error: Invalid results file. Missing analysis_results or narrative.")
            sys.exit(1)
            
        # Create NarrativeSummary from dictionary
        narrative = NarrativeSummary(
            headline=narrative_dict.get("headline", ""),
            findings=narrative_dict.get("findings", []),
            recommendations=narrative_dict.get("recommendations", [])
        )
        
        # Update config with command-line arguments
        config_updates = {"visualization.default_format": args.format}
        if args.theme:
            config_updates["visualization.default_theme"] = args.theme
        update_config(config_updates)
        
        # Generate visualizations
        if _package_import:
            engine = StorytellingEngine.load(args.input_file)
            dashboard_path = engine.visualize(args.format)
            print(f"Visualizations saved to {dashboard_path}")
        else:
            # Initialize visualization engine
            viz_engine = VisualizationEngine(theme=args.theme)
            
            # Generate dashboard
            dashboard = viz_engine.generate_dashboard(analysis_results, narrative)
            
            visualization_dir = get_value("visualization_directory", "visualizations")
            print(f"Visualizations generated in {visualization_dir} directory")
        
        return {
            "output_format": args.format,
            "theme": args.theme or get_value("visualization.default_theme", "light"),
            "visualization_directory": get_value("visualization_directory", "visualizations")
        }
            
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        sys.exit(1)


def run_calculate(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the calculate command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Results dictionary
    """
    if not _imports_succeeded:
        print("Error: Required modules not available")
        sys.exit(1)
        
    try:
        # Parse values and labels
        values = [float(v.strip()) for v in args.values.split(",")]
        labels = None
        if args.labels:
            labels = [l.strip() for l in args.labels.split(",")]
            
            # Ensure labels and values have the same length
            if len(labels) != len(values):
                print(f"Error: Number of labels ({len(labels)}) does not match number of values ({len(values)})")
                sys.exit(1)
        
        # Initialize engine or calculate directly
        if _package_import:
            engine = StorytellingEngine()
            result = engine.calculate_metric(
                metric_type=args.metric_type,
                values=values,
                labels=labels,
                baseline=args.baseline
            )
        else:
            # Import MLX
            import mlx.core as mx
            
            # Convert to MLX array
            values_mx = mx.array(values)
            
            # Calculate metric
            if args.metric_type == "percentage_change" and args.baseline is not None:
                result = float(calculate_percentage_change(values_mx, args.baseline))
                result_dict = {
                    "metric_type": "percentage_change",
                    "result": result,
                    "description": f"Percentage change: {result:.2f}%"
                }
            elif args.metric_type == "max_impact" and labels is not None:
                max_index, max_value = find_max_impact_segment(values_mx)
                result_dict = {
                    "metric_type": "max_impact",
                    "segment": labels[int(max_index)],
                    "value": float(max_value),
                    "description": f"Highest impact segment: {labels[int(max_index)]} with value {float(max_value):.2f}"
                }
            else:
                print(f"Error: Unsupported metric type: {args.metric_type} or missing required parameters")
                sys.exit(1)
                
            result = result_dict
        
        # Print result
        print(json.dumps(result, indent=2))
        
        return result
            
    except Exception as e:
        print(f"Error calculating metric: {e}")
        sys.exit(1)


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the demo command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Results dictionary
    """
    if not _imports_succeeded:
        print("Error: Required modules not available")
        sys.exit(1)
        
    try:
        # Find sample data
        data_dir = Path(__file__).parent / get_value("data_directory", "data")
        sample_file = Path(__file__).parent / "sample-analysis-results-json.json"
        
        if not os.path.exists(sample_file):
            # Try finding it in the data directory
            alternative_path = data_dir / "sample_analysis_results.json"
            if os.path.exists(alternative_path):
                sample_file = alternative_path
            else:
                print(f"Error: Could not find sample data file at {sample_file} or {alternative_path}")
                sys.exit(1)
        
        print(f"Running demo with sample data from {sample_file}")
        
        # Run analysis with sample data
        if _package_import:
            engine = StorytellingEngine()
            engine.analyze(str(sample_file))
            
            # Print narrative to console
            engine.print_narrative()
            
            # Generate visualizations
            dashboard_path = engine.visualize()
            print(f"Visualizations saved to {dashboard_path}")
            
            # Save results if requested
            if args.output:
                engine.save(args.output)
                print(f"Results saved to {args.output}")
                
            return {
                "narrative": engine.get_narrative(),
                "analysis_results": engine.analysis_results,
                "sample_file": str(sample_file)
            }
        else:
            # Initialize OpenAI client
            from openai import OpenAI
            client = OpenAI()
            
            # Initialize agent
            agent = DataStorytellerAgent(client)
            
            # Run the agent with sample data
            narrative = agent.run(str(sample_file))
            
            # Print narrative to console
            print("\n" + "="*80)
            print(f"# {narrative.headline}\n")
            print("## Key Findings")
            for finding in narrative.findings:
                print(f"- {finding}")
            print("\n## Recommendations")
            for recommendation in narrative.recommendations:
                print(f"- {recommendation}")
            print("="*80 + "\n")
            
            # Save results if requested
            if args.output:
                # Format narrative for saving
                narrative_dict = {
                    "headline": narrative.headline,
                    "findings": narrative.findings,
                    "recommendations": narrative.recommendations
                }
                
                data = {
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "analysis_results": agent.analysis_results if hasattr(agent, 'analysis_results') else None,
                    "narrative": narrative_dict,
                    "sample_file": str(sample_file)
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Results saved to {args.output}")
            
            return {
                "narrative": narrative,
                "analysis_results": agent.analysis_results if hasattr(agent, 'analysis_results') else None,
                "sample_file": str(sample_file)
            }
            
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


def main():
    """Main entry point for the command-line interface."""
    # Set up argument parser
    parser = setup_parser()
    args = parser.parse_args()
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    # Load configuration
    if args.config:
        try:
            load_config(args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
            
    # Run command
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "calculate":
        run_calculate(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 