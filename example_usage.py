#!/usr/bin/env python3
"""
Example usage of the Data Storytelling Agent with MLX.

This script demonstrates different ways to use the StorytellingEngine API
to analyze data, generate narratives, and create visualizations.
"""

import json
import os
from pathlib import Path
import sys

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import from the package
try:
    from __init__ import StorytellingEngine
    from config import load_config, update_config, get_value
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project directory.")
    sys.exit(1)
    
    
def basic_usage():
    """Demonstrate basic usage of the StorytellingEngine."""
    print("\n=== Basic Usage Example ===\n")
    
    # Initialize the engine
    engine = StorytellingEngine()
    
    # Find the sample data file
    sample_file = Path(__file__).parent / "sample-analysis-results-json.json"
    if not os.path.exists(sample_file):
        data_dir = Path(__file__).parent / get_value("data_directory", "data")
        alternative_path = data_dir / "sample_analysis_results.json"
        if os.path.exists(alternative_path):
            sample_file = alternative_path
        else:
            print(f"Error: Could not find sample data at {sample_file} or {alternative_path}")
            return
    
    print(f"Using sample data from: {sample_file}")
    
    # Analyze the data and generate a narrative
    engine.analyze(str(sample_file))
    
    # Print the narrative
    engine.print_narrative()
    
    # Generate visualizations
    visualization_path = engine.visualize()
    print(f"Visualizations saved to: {visualization_path}")
    
    
def custom_configuration():
    """Demonstrate customizing the configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # Update configuration programmatically
    update_config({
        "visualization.default_theme": "dark",
        "visualization.chart_height": 700,
        "visualization.chart_width": 900,
        "narrative.max_findings": 3,
        "narrative.max_recommendations": 2
    })
    
    print("Updated configuration with custom values:")
    print(f"  Theme: {get_value('visualization.default_theme')}")
    print(f"  Chart dimensions: {get_value('visualization.chart_width')}x{get_value('visualization.chart_height')}")
    print(f"  Max findings: {get_value('narrative.max_findings')}")
    print(f"  Max recommendations: {get_value('narrative.max_recommendations')}")
    
    # Initialize the engine with the updated config
    engine = StorytellingEngine()
    
    # Find the sample data file
    sample_file = Path(__file__).parent / "sample-analysis-results-json.json"
    if not os.path.exists(sample_file):
        data_dir = Path(__file__).parent / get_value("data_directory", "data")
        alternative_path = data_dir / "sample_analysis_results.json"
        if os.path.exists(alternative_path):
            sample_file = alternative_path
        else:
            print(f"Error: Could not find sample data at {sample_file} or {alternative_path}")
            return
    
    # Analyze the data and generate a narrative
    engine.analyze(str(sample_file))
    
    # Print the narrative (should have limited findings and recommendations)
    engine.print_narrative()
    
    # Generate visualizations with the custom theme
    visualization_path = engine.visualize("png")
    print(f"Visualizations saved to: {visualization_path}")
    
    
def custom_data_analysis():
    """Demonstrate analyzing custom data."""
    print("\n=== Custom Data Analysis Example ===\n")
    
    # Create custom analysis data
    custom_data = {
        "attribution": {
            "channels": {
                "Organic Search": 0.35,
                "Direct": 0.25,
                "Social Media": 0.20,
                "Email": 0.15,
                "Referral": 0.05
            },
            "baseline_period": {
                "Organic Search": 0.30,
                "Direct": 0.25,
                "Social Media": 0.15,
                "Email": 0.20,
                "Referral": 0.10
            },
            "time_period": "Q1 2025",
            "baseline_period_name": "Q4 2024"
        },
        "experiment_results": {
            "metadata": {
                "experiment_name": "Homepage Redesign",
                "experiment_id": "EXP-2025-001",
                "start_date": "2025-01-10",
                "end_date": "2025-02-10"
            },
            "overall": {
                "control_conversion_rate": 0.035,
                "variant_conversion_rate": 0.045,
                "lift": 0.286,
                "confidence": 0.97
            },
            "segments": {
                "Desktop Users": {
                    "control_conversion_rate": 0.042,
                    "variant_conversion_rate": 0.051,
                    "lift": 0.214,
                    "confidence": 0.96
                },
                "Mobile Users": {
                    "control_conversion_rate": 0.028,
                    "variant_conversion_rate": 0.039,
                    "lift": 0.393,
                    "confidence": 0.98
                },
                "New Visitors": {
                    "control_conversion_rate": 0.025,
                    "variant_conversion_rate": 0.038,
                    "lift": 0.520,
                    "confidence": 0.99
                },
                "Returning Visitors": {
                    "control_conversion_rate": 0.048,
                    "variant_conversion_rate": 0.055,
                    "lift": 0.146,
                    "confidence": 0.91
                }
            }
        },
        "performance_metrics": {
            "revenue": {
                "current_period": 1450000,
                "previous_period": 1350000,
                "year_over_year": 1150000
            },
            "average_order_value": {
                "current_period": 85.50,
                "previous_period": 80.25,
                "year_over_year": 75.75
            },
            "customer_lifetime_value": {
                "current_period": 450.75,
                "previous_period": 425.50,
                "year_over_year": 390.25
            },
            "customer_acquisition_cost": {
                "current_period": 25.75,
                "previous_period": 28.50,
                "year_over_year": 30.25
            }
        }
    }
    
    print("Analyzing custom data...")
    
    # Initialize the engine
    engine = StorytellingEngine()
    
    # Analyze the custom data directly (no need to save to file first)
    engine.analyze(custom_data)
    
    # Print the narrative
    engine.print_narrative()
    
    # Generate visualizations
    visualization_path = engine.visualize()
    print(f"Visualizations saved to: {visualization_path}")
    
    # Save the results for later use
    output_file = Path(__file__).parent / get_value("output_directory", "output") / "custom_analysis_results.json"
    engine.save(str(output_file))
    print(f"Results saved to: {output_file}")
    
    
def mlx_calculations():
    """Demonstrate using MLX calculations directly."""
    print("\n=== MLX Calculations Example ===\n")
    
    # Initialize the engine
    engine = StorytellingEngine()
    
    # Define some example data
    channels = ["Organic Search", "Direct", "Social Media", "Email", "Referral"]
    current_values = [0.35, 0.25, 0.20, 0.15, 0.05]
    baseline_values = [0.30, 0.25, 0.15, 0.20, 0.10]
    
    # Calculate percentage change
    print("Percentage change for each channel:")
    for i, channel in enumerate(channels):
        result = engine.calculate_metric(
            metric_type="percentage_change",
            values=[current_values[i]],
            baseline=baseline_values[i]
        )
        print(f"  {channel}: {result['result']:.2f}%")
    
    # Calculate max impact segment
    max_impact = engine.calculate_metric(
        metric_type="max_impact",
        values=current_values,
        labels=channels
    )
    print(f"\nHighest impact channel: {max_impact['segment']} with value {max_impact['value']:.2f}")
    
    # Calculate differences from mean
    diff_mean = engine.calculate_metric(
        metric_type="diff_from_mean",
        values=current_values,
        labels=channels
    )
    print("\nDifferences from mean:")
    for channel, diff in diff_mean["differences"].items():
        print(f"  {channel}: {diff:.3f}")
    
    # Calculate contribution percentages
    contributions = engine.calculate_metric(
        metric_type="contribution_percentage",
        values=current_values,
        labels=channels
    )
    print("\nContribution percentages:")
    for channel, contrib in contributions["contributions"].items():
        print(f"  {channel}: {contrib:.1f}%")
    
    
def save_and_load():
    """Demonstrate saving and loading analysis results."""
    print("\n=== Save and Load Example ===\n")
    
    # Initialize the engine
    engine = StorytellingEngine()
    
    # Find the sample data file
    sample_file = Path(__file__).parent / "sample-analysis-results-json.json"
    if not os.path.exists(sample_file):
        data_dir = Path(__file__).parent / get_value("data_directory", "data")
        alternative_path = data_dir / "sample_analysis_results.json"
        if os.path.exists(alternative_path):
            sample_file = alternative_path
        else:
            print(f"Error: Could not find sample data at {sample_file} or {alternative_path}")
            return
    
    # Analyze the data
    engine.analyze(str(sample_file))
    print("Analysis completed.")
    
    # Save the results
    output_dir = Path(__file__).parent / get_value("output_directory", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / "saved_analysis.json"
    
    engine.save(str(output_file))
    print(f"Analysis results saved to: {output_file}")
    
    # Load the saved results into a new engine instance
    print("\nLoading saved results...")
    loaded_engine = StorytellingEngine.load(str(output_file))
    
    # Print the loaded narrative
    loaded_engine.print_narrative()
    
    # Generate visualizations from the loaded results
    visualization_path = loaded_engine.visualize("html")
    print(f"Visualizations generated from loaded results: {visualization_path}")
    

if __name__ == "__main__":
    # Create directories
    for directory in ["data", "output", "visualizations"]:
        os.makedirs(Path(__file__).parent / directory, exist_ok=True)
    
    # Run the examples
    try:
        basic_usage()
        custom_configuration()
        custom_data_analysis()
        mlx_calculations()
        save_and_load()
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc() 