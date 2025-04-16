"""
Demo script for the LlamaSearch ExperimentalAgents: StoryTell framework.

This script demonstrates how to use the StorytellingEngine to generate narratives and 
visualizations from analytical data.
"""

import os
import json
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.config import get_config, update_config

def create_sample_data() -> Dict[str, Any]:
    """
    Create a sample dataset for demonstration purposes.
    
    This function generates a fictitious e-commerce performance dataset
    with attribution, experiment results, and performance metrics.
    
    Returns:
        Dictionary containing sample analytical data
    """
    # Attribution data
    attribution = {
        "channels": {
            "Organic Search": 32.5,
            "Paid Search": 25.3,
            "Social Media": 18.7,
            "Email": 12.4,
            "Direct": 7.8,
            "Referral": 3.3
        },
        "baseline_period": {
            "Organic Search": 30.2,
            "Paid Search": 28.6,
            "Social Media": 15.1,
            "Email": 13.8,
            "Direct": 8.9,
            "Referral": 3.4
        },
        "time_period": "Q2 2023",
        "baseline_period_name": "Q1 2023"
    }
    
    # Experiment results data
    experiment_results = {
        "metadata": {
            "experiment_name": "Homepage Redesign Test",
            "start_date": "2023-04-01",
            "end_date": "2023-04-30",
            "total_users": 50000,
            "hypothesis": "A simplified homepage design will increase conversion rate"
        },
        "overall": {
            "control_conversion_rate": 0.0321,
            "variant_conversion_rate": 0.0384,
            "lift": 0.1962,
            "confidence": 0.982
        },
        "segments": {
            "New Users": {
                "control_conversion_rate": 0.0254,
                "variant_conversion_rate": 0.0325,
                "lift": 0.2795,
                "confidence": 0.991
            },
            "Returning Users": {
                "control_conversion_rate": 0.0405,
                "variant_conversion_rate": 0.0428,
                "lift": 0.0568,
                "confidence": 0.648
            },
            "Mobile": {
                "control_conversion_rate": 0.0289,
                "variant_conversion_rate": 0.0362,
                "lift": 0.2526,
                "confidence": 0.974
            },
            "Desktop": {
                "control_conversion_rate": 0.0384,
                "variant_conversion_rate": 0.0412,
                "lift": 0.0729,
                "confidence": 0.723
            }
        }
    }
    
    # Performance metrics data
    performance_metrics = {
        "conversion_rate": {
            "current_period": 0.0384,
            "previous_period": 0.0321,
            "year_over_year": 0.0301
        },
        "average_order_value": {
            "current_period": 78.42,
            "previous_period": 74.36,
            "year_over_year": 68.95
        },
        "customer_acquisition_cost": {
            "current_period": 22.15,
            "previous_period": 24.82,
            "year_over_year": 28.43
        },
        "revenue_per_user": {
            "current_period": 3.01,
            "previous_period": 2.39,
            "year_over_year": 2.07
        }
    }
    
    # Combine all data
    return {
        "attribution": attribution,
        "experiment_results": experiment_results,
        "performance_metrics": performance_metrics,
        "metadata": {
            "company": "Demo E-commerce",
            "period": "Q2 2023",
            "data_generated_at": "2023-07-15"
        }
    }

def save_sample_data(data: Dict[str, Any], file_path: str) -> str:
    """
    Save sample data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the data to
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save data to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path

def main():
    """
    Run the demo script.
    """
    parser = argparse.ArgumentParser(description="Demo for Data Storytelling Agent")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="demo_output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="LLM model to use"
    )
    parser.add_argument(
        "--theme", 
        type=str, 
        choices=["light", "dark", "corporate", "minimal"],
        default="light",
        help="Visualization theme"
    )
    args = parser.parse_args()
    
    print("LlamaSearch ExperimentalAgents: StoryTell Demo")
    print("=============================================")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure the framework
    config = {
        "output_directory": args.output_dir,
        "visualization.default_theme": args.theme,
        "visualization.default_format": "html",
    }
    
    if args.model:
        config["narrative.model"] = args.model
        
    update_config(config)
    
    # Create sample data
    print("\nGenerating sample data...")
    sample_data = create_sample_data()
    data_path = save_sample_data(sample_data, os.path.join(args.output_dir, "sample_data.json"))
    print(f"Sample data saved to: {data_path}")
    
    # Initialize the storytelling engine
    print("\nInitializing StorytellingEngine...")
    engine = StorytellingEngine(
        visualization_theme=args.theme,
        llm_model=args.model
    )
    
    # Run the storytelling pipeline
    print("\nRunning storytelling pipeline...")
    results = engine.run_pipeline(
        data=sample_data,
        output_dir=args.output_dir,
        output_format="html",
        save_narrative=True,
        narrative_format="md"
    )
    
    # Print summary of results
    print("\nStoryTelling Results:")
    print(f"- Narrative title: {results['narrative'].title}")
    print(f"- Narrative saved to: {results['narrative_path']}")
    print(f"- Generated {len(results['visualization_paths'])} visualization(s)")
    for viz_type, path in results['visualization_paths'].items():
        print(f"  - {viz_type}: {path}")
    
    # Export memory dashboard for debugging
    dashboard_path = engine.export_memory_dashboard(args.output_dir)
    print(f"\nMemory dashboard exported to: {dashboard_path}")
    
    print("\nDemo completed successfully!")
    print(f"All outputs saved to directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 