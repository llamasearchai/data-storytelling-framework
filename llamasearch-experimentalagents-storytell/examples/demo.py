#!/usr/bin/env python
"""
Demo script for the Data Storytelling framework.

This script demonstrates how to use the StorytellingEngine to generate
data narratives and visualizations from sample e-commerce data.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import argparse

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.visualization import VisualizationEngine

def generate_sample_data():
    """Generate sample e-commerce data for demonstration purposes."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate monthly revenue data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    base_revenue = 100000
    growth_rate = 0.05
    seasonality = [1.0, 0.9, 1.1, 1.2, 1.3, 1.25, 1.2, 1.15, 1.25, 1.3, 1.5, 1.6]  # Holiday season effect
    
    revenue_data = pd.DataFrame({
        'month': months,
        'revenue': [base_revenue * (1 + growth_rate) ** i * seasonality[i] + np.random.normal(0, 5000) 
                   for i in range(len(months))],
        'target': [base_revenue * (1 + 0.03) ** i for i in range(len(months))]
    })
    
    # Round revenue values to integers
    revenue_data['revenue'] = revenue_data['revenue'].round().astype(int)
    revenue_data['target'] = revenue_data['target'].round().astype(int)
    
    # Generate marketing channel data
    channels = ['Email', 'Paid Search', 'Social Media', 'Direct', 'Organic Search', 'Referral']
    channel_data = pd.DataFrame({
        'channel': channels,
        'revenue': [85000, 120000, 65000, 95000, 130000, 45000],
        'conversion_rate': [0.12, 0.09, 0.08, 0.15, 0.10, 0.07],
        'cost': [15000, 45000, 25000, 5000, 10000, 8000]
    })
    
    # Calculate ROI
    channel_data['roi'] = (channel_data['revenue'] - channel_data['cost']) / channel_data['cost']
    
    # Generate device data
    devices = ['Desktop', 'Mobile', 'Tablet']
    device_data = pd.DataFrame({
        'device': devices,
        'revenue': [250000, 180000, 60000],
        'users': [15000, 25000, 5000],
        'pages_per_session': [4.5, 3.2, 3.8],
        'avg_session_duration': [180, 120, 150]  # in seconds
    })
    
    # Calculate revenue per user
    device_data['revenue_per_user'] = device_data['revenue'] / device_data['users']
    
    # Generate customer segment data
    segments = ['New', 'Returning', 'Loyal']
    segment_data = pd.DataFrame({
        'segment': segments,
        'revenue': [150000, 220000, 120000],
        'users': [22000, 15000, 8000],
        'avg_order_value': [120, 180, 250]
    })
    
    # Return all datasets in a dictionary
    return {
        'revenue_by_month': revenue_data,
        'channel_performance': channel_data,
        'device_performance': device_data,
        'customer_segments': segment_data
    }

def run_demo(api_key, output_dir=None):
    """Run the data storytelling demo with sample data."""
    # Create output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), 'storytelling_output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}")
    
    # Generate sample data
    print("Generating sample e-commerce data...")
    data = generate_sample_data()
    
    # Create visualization and storytelling engines
    print("Initializing engines...")
    viz_engine = VisualizationEngine()
    storytelling_engine = StorytellingEngine(
        visualization_engine=viz_engine,
        openai_api_key=api_key
    )
    
    # Define the analysis request
    data_description = """
    E-commerce performance data for an online retailer, including:
    - Monthly revenue vs. targets
    - Marketing channel performance (revenue, conversion rates, cost, ROI)
    - Device performance (revenue, users, engagement metrics)
    - Customer segment performance (new, returning, and loyal customers)
    """
    
    analysis_focus = """
    Identify key growth drivers and opportunities for improvement in our e-commerce business.
    Specifically:
    1. Analyze revenue trends and performance against targets
    2. Evaluate which marketing channels are most effective
    3. Assess performance across different devices and identify any issues
    4. Compare customer segments and their contribution to overall revenue
    """
    
    # Additional context for the analysis
    custom_parameters = {
        "industry": "Retail",
        "time_period": "Last 12 months",
        "business_goals": "Increase mobile conversion rate by 15%, improve ROI on marketing channels",
        "target_audience": "Marketing and product teams"
    }
    
    # Run the storytelling engine
    print("Generating data story... (this may take a minute)")
    result = storytelling_engine.run(
        data=data,
        data_description=data_description,
        analysis_focus=analysis_focus,
        output_dir=output_dir,
        custom_parameters=custom_parameters
    )
    
    # Print summary of results
    print("\n=== Data Storytelling Complete ===")
    print(f"Title: {result['narrative'].title}")
    print(f"Summary: {result['narrative'].summary}")
    print(f"\nKey Metrics:")
    for metric, value in result['narrative'].key_metrics.items():
        print(f"  - {metric}: {value}")
    
    print(f"\nInsights Generated: {len(result['narrative'].insights)}")
    for i, insight in enumerate(result['narrative'].insights, 1):
        print(f"\nInsight {i}: {insight.title}")
        print(f"  Importance: {insight.importance}/5, Confidence: {insight.confidence:.2f}")
        print(f"  Description: {insight.description}")
        if insight.recommendations:
            print(f"  Top recommendation: {insight.recommendations[0]}")
    
    print(f"\nOutputs:")
    print(f"  - Narrative (JSON): {result['narrative_path']}")
    print(f"  - Dashboard: {result['dashboard_path']}")
    print(f"  - Visualizations: {len(result['visualizations'])} generated")
    for viz_path in result['visualizations']:
        print(f"    - {os.path.basename(viz_path)}")
    
    print("\nDone! Open the dashboard file in a browser to view the complete data story.")
    
    return result

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Data Storytelling Demo")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    args = parser.parse_args()
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Use API key from args, environment, or prompt
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
        
    if not api_key:
        print("Error: OpenAI API key is required.")
        return
    
    run_demo(api_key=api_key, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 