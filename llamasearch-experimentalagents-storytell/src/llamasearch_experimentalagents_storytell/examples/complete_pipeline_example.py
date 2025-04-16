"""
Complete Data Storytelling Pipeline Example

This example demonstrates the full data storytelling process:
1. Loading and analyzing data
2. Generating a narrative using the DataStorytellerAgent
3. Creating visualizations with VisualizationEngine
4. Saving the complete story results
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary

def load_sample_data():
    """
    Creates sample e-commerce performance data for demonstration purposes.
    
    Returns:
        Dictionary containing sample datasets
    """
    # Sample e-commerce data - monthly performance metrics
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='M')
    
    # Create sample traffic and conversion data
    traffic_data = pd.DataFrame({
        'date': dates,
        'total_visitors': [45000, 42000, 47500, 52000, 58500, 63000],
        'new_visitors': [28000, 25000, 27500, 29000, 32500, 35000],
        'returning_visitors': [17000, 17000, 20000, 23000, 26000, 28000],
        'bounce_rate': [0.42, 0.44, 0.40, 0.38, 0.37, 0.36],
        'avg_session_duration': [210, 205, 220, 235, 240, 245]  # seconds
    })
    
    # Create sample conversion data
    conversion_data = pd.DataFrame({
        'date': dates,
        'conversion_rate': [0.025, 0.024, 0.027, 0.030, 0.033, 0.034],
        'transactions': [1125, 1008, 1283, 1560, 1931, 2142],
        'revenue': [78750, 70560, 94942, 124800, 163178, 193780],
        'avg_order_value': [70, 70, 74, 80, 85, 90.5],
        'cart_abandonment': [0.72, 0.74, 0.71, 0.68, 0.67, 0.65]
    })
    
    # Create sample channel performance data
    channels = ['organic_search', 'paid_search', 'social', 'email', 'direct', 'referral']
    channel_data = pd.DataFrame({
        'channel': channels,
        'visitors': [15000, 12000, 14000, 8000, 10000, 4000],
        'conversion_rate': [0.035, 0.042, 0.038, 0.055, 0.022, 0.018],
        'revenue': [52500, 50400, 53200, 44000, 22000, 7200],
        'cpa': [18, 25, 22, 12, 5, 15],
        'roas': [2.8, 2.1, 2.5, 3.2, 4.1, 1.8]
    })
    
    # Return a dictionary of datasets
    return {
        'traffic': traffic_data,
        'conversions': conversion_data,
        'channel_performance': channel_data,
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }

def main():
    """Run the complete storytelling pipeline example."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "storytelling_example_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting the Data Storytelling Pipeline example...")
    
    # Initialize the StorytellingEngine with default settings
    engine = StorytellingEngine(
        config={
            "output_dir": output_dir,
            "project_name": "e-commerce-performance-q2",
            "openai_model": "gpt-4-turbo",  # Assumes you have API access configured 
        },
        visualization_theme="light"
    )
    
    # Load sample data
    print("Loading sample e-commerce data...")
    data = load_sample_data()
    
    # Run the complete storytelling pipeline
    print("Running the storytelling pipeline...")
    results = engine.run_pipeline(
        data=data,
        context={
            "business_context": "E-commerce platform performance analysis for Q2 2023",
            "analysis_goals": "Identify key trends, opportunities and actionable recommendations",
            "target_audience": "Marketing and product teams",
            "key_questions": [
                "How is overall performance trending?", 
                "Which channels are most effective?",
                "What optimizations should we prioritize?"
            ]
        }
    )
    
    # Access the narrative summary from results
    narrative = results.get('narrative')
    if narrative:
        print("\nGenerated Narrative Summary:")
        print(f"Title: {narrative.title}")
        print(f"Summary: {narrative.summary}")
        print(f"Number of insights: {len(narrative.insights)}")
        print(f"Number of recommendations: {len(narrative.recommendations)}")
        
        # Save narrative to files in multiple formats
        narrative_json_path = os.path.join(output_dir, "narrative_summary.json")
        narrative_md_path = os.path.join(output_dir, "narrative_summary.md")
        
        narrative.save(narrative_json_path, format="json")
        narrative.save(narrative_md_path, format="markdown")
        
        print(f"\nNarrative saved to:")
        print(f"- JSON: {narrative_json_path}")
        print(f"- Markdown: {narrative_md_path}")
    
    # Access visualizations from results
    visualizations = results.get('visualizations', [])
    if visualizations:
        print(f"\nGenerated {len(visualizations)} visualizations:")
        for i, viz_path in enumerate(visualizations):
            print(f"- Visualization {i+1}: {viz_path}")
    
    # Access the dashboard if generated
    dashboard = results.get('dashboard')
    if dashboard:
        print(f"\nInteractive dashboard saved to: {dashboard}")
    
    print("\nData Storytelling Pipeline example completed successfully!")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main() 