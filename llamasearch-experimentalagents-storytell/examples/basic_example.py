#!/usr/bin/env python3
"""
Basic example demonstrating how to use the Llamasearch Data Storytelling Framework.

This script shows how to:
1. Load data from a CSV file
2. Generate a narrative summary
3. Create visualizations
4. Generate an HTML dashboard
"""

import os
import pandas as pd
from pathlib import Path
import dotenv

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.dashboard import DashboardGenerator

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

# Create output directory
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Run the basic example."""
    print("Llamasearch Data Storytelling Example")
    print("====================================")
    
    # Sample data - replace with your own dataset path
    # For this example, we'll create a simple DataFrame
    sample_data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Revenue": [10000, 12000, 15000, 14000, 16000, 18000],
        "Expenses": [8000, 8500, 9000, 9500, 10000, 11000],
        "Customers": [120, 125, 140, 155, 160, 175]
    }
    data = pd.DataFrame(sample_data)
    
    # Save the sample data to a CSV file
    data_path = output_dir / "sample_data.csv"
    data.to_csv(data_path, index=False)
    print(f"Created sample data at {data_path}")
    
    # Initialize the storytelling engine
    engine = StorytellingEngine(
        openai_api_key=openai_api_key,
        model="gpt-4-1106-preview"  # Change to your preferred model
    )
    
    print("\nGenerating narrative...")
    # Generate a narrative from the data
    narrative = engine.generate_narrative(
        data=data,
        context="This is a small business's financial data for the first half of the year. "
                "Analyze trends in revenue, expenses, and customer growth.",
        title="Small Business Financial Analysis"
    )
    
    # Save the narrative as JSON and Markdown
    narrative_json_path = output_dir / "narrative.json"
    narrative_md_path = output_dir / "narrative.md"
    
    narrative.save(str(narrative_json_path), format="json")
    narrative.save(str(narrative_md_path), format="markdown")
    
    print(f"Narrative saved to {narrative_json_path} and {narrative_md_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizations = engine.generate_visualizations(
        data=data,
        narrative=narrative,
        output_dir=str(output_dir)
    )
    
    print(f"Generated {len(visualizations)} visualizations.")
    
    # Generate a dashboard
    print("\nGenerating dashboard...")
    dashboard_generator = DashboardGenerator()
    dashboard_path = output_dir / "dashboard.html"
    
    dashboard_generator.generate_dashboard(
        narrative=narrative,
        visualizations=visualizations,
        output_file=str(dashboard_path)
    )
    
    print(f"Dashboard saved to {dashboard_path}")
    print("\nExample complete! Check the output directory for results.")

if __name__ == "__main__":
    main() 