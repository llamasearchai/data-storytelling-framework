"""
Mock Storyteller Demo

This example demonstrates the data storytelling framework using mocked responses,
allowing users to test the pipeline without needing API access.
"""

import os
import json
import pandas as pd
from datetime import datetime

from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary, Insight
from llamasearch_experimentalagents_storytell.core.visualization import VisualizationEngine

class MockStorytellerDemo:
    """
    Demonstrates the data storytelling framework using mocked data and responses.
    This allows users to see the workflow without requiring API access.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the demo with an output directory.
        
        Args:
            output_dir: Directory to save outputs (default: creates a timestamped directory)
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.getcwd(), f"storyteller_demo_{timestamp}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualization engine
        self.viz_engine = VisualizationEngine(
            output_dir=self.output_dir,
            theme="light"
        )
        
        print(f"Mock Storyteller Demo initialized. Outputs will be saved to: {self.output_dir}")
    
    def load_sample_data(self):
        """Create sample e-commerce data for demonstration"""
        # Create date range for 6 months
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='M')
        
        # Monthly revenue data
        revenue_data = pd.DataFrame({
            'date': dates,
            'revenue': [78750, 82560, 94942, 112800, 133178, 153780],
            'target': [80000, 85000, 90000, 100000, 120000, 140000]
        })
        
        # Channel performance
        channel_data = pd.DataFrame({
            'channel': ['Organic Search', 'Paid Search', 'Social', 'Email', 'Direct', 'Referral'],
            'visitors': [15000, 12000, 14000, 8000, 10000, 4000],
            'conversion_rate': [0.035, 0.042, 0.038, 0.055, 0.022, 0.018],
            'revenue': [52500, 50400, 53200, 44000, 22000, 7200]
        })
        
        # Mobile vs Desktop
        device_data = pd.DataFrame({
            'device': ['Desktop', 'Mobile', 'Tablet'],
            'sessions': [32000, 26000, 5000],
            'conversion_rate': [0.042, 0.028, 0.033],
            'revenue': [80640, 58240, 14820]
        })
        
        return {
            'revenue': revenue_data,
            'channels': channel_data,
            'devices': device_data
        }
    
    def generate_mock_narrative(self, data):
        """
        Create a mock narrative based on the data.
        
        Args:
            data: Dictionary containing dataframes
            
        Returns:
            NarrativeSummary object
        """
        # Create the narrative summary
        narrative = NarrativeSummary(
            title="Q2 2023 E-commerce Performance Analysis",
            summary="Our e-commerce platform showed strong growth in Q2 2023, exceeding revenue targets by 10%. Key drivers include email marketing performance and mobile optimization improvements.",
            key_metrics={
                "Total Revenue": "$400,758",
                "Revenue Growth": "+12.3% QoQ",
                "Conversion Rate": "3.42%",
                "Average Order Value": "$82.75"
            },
            metadata={
                "date_range": "Q2 2023 (Apr-Jun)",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "data_sources": ["Web Analytics", "CRM", "Payment Processor"]
            }
        )
        
        # Add insights
        narrative.add_insight(Insight(
            insight_id="INS-001",
            title="Email Marketing Shows Highest Conversion Rate",
            description="Email campaigns achieved a 5.5% conversion rate, outperforming all other channels by at least 30%.",
            confidence=0.92,
            importance=5,
            supporting_data={
                "email_conversion": "5.5%",
                "next_highest_channel": "Paid Search (4.2%)",
                "email_revenue": "$44,000"
            },
            recommendations=[
                "Increase email campaign frequency from bi-weekly to weekly",
                "A/B test new email templates to optimize conversions further"
            ],
            tags=["marketing", "email", "high-impact"]
        ))
        
        narrative.add_insight(Insight(
            insight_id="INS-002",
            title="Revenue Consistently Exceeding Targets",
            description="Monthly revenue has exceeded targets for 4 consecutive months, with June showing the largest positive gap of $13,780.",
            confidence=0.97,
            importance=4,
            supporting_data={
                "june_revenue": "$153,780",
                "june_target": "$140,000",
                "q2_total_overage": "$39,758"
            },
            recommendations=[
                "Revise Q3 targets upward by 8-10%",
                "Reinvest 5% of excess revenue into best-performing channels"
            ],
            tags=["forecasting", "performance"]
        ))
        
        narrative.add_insight(Insight(
            insight_id="INS-003",
            title="Mobile Revenue Lags Despite Traffic Share",
            description="Mobile accounts for 41% of sessions but only 38% of revenue, indicating conversion optimization opportunities.",
            confidence=0.85,
            importance=4,
            supporting_data={
                "mobile_sessions": "26,000 (41%)",
                "mobile_revenue": "$58,240 (38%)",
                "mobile_conversion": "2.8% vs 4.2% desktop"
            },
            recommendations=[
                "Prioritize mobile checkout flow optimization",
                "Implement one-click payments for mobile users",
                "Conduct usability testing on mobile checkout"
            ],
            tags=["mobile", "conversion", "optimization"]
        ))
        
        # Add recommendations
        narrative.add_recommendation("Expand email subscriber list through a site-wide promotion offering 10% discount")
        narrative.add_recommendation("Optimize mobile checkout flow to reduce abandonment rate")
        narrative.add_recommendation("Increase budget allocation for paid search by 15%")
        narrative.add_recommendation("Develop retargeting campaign for cart abandoners")
        
        # Generate narrative text
        narrative_text = f"""
# {narrative.title}

## Executive Summary
{narrative.summary}

## Key Performance Indicators
"""
        
        # Add key metrics to narrative text
        for metric, value in narrative.key_metrics.items():
            narrative_text += f"- **{metric}**: {value}\n"
        
        # Add insights section
        narrative_text += "\n## Key Insights\n\n"
        for insight in narrative.insights:
            narrative_text += f"### {insight.title}\n"
            narrative_text += f"{insight.description}\n\n"
            narrative_text += f"**Confidence**: {insight.confidence:.0%} | **Importance**: {insight.importance}/5\n\n"
            narrative_text += "**Recommendations:**\n"
            for rec in insight.recommendations:
                narrative_text += f"- {rec}\n"
            narrative_text += "\n"
        
        # Add overall recommendations
        narrative_text += "\n## Strategic Recommendations\n\n"
        for i, rec in enumerate(narrative.recommendations, 1):
            narrative_text += f"{i}. {rec}\n"
        
        # Add metadata
        narrative_text += f"\n---\n*Analysis Period: {narrative.metadata.get('date_range')}*\n"
        narrative_text += f"*Generated on: {narrative.metadata.get('analysis_date')}*\n"
        
        # Set the narrative text
        narrative.narrative_text = narrative_text
        
        return narrative
    
    def generate_visualizations(self, data):
        """
        Generate visualizations from the data.
        
        Args:
            data: Dictionary of dataframes
            
        Returns:
            List of paths to visualization files
        """
        viz_paths = []
        
        # 1. Revenue vs Target Line Chart
        revenue_viz_path = os.path.join(self.output_dir, "revenue_vs_target.png")
        self.viz_engine.create_time_series_chart(
            data=data['revenue'],
            x_column='date',
            y_columns=['revenue', 'target'],
            labels=['Actual Revenue', 'Target Revenue'],
            title='Monthly Revenue vs Target',
            y_label='Revenue ($)',
            output_path=revenue_viz_path
        )
        viz_paths.append(revenue_viz_path)
        
        # 2. Channel Performance Bar Chart
        channel_viz_path = os.path.join(self.output_dir, "channel_performance.png")
        self.viz_engine.create_bar_chart(
            data=data['channels'],
            x_column='channel',
            y_column='revenue',
            title='Revenue by Channel',
            x_label='Channel',
            y_label='Revenue ($)',
            output_path=channel_viz_path
        )
        viz_paths.append(channel_viz_path)
        
        # 3. Device Performance Comparison
        device_viz_path = os.path.join(self.output_dir, "device_performance.png")
        self.viz_engine.create_bar_chart(
            data=data['devices'],
            x_column='device',
            y_column='revenue',
            title='Revenue by Device Type',
            x_label='Device Type',
            y_label='Revenue ($)',
            output_path=device_viz_path
        )
        viz_paths.append(device_viz_path)
        
        return viz_paths
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("Starting mock storyteller demonstration...")
        
        # Load sample data
        print("Loading sample data...")
        data = self.load_sample_data()
        
        # Generate mock narrative
        print("Generating narrative...")
        narrative = self.generate_mock_narrative(data)
        
        # Save narrative to files
        narrative_json_path = os.path.join(self.output_dir, "narrative_summary.json")
        narrative_md_path = os.path.join(self.output_dir, "narrative_summary.md")
        
        narrative.save(narrative_json_path, format="json")
        narrative.save(narrative_md_path, format="markdown")
        
        print(f"Narrative saved to:")
        print(f"- JSON: {narrative_json_path}")
        print(f"- Markdown: {narrative_md_path}")
        
        # Generate visualizations
        print("Generating visualizations...")
        viz_paths = self.generate_visualizations(data)
        
        print(f"Generated {len(viz_paths)} visualizations:")
        for i, path in enumerate(viz_paths, 1):
            print(f"- Visualization {i}: {path}")
        
        # Create a basic HTML dashboard
        print("Creating a simple dashboard...")
        dashboard_path = os.path.join(self.output_dir, "dashboard.html")
        
        with open(dashboard_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{narrative.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }}
        .container {{ display: flex; flex-wrap: wrap; }}
        .narrative {{ flex: 1; min-width: 300px; padding: 20px; }}
        .visualizations {{ flex: 1; min-width: 300px; padding: 20px; }}
        .viz-container {{ margin-bottom: 30px; text-align: center; }}
        h1, h2 {{ color: #333; }}
        img {{ max-width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>{narrative.title}</h1>
    <div class="container">
        <div class="narrative">
            <h2>Analysis Summary</h2>
            <p>{narrative.summary}</p>
            
            <h2>Key Metrics</h2>
            <ul>
            """)
            
            for metric, value in narrative.key_metrics.items():
                f.write(f"    <li><strong>{metric}:</strong> {value}</li>\n")
                
            f.write("""
            </ul>
            
            <h2>Key Insights</h2>
            """)
            
            for insight in narrative.insights:
                f.write(f"""
                <div>
                    <h3>{insight.title}</h3>
                    <p>{insight.description}</p>
                    <p><strong>Confidence:</strong> {insight.confidence:.0%} | <strong>Importance:</strong> {insight.importance}/5</p>
                </div>
                """)
            
            f.write("""
            <h2>Recommendations</h2>
            <ol>
            """)
            
            for rec in narrative.recommendations:
                f.write(f"    <li>{rec}</li>\n")
                
            f.write("""
            </ol>
        </div>
        
        <div class="visualizations">
            <h2>Visualizations</h2>
            """)
            
            for i, viz_path in enumerate(viz_paths):
                viz_filename = os.path.basename(viz_path)
                f.write(f"""
                <div class="viz-container">
                    <h3>Figure {i+1}</h3>
                    <img src="{viz_filename}" alt="Visualization {i+1}">
                </div>
                """)
            
            f.write("""
        </div>
    </div>
    
    <footer>
        <p><em>Generated on: """ + datetime.now().strftime("%Y-%m-%d") + """</em></p>
    </footer>
</body>
</html>""")
        
        print(f"Dashboard saved to: {dashboard_path}")
        print("\nDemo completed successfully!")
        print(f"All outputs are available in: {self.output_dir}")
        print("\nTo view the dashboard, open the HTML file in your browser.")

def main():
    """Run the mock storyteller demo"""
    demo = MockStorytellerDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 