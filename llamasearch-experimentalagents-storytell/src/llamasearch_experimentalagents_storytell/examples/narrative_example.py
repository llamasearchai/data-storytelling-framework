"""
Example demonstrating the usage of the NarrativeSummary class.

This example shows how to create, modify, save, and load NarrativeSummary objects,
which represent the output of the DataStorytellerAgent.
"""

import os
import json
from datetime import datetime
from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary, Insight

def create_sample_narrative() -> NarrativeSummary:
    """
    Create a sample NarrativeSummary object with demo data.
    
    Returns:
        NarrativeSummary object with sample data
    """
    # Create a new NarrativeSummary
    narrative = NarrativeSummary(
        title="Q2 E-commerce Performance Analysis",
        summary="Our e-commerce platform showed strong growth in Q2, with increased conversion rates and revenue metrics across most channels.",
        insights=[],  # We'll add insights separately
        key_metrics={
            "conversion_rate": 3.2,
            "average_order_value": 78.5,
            "revenue_per_user": 12.4,
            "customer_acquisition_cost": 22.7
        },
        recommendations=[
            "Increase budget allocation to social media channels which showed the highest ROI",
            "Optimize the checkout flow to reduce cart abandonment rates",
            "Launch targeted campaigns for high-value customer segments"
        ],
        metadata={
            "analysis_date": datetime.now().isoformat(),
            "data_source": "E-commerce Analytics Platform",
            "analyst": "LlamaSearch ExperimentalAgents"
        }
    )
    
    # Add insights to the narrative
    narrative.add_insight(
        title="Social Media Drives Highest Conversion Rate",
        description="Social media channels have surpassed search as the highest converting traffic source with a 4.2% conversion rate.",
        confidence=0.92,
        importance=8,
        supporting_data={
            "conversion_rates": {
                "social": 4.2,
                "search": 3.8,
                "email": 3.5,
                "direct": 2.9,
                "referral": 2.3
            }
        },
        recommendations=["Increase social media ad spend by 15%", "Test new creative formats on Instagram and TikTok"]
    )
    
    narrative.add_insight(
        title="Mobile Revenue Growth Outpacing Desktop",
        description="Mobile revenue increased by 28% YoY compared to 12% for desktop, indicating a shift in user preferences.",
        confidence=0.95,
        importance=9,
        supporting_data={
            "revenue_growth": {
                "mobile": 28,
                "desktop": 12,
                "tablet": 5
            },
            "revenue_share": {
                "mobile": 58,
                "desktop": 36,
                "tablet": 6
            }
        },
        recommendations=["Prioritize mobile UX optimizations", "Implement mobile-specific promotional strategies"]
    )
    
    narrative.add_insight(
        title="New Customer Acquisition Slowing",
        description="New customer acquisition rate has decreased by 8% while repeat customer transactions increased by 14%.",
        confidence=0.88,
        importance=7,
        supporting_data={
            "customer_trends": {
                "new_customers": -8,
                "repeat_customers": 14,
                "churn_rate": -3
            }
        },
        recommendations=["Review and optimize acquisition channels", "Develop targeted campaigns for new customer segments"]
    )
    
    # Generate narrative text based on insights and recommendations
    narrative.narrative_text = f"""
# {narrative.title}

{narrative.summary}

## Key Insights

1. **{narrative.insights[0].title}** 
   {narrative.insights[0].description}
   
2. **{narrative.insights[1].title}** 
   {narrative.insights[1].description}
   
3. **{narrative.insights[2].title}** 
   {narrative.insights[2].description}

## Key Metrics
- Conversion Rate: {narrative.key_metrics["conversion_rate"]}%
- Average Order Value: ${narrative.key_metrics["average_order_value"]}
- Revenue Per User: ${narrative.key_metrics["revenue_per_user"]}
- Customer Acquisition Cost: ${narrative.key_metrics["customer_acquisition_cost"]}

## Recommendations

1. {narrative.recommendations[0]}
2. {narrative.recommendations[1]}
3. {narrative.recommendations[2]}
"""
    
    return narrative

def main():
    """Run the narrative example."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "narrative_example_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample narrative
    print("Creating sample narrative...")
    narrative = create_sample_narrative()
    
    # Print the narrative as markdown
    print("\nNarrative as Markdown:")
    print(narrative.to_markdown())
    
    # Save the narrative in JSON format
    json_path = os.path.join(output_dir, "narrative_example.json")
    narrative.save(json_path, format="json")
    print(f"\nSaved narrative to JSON: {json_path}")
    
    # Save the narrative in Markdown format
    md_path = os.path.join(output_dir, "narrative_example.md")
    narrative.save(md_path, format="markdown")
    print(f"Saved narrative to Markdown: {md_path}")
    
    # Demonstrate loading a narrative from file
    print("\nLoading narrative from JSON file...")
    loaded_narrative = NarrativeSummary.load(json_path)
    
    # Modify the loaded narrative
    loaded_narrative.title = "Updated: " + loaded_narrative.title
    loaded_narrative.add_recommendation("Implement a loyalty program for frequent customers")
    
    # Save the modified narrative
    updated_path = os.path.join(output_dir, "updated_narrative.json")
    loaded_narrative.save(updated_path)
    print(f"Saved updated narrative to: {updated_path}")
    
    # Demonstrate converting to dictionary
    print("\nNarrative as dictionary:")
    narrative_dict = narrative.to_dict()
    print(f"Dictionary keys: {list(narrative_dict.keys())}")
    print(f"Number of insights: {len(narrative_dict['insights'])}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 