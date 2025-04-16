# Data Storytelling Agent with MLX

A Python repository demonstrating an AI agent that assists with data storytelling by synthesizing analysis results and performing secondary MLX calculations to enhance narratives for business stakeholders.

## Project Overview

This project showcases an AI agent that:

1. Takes structured analytical findings (attribution, experimentation results, etc.)
2. **Key Findings**: 3-5 bullet points highlighting the most important insights
3. **Recommendations**: 2-3 actionable recommendations based on the findings

Example output:

```
# Mobile Redesign Drives 50% Conversion Lift, Outperforming All Other Segments

## Key Findings
- Mobile users experienced the highest conversion lift at 50%, significantly outperforming the overall lift of 21.4%
- New visitors were highly responsive with a 35.5% lift, suggesting the new design better meets their needs
- Paid Search attribution increased by 14.3% from Q1 to Q2, indicating improved search performance
- Revenue is up 8.7% from previous period, with conversion rate and AOV both showing positive trends

## Recommendations
- Prioritize further optimization for the mobile experience given the exceptional performance
- Create targeted campaigns to attract more new visitors who are responding well to the redesign
- Increase investment in Paid Search channels to capitalize on improved attribution
```

## Limitations and Requirements

- This project requires an Apple Silicon Mac to run MLX
- You'll need an OpenAI API key to use the agent functionality
- The narrative quality depends on the structure and completeness of your input data

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request Uses MLX (Apple's machine learning framework) for relevant secondary calculations
3. Generates a concise, compelling narrative summary for non-technical stakeholders

The agent focuses on the "so what?" - extracting key takeaways and actionable recommendations from complex data.

## Repository Structure

- `main.py`: Orchestration script that defines and runs the DataStorytellerAgent
- `storytelling_agent.py`: Implementation of the DataStorytellerAgent class
- `mlx_analyzer.py`: MLX computation module with functions for secondary calculations
- `data/sample_analysis_results.json`: Sample structured analysis results
- `requirements.txt`: Project dependencies

## Setup

### Prerequisites

- Python 3.8 or higher
- Apple Silicon Mac (for MLX support)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/data-storyteller-agent.git
   cd data-storyteller-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

1. Prepare your analysis results in JSON format (see Input Format section below)
2. Run the agent:
   ```bash
   python main.py
   ```

This will:
- Load the sample analysis results from `data/sample_analysis_results.json`
- Identify opportunities for secondary calculations using MLX
- Generate a narrative summary with headlines, key findings, and recommendations

### Customizing the Agent

To use your own data:
1. Replace `sample_analysis_results.json` with your own data following the same structure
2. Modify the `metric_type` parameter in `calculate_impact_metric()` calls to use different calculations based on your data needs

## Data Storytelling Workflow

The data storytelling process follows these steps:

1. **Data Ingestion**: The agent loads structured analysis results from a JSON file
2. **Data Exploration**: The agent examines the data to identify opportunities for secondary calculations
3. **Secondary Calculations**: The agent uses MLX to perform calculations that enhance the narrative:
   - Percentage changes to highlight growth/decline
   - Finding max impact segments to spotlight the best/worst performers
   - Calculating differences from means to identify outliers
   - Computing contribution percentages to show relative importance
4. **Narrative Generation**: The agent combines the primary results and secondary calculations to create a compelling story
5. **Output**: The agent returns a structured narrative with headline, findings, and recommendations

## Input Format (analysis_results.json)

The input file should be a JSON file with a structure similar to the sample provided:

```json
{
  "attribution": {
    "channels": { ... },
    "baseline_period": { ... },
    "time_period": "...",
    "baseline_period_name": "..."
  },
  "experiment_results": {
    "metadata": { ... },
    "overall": { ... },
    "segments": { ... }
  },
  "performance_metrics": {
    "revenue": { ... },
    "conversion_rate": { ... },
    "average_order_value": { ... },
    "customer_acquisition_cost": { ... }
  }
}
```

## MLX Calculations

The repository uses MLX (Apple's machine learning framework) to perform efficient calculations that enhance the narrative:

1. **Percentage Change**: Calculates relative changes between periods or variants
   - Implementation: `((new_value - old_value) / old_value) * 100`
   - MLX Operations: `mx.subtract`, `mx.divide`, `mx.multiply`
   - Narrative Value: Quantifies growth/decline in metrics

2. **Max Impact Segment**: Finds the segment with the highest impact
   - Implementation: Uses `mx.argmax` on an array of segment metrics
   - MLX Operations: `mx.argmax`
   - Narrative Value: Identifies standout performers or most affected areas

3. **Difference from Mean**: Calculates how each value compares to the average
   - Implementation: `segment_value - mx.mean(all_segment_values)`
   - MLX Operations: `mx.subtract`, `mx.mean`
   - Narrative Value: Highlights over/underperformers

4. **Contribution Percentage**: Calculates relative importance of segments
   - Implementation: `(value / sum(values)) * 100`
   - MLX Operations: `mx.sum`, `mx.divide`, `mx.multiply`
   - Narrative Value: Shows relative importance of different segments

5. **Rolling Average**: Smooths time-series data to reveal trends
   - Implementation: Moving window averaging
   - MLX Operations: `mx.mean`, `mx.zeros`
   - Narrative Value: Reveals underlying trends in noisy data

## Output Format

The agent produces a structured narrative in the form of a `NarrativeSummary` object with:

1. **Headline**: An attention-grabbing statement that captures the key insight
2.