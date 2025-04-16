# Data Storytelling Agent with MLX

A Python repository demonstrating an AI agent that assists with data storytelling by synthesizing analysis results and performing secondary MLX calculations to enhance narratives for business stakeholders.

## Project Overview

This project showcases an AI agent that:

1. Takes structured analytical findings (attribution, experimentation results, etc.)
2. Uses MLX (Apple's machine learning framework) for relevant secondary calculations
3. Generates a concise, compelling narrative summary for non-technical stakeholders
4. Creates interactive visualizations to enhance understanding of the insights

The agent focuses on the "so what?" - extracting key takeaways and actionable recommendations from complex data.

## Repository Structure

- `main.py`: Orchestration script that defines and runs the DataStorytellerAgent
- `storytelling_agent.py`: Implementation of the DataStorytellerAgent class
- `mlx_analyzer.py`: MLX computation module with functions for secondary calculations
- `visualization.py`: Module for creating interactive visualizations of insights
- `config.py`: Configuration system for customizing behavior across all modules
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
- Create interactive visualizations of the key insights

### Customizing the Agent

#### Configuration System

The project uses a flexible configuration system that allows you to customize behavior across all modules. You can:

1. Modify the default configuration in `config.py`
2. Create a custom JSON configuration file and specify its path:
   ```bash
   export CONFIG_PATH="/path/to/your/config.json"
   python main.py
   ```
3. Programmatically update the configuration:
   ```python
   from config import update_config
   
   update_config({
       "narrative.model": "gpt-4o",
       "visualization.default_theme": "dark",
       "mlx.confidence_threshold": 0.8
   })
   ```

#### Configuration Options

| Category | Option | Default | Description |
|----------|--------|---------|-------------|
| **Paths** | `data_directory` | `"data"` | Directory for input data files |
| | `output_directory` | `"output"` | Directory for output files |
| | `visualization_directory` | `"visualizations"` | Directory for visualization files |
| **MLX Settings** | `mlx.min_data_points` | `10` | Minimum data points for trend analysis |
| | `mlx.confidence_threshold` | `0.7` | Threshold for filtering low-confidence insights |
| | `mlx.rolling_window_size` | `3` | Window size for rolling averages |
| **Narrative** | `narrative.max_findings` | `5` | Maximum number of findings to include |
| | `narrative.max_recommendations` | `3` | Maximum number of recommendations to include |
| | `narrative.temperature` | `0.7` | Temperature for LLM generation |
| | `narrative.model` | `"gpt-4o"` | OpenAI model to use |
| **Visualization** | `visualization.default_theme` | `"light"` | Visual theme (light, dark, corporate, minimal) |
| | `visualization.default_format` | `"html"` | Output format (html, png, jpg, pdf) |
| | `visualization.chart_height` | `600` | Default chart height in pixels |
| | `visualization.chart_width` | `800` | Default chart width in pixels |
| | `visualization.color_palette` | `"viridis"` | Default color palette |
| **Analysis** | `analysis.min_segment_size` | `100` | Minimum segment size for analysis |
| | `analysis.significance_level` | `0.05` | Statistical significance threshold |
| | `analysis.highlight_threshold` | `0.1` | Threshold for highlighting notable changes |

#### Custom Data

To use your own data:
1. Replace `sample_analysis_results.json` with your own data following the same structure
2. Modify the `metric_type` parameter in `calculate_impact_metric()` calls to use different calculations based on your data needs
3. Adjust the configuration parameters to match your data characteristics

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
5. **Visualization Creation**: The agent generates interactive dashboards to complement the narrative
6. **Output**: The agent returns a structured narrative with headline, findings, and recommendations along with visualizations

## Visualization Features

The visualization module provides several types of interactive dashboards:

1. **Attribution Dashboard**: Visualizes channel attribution data with comparison to baseline periods
2. **Experiment Results**: Shows experiment performance across different segments
3. **Performance Metrics**: Compares current metrics against previous periods and year-over-year data
4. **Narrative Visualization**: Presents the narrative findings and recommendations in a visual format
5. **Combined Dashboard**: Integrates all visualizations into a comprehensive dashboard

Visualizations are saved in the configured `visualization_directory` in the specified format (HTML by default for interactivity).

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

6. **Trend Significance**: Calculates the statistical significance of trends
   - Implementation: Linear regression with R-squared calculation
   - MLX Operations: Various statistical operations
   - Narrative Value: Provides confidence in trend statements

## Output Format

The agent produces a structured narrative in the form of a `NarrativeSummary` object with:

1. **Headline**: An attention-grabbing statement that captures the key insight
2. **Findings**: A list of key findings from the analysis (limited by config)
3. **Recommendations**: Actionable recommendations based on the findings (limited by config)

Additionally, interactive visualizations are saved to the configured output directory.

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request