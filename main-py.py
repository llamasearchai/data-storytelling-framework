"""
Main orchestration script for the Data Storytelling Agent.

This script defines and executes a DataStorytellerAgent that takes analytical findings,
performs secondary calculations using MLX, and generates a narrative summary.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel, Field
import mlx.core as mx
import os

from openai import OpenAI
from openai.types.beta.threads import Run
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall

# Local imports
from config import load_config, get_config, get_value
from mlx_analyzer import calculate_percentage_change, find_max_impact_segment, calculate_difference_from_mean
from visualization import VisualizationEngine

class NarrativeSummary(BaseModel):
    """Structured output format for the data storyteller agent."""
    headline: str = Field(description="An attention-grabbing headline summarizing the key insight")
    findings: List[str] = Field(description="Bulleted list of key findings from the analysis")
    recommendations: List[str] = Field(description="Actionable recommendations based on the findings")

class DataStorytellerAgent:
    """
    An AI agent that converts analytical findings into compelling narratives.
    
    The agent loads structured analysis results, identifies opportunities for
    secondary calculations using MLX, and synthesizes the information into
    a clear narrative for non-technical stakeholders.
    """
    
    def __init__(self, client: OpenAI, model: Optional[str] = None):
        """
        Initialize the DataStorytellerAgent.
        
        Args:
            client: OpenAI client instance
            model: The OpenAI model to use (optional, uses config if not provided)
        """
        self.client = client
        self.model = model or get_value("narrative.model", "gpt-4o")
        self.instructions = """
        You are an AI Analytics Translator. Your goal is to convert structured analysis results and supplementary 
        calculations into a clear, concise, and compelling narrative summary for a non-technical business audience. 
        
        Focus on the 'so what?' - the key takeaways and actionable recommendations. Use the provided 
        calculate_impact_metric tool to compute relevant secondary metrics (e.g., percentage change, segment 
        contribution) to strengthen the narrative.
        
        Structure your output with a headline, key findings (bullet points), and clear recommendations.
        """
        self.config = get_config()
        self.visualization_engine = VisualizationEngine()
    
    def load_analysis_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load structured analysis results from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing analysis results
            
        Returns:
            Dictionary containing the structured analysis results
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading analysis results: {e}")
    
    def calculate_impact_metric(self, 
                               metric_type: str, 
                               values: List[float],
                               labels: Optional[List[str]] = None,
                               baseline: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate a secondary metric using MLX to enhance the narrative.
        
        Args:
            metric_type: Type of metric to calculate ('percentage_change', 'max_impact', or 'diff_from_mean')
            values: List of numerical values to analyze
            labels: Optional list of labels corresponding to values
            baseline: Optional baseline value for comparison
            
        Returns:
            Dictionary containing the calculated metric and associated information
        """
        # Convert values to MLX arrays for calculation
        values_mx = mx.array(values)
        
        if metric_type == "percentage_change" and baseline is not None:
            result = calculate_percentage_change(values_mx, baseline)
            return {
                "metric_type": "percentage_change",
                "result": float(result),
                "description": f"Percentage change: {float(result):.2f}%"
            }
        
        elif metric_type == "max_impact" and labels is not None:
            max_index, max_value = find_max_impact_segment(values_mx)
            return {
                "metric_type": "max_impact",
                "segment": labels[int(max_index)],
                "value": float(max_value),
                "description": f"Highest impact segment: {labels[int(max_index)]} with value {float(max_value):.2f}"
            }
        
        elif metric_type == "diff_from_mean":
            differences = calculate_difference_from_mean(values_mx)
            
            if labels is not None:
                diff_dict = {labels[i]: float(diff) for i, diff in enumerate(differences)}
                return {
                    "metric_type": "diff_from_mean",
                    "differences": diff_dict,
                    "description": "Differences from mean calculated for each segment"
                }
            else:
                return {
                    "metric_type": "diff_from_mean",
                    "differences": [float(d) for d in differences],
                    "description": "Differences from mean calculated"
                }
        
        else:
            raise ValueError(f"Unsupported metric type: {metric_type} or missing required parameters")
    
    def run(self, analysis_file: str, output_visualization: bool = True) -> NarrativeSummary:
        """
        Execute the agent to generate a narrative summary from analysis results.
        
        Args:
            analysis_file: Path to the JSON file containing analysis results
            output_visualization: Whether to generate and save visualizations
            
        Returns:
            Structured narrative summary
        """
        # Create a new thread for the conversation
        thread = self.client.beta.threads.create()
        
        # Define the available tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "load_analysis_results",
                    "description": "Load structured analysis results from a JSON file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the JSON file containing analysis results"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_impact_metric",
                    "description": "Calculate a secondary metric using MLX to enhance the narrative",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric_type": {
                                "type": "string",
                                "enum": ["percentage_change", "max_impact", "diff_from_mean"],
                                "description": "Type of metric to calculate"
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numerical values to analyze"
                            },
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of labels corresponding to values"
                            },
                            "baseline": {
                                "type": "number",
                                "description": "Optional baseline value for comparison"
                            }
                        },
                        "required": ["metric_type", "values"]
                    }
                }
            }
        ]
        
        # Add the user's request to the thread
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please analyze the results in {analysis_file} and create a compelling narrative summary focusing on the key insights and recommendations."
        )
        
        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id="YOUR_ASSISTANT_ID",  # Replace with your actual assistant ID
            instructions=self.instructions,
            tools=tools
        )
        
        # Process the run until it's completed
        while run.status != "completed":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "load_analysis_results":
                        output = self.load_analysis_results(analysis_file)
                        self.analysis_results = output  # Store for later use in visualizations
                    elif function_name == "calculate_impact_metric":
                        output = self.calculate_impact_metric(**function_args)
                    else:
                        output = {"error": f"Unknown function: {function_name}"}
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(output)
                    })
                
                # Submit the tool outputs and continue the run
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
        
        # Retrieve the assistant's response
        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Get the last message from the assistant
        assistant_message = next((m for m in messages.data if m.role == "assistant"), None)
        
        if not assistant_message:
            raise ValueError("No assistant message found in the thread")
        
        # Extract the content from the assistant's message
        content = assistant_message.content[0].text.value
        
        # Parse the content to extract the narrative structure
        narrative = self._parse_narrative(content)
        
        # Generate visualizations if requested
        if output_visualization and hasattr(self, 'analysis_results'):
            try:
                dashboard = self.visualization_engine.generate_dashboard(
                    self.analysis_results, narrative
                )
                print(f"Visualizations generated in {get_value('visualization_directory', 'visualizations')} directory")
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
        
        return narrative
    
    def _parse_narrative(self, content: str) -> NarrativeSummary:
        """
        Parse the narrative content from the assistant's response.
        
        Args:
            content: Raw content from the assistant's message
            
        Returns:
            Structured NarrativeSummary object
        """
        try:
            lines = content.strip().split('\n')
            headline = lines[0].strip('# ')
            
            findings_start = content.find("## Key Findings")
            recommendations_start = content.find("## Recommendations")
            
            if findings_start != -1 and recommendations_start != -1:
                findings_text = content[findings_start:recommendations_start].split('\n')[1:]
                recommendations_text = content[recommendations_start:].split('\n')[1:]
                
                findings = [f.strip('- ').strip() for f in findings_text if f.strip().startswith('-')]
                recommendations = [r.strip('- ').strip() for r in recommendations_text if r.strip().startswith('-')]
                
                # Limit findings and recommendations based on config
                max_findings = get_value("narrative.max_findings", 5)
                max_recommendations = get_value("narrative.max_recommendations", 3)
                
                findings = findings[:max_findings]
                recommendations = recommendations[:max_recommendations]
                
                return NarrativeSummary(
                    headline=headline,
                    findings=findings,
                    recommendations=recommendations
                )
            else:
                # Fallback parsing if markdown structure isn't found
                return NarrativeSummary(
                    headline=headline,
                    findings=[line.strip('- ').strip() for line in lines if line.strip().startswith('- ') and "recommendation" not in line.lower()][:get_value("narrative.max_findings", 5)],
                    recommendations=[line.strip('- ').strip() for line in lines if line.strip().startswith('- ') and "recommendation" in line.lower()][:get_value("narrative.max_recommendations", 3)]
                )
        except Exception as e:
            raise ValueError(f"Error parsing model response: {e}, response: {content}")


def main():
    """Main function to run the data storyteller agent."""
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH")
    config = load_config(config_path)
    
    # Initialize OpenAI client
    client = OpenAI()  # Initialize the OpenAI client
    
    # Define the path to the analysis results
    data_dir = Path(__file__).parent / get_value("data_directory", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Use provided sample data or look for a file
    sample_data_file = "sample-analysis-results-json.json"
    analysis_file = str(Path(__file__).parent / sample_data_file)
    
    if not os.path.exists(analysis_file):
        # Try finding it in the data directory
        alternative_path = data_dir / "sample_analysis_results.json"
        if os.path.exists(alternative_path):
            analysis_file = str(alternative_path)
        else:
            print(f"Warning: Could not find analysis file at {analysis_file} or {alternative_path}")
            return
    
    # Initialize and run the agent
    print(f"Initializing DataStorytellerAgent with model: {get_value('narrative.model', 'gpt-4o')}")
    agent = DataStorytellerAgent(client)
    
    print(f"Running analysis on file: {analysis_file}")
    narrative = agent.run(analysis_file)
    
    # Print the narrative summary
    print("\n" + "="*80)
    print(f"# {narrative.headline}\n")
    print("## Key Findings")
    for finding in narrative.findings:
        print(f"- {finding}")
    print("\n## Recommendations")
    for recommendation in narrative.recommendations:
        print(f"- {recommendation}")
    print("="*80 + "\n")
    
    print(f"Narrative generation complete. Check the {get_value('visualization_directory', 'visualizations')} directory for visualizations.")


if __name__ == "__main__":
    main()
