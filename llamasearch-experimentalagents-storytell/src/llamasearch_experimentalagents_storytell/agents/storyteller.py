"""
Storyteller Agent module for the LlamaSearch ExperimentalAgents: StoryTell framework.

This module provides the DataStorytellerAgent class which orchestrates the generation
of compelling narratives from analytical data using OpenAI's API with function calling.
"""

import json
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import openai
from openai import OpenAI
from openai.types.beta.threads import Run
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall

# Local imports
from ..core.config import get_config, get_value
from ..core.narrative import NarrativeSummary
from ..core.types import AnalysisResults, NarrativeSummary as NarrativeSummaryModel
from ..utils.memory import Memory

logger = logging.getLogger(__name__)

class LLMProviderError(Exception):
    """Exception raised when all LLM providers fail."""
    pass

class DataStorytellerAgent:
    """
    An AI agent that converts analytical findings into compelling narratives.
    
    This agent uses OpenAI's API with function calling capability to analyze data,
    perform secondary calculations, and generate narrative summaries. It includes
    fallback mechanisms for multiple LLM providers and integrates with the memory
    system for logging and persistence.
    """
    
    def __init__(
        self, 
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        memory: Optional[Memory] = None,
    ):
        """
        Initialize the DataStorytellerAgent.
        
        Args:
            client: OpenAI client instance (optional, created if not provided)
            model: The model to use (optional, uses config if not provided)
            memory: Memory instance for logging (optional, created if not provided)
        """
        # Load configuration
        self.config = get_config()
        
        # Set up OpenAI client
        self.client = client or self._create_openai_client()
        
        # Set up model
        self.model = model or get_value("narrative.model", "gpt-4o")
        self.fallback_models = get_value(
            "narrative.fallback_models", 
            ["gpt-3.5-turbo", "gemini-pro", "claude-3-haiku"]
        )
        
        # Set up memory
        self.memory = memory or Memory()
        
        # Agent instructions for narrative generation
        self.instructions = """
        You are an AI Analytics Translator for LlamaSearch ExperimentalAgents. Your goal is to convert
        structured analysis results and supplementary calculations into a clear, concise, and compelling
        narrative summary for a non-technical business audience.
        
        Focus on the 'so what?' - the key takeaways and actionable recommendations from the data.
        Use the provided calculate_impact_metric tool to compute relevant secondary metrics 
        (e.g., percentage change, segment contribution) to strengthen the narrative when needed.
        
        Structure your output with:
        1. A compelling headline that captures the most important insight
        2. 3-5 key findings as bullet points (be specific with numbers and facts)
        3. 2-3 actionable, specific recommendations based on the findings
        
        Be direct, concise, and emphasize the business impact of the data.
        """
        
        # Set up conversation for tracking
        self.conversation_id = None
        
        # Log initialization
        logger.info(f"Initialized DataStorytellerAgent with model: {self.model}")
        self.memory.log(
            level="info",
            component="storyteller",
            message=f"Initialized DataStorytellerAgent with model: {self.model}",
            metadata={"model": self.model, "fallback_models": self.fallback_models}
        )
    
    def _create_openai_client(self) -> OpenAI:
        """Create an OpenAI client with API key from config."""
        api_key = get_value("openai.api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set it in the config or as the OPENAI_API_KEY environment variable."
            )
        
        organization = get_value("openai.organization", os.environ.get("OPENAI_ORGANIZATION", None))
        
        return OpenAI(api_key=api_key, organization=organization)
    
    def _with_fallback(self, function: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with OpenAI API and fall back to alternative models if needed.
        
        Args:
            function: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            LLMProviderError: If all LLM providers fail
        """
        # First try with the primary model
        original_model = self.model
        
        try:
            return function(*args, **kwargs)
        except (openai.APIError, openai.RateLimitError, openai.APIConnectionError) as e:
            # Log the error
            error_message = f"Error with primary model {self.model}: {str(e)}"
            logger.warning(error_message)
            self.memory.log(
                level="warning",
                component="storyteller",
                message=error_message,
                metadata={"model": self.model, "error": str(e)}
            )
            
            # Try fallback models
            for fallback_model in self.fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    self.model = fallback_model
                    
                    self.memory.log(
                        level="info",
                        component="storyteller",
                        message=f"Attempting fallback to model: {fallback_model}",
                        metadata={"fallback_model": fallback_model}
                    )
                    
                    return function(*args, **kwargs)
                except Exception as fallback_error:
                    error_message = f"Error with fallback model {fallback_model}: {str(fallback_error)}"
                    logger.warning(error_message)
                    self.memory.log(
                        level="warning",
                        component="storyteller",
                        message=error_message,
                        metadata={"model": fallback_model, "error": str(fallback_error)}
                    )
            
            # If we reached here, all models failed
            error_message = "All LLM providers failed."
            logger.error(error_message)
            self.memory.log(
                level="error",
                component="storyteller",
                message=error_message,
                metadata={"attempted_models": [original_model] + self.fallback_models}
            )
            
            # Restore original model
            self.model = original_model
            
            raise LLMProviderError("All LLM providers failed. Check logs for details.")
    
    def load_analysis_results(self, file_path_or_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load structured analysis results from a JSON file or dictionary.
        
        Args:
            file_path_or_data: Path to the JSON file or dictionary containing analysis results
            
        Returns:
            Dictionary containing the structured analysis results
        """
        if isinstance(file_path_or_data, dict):
            # Already a dictionary
            analysis_results = file_path_or_data
            
            # Log the data loading
            self.memory.log(
                level="info",
                component="storyteller",
                message="Loaded analysis results from dictionary",
                metadata={"data_keys": list(analysis_results.keys())}
            )
            
            return analysis_results
        else:
            # Assume it's a file path
            try:
                with open(file_path_or_data, 'r') as f:
                    analysis_results = json.load(f)
                
                # Log the file loading
                self.memory.log(
                    level="info",
                    component="storyteller",
                    message="Loaded analysis results from file",
                    metadata={"file_path": file_path_or_data, "data_keys": list(analysis_results.keys())}
                )
                
                return analysis_results
            except Exception as e:
                error_message = f"Error loading analysis results: {e}"
                logger.error(error_message)
                self.memory.log(
                    level="error",
                    component="storyteller",
                    message=error_message,
                    metadata={"file_path": file_path_or_data}
                )
                raise ValueError(error_message)
    
    def calculate_impact_metric(
        self, 
        metric_type: str, 
        values: List[float],
        labels: Optional[List[str]] = None,
        baseline: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate a secondary metric to enhance the narrative.
        
        Args:
            metric_type: Type of metric to calculate ('percentage_change', 'max_impact', 'diff_from_mean')
            values: List of numerical values to analyze
            labels: Optional list of labels corresponding to values
            baseline: Optional baseline value for comparison
            
        Returns:
            Dictionary containing the calculated metric and associated information
        """
        import numpy as np
        
        # Try to use MLX for calculation if available
        try:
            import mlx.core as mx
            
            # Convert values to MLX arrays
            values_mx = mx.array(values)
            
            if metric_type == "percentage_change" and baseline is not None:
                result = ((values_mx - baseline) / baseline) * 100
                return {
                    "metric_type": "percentage_change",
                    "result": float(mx.mean(result)),
                    "description": f"Percentage change: {float(mx.mean(result)):.2f}%"
                }
            
            elif metric_type == "max_impact" and labels is not None:
                max_index = mx.argmax(values_mx)
                max_value = values_mx[max_index]
                return {
                    "metric_type": "max_impact",
                    "segment": labels[int(max_index)],
                    "value": float(max_value),
                    "description": f"Highest impact segment: {labels[int(max_index)]} with value {float(max_value):.2f}"
                }
            
            elif metric_type == "diff_from_mean":
                mean_value = mx.mean(values_mx)
                differences = values_mx - mean_value
                
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
                
        except ImportError:
            # Fall back to numpy if MLX is not available
            logger.info("MLX not available, falling back to numpy for calculations")
            
            # Convert to numpy array
            values_np = np.array(values)
            
            if metric_type == "percentage_change" and baseline is not None:
                result = ((values_np - baseline) / baseline) * 100
                return {
                    "metric_type": "percentage_change",
                    "result": float(np.mean(result)),
                    "description": f"Percentage change: {float(np.mean(result)):.2f}%"
                }
            
            elif metric_type == "max_impact" and labels is not None:
                max_index = np.argmax(values_np)
                max_value = values_np[max_index]
                return {
                    "metric_type": "max_impact",
                    "segment": labels[int(max_index)],
                    "value": float(max_value),
                    "description": f"Highest impact segment: {labels[int(max_index)]} with value {float(max_value):.2f}"
                }
            
            elif metric_type == "diff_from_mean":
                mean_value = np.mean(values_np)
                differences = values_np - mean_value
                
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
    
    def _handle_tool_calls(self, thread_id: str, run: Run) -> None:
        """
        Handle any tool calls required by the OpenAI assistant.
        
        Args:
            thread_id: Thread ID
            run: Run object containing tool calls
        """
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Log the tool call
            self.memory.log(
                level="info",
                component="storyteller",
                message=f"Tool call: {function_name}",
                metadata={"function_name": function_name, "arguments": function_args}
            )
            
            try:
                # Execute the appropriate tool
                if function_name == "load_analysis_results":
                    result = self.load_analysis_results(function_args.get("file_path"))
                elif function_name == "calculate_impact_metric":
                    result = self.calculate_impact_metric(
                        metric_type=function_args.get("metric_type"),
                        values=function_args.get("values"),
                        labels=function_args.get("labels"),
                        baseline=function_args.get("baseline")
                    )
                else:
                    raise ValueError(f"Unknown tool: {function_name}")
                
                # Add the result to tool outputs
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result)
                })
                
                # Record successful tool use
                if self.conversation_id:
                    self.memory.record_tool_use(
                        message_id=tool_call.id,
                        tool_name=function_name,
                        tool_input=function_args,
                        tool_output=result,
                        success=True
                    )
                
            except Exception as e:
                error_message = f"Error executing tool {function_name}: {str(e)}"
                logger.error(error_message)
                
                # Record failed tool use
                if self.conversation_id:
                    self.memory.record_tool_use(
                        message_id=tool_call.id,
                        tool_name=function_name,
                        tool_input=function_args,
                        tool_output={},
                        success=False,
                        error_message=str(e)
                    )
                
                # Add error message as tool output
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps({"error": str(e)})
                })
        
        # Submit the tool outputs
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    
    def _wait_for_run_completion(self, thread_id: str, run_id: str) -> Run:
        """
        Wait for a run to complete.
        
        Args:
            thread_id: Thread ID
            run_id: Run ID
            
        Returns:
            Completed run
        """
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status in ["completed", "failed", "cancelled"]:
                return run
            
            if run.status == "requires_action":
                self._handle_tool_calls(thread_id, run)
            
            time.sleep(1)
    
    def _parse_narrative(self, content: str) -> NarrativeSummary:
        """
        Parse a narrative summary from OpenAI response content.
        
        Args:
            content: Response content from OpenAI
            
        Returns:
            NarrativeSummary object
        """
        return NarrativeSummary.parse_from_text(content)
    
    def generate_narrative(self, analysis_results: Dict[str, Any]) -> NarrativeSummary:
        """
        Generate a narrative summary from analysis results.
        
        Args:
            analysis_results: Dictionary containing structured analysis results
            
        Returns:
            NarrativeSummary object
        """
        # Log the narrative generation
        self.memory.log(
            level="info",
            component="storyteller",
            message="Generating narrative from analysis results",
            metadata={"analysis_keys": list(analysis_results.keys())}
        )
        
        def _generate():
            # Create a new thread
            thread = self.client.beta.threads.create()
            self.conversation_id = thread.id
            
            # Add analysis results to the thread
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"""
                Please analyze the following data and create a compelling narrative summary for business stakeholders.
                
                Analysis Results:
                {json.dumps(analysis_results, indent=2)}
                
                Structure your response with:
                1. A headline that captures the key insight
                2. 3-5 key findings as bullet points
                3. 2-3 actionable recommendations based on the findings
                """
            )
            
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
                        "description": "Calculate a secondary metric to enhance the narrative",
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
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=get_value("openai.assistant_id", None),
                model=self.model,
                instructions=self.instructions,
                tools=tools
            )
            
            # Wait for completion
            run = self._wait_for_run_completion(thread.id, run.id)
            
            if run.status != "completed":
                error_message = f"Run failed with status: {run.status}"
                logger.error(error_message)
                self.memory.log(
                    level="error",
                    component="storyteller",
                    message=error_message,
                    metadata={"run_id": run.id, "status": run.status}
                )
                raise RuntimeError(error_message)
            
            # Get the assistant's response
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )
            
            # Extract content
            content = messages.data[0].content[0].text.value
            
            # Parse the narrative
            narrative = self._parse_narrative(content)
            
            # Store the narrative in memory
            analysis_id = self.memory.store_analysis_result(
                data=analysis_results,
                title=narrative.headline,
                metadata={"model": self.model, "thread_id": thread.id}
            )
            
            self.memory.store_narrative(
                analysis_id=analysis_id,
                headline=narrative.headline,
                findings=narrative.findings,
                recommendations=narrative.recommendations,
                metadata={"model": self.model, "thread_id": thread.id}
            )
            
            return narrative
        
        # Use the fallback mechanism
        return self._with_fallback(_generate)
    
    def analyze(self, file_path_or_data: Union[str, Dict[str, Any]]) -> NarrativeSummary:
        """
        Analyze data and generate a narrative summary.
        
        This is a convenience method that combines loading analysis results
        and generating a narrative.
        
        Args:
            file_path_or_data: Path to the JSON file or dictionary containing analysis results
            
        Returns:
            NarrativeSummary object
        """
        # Load analysis results
        analysis_results = self.load_analysis_results(file_path_or_data)
        
        # Generate narrative
        return self.generate_narrative(analysis_results) 