"""
Data storytelling agent module for the LlamaSearch ExperimentalAgents: StoryTell framework.

This module provides the DataStorytellerAgent class which handles interaction with LLMs 
to analyze data and generate compelling narratives with insights and recommendations.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.config import get_config
from ..core.narrative import NarrativeSummary, Insight
from ..core.memory import Memory
from ..analysis.results import AnalysisResults

logger = logging.getLogger(__name__)

class DataStorytellerAgent:
    """
    Agent that uses LLMs to transform data analysis into compelling narratives.
    
    This agent leverages OpenAI API (or other LLM providers) to analyze data,
    identify insights, and generate narrative summaries.
    """
    
    def __init__(
        self,
        memory: Optional[Memory] = None,
        model: str = None,
        api_key: str = None,
        enable_fallbacks: bool = True,
        max_retries: int = 3,
        verbose: bool = False
    ):
        """
        Initialize the DataStorytellerAgent.
        
        Args:
            memory: Memory module for logging and persistence
            model: LLM model to use (defaults to config)
            api_key: API key for LLM provider (defaults to env/config)
            enable_fallbacks: Whether to enable fallback to simpler models if needed
            max_retries: Maximum number of retries for API calls
            verbose: Whether to enable verbose logging
        """
        config = get_config()
        self.memory = memory or Memory()
        self.model = model or config.get("llm", {}).get("default_model", "gpt-4o")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.get("llm", {}).get("api_key")
        self.enable_fallbacks = enable_fallbacks
        self.max_retries = max_retries
        self.verbose = verbose
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via api_key parameter, OPENAI_API_KEY environment variable, or config file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.fallback_models = config.get("llm", {}).get("fallback_models", ["gpt-3.5-turbo"])
        
        # Register tools
        self.tools = self._register_tools()
        
        logger.info(f"Initialized DataStorytellerAgent with model {self.model}")
        if memory:
            self.memory.log("init", "DataStorytellerAgent", {
                "model": self.model,
                "enable_fallbacks": self.enable_fallbacks
            })
    
    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register available tools for the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_insight",
                    "description": "Create an insight based on data analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "insight_id": {
                                "type": "string",
                                "description": "Unique identifier for the insight"
                            },
                            "title": {
                                "type": "string",
                                "description": "Short, attention-grabbing title for the insight"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the insight"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level of the insight (0.0-1.0)"
                            },
                            "importance": {
                                "type": "number",
                                "description": "Importance level of the insight (0.0-1.0)"
                            },
                            "supporting_data": {
                                "type": "object",
                                "description": "Data points supporting this insight"
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Recommended actions based on this insight"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags categorizing this insight"
                            }
                        },
                        "required": ["insight_id", "title", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_narrative_summary",
                    "description": "Create a narrative summary from insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title for the overall narrative"
                            },
                            "summary": {
                                "type": "string",
                                "description": "Executive summary of the narrative"
                            },
                            "narrative_text": {
                                "type": "string",
                                "description": "Detailed narrative text that tells the story of the data"
                            },
                            "key_metrics": {
                                "type": "object",
                                "description": "Key metrics highlighted in the narrative"
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Overall recommendations based on the analysis"
                            }
                        },
                        "required": ["title", "summary"]
                    }
                }
            }
        ]
    
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Any:
        """
        Call the LLM API with retry logic.
        
        Args:
            messages: List of message objects for the API
            model: Model override (uses self.model by default)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            API response
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=self.tools,
                tool_choice="auto"
            )
            
            if self.verbose:
                logger.debug(f"LLM response: {response}")
                
            # Log to memory
            if self.memory:
                self.memory.log(
                    action="llm_call",
                    entity="openai",
                    details={
                        "model": model or self.model,
                        "messages": messages,
                        "response": response.model_dump(),
                        "duration_ms": int((time.time() - start_time) * 1000)
                    }
                )
                
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            if self.memory:
                self.memory.log(
                    action="llm_error",
                    entity="openai",
                    details={
                        "model": model or self.model,
                        "error": str(e),
                        "messages": messages,
                        "duration_ms": int((time.time() - start_time) * 1000)
                    }
                )
            raise
    
    def _handle_tool_calls(self, response, collected_insights: List[Insight]) -> Dict[str, Any]:
        """
        Handle tool calls from LLM response.
        
        Args:
            response: LLM response with tool calls
            collected_insights: List to store generated insights
            
        Returns:
            Dictionary with processed results
        """
        result = {}
        
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return result
            
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "create_insight":
                    insight = Insight(**function_args)
                    collected_insights.append(insight)
                    result["insight"] = insight.to_dict()
                    
                elif function_name == "create_narrative_summary":
                    result["narrative_summary"] = function_args
                    
                # Log tool call
                if self.memory:
                    self.memory.log(
                        action="tool_call",
                        entity=function_name,
                        details={
                            "arguments": function_args,
                            "tool_call_id": tool_call.id
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Error processing tool call {function_name}: {e}")
                if self.memory:
                    self.memory.log(
                        action="tool_call_error",
                        entity=function_name,
                        details={
                            "error": str(e),
                            "tool_call_id": tool_call.id,
                            "arguments": tool_call.function.arguments
                        }
                    )
        
        return result
    
    def analyze_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        analysis_results: Optional[AnalysisResults] = None,
        context: Optional[str] = None,
        max_insights: int = 5
    ) -> List[Insight]:
        """
        Analyze data to extract key insights.
        
        Args:
            data: Data to analyze (dictionary or list of dictionaries)
            analysis_results: Optional pre-computed analysis results
            context: Optional context about the data
            max_insights: Maximum number of insights to generate
            
        Returns:
            List of Insights extracted from the data
        """
        logger.info("Analyzing data to extract insights")
        
        # Prepare prompt for the LLM
        system_message = """
        You are an expert data analyst and storyteller. Your task is to analyze the provided data 
        and extract the most important insights. For each insight, you should:
        
        1. Identify a clear, specific finding that is supported by the data
        2. Assign a confidence level based on the strength of evidence
        3. Assign an importance level based on potential impact
        4. Provide supporting data points
        5. Suggest actionable recommendations
        
        Use the create_insight tool to generate each insight.
        """
        
        user_message = f"""
        Please analyze the following data and extract up to {max_insights} key insights:
        
        ```
        {json.dumps(data, indent=2)[:4000]}  # Truncate if too large
        ```
        """
        
        if analysis_results:
            user_message += f"""
            For reference, here are the statistical analysis results:
            
            ```
            {analysis_results.to_dict()}
            ```
            """
            
        if context:
            user_message += f"""
            Additional context: {context}
            """
            
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Call LLM with fallback logic
        collected_insights = []
        try:
            response = self._call_llm(messages, temperature=0.5)
            self._handle_tool_calls(response, collected_insights)
            
            # If needed, keep generating insights until we reach max_insights
            remaining_attempts = 2  # Limit additional attempts
            while len(collected_insights) < max_insights and remaining_attempts > 0:
                if len(collected_insights) == 0:
                    # No insights generated yet, try again with more explicit instructions
                    followup = "No insights detected. Please analyze the data again and use the create_insight tool to generate specific insights from the data."
                else:
                    # Some insights generated, ask for more
                    followup = f"Thank you. Please generate {max_insights - len(collected_insights)} more insights from the data that are different from what you've already provided."
                
                messages.append({"role": "user", "content": followup})
                response = self._call_llm(messages, temperature=0.7)  # Increase temperature for diversity
                self._handle_tool_calls(response, collected_insights)
                remaining_attempts -= 1
                
        except Exception as e:
            logger.error(f"Error in analyze_data: {e}")
            if self.enable_fallbacks and self.fallback_models:
                logger.info(f"Trying fallback model {self.fallback_models[0]}")
                try:
                    response = self._call_llm(messages, model=self.fallback_models[0], temperature=0.5)
                    self._handle_tool_calls(response, collected_insights)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
        
        logger.info(f"Generated {len(collected_insights)} insights from data")
        return collected_insights
    
    def generate_narrative(
        self, 
        insights: List[Insight],
        data_context: Optional[str] = None,
        audience: str = "general",
        tone: str = "professional",
        focus_areas: Optional[List[str]] = None
    ) -> NarrativeSummary:
        """
        Generate a narrative summary from insights.
        
        Args:
            insights: List of insights to include in the narrative
            data_context: Optional context about the data source
            audience: Target audience for the narrative
            tone: Desired tone of the narrative
            focus_areas: Optional list of areas to focus on
            
        Returns:
            NarrativeSummary object containing the generated narrative
        """
        logger.info(f"Generating narrative from {len(insights)} insights")
        
        # Prepare prompt for the LLM
        system_message = f"""
        You are an expert data storyteller. Your task is to create a compelling narrative based on 
        the provided insights. The narrative should be tailored for a {audience} audience with a {tone} tone.
        
        Use the create_narrative_summary tool to generate the narrative structure, which should include:
        1. A compelling title
        2. An executive summary
        3. A detailed narrative that tells the story hidden in the data
        4. References to key metrics
        5. Clear, actionable recommendations
        """
        
        insights_json = json.dumps([insight.to_dict() for insight in insights], indent=2)
        
        user_message = f"""
        Please create a narrative based on these insights:
        
        ```
        {insights_json}
        ```
        """
        
        if data_context:
            user_message += f"""
            Context about the data: {data_context}
            """
            
        if focus_areas:
            user_message += f"""
            Please focus the narrative particularly on these areas: {', '.join(focus_areas)}
            """
            
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Call LLM with fallback logic
        narrative_data = {}
        try:
            response = self._call_llm(messages, temperature=0.7)
            result = self._handle_tool_calls(response, [])  # Empty list as we don't collect insights here
            narrative_data = result.get("narrative_summary", {})
            
            # If no narrative summary was created, try to extract from the response text
            if not narrative_data and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.info("No tool call detected, attempting to parse narrative from text response")
                
                # Simple extraction (this could be improved)
                title = content.split("\n")[0].strip("# ")
                parts = content.split("## ")
                summary = parts[1].split("\n", 1)[1].strip() if len(parts) > 1 else ""
                
                narrative_data = {
                    "title": title,
                    "summary": summary,
                    "narrative_text": content,
                    "recommendations": []
                }
                
                # Try to extract recommendations
                if "recommendation" in content.lower():
                    for line in content.lower().split("recommendation")[1].split("\n"):
                        if line.strip().startswith(("- ", "* ", "• ")):
                            rec = line.strip().lstrip("- *•").strip()
                            narrative_data["recommendations"].append(rec)
                
        except Exception as e:
            logger.error(f"Error in generate_narrative: {e}")
            if self.enable_fallbacks and self.fallback_models:
                logger.info(f"Trying fallback model {self.fallback_models[0]}")
                try:
                    response = self._call_llm(messages, model=self.fallback_models[0], temperature=0.7)
                    result = self._handle_tool_calls(response, [])
                    narrative_data = result.get("narrative_summary", {})
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
        
        # Create NarrativeSummary object
        if not narrative_data:
            logger.warning("Failed to generate narrative, creating a basic summary")
            narrative_data = {
                "title": "Data Analysis Summary",
                "summary": "A summary of the analyzed data.",
                "narrative_text": "The analysis process encountered issues generating a detailed narrative."
            }
        
        # Add metadata
        narrative_data["metadata"] = {
            "audience": audience,
            "tone": tone,
            "focus_areas": focus_areas,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create narrative summary object
        narrative = NarrativeSummary(
            title=narrative_data.get("title", "Data Analysis"),
            summary=narrative_data.get("summary", "Summary of data analysis"),
            insights=insights,
            key_metrics=narrative_data.get("key_metrics", {}),
            recommendations=narrative_data.get("recommendations", []),
            narrative_text=narrative_data.get("narrative_text", ""),
            metadata=narrative_data.get("metadata", {})
        )
        
        logger.info(f"Generated narrative with title: {narrative.title}")
        
        # Log to memory
        if self.memory:
            self.memory.log(
                action="generate_narrative",
                entity="narrative",
                details={
                    "title": narrative.title,
                    "summary_length": len(narrative.summary),
                    "insight_count": len(narrative.insights),
                    "recommendation_count": len(narrative.recommendations)
                }
            )
            
        return narrative
    
    def analyze_and_narrate(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        analysis_results: Optional[AnalysisResults] = None,
        data_context: Optional[str] = None,
        audience: str = "general",
        tone: str = "professional",
        focus_areas: Optional[List[str]] = None,
        max_insights: int = 5
    ) -> NarrativeSummary:
        """
        Complete pipeline to analyze data and generate a narrative.
        
        Args:
            data: Data to analyze
            analysis_results: Optional pre-computed analysis results
            data_context: Optional context about the data source
            audience: Target audience for the narrative
            tone: Desired tone of the narrative
            focus_areas: Optional list of areas to focus on
            max_insights: Maximum number of insights to generate
            
        Returns:
            NarrativeSummary object containing the generated narrative
        """
        logger.info("Running complete analyze and narrate pipeline")
        
        # Step 1: Extract insights
        insights = self.analyze_data(
            data=data,
            analysis_results=analysis_results,
            context=data_context,
            max_insights=max_insights
        )
        
        # Log progress
        if self.memory:
            self.memory.log(
                action="analyze_data_complete",
                entity="insights",
                details={"insight_count": len(insights)}
            )
        
        # Step 2: Generate narrative
        narrative = self.generate_narrative(
            insights=insights,
            data_context=data_context,
            audience=audience,
            tone=tone,
            focus_areas=focus_areas
        )
        
        logger.info("Completed analyze and narrate pipeline")
        return narrative 