"""
Storytelling Agent module for data narrative generation.

This module provides the DataStorytellerAgent class which acts as a wrapper
around the OpenAI API to create compelling narratives from analytical findings.
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from openai import OpenAI


class NarrativeSummary(BaseModel):
    """Structured output format for the data storyteller agent."""
    headline: str = Field(description="An attention-grabbing headline summarizing the key insight")
    findings: List[str] = Field(description="Bulleted list of key findings from the analysis")
    recommendations: List[str] = Field(description="Actionable recommendations based on the findings")


class DataStorytellerAgent:
    """
    An AI agent that converts analytical findings into compelling narratives.
    
    The agent is implemented as a wrapper around the OpenAI API, using the
    function calling capability to interact with external tools for loading
    analysis results and performing secondary calculations.
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        """
        Initialize the DataStorytellerAgent.
        
        Args:
            client: OpenAI client instance
            model: The OpenAI model to use
        """
        self.client = client
        self.model = model
        self.instructions = """
        You are an AI Analytics Translator. Your goal is to convert structured analysis results and supplementary 
        calculations into a clear, concise, and compelling narrative summary for a non-technical business audience. 
        
        Focus on the 'so what?' - the key takeaways and actionable recommendations. Use the provided 
        calculate_impact_metric tool to compute relevant secondary metrics (e.g., percentage change, segment 
        contribution) to strengthen the narrative.
        
        Structure your output with a headline, key findings (bullet points), and clear recommendations.
        """
    
    def generate_narrative(self, analysis_results: Dict[str, Any], impact_metrics: Dict[str, Any]) -> NarrativeSummary:
        """
        Generate a narrative summary from analysis results and impact metrics.
        
        Args:
            analysis_results: Dictionary containing structured analysis results
            impact_metrics: Dictionary containing calculated impact metrics
            
        Returns:
            Structured narrative summary
        """
        # Create a message to the model describing what we want
        prompt = f"""
        Please analyze the following data and create a compelling narrative summary for business stakeholders.
        
        Analysis Results:
        {json.dumps(analysis_results, indent=2)}
        
        Impact Metrics:
        {json.dumps(impact_metrics, indent=2)}
        
        Structure your response with:
        1. A headline that captures the key insight
        2. 3-5 key findings as bullet points
        3. 2-3 actionable recommendations based on the findings
        """
        
        # Call the model to generate the narrative
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Extract the narrative content
        content = response.choices[0].message.content
        
        # Parse the content to extract the headline, findings, and recommendations
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
                
                return NarrativeSummary(
                    headline=headline,
                    findings=findings,
                    recommendations=recommendations
                )
            else:
                # Fallback parsing if markdown structure isn't found
                return NarrativeSummary(
                    headline=headline,
                    findings=[line.strip('- ').strip() for line in lines if line.strip().startswith('- ') and "recommendation" not in line.lower()],
                    recommendations=[line.strip('- ').strip() for line in lines if line.strip().startswith('- ') and "recommendation" in line.lower()]
                )
        except Exception as e:
            raise ValueError(f"Error parsing model response: {e}, response: {content}")
