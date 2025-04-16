"""
Type definitions for the LlamaSearch ExperimentalAgents: StoryTell framework.

This module provides Pydantic models for structured data types used throughout
the framework, ensuring type safety and validation.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class ExperimentSegment(BaseModel):
    """Experiment results for a specific user segment."""
    
    control_conversion_rate: float = Field(..., description="Conversion rate for the control group")
    variant_conversion_rate: float = Field(..., description="Conversion rate for the variant group")
    lift: float = Field(..., description="Relative lift between control and variant")
    confidence: float = Field(..., description="Statistical confidence in the result (0-1)")

class ExperimentOverall(BaseModel):
    """Overall experiment results."""
    
    control_conversion_rate: float = Field(..., description="Overall conversion rate for the control group")
    variant_conversion_rate: float = Field(..., description="Overall conversion rate for the variant group")
    lift: float = Field(..., description="Relative lift between control and variant")
    confidence: float = Field(..., description="Statistical confidence in the result (0-1)")

class ExperimentMetadata(BaseModel):
    """Metadata about an A/B test experiment."""
    
    experiment_name: str = Field(..., description="Name of the experiment")
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    start_date: str = Field(..., description="Start date of the experiment")
    end_date: str = Field(..., description="End date of the experiment")

class ExperimentResults(BaseModel):
    """Complete results from an A/B test experiment."""
    
    metadata: ExperimentMetadata = Field(..., description="Metadata about the experiment")
    overall: ExperimentOverall = Field(..., description="Overall experiment results")
    segments: Dict[str, ExperimentSegment] = Field(..., description="Results broken down by segment")

class AttributionData(BaseModel):
    """Marketing attribution data for different channels."""
    
    channels: Dict[str, float] = Field(..., description="Current period attribution by channel")
    baseline_period: Dict[str, float] = Field(..., description="Baseline period attribution by channel")
    time_period: str = Field(..., description="Current time period description")
    baseline_period_name: str = Field(..., description="Baseline period description")
    
    @field_validator('channels', 'baseline_period')
    @classmethod
    def validate_attribution_sums_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # Allow small float precision errors
            raise ValueError(f"Attribution values must sum to approximately 1.0, got {total}")
        return v

class MetricPeriods(BaseModel):
    """Metric values across different time periods."""
    
    current_period: float = Field(..., description="Current period value")
    previous_period: float = Field(..., description="Previous period value")
    year_over_year: float = Field(..., description="Year-over-year value")

class PerformanceMetrics(BaseModel):
    """Performance metrics across different time periods."""
    
    __root__: Dict[str, MetricPeriods] = Field(..., description="Various performance metrics")

class AnalysisResults(BaseModel):
    """Complete analysis results combining multiple data types."""
    
    attribution: Optional[AttributionData] = Field(None, description="Marketing attribution data")
    experiment_results: Optional[ExperimentResults] = Field(None, description="A/B test experiment results")
    performance_metrics: Optional[Dict[str, MetricPeriods]] = Field(None, description="Performance metrics")
    
    class Config:
        """Pydantic model configuration."""
        extra = "allow"  # Allow additional fields for extensibility

class NarrativeSummary(BaseModel):
    """Structured output format for the data storyteller agent."""
    
    headline: str = Field(..., description="An attention-grabbing headline summarizing the key insight")
    findings: List[str] = Field(..., description="Bulleted list of key findings from the analysis")
    recommendations: List[str] = Field(..., description="Actionable recommendations based on the findings")
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "headline": "Mobile Experience Redesign Drives 50% Conversion Increase for New Visitors",
                "findings": [
                    "The new product page design increased overall conversion rate by 21.4%",
                    "Mobile users saw the largest improvement with a 50% conversion lift",
                    "New visitors responded especially well to the redesign with a 35.5% lift"
                ],
                "recommendations": [
                    "Roll out the new design to all users, with emphasis on mobile optimization",
                    "Further optimize the experience for returning visitors who saw smaller gains",
                    "Apply similar design principles to other key conversion pages"
                ]
            }
        }

class ImpactMetric(BaseModel):
    """Secondary metric calculated to enhance the narrative."""
    
    metric_type: str = Field(..., description="Type of metric calculation")
    result: Optional[float] = Field(None, description="Numerical result of calculation")
    description: str = Field(..., description="Human-readable description of the metric")
    segment: Optional[str] = Field(None, description="Segment associated with the metric, if applicable")
    value: Optional[float] = Field(None, description="Value associated with the metric, if applicable")
    differences: Optional[Union[Dict[str, float], List[float]]] = Field(None, description="Differences values, if applicable")
    
    class Config:
        """Pydantic model configuration."""
        extra = "allow"  # Allow additional fields for specialized metrics 