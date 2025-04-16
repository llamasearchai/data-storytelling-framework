"""
Narrative summary and insight models for the Data Storytelling Framework.

Provides robust, extensible, and serializable classes for representing narrative outputs.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator

class Insight(BaseModel):
    """
    Represents a single insight extracted from data analysis.
    """
    insight_id: Optional[str] = Field(default=None, description="Unique identifier for the insight")
    title: str = Field(..., description="Title of the insight")
    description: str = Field(..., description="Detailed description of the insight")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1)")
    importance: Optional[Union[int, float]] = Field(default=None, description="Importance or impact score")
    supporting_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Supporting data for the insight")
    recommendations: Optional[List[str]] = Field(default_factory=list, description="Recommendations related to this insight")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags or categories for the insight")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the insight to a dictionary."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        """Create an Insight from a dictionary."""
        return cls(**data)

class NarrativeSummary(BaseModel):
    """
    Represents the output of the Data Storyteller Agent: a structured narrative summary.
    """
    title: str = Field(..., description="Narrative title")
    summary: str = Field(..., description="Executive summary of the narrative")
    insights: List[Insight] = Field(default_factory=list, description="List of key insights")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Key metrics highlighted in the narrative")
    recommendations: List[str] = Field(default_factory=list, description="Overall recommendations")
    narrative_text: Optional[str] = Field(default=None, description="Full narrative text (markdown or prose)")
    creation_timestamp: Union[datetime, str, None] = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of creation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('creation_timestamp', pre=True, always=True)
    def ensure_datetime(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str):
            try:
                # Accept both ISO and already string
                datetime.fromisoformat(v)
                return v
            except Exception:
                return datetime.now().isoformat()
        return datetime.now().isoformat()

    def add_insight(self, 
                    title: str,
                    description: str,
                    confidence: Optional[float] = None,
                    importance: Optional[Union[int, float]] = None,
                    supporting_data: Optional[Dict[str, Any]] = None,
                    recommendations: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    insight_id: Optional[str] = None) -> None:
        """
        Add an insight to the narrative summary.
        """
        self.insights.append(Insight(
            insight_id=insight_id,
            title=title,
            description=description,
            confidence=confidence,
            importance=importance,
            supporting_data=supporting_data or {},
            recommendations=recommendations or [],
            tags=tags or []
        ))

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the narrative summary."""
        self.recommendations.append(recommendation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the narrative summary to a dictionary."""
        d = self.dict()
        d['insights'] = [i.to_dict() for i in self.insights]
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert the narrative summary to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Convert the narrative summary to a markdown string."""
        md = f"# {self.title}\n\n"
        md += f"**Created:** {self.creation_timestamp}\n\n"
        md += "## Summary\n\n"
        md += f"{self.summary}\n\n"
        if self.narrative_text:
            md += "## Detailed Narrative\n\n"
            md += f"{self.narrative_text}\n\n"
        if self.key_metrics:
            md += "## Key Metrics\n\n"
            for metric, value in self.key_metrics.items():
                md += f"- **{metric}:** {value}\n"
            md += "\n"
        if self.insights:
            md += "## Key Insights\n\n"
            for insight in self.insights:
                md += f"### {insight.title}\n\n"
                md += f"{insight.description}\n\n"
                if insight.confidence is not None:
                    md += f"**Confidence:** {insight.confidence:.2f}  "
                if insight.importance is not None:
                    md += f"**Importance:** {insight.importance}\n\n"
                if insight.tags:
                    md += f"**Tags:** {', '.join(insight.tags)}\n\n"
                if insight.recommendations:
                    md += "**Recommendations:**\n\n"
                    for rec in insight.recommendations:
                        md += f"- {rec}\n"
                    md += "\n"
        if self.recommendations:
            md += "## Overall Recommendations\n\n"
            for rec in self.recommendations:
                md += f"- {rec}\n"
        return md

    def save(self, filepath: str, format: str = "json") -> str:
        """
        Save the narrative summary to a file.
        Args:
            filepath: Path to save the narrative summary to
            format: Format to save the narrative summary in (json or md)
        Returns:
            Path to the saved file
        """
        fmt = format.lower()
        if fmt == "json":
            content = self.to_json()
            ext = "json"
        elif fmt in ["md", "markdown"]:
            content = self.to_markdown()
            ext = "md"
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'md'.")
        if not filepath.endswith(f".{ext}"):
            filepath = f"{filepath}.{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NarrativeSummary':
        """Create a narrative summary from a dictionary."""
        # Convert insight dictionaries to Insight objects
        if 'insights' in data:
            data['insights'] = [Insight.from_dict(i) if isinstance(i, dict) else i for i in data['insights']]
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'NarrativeSummary':
        """Create a narrative summary from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, filepath: str) -> 'NarrativeSummary':
        """
        Load a narrative summary from a file.
        Args:
            filepath: Path to load the narrative summary from
        Returns:
            Loaded NarrativeSummary object
        """
        with open(filepath, "r") as f:
            if filepath.endswith(".json"):
                return cls.from_json(f.read())
            elif filepath.endswith(".md"):
                raise NotImplementedError("Loading from markdown is not yet supported.")
            else:
                raise ValueError(f"Unsupported file format: {filepath}") 