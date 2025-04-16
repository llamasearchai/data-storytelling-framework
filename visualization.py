"""
Visualization module for Data Storytelling Agent with MLX.

This module provides functions for visualizing analytical results and narrative
outputs from the storytelling agent, enabling the creation of compelling
visual presentations for non-technical stakeholders.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

# Local imports
from config import get_config, get_value

class VisualizationEngine:
    """Engine for generating visualizations for data storytelling."""
    
    def __init__(self, theme: Optional[str] = None):
        """
        Initialize the visualization engine.
        
        Args:
            theme: Visual theme for plots (optional, uses config if not provided)
        """
        self.config = get_config()
        self.theme = theme or get_value("visualization.default_theme", "light")
        self._setup_visual_style()
        self.output_dir = get_value("visualization_directory", "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_visual_style(self):
        """Configure visualization styles based on selected theme."""
        themes = {
            "light": {
                "bg_color": "#ffffff",
                "text_color": "#333333",
                "grid_color": "#eeeeee",
                "palette": get_value("visualization.color_palette", "viridis")
            },
            "dark": {
                "bg_color": "#222222",
                "text_color": "#ffffff",
                "grid_color": "#333333",
                "palette": "plasma"
            },
            "corporate": {
                "bg_color": "#f9f9f9",
                "text_color": "#2c3e50",
                "grid_color": "#ecf0f1",
                "palette": "Blues"
            },
            "minimal": {
                "bg_color": "#ffffff",
                "text_color": "#555555",
                "grid_color": "#f5f5f5",
                "palette": "Greys"
            }
        }
        
        self.style = themes.get(self.theme, themes["light"])
        sns.set_style("whitegrid")
        plt.rcParams['axes.facecolor'] = self.style["bg_color"]
        plt.rcParams['axes.edgecolor'] = self.style["grid_color"]
        plt.rcParams['axes.labelcolor'] = self.style["text_color"]
        plt.rcParams['text.color'] = self.style["text_color"]
        plt.rcParams['xtick.color'] = self.style["text_color"]
        plt.rcParams['ytick.color'] = self.style["text_color"]
        
    def attribution_dashboard(self, attribution_data: Dict[str, Any]) -> go.Figure:
        """
        Generate an attribution dashboard visualization.
        
        Args:
            attribution_data: Dictionary containing attribution data
            
        Returns:
            Interactive Plotly figure with attribution dashboard
        """
        # Extract data from attribution dictionary
        channels = attribution_data.get("channels", {})
        baseline = attribution_data.get("baseline_period", {})
        time_period = attribution_data.get("time_period", "Current Period")
        baseline_period_name = attribution_data.get("baseline_period_name", "Baseline")
        
        # Create comparison dataframe
        df = pd.DataFrame({
            'Channel': list(channels.keys()),
            'Current': list(channels.values()),
            'Baseline': [baseline.get(channel, 0) for channel in channels.keys()]
        })
        
        # Sort by current period values
        df = df.sort_values('Current', ascending=False)
        
        # Calculate changes
        df['Change'] = df['Current'] - df['Baseline']
        df['Percent_Change'] = (df['Change'] / df['Baseline'].clip(lower=0.001)) * 100
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Channel Attribution - {time_period}", 
                f"Change from {baseline_period_name}",
                "Channel Distribution",
                "Percent Change by Channel"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # Current Attribution Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['Channel'], 
                y=df['Current'],
                name=time_period,
                marker_color='#4285F4'
            ),
            row=1, col=1
        )
        
        # Comparison Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['Channel'], 
                y=df['Change'],
                name='Change',
                marker_color=df['Change'].apply(
                    lambda x: '#34A853' if x >= 0 else '#EA4335'
                )
            ),
            row=1, col=2
        )
        
        # Pie Chart of Current Distribution
        fig.add_trace(
            go.Pie(
                labels=df['Channel'],
                values=df['Current'],
                hole=0.4,
                marker_colors=px.colors.sequential.Viridis
            ),
            row=2, col=1
        )
        
        # Percent Change Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['Channel'],
                y=df['Percent_Change'],
                name='% Change',
                marker_color=df['Percent_Change'].apply(
                    lambda x: '#34A853' if x >= 0 else '#EA4335'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=get_value("visualization.chart_height", 800),
            width=get_value("visualization.chart_width", 1000),
            title_text=f"Attribution Analysis: {time_period} vs {baseline_period_name}",
            paper_bgcolor=self.style["bg_color"],
            plot_bgcolor=self.style["bg_color"],
            font=dict(color=self.style["text_color"]),
            showlegend=False
        )
        
        return fig
    
    def experiment_results_visualization(self, experiment_data: Dict[str, Any]) -> go.Figure:
        """
        Visualize experiment results comparing control and variant performance.
        
        Args:
            experiment_data: Dictionary containing experiment results
            
        Returns:
            Plotly figure showing experiment results
        """
        # Extract data
        metadata = experiment_data.get("metadata", {})
        overall = experiment_data.get("overall", {})
        segments = experiment_data.get("segments", {})
        
        # Create dataframe for overall results
        overall_df = pd.DataFrame({
            'Variant': ['Control', 'Test'],
            'Conversion Rate': [
                overall.get('control_conversion_rate', 0),
                overall.get('variant_conversion_rate', 0)
            ]
        })
        
        # Create dataframe for segment results
        segment_data = []
        for segment_name, segment_results in segments.items():
            segment_data.append({
                'Segment': segment_name,
                'Control': segment_results.get('control_conversion_rate', 0),
                'Variant': segment_results.get('variant_conversion_rate', 0),
                'Lift': segment_results.get('lift', 0),
                'Confidence': segment_results.get('confidence', 0)
            })
        
        segment_df = pd.DataFrame(segment_data)
        
        # Sort segments by lift
        segment_df = segment_df.sort_values('Lift', ascending=False)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Overall Conversion Rate", 
                "Conversion Lift by Segment",
                "Segment Comparison: Control vs Variant",
                "Confidence Level by Segment"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Overall results
        fig.add_trace(
            go.Bar(
                x=overall_df['Variant'],
                y=overall_df['Conversion Rate'],
                marker_color=['#4285F4', '#34A853'],
                text=overall_df['Conversion Rate'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Lift by segment
        fig.add_trace(
            go.Bar(
                x=segment_df['Segment'],
                y=segment_df['Lift'],
                marker_color='#FBBC05',
                text=segment_df['Lift'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Control vs Variant by segment
        fig.add_trace(
            go.Bar(
                x=segment_df['Segment'],
                y=segment_df['Control'],
                name='Control',
                marker_color='#4285F4',
                text=segment_df['Control'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=segment_df['Segment'],
                y=segment_df['Variant'],
                name='Variant',
                marker_color='#34A853',
                text=segment_df['Variant'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Confidence by segment
        fig.add_trace(
            go.Bar(
                x=segment_df['Segment'],
                y=segment_df['Confidence'],
                marker_color=segment_df['Confidence'].apply(
                    lambda x: '#34A853' if x >= 0.95 else 
                              '#FBBC05' if x >= 0.8 else '#EA4335'
                ),
                text=segment_df['Confidence'].apply(lambda x: f"{x:.0%}"),
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Add a threshold line for 95% confidence
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0.95,
            x1=len(segment_df['Segment'])-0.5,
            y1=0.95,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=get_value("visualization.chart_height", 800),
            width=get_value("visualization.chart_width", 1000),
            title_text=f"Experiment Results: {metadata.get('experiment_name', 'Experiment Analysis')}",
            paper_bgcolor=self.style["bg_color"],
            plot_bgcolor=self.style["bg_color"],
            font=dict(color=self.style["text_color"]),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def performance_metrics_visualization(self, performance_data: Dict[str, Any]) -> go.Figure:
        """
        Visualize performance metrics over different time periods.
        
        Args:
            performance_data: Dictionary containing performance metrics
            
        Returns:
            Plotly figure showing performance metrics
        """
        # Process metrics data
        metrics = []
        for metric_name, periods in performance_data.items():
            metrics.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Current': periods.get('current_period', 0),
                'Previous': periods.get('previous_period', 0),
                'YoY': periods.get('year_over_year', 0)
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Calculate changes
        metrics_df['Current_vs_Previous'] = (
            (metrics_df['Current'] - metrics_df['Previous']) / 
            metrics_df['Previous'].clip(lower=0.001)
        ) * 100
        
        metrics_df['Current_vs_YoY'] = (
            (metrics_df['Current'] - metrics_df['YoY']) / 
            metrics_df['YoY'].clip(lower=0.001)
        ) * 100
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Current vs Previous Period",
                "Current vs Year Over Year",
                "Percent Change from Previous Period",
                "Percent Change from Year Over Year"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Current vs Previous
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Current'],
                name='Current',
                marker_color='#4285F4'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Previous'],
                name='Previous',
                marker_color='#7BAAF7'
            ),
            row=1, col=1
        )
        
        # Current vs YoY
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Current'],
                name='Current',
                marker_color='#4285F4',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['YoY'],
                name='Year Ago',
                marker_color='#EA4335'
            ),
            row=1, col=2
        )
        
        # Percent Change from Previous
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Current_vs_Previous'],
                name='% Change vs Previous',
                marker_color=metrics_df['Current_vs_Previous'].apply(
                    lambda x: '#34A853' if x >= 0 else '#EA4335'
                ),
                text=metrics_df['Current_vs_Previous'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Percent Change from YoY
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Current_vs_YoY'],
                name='% Change vs YoY',
                marker_color=metrics_df['Current_vs_YoY'].apply(
                    lambda x: '#34A853' if x >= 0 else '#EA4335'
                ),
                text=metrics_df['Current_vs_YoY'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=get_value("visualization.chart_height", 800),
            width=get_value("visualization.chart_width", 1000),
            title_text="Performance Metrics Analysis",
            paper_bgcolor=self.style["bg_color"],
            plot_bgcolor=self.style["bg_color"],
            font=dict(color=self.style["text_color"]),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def narrative_visualization(self, narrative_summary, analysis_results: Dict[str, Any]) -> go.Figure:
        """
        Create a visual presentation of the narrative summary.
        
        Args:
            narrative_summary: NarrativeSummary object with headline, findings, and recommendations
            analysis_results: Dictionary containing the original analysis results
            
        Returns:
            Plotly figure with narrative visualization
        """
        # Create a dashboard layout for the narrative
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.2, 0.4, 0.4],
            subplot_titles=("", "Key Findings", "Recommendations")
        )
        
        # Add headline as annotation
        fig.add_annotation(
            text=narrative_summary.headline,
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=24, color=self.style["text_color"]),
            align="center"
        )
        
        # Create findings visualization
        for i, finding in enumerate(narrative_summary.findings):
            fig.add_annotation(
                text=f"• {finding}",
                xref="paper", yref="paper",
                x=0.5, y=0.7 - (i * 0.08),
                showarrow=False,
                font=dict(size=16, color=self.style["text_color"]),
                align="center",
                width=800
            )
        
        # Create recommendations visualization
        for i, recommendation in enumerate(narrative_summary.recommendations):
            fig.add_annotation(
                text=f"• {recommendation}",
                xref="paper", yref="paper",
                x=0.5, y=0.3 - (i * 0.08),
                showarrow=False,
                font=dict(size=16, color=self.style["text_color"], color_discrete_map={
                    "• ": "#34A853"  # Green bullet points for recommendations
                }),
                align="center",
                width=800
            )
        
        # Update layout
        fig.update_layout(
            height=get_value("visualization.chart_height", 800),
            width=get_value("visualization.chart_width", 1000),
            paper_bgcolor=self.style["bg_color"],
            plot_bgcolor=self.style["bg_color"],
            font=dict(color=self.style["text_color"]),
            showlegend=False
        )
        
        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def generate_dashboard(self, analysis_results: Dict[str, Any], narrative_summary) -> go.Figure:
        """
        Generate a complete dashboard combining all visualizations.
        
        Args:
            analysis_results: Dictionary containing the complete analysis results
            narrative_summary: NarrativeSummary object with headline, findings, and recommendations
            
        Returns:
            Plotly figure with complete dashboard
        """
        dashboard = {}
        
        # Generate individual visualizations
        if "attribution" in analysis_results:
            dashboard["attribution"] = self.attribution_dashboard(analysis_results["attribution"])
        
        if "experiment_results" in analysis_results:
            dashboard["experiment"] = self.experiment_results_visualization(analysis_results["experiment_results"])
        
        if "performance_metrics" in analysis_results:
            dashboard["performance"] = self.performance_metrics_visualization(analysis_results["performance_metrics"])
        
        # Generate narrative visualization
        dashboard["narrative"] = self.narrative_visualization(narrative_summary, analysis_results)
        
        # Save individual components
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, fig in dashboard.items():
            output_format = get_value("visualization.default_format", "html")
            output_path = os.path.join(self.output_dir, f"{name}_{timestamp}.{output_format}")
            if output_format == "html":
                fig.write_html(output_path)
            else:
                fig.write_image(output_path)
        
        # Return the narrative visualization as the main dashboard
        return dashboard["narrative"]


def save_visualization(fig, filename: str, format: str = None):
    """
    Save a visualization to file.
    
    Args:
        fig: Plotly figure to save
        filename: Filename to save (without extension)
        format: Output format (html, png, jpg, pdf, svg)
    """
    config = get_config()
    output_dir = get_value("visualization_directory", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    format = format or get_value("visualization.default_format", "html")
    path = os.path.join(output_dir, f"{filename}.{format}")
    
    if format == "html":
        fig.write_html(path)
    else:
        fig.write_image(path)
    
    return path 