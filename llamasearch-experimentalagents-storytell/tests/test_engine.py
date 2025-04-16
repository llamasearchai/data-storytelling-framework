"""Unit tests for the StorytellingEngine class"""

import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil

from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary, Insight
from llamasearch_experimentalagents_storytell.core.visualization import VisualizationEngine

class TestStorytellingEngine(unittest.TestCase):
    """Test cases for the StorytellingEngine class using mock data"""
    
    def setUp(self):
        """Set up test fixtures, including mock data and engines"""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock visualization engine
        self.mock_viz_engine = MagicMock(spec=VisualizationEngine)
        
        # Create sample data frames
        self.revenue_data = pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'revenue': [120000, 125000, 140000, 160000, 180000, 210000],
            'target': [125000, 130000, 135000, 140000, 145000, 150000]
        })
        
        self.channel_data = pd.DataFrame({
            'channel': ['Email', 'Social', 'Direct', 'Organic', 'Referral'],
            'revenue': [85000, 65000, 120000, 95000, 45000],
            'conversion_rate': [0.12, 0.08, 0.15, 0.10, 0.07]
        })
        
        self.device_data = pd.DataFrame({
            'device': ['Desktop', 'Mobile', 'Tablet'],
            'revenue': [250000, 130000, 80000],
            'users': [15000, 25000, 5000]
        })
        
        # Create a mock narrative
        self.mock_narrative = NarrativeSummary(
            title="E-commerce Performance Analysis",
            summary="Strong revenue growth exceeding targets with some concerns about mobile performance",
            key_metrics={
                "Total Revenue": "$410,000",
                "Growth Rate": "15%",
                "Desktop Conversion": "14%",
                "Mobile Conversion": "7%"
            },
            narrative_text="# E-commerce Performance\n\nThe e-commerce platform shows strong growth...",
        )
        
        # Add insights to the narrative
        insight1 = Insight(
            insight_id="INS-001",
            title="Email marketing outperforming other channels",
            description="Email campaigns have the highest conversion rate at 12%",
            confidence=0.92,
            importance=4,
            supporting_data={"conversion_rate": 0.12, "revenue": 85000},
            recommendations=["Increase email campaign frequency", "A/B test subject lines"],
            tags=["email", "marketing", "high-priority"]
        )
        
        insight2 = Insight(
            insight_id="INS-002",
            title="Mobile revenue underperforming",
            description="Mobile has high traffic but low conversion rate at 7%",
            confidence=0.88,
            importance=5,
            supporting_data={"conversion_rate": 0.07, "users": 25000},
            recommendations=["Optimize mobile checkout flow", "Implement mobile-specific offers"],
            tags=["mobile", "conversion", "high-priority"]
        )
        
        self.mock_narrative.add_insight(insight1)
        self.mock_narrative.add_insight(insight2)
        
        # Create the storytelling engine
        self.engine = StorytellingEngine(
            visualization_engine=self.mock_viz_engine,
            openai_api_key="mock-api-key"  # This won't be used as we'll mock the API call
        )
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        plt.close('all')  # Close any open plots
    
    @patch('llamasearch_experimentalagents_storytell.core.storyteller.DataStorytellerAgent.generate_narrative')
    def test_engine_workflow(self, mock_generate_narrative):
        """Test the complete workflow of the storytelling engine with mocked narrative generation"""
        # Set up the mock to return our prepared narrative
        mock_generate_narrative.return_value = self.mock_narrative
        
        # Set up the visualization mock to return a file path
        self.mock_viz_engine.create_line_chart.return_value = os.path.join(self.test_dir, "revenue_chart.png")
        self.mock_viz_engine.create_bar_chart.return_value = os.path.join(self.test_dir, "channel_chart.png")
        self.mock_viz_engine.create_pie_chart.return_value = os.path.join(self.test_dir, "device_chart.png")
        self.mock_viz_engine.generate_dashboard.return_value = os.path.join(self.test_dir, "dashboard.html")
        
        # Create a dictionary of data to analyze
        data_to_analyze = {
            "revenue_data": self.revenue_data,
            "channel_data": self.channel_data, 
            "device_data": self.device_data
        }
        
        # Run the engine
        result = self.engine.run(
            data=data_to_analyze,
            data_description="E-commerce performance data including revenue by month, channel, and device",
            analysis_focus="Identify growth trends and opportunities for improvement in the e-commerce platform",
            output_dir=self.test_dir
        )
        
        # Verify that the storyteller agent was called
        mock_generate_narrative.assert_called_once()
        
        # Verify the visualization engine was called for each visualization type
        self.mock_viz_engine.create_line_chart.assert_called()
        self.mock_viz_engine.create_bar_chart.assert_called()
        self.mock_viz_engine.create_pie_chart.assert_called()
        self.mock_viz_engine.generate_dashboard.assert_called()
        
        # Verify the result contains the expected components
        self.assertIsInstance(result["narrative"], NarrativeSummary)
        self.assertEqual(result["narrative"].title, "E-commerce Performance Analysis")
        self.assertEqual(len(result["narrative"].insights), 2)
        
        # Check that output files were specified
        self.assertIn("dashboard_path", result)
        self.assertIn("narrative_path", result)
        self.assertIn("visualizations", result)
    
    @patch('llamasearch_experimentalagents_storytell.core.storyteller.DataStorytellerAgent.generate_narrative')
    def test_custom_analysis_parameters(self, mock_generate_narrative):
        """Test that custom analysis parameters are passed correctly"""
        # Set up the mock to return our prepared narrative
        mock_generate_narrative.return_value = self.mock_narrative
        
        # Set up visualization mocks
        self.mock_viz_engine.create_line_chart.return_value = os.path.join(self.test_dir, "revenue_chart.png")
        self.mock_viz_engine.generate_dashboard.return_value = os.path.join(self.test_dir, "dashboard.html")
        
        # Create a dictionary of data to analyze
        data_to_analyze = {
            "revenue_data": self.revenue_data,
        }
        
        # Custom parameters for analysis
        custom_parameters = {
            "industry": "Retail",
            "time_period": "H1 2023",
            "target_audience": "Online shoppers aged 25-34",
            "business_goals": "Increase mobile conversion rate by 20%"
        }
        
        # Run the engine with custom parameters
        self.engine.run(
            data=data_to_analyze,
            data_description="Monthly e-commerce revenue data",
            analysis_focus="Focus on revenue growth patterns",
            output_dir=self.test_dir,
            custom_parameters=custom_parameters
        )
        
        # Verify that custom parameters were passed to the storyteller agent
        args, kwargs = mock_generate_narrative.call_args
        self.assertIn("custom_parameters", kwargs)
        self.assertEqual(kwargs["custom_parameters"], custom_parameters)
    
    @patch('llamasearch_experimentalagents_storytell.core.storyteller.DataStorytellerAgent.generate_narrative')
    def test_visualization_generation(self, mock_generate_narrative):
        """Test that visualizations are generated correctly based on data types"""
        # Set up the mock to return our prepared narrative
        mock_generate_narrative.return_value = self.mock_narrative
        
        # Set up visualization mocks
        self.mock_viz_engine.create_line_chart.return_value = os.path.join(self.test_dir, "revenue_chart.png")
        self.mock_viz_engine.create_bar_chart.return_value = os.path.join(self.test_dir, "channel_chart.png")
        self.mock_viz_engine.generate_dashboard.return_value = os.path.join(self.test_dir, "dashboard.html")
        
        # Create a dictionary with just time series and categorical data
        data_to_analyze = {
            "revenue_over_time": self.revenue_data,
            "channel_performance": self.channel_data
        }
        
        # Run the engine
        result = self.engine.run(
            data=data_to_analyze,
            data_description="E-commerce performance data",
            analysis_focus="Analyze revenue trends",
            output_dir=self.test_dir
        )
        
        # Verify that the appropriate visualization methods were called
        self.mock_viz_engine.create_line_chart.assert_called_once()
        self.mock_viz_engine.create_bar_chart.assert_called_once()
        self.mock_viz_engine.create_pie_chart.assert_not_called()  # Should not be called
        
        # Check the visualizations in the result
        self.assertEqual(len(result["visualizations"]), 2)  # Should have 2 visualizations

class TestStorytellingEngineIntegration(unittest.TestCase):
    """Integration test for StorytellingEngine with minimal mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.revenue_data = pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'revenue': [120000, 125000, 140000, 160000, 180000, 210000],
            'target': [125000, 130000, 135000, 140000, 145000, 150000]
        })
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        plt.close('all')  # Close any open plots
    
    @unittest.skip("Skipping integration test that requires OpenAI API key")
    def test_integration_with_real_viz_engine(self):
        """Integration test with actual visualization engine but mocked storyteller"""
        # Create a real visualization engine
        viz_engine = VisualizationEngine()
        
        # Create storytelling engine with mocked API key
        engine = StorytellingEngine(
            visualization_engine=viz_engine,
            openai_api_key="sk-mock-key"  # Replace with actual key for real testing
        )
        
        # Create mock narrative for patching
        mock_narrative = NarrativeSummary(
            title="Revenue Analysis",
            summary="Revenue is growing steadily month over month",
            key_metrics={"Total Revenue": "$935,000", "Growth Rate": "15%"},
            narrative_text="# Revenue Analysis\n\nRevenue has been growing steadily..."
        )
        
        # Patch the generate_narrative method
        with patch('llamasearch_experimentalagents_storytell.core.storyteller.DataStorytellerAgent.generate_narrative', 
                  return_value=mock_narrative):
            
            # Run the engine with minimal data
            result = engine.run(
                data={"revenue_data": self.revenue_data},
                data_description="Monthly revenue data for H1 2023",
                analysis_focus="Analyze revenue growth trends",
                output_dir=self.test_dir
            )
            
            # Check that files were actually created
            self.assertTrue(os.path.exists(result["narrative_path"]))
            self.assertTrue(os.path.exists(result["dashboard_path"]))
            
            # Check that at least one visualization was created
            self.assertGreaterEqual(len(result["visualizations"]), 1)
            for viz_path in result["visualizations"]:
                self.assertTrue(os.path.exists(viz_path))

if __name__ == "__main__":
    unittest.main() 