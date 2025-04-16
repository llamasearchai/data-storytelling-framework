"""Unit tests for narrative.py"""

import os
import json
import tempfile
import unittest
from datetime import datetime

from llamasearch_experimentalagents_storytell.core.narrative import NarrativeSummary, Insight

class TestNarrativeSummary(unittest.TestCase):
    """Test cases for the NarrativeSummary class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_timestamp = datetime.now()
        
        # Create a sample narrative summary
        self.narrative = NarrativeSummary(
            title="Test Narrative",
            summary="This is a test summary",
            key_metrics={"metric1": "value1", "metric2": "value2"},
            narrative_text="# Test Narrative\n\nTest content",
            creation_timestamp=self.test_timestamp,
            metadata={"source": "test", "version": "1.0"}
        )
        
        # Add sample insights
        self.insight1 = Insight(
            insight_id="INS-001",
            title="Test Insight 1",
            description="This is a test insight",
            confidence=0.85,
            importance=4,
            supporting_data={"data1": "value1"},
            recommendations=["Recommendation 1", "Recommendation 2"],
            tags=["test", "important"]
        )
        
        self.insight2 = Insight(
            insight_id="INS-002",
            title="Test Insight 2",
            description="This is another test insight",
            confidence=0.75,
            importance=3,
            supporting_data={"data2": "value2"},
            recommendations=["Recommendation 3"],
            tags=["test"]
        )
        
        self.narrative.add_insight(self.insight1)
        self.narrative.add_insight(self.insight2)
        
        # Add recommendations
        self.narrative.add_recommendation("Overall recommendation 1")
        self.narrative.add_recommendation("Overall recommendation 2")
    
    def test_add_insight(self):
        """Test adding insights to the narrative"""
        # Check that insights were added correctly
        self.assertEqual(len(self.narrative.insights), 2)
        self.assertEqual(self.narrative.insights[0].title, "Test Insight 1")
        self.assertEqual(self.narrative.insights[1].title, "Test Insight 2")
    
    def test_add_recommendation(self):
        """Test adding recommendations to the narrative"""
        # Check that recommendations were added correctly
        self.assertEqual(len(self.narrative.recommendations), 2)
        self.assertEqual(self.narrative.recommendations[0], "Overall recommendation 1")
        self.assertEqual(self.narrative.recommendations[1], "Overall recommendation 2")
    
    def test_to_dict(self):
        """Test converting the narrative to a dictionary"""
        narrative_dict = self.narrative.to_dict()
        
        # Check key attributes
        self.assertEqual(narrative_dict["title"], "Test Narrative")
        self.assertEqual(narrative_dict["summary"], "This is a test summary")
        self.assertEqual(narrative_dict["key_metrics"], {"metric1": "value1", "metric2": "value2"})
        self.assertEqual(narrative_dict["metadata"], {"source": "test", "version": "1.0"})
        
        # Check insights
        self.assertEqual(len(narrative_dict["insights"]), 2)
        self.assertEqual(narrative_dict["insights"][0]["title"], "Test Insight 1")
        self.assertEqual(narrative_dict["insights"][1]["title"], "Test Insight 2")
        
        # Check recommendations
        self.assertEqual(len(narrative_dict["recommendations"]), 2)
        self.assertEqual(narrative_dict["recommendations"][0], "Overall recommendation 1")
        self.assertEqual(narrative_dict["recommendations"][1], "Overall recommendation 2")
    
    def test_from_dict(self):
        """Test creating a narrative from a dictionary"""
        # Convert to dict and back
        narrative_dict = self.narrative.to_dict()
        new_narrative = NarrativeSummary.from_dict(narrative_dict)
        
        # Check key attributes
        self.assertEqual(new_narrative.title, "Test Narrative")
        self.assertEqual(new_narrative.summary, "This is a test summary")
        self.assertEqual(new_narrative.key_metrics, {"metric1": "value1", "metric2": "value2"})
        self.assertEqual(new_narrative.metadata, {"source": "test", "version": "1.0"})
        
        # Check insights
        self.assertEqual(len(new_narrative.insights), 2)
        self.assertEqual(new_narrative.insights[0].title, "Test Insight 1")
        self.assertEqual(new_narrative.insights[1].title, "Test Insight 2")
        
        # Check recommendations
        self.assertEqual(len(new_narrative.recommendations), 2)
        self.assertEqual(new_narrative.recommendations[0], "Overall recommendation 1")
        self.assertEqual(new_narrative.recommendations[1], "Overall recommendation 2")
    
    def test_to_json(self):
        """Test converting the narrative to JSON format"""
        json_str = self.narrative.to_json()
        
        # Parse the JSON and check contents
        parsed_json = json.loads(json_str)
        self.assertEqual(parsed_json["title"], "Test Narrative")
        self.assertEqual(parsed_json["summary"], "This is a test summary")
        self.assertEqual(len(parsed_json["insights"]), 2)
        self.assertEqual(len(parsed_json["recommendations"]), 2)
    
    def test_to_markdown(self):
        """Test converting the narrative to Markdown format"""
        markdown = self.narrative.to_markdown()
        
        # Check that the markdown contains key elements
        self.assertIn("# Test Narrative", markdown)
        self.assertIn("This is a test summary", markdown)
        self.assertIn("Test Insight 1", markdown)
        self.assertIn("Test Insight 2", markdown)
        self.assertIn("Overall recommendation 1", markdown)
        self.assertIn("Overall recommendation 2", markdown)
    
    def test_save_and_load_json(self):
        """Test saving and loading the narrative in JSON format"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the narrative
            self.narrative.save(temp_path, format="json")
            
            # Check that the file exists and has content
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
            
            # Load the narrative
            loaded_narrative = NarrativeSummary.load(temp_path)
            
            # Check that the loaded narrative has the correct content
            self.assertEqual(loaded_narrative.title, "Test Narrative")
            self.assertEqual(loaded_narrative.summary, "This is a test summary")
            self.assertEqual(len(loaded_narrative.insights), 2)
            self.assertEqual(len(loaded_narrative.recommendations), 2)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_markdown(self):
        """Test saving the narrative in Markdown format"""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the narrative
            self.narrative.save(temp_path, format="markdown")
            
            # Check that the file exists and has content
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
            
            # Read the file and check its content
            with open(temp_path, 'r') as f:
                content = f.read()
            
            self.assertIn("# Test Narrative", content)
            self.assertIn("This is a test summary", content)
            self.assertIn("Test Insight 1", content)
            self.assertIn("Test Insight 2", content)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_unsupported_format(self):
        """Test that loading an unsupported format raises an error"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Try to load an unsupported format
            with self.assertRaises(ValueError):
                NarrativeSummary.load(temp_path)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestInsight(unittest.TestCase):
    """Test cases for the Insight class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.insight = Insight(
            insight_id="INS-001",
            title="Test Insight",
            description="This is a test insight",
            confidence=0.85,
            importance=4,
            supporting_data={"data1": "value1", "data2": "value2"},
            recommendations=["Recommendation 1", "Recommendation 2"],
            tags=["test", "important"]
        )
    
    def test_to_dict(self):
        """Test converting the insight to a dictionary"""
        insight_dict = self.insight.to_dict()
        
        # Check key attributes
        self.assertEqual(insight_dict["insight_id"], "INS-001")
        self.assertEqual(insight_dict["title"], "Test Insight")
        self.assertEqual(insight_dict["description"], "This is a test insight")
        self.assertEqual(insight_dict["confidence"], 0.85)
        self.assertEqual(insight_dict["importance"], 4)
        self.assertEqual(insight_dict["supporting_data"], {"data1": "value1", "data2": "value2"})
        self.assertEqual(insight_dict["recommendations"], ["Recommendation 1", "Recommendation 2"])
        self.assertEqual(insight_dict["tags"], ["test", "important"])
    
    def test_from_dict(self):
        """Test creating an insight from a dictionary"""
        # Convert to dict and back
        insight_dict = self.insight.to_dict()
        new_insight = Insight.from_dict(insight_dict)
        
        # Check key attributes
        self.assertEqual(new_insight.insight_id, "INS-001")
        self.assertEqual(new_insight.title, "Test Insight")
        self.assertEqual(new_insight.description, "This is a test insight")
        self.assertEqual(new_insight.confidence, 0.85)
        self.assertEqual(new_insight.importance, 4)
        self.assertEqual(new_insight.supporting_data, {"data1": "value1", "data2": "value2"})
        self.assertEqual(new_insight.recommendations, ["Recommendation 1", "Recommendation 2"])
        self.assertEqual(new_insight.tags, ["test", "important"])

if __name__ == "__main__":
    unittest.main() 