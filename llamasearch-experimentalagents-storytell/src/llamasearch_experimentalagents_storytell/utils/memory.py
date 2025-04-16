"""
Memory module for SQLite-based persistence in the LlamaSearch ExperimentalAgents framework.

This module provides utilities for storing and retrieving conversation history,
logs, and agent memory using SQLite.
"""

import json
import os
import sqlite_utils
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from ..core.config import get_value

logger = logging.getLogger(__name__)

class Memory:
    """
    Persistent memory for storing agent conversations, logs, and other data in SQLite.
    
    This class provides a simple interface to a SQLite database for storing
    and retrieving data, with a focus on agent interactions and analysis results.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the memory.
        
        Args:
            db_path: Path to the SQLite database file (uses config if not provided)
        """
        self.db_path = db_path or get_value(
            "memory.sqlite_path", 
            os.path.join(
                os.path.expanduser("~"),
                ".llamasearch",
                "experimentalagents",
                "storytell",
                "memory.db"
            )
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Initialize database
        self.db = sqlite_utils.Database(self.db_path)
        
        # Initialize tables
        self._init_tables()
    
    def _init_tables(self):
        """Initialize tables in the database."""
        # Conversations table
        if "conversations" not in self.db.table_names():
            self.db["conversations"].create({
                "id": str,
                "title": str,
                "created_at": str,
                "updated_at": str,
                "metadata": str  # JSON string
            }, pk="id")
        
        # Messages table
        if "messages" not in self.db.table_names():
            self.db["messages"].create({
                "id": str,
                "conversation_id": str,
                "role": str,
                "content": str,
                "created_at": str,
                "metadata": str  # JSON string
            }, pk="id")
            self.db["messages"].create_index(["conversation_id"])
        
        # Analysis results table
        if "analysis_results" not in self.db.table_names():
            self.db["analysis_results"].create({
                "id": str,
                "title": str,
                "data": str,  # JSON string
                "created_at": str,
                "metadata": str  # JSON string
            }, pk="id")
        
        # Narratives table
        if "narratives" not in self.db.table_names():
            self.db["narratives"].create({
                "id": str,
                "analysis_id": str,
                "headline": str,
                "findings": str,  # JSON string
                "recommendations": str,  # JSON string
                "created_at": str,
                "metadata": str  # JSON string
            }, pk="id")
            self.db["narratives"].create_index(["analysis_id"])
        
        # Logs table
        if "logs" not in self.db.table_names():
            self.db["logs"].create({
                "id": str,
                "timestamp": str,
                "level": str,
                "component": str,
                "message": str,
                "metadata": str  # JSON string
            }, pk="id")
            self.db["logs"].create_index(["component", "level"])
            self.db["logs"].create_index(["timestamp"])
        
        # Agent tools table
        if "agent_tools" not in self.db.table_names():
            self.db["agent_tools"].create({
                "id": str,
                "message_id": str,
                "tool_name": str,
                "tool_input": str,  # JSON string
                "tool_output": str,  # JSON string
                "created_at": str,
                "success": int,  # Boolean as integer
                "error_message": str
            }, pk="id")
            self.db["agent_tools"].create_index(["message_id"])
            self.db["agent_tools"].create_index(["tool_name"])
    
    def create_conversation(self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Title of the conversation
            metadata: Additional metadata for the conversation
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["conversations"].insert({
            "id": conversation_id,
            "title": title or f"Conversation {now}",
            "created_at": now,
            "updated_at": now,
            "metadata": json.dumps(metadata or {})
        })
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender (user, assistant, system)
            content: Content of the message
            metadata: Additional metadata for the message
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["messages"].insert({
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now,
            "metadata": json.dumps(metadata or {})
        })
        
        # Update conversation updated_at
        self.db["conversations"].update(conversation_id, {
            "updated_at": now
        })
        
        return message_id
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation details.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation details
        """
        conversation = self.db["conversations"].get(conversation_id)
        if conversation:
            conversation["metadata"] = json.loads(conversation["metadata"])
        return conversation
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of messages
        """
        messages = list(self.db["messages"].rows_where(
            "conversation_id = ?", [conversation_id], order_by="created_at"
        ))
        
        for message in messages:
            message["metadata"] = json.loads(message["metadata"])
        
        return messages
    
    def list_conversations(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            
        Returns:
            List of conversations
        """
        conversations = list(self.db["conversations"].rows_where(
            order_by="updated_at desc", limit=limit, offset=offset
        ))
        
        for conversation in conversations:
            conversation["metadata"] = json.loads(conversation["metadata"])
        
        return conversations
    
    def store_analysis_result(self, data: Dict[str, Any], title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an analysis result.
        
        Args:
            data: Analysis result data
            title: Title of the analysis
            metadata: Additional metadata for the analysis
            
        Returns:
            Analysis ID
        """
        analysis_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["analysis_results"].insert({
            "id": analysis_id,
            "title": title or f"Analysis {now}",
            "data": json.dumps(data),
            "created_at": now,
            "metadata": json.dumps(metadata or {})
        })
        
        return analysis_id
    
    def get_analysis_result(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get an analysis result.
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Analysis result
        """
        analysis = self.db["analysis_results"].get(analysis_id)
        if analysis:
            analysis["data"] = json.loads(analysis["data"])
            analysis["metadata"] = json.loads(analysis["metadata"])
        return analysis
    
    def store_narrative(self, analysis_id: str, headline: str, findings: List[str], recommendations: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a narrative summary.
        
        Args:
            analysis_id: ID of the associated analysis
            headline: Headline of the narrative
            findings: Key findings
            recommendations: Recommendations
            metadata: Additional metadata for the narrative
            
        Returns:
            Narrative ID
        """
        narrative_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["narratives"].insert({
            "id": narrative_id,
            "analysis_id": analysis_id,
            "headline": headline,
            "findings": json.dumps(findings),
            "recommendations": json.dumps(recommendations),
            "created_at": now,
            "metadata": json.dumps(metadata or {})
        })
        
        return narrative_id
    
    def get_narrative(self, narrative_id: str) -> Dict[str, Any]:
        """
        Get a narrative summary.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Narrative summary
        """
        narrative = self.db["narratives"].get(narrative_id)
        if narrative:
            narrative["findings"] = json.loads(narrative["findings"])
            narrative["recommendations"] = json.loads(narrative["recommendations"])
            narrative["metadata"] = json.loads(narrative["metadata"])
        return narrative
    
    def get_narratives_for_analysis(self, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Get narrative summaries for an analysis.
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            List of narrative summaries
        """
        narratives = list(self.db["narratives"].rows_where(
            "analysis_id = ?", [analysis_id], order_by="created_at desc"
        ))
        
        for narrative in narratives:
            narrative["findings"] = json.loads(narrative["findings"])
            narrative["recommendations"] = json.loads(narrative["recommendations"])
            narrative["metadata"] = json.loads(narrative["metadata"])
        
        return narratives
    
    def log(self, level: str, component: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a message.
        
        Args:
            level: Log level (info, warning, error, debug)
            component: Component generating the log
            message: Log message
            metadata: Additional metadata for the log
            
        Returns:
            Log ID
        """
        log_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["logs"].insert({
            "id": log_id,
            "timestamp": now,
            "level": level,
            "component": component,
            "message": message,
            "metadata": json.dumps(metadata or {})
        })
        
        return log_id
    
    def get_logs(self, component: Optional[str] = None, level: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get logs.
        
        Args:
            component: Filter by component
            level: Filter by log level
            limit: Maximum number of logs to return
            offset: Offset for pagination
            
        Returns:
            List of logs
        """
        where_clauses = []
        params = []
        
        if component:
            where_clauses.append("component = ?")
            params.append(component)
        
        if level:
            where_clauses.append("level = ?")
            params.append(level)
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        logs = list(self.db["logs"].rows_where(
            where_clause, params, order_by="timestamp desc", limit=limit, offset=offset
        ))
        
        for log in logs:
            log["metadata"] = json.loads(log["metadata"])
        
        return logs
    
    def record_tool_use(self, message_id: str, tool_name: str, tool_input: Dict[str, Any], tool_output: Optional[Dict[str, Any]] = None, success: bool = True, error_message: Optional[str] = None) -> str:
        """
        Record a tool use.
        
        Args:
            message_id: ID of the message associated with the tool use
            tool_name: Name of the tool
            tool_input: Input to the tool
            tool_output: Output from the tool
            success: Whether the tool use was successful
            error_message: Error message if the tool use failed
            
        Returns:
            Tool use ID
        """
        tool_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db["agent_tools"].insert({
            "id": tool_id,
            "message_id": message_id,
            "tool_name": tool_name,
            "tool_input": json.dumps(tool_input),
            "tool_output": json.dumps(tool_output or {}),
            "created_at": now,
            "success": 1 if success else 0,
            "error_message": error_message or ""
        })
        
        return tool_id
    
    def get_tool_uses(self, message_id: Optional[str] = None, tool_name: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get tool uses.
        
        Args:
            message_id: Filter by message ID
            tool_name: Filter by tool name
            limit: Maximum number of tool uses to return
            offset: Offset for pagination
            
        Returns:
            List of tool uses
        """
        where_clauses = []
        params = []
        
        if message_id:
            where_clauses.append("message_id = ?")
            params.append(message_id)
        
        if tool_name:
            where_clauses.append("tool_name = ?")
            params.append(tool_name)
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        tool_uses = list(self.db["agent_tools"].rows_where(
            where_clause, params, order_by="created_at desc", limit=limit, offset=offset
        ))
        
        for tool_use in tool_uses:
            tool_use["tool_input"] = json.loads(tool_use["tool_input"])
            tool_use["tool_output"] = json.loads(tool_use["tool_output"])
            tool_use["success"] = bool(tool_use["success"])
        
        return tool_uses
    
    def clear_data(self, table: Optional[str] = None):
        """
        Clear data from the memory.
        
        Args:
            table: Table to clear (clears all tables if None)
        """
        if table:
            if table in self.db.table_names():
                self.db[table].delete_where("1=1")
        else:
            for table_name in self.db.table_names():
                self.db[table_name].delete_where("1=1")

    def export_datasette(self, output_dir: Optional[str] = None) -> str:
        """
        Export a Datasette configuration for exploring the memory database.
        
        Args:
            output_dir: Directory to save the Datasette configuration to
            
        Returns:
            Path to the Datasette configuration file
        """
        output_dir = output_dir or os.path.dirname(self.db_path)
        os.makedirs(output_dir, exist_ok=True)
        
        config_path = os.path.join(output_dir, "datasette.json")
        config = {
            "title": "LlamaSearch ExperimentalAgents: StoryTell Memory",
            "description": "Agent conversations, analysis results, and logs",
            "databases": {
                "memory": {
                    "path": self.db_path,
                    "tables": {
                        "conversations": {
                            "title": "Conversations",
                            "sort_desc": "updated_at"
                        },
                        "messages": {
                            "title": "Messages",
                            "sort_desc": "created_at",
                            "foreign_keys": {
                                "conversation_id": "conversations.id"
                            }
                        },
                        "analysis_results": {
                            "title": "Analysis Results",
                            "sort_desc": "created_at"
                        },
                        "narratives": {
                            "title": "Narrative Summaries",
                            "sort_desc": "created_at",
                            "foreign_keys": {
                                "analysis_id": "analysis_results.id"
                            }
                        },
                        "logs": {
                            "title": "Logs",
                            "sort_desc": "timestamp"
                        },
                        "agent_tools": {
                            "title": "Agent Tool Uses",
                            "sort_desc": "created_at",
                            "foreign_keys": {
                                "message_id": "messages.id"
                            }
                        }
                    }
                }
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Create launch script
        script_path = os.path.join(output_dir, "launch_datasette.sh")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Launch Datasette to explore the memory database\n")
            f.write("datasette serve {} -m datasette.json --metadata datasette.json\n".format(self.db_path))
        
        os.chmod(script_path, 0o755)  # Make executable
        
        return config_path 