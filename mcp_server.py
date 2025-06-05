#!/usr/bin/env python3
"""
MCP Server for RL-A2A Agent Communication Protocol
Exposes agent communication and management tools via Model Context Protocol
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Union
from mcp.server.models import InitializeResult
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

# Import your existing A2A components
import requests
import msgpack
import numpy as np
from dataclasses import dataclass

# Configuration
A2A_SERVER_URL = "http://localhost:8000"
MCP_SERVER_NAME = "rl-a2a-mcp-server"
MCP_SERVER_VERSION = "1.0.0"

class A2AClient:
    """Client for interacting with A2A server from MCP"""
    
    def __init__(self, server_url: str = A2A_SERVER_URL):
        self.server_url = server_url
        self.session_id = None
        self.agent_id = None
    
    async def register_agent(self, agent_id: str) -> Dict[str, str]:
        """Register an agent with the A2A server"""
        try:
            response = requests.post(
                f"{self.server_url}/register",
                params={"agent_id": agent_id}
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            self.agent_id = data["agent_id"]
            return data
        except Exception as e:
            raise Exception(f"Failed to register agent: {str(e)}")
    
    async def send_feedback(self, agent_id: str, action_id: str, reward: float, context: Dict[str, Any]) -> bool:
        """Send feedback to A2A server for reinforcement learning"""
        try:
            response = requests.post(
                f"{self.server_url}/feedback",
                json={
                    "agent_id": agent_id,
                    "action_id": action_id,
                    "reward": reward,
                    "context": context
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise Exception(f"Failed to send feedback: {str(e)}")

# Initialize A2A client
a2a_client = A2AClient()

# Create the MCP server
server = Server(MCP_SERVER_NAME)

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available A2A resources"""
    return [
        Resource(
            uri="a2a://agents",
            name="Active Agents",
            description="Information about currently active agents in the A2A system",
            mimeType="application/json"
        ),
        Resource(
            uri="a2a://models", 
            name="RL Models",
            description="Reinforcement learning models for agents",
            mimeType="application/json"
        ),
        Resource(
            uri="a2a://communication",
            name="Agent Communication Log",
            description="Recent agent communications and messages",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read A2A system resources"""
    if uri == "a2a://agents":
        # Return information about active agents
        return json.dumps({
            "active_agents": [],
            "total_sessions": 0,
            "server_status": "running"
        })
    elif uri == "a2a://models":
        # Return RL model information
        return json.dumps({
            "models": {},
            "learning_progress": {}
        })
    elif uri == "a2a://communication":
        # Return communication logs
        return json.dumps({
            "recent_messages": [],
            "message_count": 0
        })
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available A2A tools"""
    return [
        Tool(
            name="register_agent",
            description="Register a new agent with the A2A communication system",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Unique identifier for the agent"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="send_agent_feedback",
            description="Send reinforcement learning feedback for an agent action",
            inputSchema={
                "type": "object", 
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the agent providing feedback"
                    },
                    "action_id": {
                        "type": "string",
                        "description": "ID of the action being evaluated"
                    },
                    "reward": {
                        "type": "number",
                        "description": "Reward value (-1.0 to 1.0)"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context information",
                        "properties": {
                            "emotion": {"type": "string"},
                            "environment": {"type": "string"},
                            "success": {"type": "boolean"}
                        }
                    }
                },
                "required": ["agent_id", "action_id", "reward"]
            }
        ),
        Tool(
            name="create_agent_observation",
            description="Create an observation message for agent communication",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent sending the observation"
                    },
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "description": "Agent position coordinates"
                    },
                    "velocity": {
                        "type": "object", 
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "description": "Agent velocity vector"
                    },
                    "emotion": {
                        "type": "string",
                        "enum": ["happy", "sad", "neutral", "excited", "angry"],
                        "description": "Current emotional state"
                    },
                    "target_agent_id": {
                        "type": "string",
                        "description": "Optional target agent for direct communication"
                    }
                },
                "required": ["agent_id", "position", "velocity", "emotion"]
            }
        ),
        Tool(
            name="start_a2a_server",
            description="Start the A2A communication server",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {
                        "type": "number",
                        "default": 8000,
                        "description": "Port to run the server on"
                    },
                    "host": {
                        "type": "string", 
                        "default": "localhost",
                        "description": "Host to bind the server to"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls for A2A operations"""
    
    if name == "register_agent":
        agent_id = arguments.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required")
        
        try:
            result = await a2a_client.register_agent(agent_id)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "session_id": result["session_id"],
                        "agent_id": result["agent_id"],
                        "message": f"Agent {agent_id} registered successfully"
                    }, indent=2)
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", 
                    text=json.dumps({
                        "success": False,
                        "error": str(e)
                    }, indent=2)
                )
            ]
    
    elif name == "send_agent_feedback":
        agent_id = arguments.get("agent_id")
        action_id = arguments.get("action_id")
        reward = arguments.get("reward")
        context = arguments.get("context", {})
        
        try:
            await a2a_client.send_feedback(agent_id, action_id, reward, context)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Feedback sent for agent {agent_id}, action {action_id}",
                        "reward": reward,
                        "context": context
                    }, indent=2)
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e)
                    }, indent=2)
                )
            ]
    
    elif name == "create_agent_observation":
        observation = {
            "agent_id": arguments.get("agent_id"),
            "position": arguments.get("position", {}),
            "velocity": arguments.get("velocity", {}),
            "emotion": arguments.get("emotion"),
            "target_agent_id": arguments.get("target_agent_id"),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "observation": observation,
                    "message": "Agent observation created successfully"
                }, indent=2)
            )
        ]
    
    elif name == "start_a2a_server":
        port = arguments.get("port", 8000)
        host = arguments.get("host", "localhost")
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"A2A server configuration prepared for {host}:{port}",
                    "note": "Use 'uvicorn a2a_server:app --host {host} --port {port}' to start the server",
                    "command": f"uvicorn a2a_server:app --host {host} --port {port}"
                }, indent=2)
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the MCP server"""
    # Run the MCP server using stdin/stdout streams
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializeResult(
                serverName=MCP_SERVER_NAME,
                serverVersion=MCP_SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())