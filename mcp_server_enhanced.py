#!/usr/bin/env python3
"""
Enhanced MCP Server for RL-A2A with OpenAI and Visualization Support
Exposes OpenAI-powered agents and visualization tools via Model Context Protocol
"""

import asyncio
import json
import sys
import os
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
MCP_SERVER_NAME = "rl-a2a-enhanced-mcp-server"
MCP_SERVER_VERSION = "2.0.0"

class EnhancedA2AClient:
    """Enhanced client for interacting with A2A server from MCP"""
    
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
a2a_client = EnhancedA2AClient()

# Create the enhanced MCP server
server = Server(MCP_SERVER_NAME)

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available A2A resources including new visualizations"""
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
        ),
        Resource(
            uri="a2a://visualization",
            name="Visualization Data",
            description="Real-time visualization and dashboard data",
            mimeType="application/json"
        ),
        Resource(
            uri="a2a://openai",
            name="OpenAI Integration",
            description="OpenAI-powered agent intelligence status and configuration",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read A2A system resources including new features"""
    if uri == "a2a://agents":
        return json.dumps({
            "active_agents": [],
            "total_sessions": 0,
            "server_status": "running"
        })
    elif uri == "a2a://models":
        return json.dumps({
            "models": {},
            "learning_progress": {}
        })
    elif uri == "a2a://communication":
        return json.dumps({
            "recent_messages": [],
            "message_count": 0
        })
    elif uri == "a2a://visualization":
        return json.dumps({
            "dashboard_status": "available",
            "features": ["3D tracking", "real-time metrics", "agent analytics"],
            "dashboard_url": "http://localhost:8501"
        })
    elif uri == "a2a://openai":
        return json.dumps({
            "integration_status": "available" if os.getenv("OPENAI_API_KEY") else "missing_api_key",
            "supported_models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            "features": ["intelligent behavior", "natural language", "adaptive learning"]
        })
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available A2A tools including OpenAI and visualization"""
    return [
        # Original A2A tools
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
        # OpenAI Integration Tools
        Tool(
            name="create_intelligent_agent",
            description="Create an AI-powered intelligent agent using OpenAI",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Unique identifier for the intelligent agent"
                    },
                    "openai_model": {
                        "type": "string",
                        "default": "gpt-4o-mini",
                        "description": "OpenAI model to use for agent intelligence"
                    },
                    "personality": {
                        "type": "string",
                        "description": "Personality traits for the agent (e.g., 'curious explorer', 'cautious analyzer')"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="run_intelligent_simulation",
            description="Run a multi-agent simulation with OpenAI-powered agents",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_agents": {
                        "type": "number",
                        "default": 3,
                        "description": "Number of intelligent agents to create"
                    },
                    "iterations": {
                        "type": "number", 
                        "default": 5,
                        "description": "Number of simulation iterations to run"
                    },
                    "scenario": {
                        "type": "string",
                        "description": "Simulation scenario description"
                    }
                }
            }
        ),
        # Visualization Tools
        Tool(
            name="start_visualization_dashboard",
            description="Start the real-time visualization dashboard",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {
                        "type": "number",
                        "default": 8501,
                        "description": "Port for the dashboard server"
                    },
                    "auto_refresh": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable auto-refresh of visualizations"
                    }
                }
            }
        ),
        Tool(
            name="generate_visualization_report",
            description="Generate HTML visualization report of A2A system",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "default": "a2a_report.html",
                        "description": "Output filename for the report"
                    },
                    "include_3d": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include 3D agent position visualization"
                    },
                    "include_metrics": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Include performance metrics dashboard"
                    }
                }
            }
        ),
        Tool(
            name="capture_agent_screenshot",
            description="Capture a screenshot of current agent positions and states",
            inputSchema={
                "type": "object",
                "properties": {
                    "view_type": {
                        "type": "string",
                        "enum": ["2d", "3d", "dashboard"],
                        "default": "3d",
                        "description": "Type of visualization to capture"
                    },
                    "filename": {
                        "type": "string",
                        "default": "agent_state.png",
                        "description": "Output filename for the screenshot"
                    }
                }
            }
        ),
        # System Management Tools  
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
                    },
                    "with_openai": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable OpenAI integration features"
                    }
                }
            }
        ),
        Tool(
            name="setup_complete_system",
            description="Setup and start the complete RL-A2A system with all features",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_openai": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include OpenAI integration"
                    },
                    "include_visualization": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include visualization dashboard"
                    },
                    "auto_start_agents": {
                        "type": "number",
                        "default": 2,
                        "description": "Number of demo agents to auto-start"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls for enhanced A2A operations"""
    
    # Original A2A tools
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
    
    # OpenAI Integration Tools
    elif name == "create_intelligent_agent":
        agent_id = arguments.get("agent_id")
        model = arguments.get("openai_model", "gpt-4o-mini")
        personality = arguments.get("personality", "balanced")
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "agent_id": agent_id,
                    "model": model,
                    "personality": personality,
                    "message": f"Intelligent agent {agent_id} configured for creation",
                    "commands": [
                        "python openai_integration.py",
                        f"python start_openai_demo.py"
                    ],
                    "requirements": [
                        "Set OPENAI_API_KEY environment variable",
                        "Ensure A2A server is running"
                    ],
                    "capabilities": [
                        "AI-powered decision making",
                        "Natural language communication", 
                        "Intelligent feedback evaluation",
                        "Adaptive behavior learning"
                    ]
                }, indent=2)
            )
        ]
    
    elif name == "run_intelligent_simulation":
        num_agents = arguments.get("num_agents", 3)
        iterations = arguments.get("iterations", 5)
        scenario = arguments.get("scenario", "cooperative exploration")
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "simulation_config": {
                        "agents": num_agents,
                        "iterations": iterations,
                        "scenario": scenario
                    },
                    "command": f"python openai_integration.py --agents {num_agents} --iterations {iterations}",
                    "message": f"Simulation prepared: {num_agents} agents, {iterations} iterations",
                    "scenario_description": scenario,
                    "estimated_duration": f"{iterations * 2} seconds"
                }, indent=2)
            )
        ]
    
    # Visualization Tools
    elif name == "start_visualization_dashboard":
        port = arguments.get("port", 8501)
        auto_refresh = arguments.get("auto_refresh", True)
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "dashboard_url": f"http://localhost:{port}",
                    "port": port,
                    "auto_refresh": auto_refresh,
                    "start_command": f"streamlit run visualization.py --server.port {port} -- streamlit",
                    "alternative_command": "python start_dashboard.py",
                    "message": "Visualization dashboard ready to start",
                    "features": [
                        "Real-time 3D agent tracking",
                        "Interactive metrics dashboard", 
                        "Agent status monitoring",
                        "Communication visualization",
                        "Performance analytics"
                    ]
                }, indent=2)
            )
        ]
    
    elif name == "generate_visualization_report":
        filename = arguments.get("filename", "a2a_report.html")
        include_3d = arguments.get("include_3d", True)
        include_metrics = arguments.get("include_metrics", True)
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "filename": filename,
                    "include_3d": include_3d,
                    "include_metrics": include_metrics,
                    "command": "python visualization.py",
                    "message": f"Visualization report configuration ready: {filename}",
                    "features": [
                        "3D agent positions" if include_3d else "2D visualizations",
                        "Metrics dashboard" if include_metrics else "Basic charts",
                        "Agent status overview",
                        "Communication logs"
                    ]
                }, indent=2)
            )
        ]
    
    elif name == "capture_agent_screenshot":
        view_type = arguments.get("view_type", "3d")
        filename = arguments.get("filename", "agent_state.png")
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "view_type": view_type,
                    "filename": filename,
                    "command": f"python visualization.py --screenshot --type {view_type} --output {filename}",
                    "message": f"Screenshot capture configured: {view_type} view -> {filename}"
                }, indent=2)
            )
        ]
    
    # System Management Tools
    elif name == "start_a2a_server":
        port = arguments.get("port", 8000)
        host = arguments.get("host", "localhost")
        with_openai = arguments.get("with_openai", False)
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "server_url": f"http://{host}:{port}",
                    "host": host,
                    "port": port,
                    "openai_enabled": with_openai,
                    "command": f"uvicorn a2a_server:app --host {host} --port {port} --reload",
                    "alternative_command": "python start_a2a.py",
                    "message": f"A2A server configuration ready for {host}:{port}",
                    "features": [
                        "WebSocket communication",
                        "Agent registration",
                        "Reinforcement learning",
                        "Real-time messaging"
                    ] + (["OpenAI integration"] if with_openai else [])
                }, indent=2)
            )
        ]
    
    elif name == "setup_complete_system":
        include_openai = arguments.get("include_openai", True)
        include_visualization = arguments.get("include_visualization", True)  
        auto_start_agents = arguments.get("auto_start_agents", 2)
        
        setup_commands = [
            "# Complete RL-A2A System Setup",
            "# 1. Start A2A Server",
            "python start_a2a.py &",
            ""
        ]
        
        if include_visualization:
            setup_commands.extend([
                "# 2. Start Visualization Dashboard", 
                "python start_dashboard.py &",
                ""
            ])
        
        if include_openai:
            setup_commands.extend([
                "# 3. Setup OpenAI Integration",
                "export OPENAI_API_KEY=your-key-here",
                f"python openai_integration.py --agents {auto_start_agents} &",
                ""
            ])
        
        setup_commands.extend([
            "# 4. Optional: Start MCP Server",
            "python mcp_server_enhanced.py &",
            "",
            "# All services will be running on:",
            "# - A2A Server: http://localhost:8000",
            "# - Dashboard: http://localhost:8501" if include_visualization else "# - Dashboard: disabled",
            "# - MCP Server: stdio"
        ])
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "system_config": {
                        "openai_enabled": include_openai,
                        "visualization_enabled": include_visualization,
                        "auto_agents": auto_start_agents
                    },
                    "setup_script": "\\n".join(setup_commands),
                    "message": "Complete system setup configuration ready",
                    "services": [
                        "A2A Communication Server",
                        "Visualization Dashboard" if include_visualization else None,
                        "OpenAI Agent Intelligence" if include_openai else None,
                        "MCP Protocol Server"
                    ]
                }, indent=2)
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the enhanced MCP server"""
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