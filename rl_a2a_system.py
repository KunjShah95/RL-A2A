#!/usr/bin/env python3
"""
Complete RL-A2A System: Agent-to-Agent Communication with OpenAI and Visualization
Combines A2A server, OpenAI integration, visualization, and MCP support in one file
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue
import signal
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import subprocess
import requests
import msgpack
import numpy as np

# Core dependencies
try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    import websockets
    
    # OpenAI integration
    from openai import AsyncOpenAI
    
    # Visualization dependencies
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    import pandas as pd
    
    # MCP dependencies
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent
    import mcp.types as types
    
    print("✅ All dependencies loaded successfully")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install -r requirements-full.txt")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:8000")
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "localhost")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AgentState:
    """Agent state for A2A system"""
    agent_id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    timestamp: float
    session_id: str = ""
    action: str = ""
    reward: float = 0.0

@dataclass
class Communication:
    """Communication event"""
    sender: str
    receiver: str
    message: str
    timestamp: float
    message_type: str = "direct"

@dataclass
class AgentContext:
    """Context for OpenAI agent intelligence"""
    agent_id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    observations: List[Dict]
    recent_actions: List[str]
    feedback_history: List[float]

class RegistrationRequest(BaseModel):
    """Agent registration request"""
    agent_id: str = Field(..., description="Unique agent identifier")

class FeedbackRequest(BaseModel):
    """RL feedback request"""
    agent_id: str
    action_id: str
    reward: float = Field(..., ge=-1.0, le=1.0)
    context: Dict[str, Any] = {}

# =============================================================================
# A2A COMMUNICATION SERVER
# =============================================================================

class A2AServer:
    """Complete A2A communication server"""
    
    def __init__(self):
        self.app = FastAPI(title="RL-A2A System", version="2.0.0")
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_sessions: Dict[str, dict] = {}
        self.agent_models: Dict[str, dict] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/register")
        async def register_agent(agent_id: str):
            """Register a new agent"""
            session_id = str(uuid.uuid4())
            
            self.agent_sessions[session_id] = {
                "agent_id": agent_id,
                "session_id": session_id,
                "created_at": time.time(),
                "last_seen": time.time()
            }
            
            # Initialize Q-learning model
            self.agent_models[agent_id] = {
                "q_table": np.random.rand(5, 4) * 0.1,
                "epsilon": 0.1,
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "total_rewards": 0.0,
                "action_count": 0
            }
            
            self.message_queues[agent_id] = queue.Queue()
            
            print(f"✅ Agent {agent_id} registered with session {session_id}")
            return {"session_id": session_id, "agent_id": agent_id}
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication"""
            await websocket.accept()
            self.active_connections[session_id] = websocket
            
            try:
                while True:
                    data = await websocket.receive_bytes()
                    parsed_data = msgpack.unpackb(data)
                    
                    # Process agent observation
                    agent_id = parsed_data.get("agent_id")
                    if agent_id and agent_id in self.agent_models:
                        # Simple RL action selection
                        model = self.agent_models[agent_id]
                        state_idx = np.random.randint(0, 5)
                        
                        if np.random.random() < model["epsilon"]:
                            action_idx = np.random.randint(0, 4)
                        else:
                            action_idx = np.argmax(model["q_table"][state_idx])
                        
                        actions = ["move_forward", "turn_left", "turn_right", "communicate", "observe"]
                        selected_action = actions[action_idx]
                        
                        response = {
                            "agent_id": agent_id,
                            "command": selected_action,
                            "message": f"Action for {agent_id}",
                            "success_probability": float(np.random.random())
                        }
                        
                        await websocket.send(msgpack.packb(response))
                        model["action_count"] += 1
            
            except Exception as e:
                print(f"WebSocket error for {session_id}: {e}")
            finally:
                if session_id in self.active_connections:
                    del self.active_connections[session_id]
        
        @self.app.post("/feedback")
        async def receive_feedback(request: FeedbackRequest):
            """Receive RL feedback"""
            agent_id = request.agent_id
            
            if agent_id in self.agent_models:
                model = self.agent_models[agent_id]
                model["total_rewards"] += request.reward
                
                # Simple Q-learning update
                state_idx = np.random.randint(0, 5)
                action_idx = np.random.randint(0, 4)
                
                old_q = model["q_table"][state_idx, action_idx]
                model["q_table"][state_idx, action_idx] = old_q + model["learning_rate"] * (
                    request.reward - old_q
                )
                
                print(f"✅ Feedback received: {agent_id} -> {request.reward}")
                return {"success": True, "message": "Feedback processed"}
            
            return {"success": False, "message": "Agent not found"}
    
    async def start_server(self):
        """Start the A2A server"""
        config = uvicorn.Config(
            self.app,
            host=A2A_SERVER_HOST,
            port=A2A_SERVER_PORT,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# =============================================================================
# OPENAI INTELLIGENCE
# =============================================================================

class OpenAIAgentBrain:
    """OpenAI-powered agent intelligence"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.model = model
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        self.system_prompt = """You are an intelligent autonomous agent in a multi-agent system. 
        Analyze situations, make strategic decisions, and communicate effectively with other agents.
        Always respond in JSON format with clear action recommendations."""
    
    async def analyze_situation(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze current situation and generate intelligent response"""
        if not self.client:
            return {
                "situation_analysis": "OpenAI not available",
                "recommended_action": "observe",
                "confidence": 0.1
            }
        
        situation_prompt = f"""
        Agent Status:
        - ID: {context.agent_id}
        - Position: {context.position}
        - Emotion: {context.emotion}
        - Recent Actions: {context.recent_actions[-3:]}
        - Feedback: {context.feedback_history[-3:]}
        
        Provide analysis and action recommendation in JSON format.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": situation_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return {
                "error": str(e),
                "situation_analysis": "AI analysis failed",
                "recommended_action": "continue_previous_behavior",
                "confidence": 0.0
            }

class IntelligentAgent:
    """OpenAI-powered intelligent agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.brain = OpenAIAgentBrain()
        self.context = AgentContext(
            agent_id=agent_id,
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            velocity={"x": 0.0, "y": 0.0, "z": 0.0},
            emotion="neutral",
            observations=[],
            recent_actions=[],
            feedback_history=[]
        )
        self.session_id = None
    
    async def register_with_a2a(self) -> bool:
        """Register with A2A server"""
        try:
            response = requests.post(
                f"{A2A_SERVER_URL}/register",
                params={"agent_id": self.agent_id}
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            return True
        except Exception as e:
            print(f"Registration failed: {e}")
            return False
    
    async def think_and_act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and generate intelligent response"""
        self.context.observations.append(observation)
        if len(self.context.observations) > 10:
            self.context.observations = self.context.observations[-10:]
        
        analysis = await self.brain.analyze_situation(self.context)
        
        action = {
            "agent_id": self.agent_id,
            "action": analysis.get("recommended_action", "observe"),
            "reasoning": analysis.get("situation_analysis", ""),
            "confidence": analysis.get("confidence", 0.5),
            "timestamp": time.time()
        }
        
        self.context.recent_actions.append(action["action"])
        if len(self.context.recent_actions) > 20:
            self.context.recent_actions = self.context.recent_actions[-20:]
        
        return action

# =============================================================================
# VISUALIZATION SYSTEM
# =============================================================================

class DataCollector:
    """Collects data from A2A system for visualization"""
    
    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.communications: List[Communication] = []
        self.rewards: List[Tuple[str, float, float]] = []
        self.is_collecting = False
        self.data_queue = queue.Queue()
    
    def start_collection(self):
        """Start data collection"""
        self.is_collecting = True
        # Simulate data collection for demo
        threading.Thread(target=self._simulate_data, daemon=True).start()
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
    
    def _simulate_data(self):
        """Simulate agent data for visualization"""
        agent_count = 3
        
        while self.is_collecting:
            for i in range(agent_count):
                agent_id = f"agent_{i+1}"
                
                # Simulate agent movement
                if agent_id not in self.agents:
                    self.agents[agent_id] = AgentState(
                        agent_id=agent_id,
                        position={"x": np.random.uniform(-10, 10), "y": np.random.uniform(-10, 10), "z": 0},
                        velocity={"x": 0, "y": 0, "z": 0},
                        emotion=np.random.choice(["happy", "neutral", "excited"]),
                        timestamp=time.time()
                    )
                else:
                    agent = self.agents[agent_id]
                    # Update position
                    agent.position["x"] += np.random.uniform(-1, 1)
                    agent.position["y"] += np.random.uniform(-1, 1)
                    agent.emotion = np.random.choice(["happy", "neutral", "excited", "sad"])
                    agent.timestamp = time.time()
            
            time.sleep(1)

class PlotlyVisualizer:
    """Plotly-based visualization system"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
    
    def create_3d_agent_plot(self) -> go.Figure:
        """Create 3D visualization of agent positions"""
        fig = go.Figure()
        
        if self.data_collector.agents:
            agents = list(self.data_collector.agents.values())
            
            emotion_colors = {
                "happy": "gold",
                "sad": "blue",
                "neutral": "gray",
                "excited": "orange",
                "angry": "red"
            }
            
            fig.add_trace(go.Scatter3d(
                x=[agent.position["x"] for agent in agents],
                y=[agent.position["y"] for agent in agents],
                z=[agent.position["z"] for agent in agents],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=[emotion_colors.get(agent.emotion, "gray") for agent in agents],
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                text=[agent.agent_id for agent in agents],
                textposition="top center",
                name='Agents'
            ))
        
        fig.update_layout(
            title="RL-A2A Agent Positions",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position"
            ),
            height=600
        )
        
        return fig
    
    def create_metrics_dashboard(self) -> go.Figure:
        """Create metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Agent Emotions", "Agent Positions", "Activity", "System Status"),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        if self.data_collector.agents:
            agents = list(self.data_collector.agents.values())
            
            # Emotions pie chart
            emotions = [agent.emotion for agent in agents]
            emotion_counts = pd.Series(emotions).value_counts()
            
            fig.add_trace(
                go.Pie(values=emotion_counts.values, labels=emotion_counts.index),
                row=1, col=1
            )
            
            # Agent positions
            fig.add_trace(
                go.Scatter(
                    x=[agent.position["x"] for agent in agents],
                    y=[agent.position["y"] for agent in agents],
                    mode="markers+text",
                    text=[agent.agent_id for agent in agents],
                    textposition="top center"
                ),
                row=1, col=2
            )
            
            # Activity bars
            fig.add_trace(
                go.Bar(
                    x=[agent.agent_id for agent in agents],
                    y=[1 for _ in agents]  # Placeholder activity
                ),
                row=2, col=1
            )
            
            # System status indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=len(agents),
                    title={"text": "Active Agents"},
                    gauge={"bar": {"color": "green"}}
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="RL-A2A System Dashboard")
        return fig
    
    def save_html_report(self, filename: str = "a2a_report.html"):
        """Generate HTML report"""
        fig_3d = self.create_3d_agent_plot()
        fig_metrics = self.create_metrics_dashboard()
        
        # Create combined HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL-A2A System Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RL-A2A System Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Active Agents: {len(self.data_collector.agents)}</p>
            </div>
            <div class="plot" id="plot3d"></div>
            <div class="plot" id="metrics"></div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        # Save individual plots
        pyo.plot(fig_3d, filename=filename, auto_open=False)
        print(f"✅ Visualization report saved: {filename}")

# =============================================================================
# MCP SERVER INTEGRATION
# =============================================================================

class MCPServer:
    """Enhanced MCP server for RL-A2A system"""
    
    def __init__(self):
        self.server = Server("rl-a2a-complete")
        self.data_collector = DataCollector()
        self.visualizer = PlotlyVisualizer(self.data_collector)
        self.setup_mcp_handlers()
    
    def setup_mcp_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="a2a://system",
                    name="A2A System Status",
                    description="Complete RL-A2A system information",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "a2a://system":
                return json.dumps({
                    "system": "RL-A2A Complete",
                    "agents": len(self.data_collector.agents),
                    "features": ["A2A Communication", "OpenAI Intelligence", "Visualization", "MCP Integration"],
                    "status": "running"
                })
            raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="start_a2a_system",
                    description="Start the complete RL-A2A system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_openai": {"type": "boolean", "default": True},
                            "include_visualization": {"type": "boolean", "default": True},
                            "num_demo_agents": {"type": "number", "default": 3}
                        }
                    }
                ),
                Tool(
                    name="create_agent",
                    description="Create a new agent in the system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "intelligent": {"type": "boolean", "default": True, "description": "Use OpenAI intelligence"}
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="generate_report",
                    description="Generate visualization report",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "default": "a2a_report.html"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            
            if name == "start_a2a_system":
                include_openai = arguments.get("include_openai", True)
                include_viz = arguments.get("include_visualization", True)
                num_agents = arguments.get("num_demo_agents", 3)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": "RL-A2A system configuration ready",
                        "command": f"python rl_a2a_system.py --mode complete --agents {num_agents}",
                        "features": {
                            "openai": include_openai,
                            "visualization": include_viz,
                            "demo_agents": num_agents
                        }
                    }, indent=2)
                )]
            
            elif name == "create_agent":
                agent_id = arguments.get("agent_id")
                intelligent = arguments.get("intelligent", True)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "agent_id": agent_id,
                        "intelligent": intelligent,
                        "message": f"Agent {agent_id} configured",
                        "type": "AI-powered" if intelligent else "Basic"
                    }, indent=2)
                )]
            
            elif name == "generate_report":
                filename = arguments.get("filename", "a2a_report.html")
                self.data_collector.start_collection()
                await asyncio.sleep(2)
                self.visualizer.save_html_report(filename)
                self.data_collector.stop_collection()
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "filename": filename,
                        "message": "Visualization report generated"
                    }, indent=2)
                )]
            
            raise ValueError(f"Unknown tool: {name}")

# =============================================================================
# MAIN SYSTEM ORCHESTRATOR
# =============================================================================

class RLA2ASystem:
    """Complete RL-A2A system orchestrator"""
    
    def __init__(self):
        self.a2a_server = A2AServer()
        self.data_collector = DataCollector()
        self.visualizer = PlotlyVisualizer(self.data_collector)
        self.mcp_server = MCPServer()
        self.agents: List[IntelligentAgent] = []
        self.running = False
    
    async def start_complete_system(self, num_agents: int = 3, include_viz: bool = True):
        """Start the complete system"""
        print("🚀 Starting Complete RL-A2A System")
        print("="*50)
        
        self.running = True
        
        # Start A2A server in background
        server_task = asyncio.create_task(self.a2a_server.start_server())
        print(f"✅ A2A Server starting on {A2A_SERVER_HOST}:{A2A_SERVER_PORT}")
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Start data collection
        self.data_collector.start_collection()
        print("✅ Data collection started")
        
        # Create demo agents
        if OPENAI_API_KEY:
            print(f"🧠 Creating {num_agents} intelligent agents...")
            for i in range(num_agents):
                agent = IntelligentAgent(f"ai_agent_{i+1}")
                success = await agent.register_with_a2a()
                if success:
                    self.agents.append(agent)
                    print(f"✅ Agent {agent.agent_id} created and registered")
        else:
            print("⚠️  OpenAI API key not set - using basic agents")
        
        # Generate initial report
        if include_viz:
            print("📊 Generating visualization report...")
            await asyncio.sleep(3)  # Let some data collect
            self.visualizer.save_html_report()
            print("✅ Report generated: a2a_report.html")
        
        print("\\n🎉 RL-A2A System is running!")
        print(f"   • A2A Server: http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}")
        print(f"   • Visualization: a2a_report.html")
        print(f"   • Active Agents: {len(self.agents)}")
        print("   • Press Ctrl+C to stop")
        
        try:
            await server_task
        except asyncio.CancelledError:
            print("\\n🛑 System stopped")
        except KeyboardInterrupt:
            print("\\n🛑 System interrupted")
        finally:
            self.cleanup()
    
    async def run_mcp_server(self):
        """Run MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.server.run(
                read_stream,
                write_stream,
                InitializeResult(
                    serverName="rl-a2a-complete",
                    serverVersion="2.0.0",
                    capabilities=self.mcp_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.data_collector.stop_collection()
        print("✅ System cleanup completed")

# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL-A2A Complete System")
    parser.add_argument("--mode", choices=["server", "complete", "mcp", "demo"], 
                       default="complete", help="Run mode")
    parser.add_argument("--agents", type=int, default=3, help="Number of demo agents")
    parser.add_argument("--host", default=A2A_SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=A2A_SERVER_PORT, help="Server port")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Update globals
    global A2A_SERVER_HOST, A2A_SERVER_PORT
    A2A_SERVER_HOST = args.host
    A2A_SERVER_PORT = args.port
    
    system = RLA2ASystem()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\\n🛑 Shutting down...")
        system.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.mode == "server":
            print("Starting A2A Server only...")
            await system.a2a_server.start_server()
        
        elif args.mode == "mcp":
            print("Starting MCP Server...")
            await system.run_mcp_server()
        
        elif args.mode == "demo":
            print("Running demo simulation...")
            system.data_collector.start_collection()
            await asyncio.sleep(10)
            system.visualizer.save_html_report("demo_report.html")
            system.data_collector.stop_collection()
            print("✅ Demo completed: demo_report.html")
        
        else:  # complete mode
            await system.start_complete_system(
                num_agents=args.agents,
                include_viz=not args.no_viz
            )
    
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())