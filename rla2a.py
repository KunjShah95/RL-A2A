"""
RL-A2A: Complete Agent-to-Agent Communication System
All-in-one file with A2A server, OpenAI agents, visualization, and MCP support
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue
import signal
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import uuid
import argparse

# Core dependencies check and import
def check_and_install_dependencies():
    """Check and install required dependencies"""
    required = [
        "fastapi", "uvicorn", "websockets", "msgpack", "numpy", "pydantic", 
        "requests", "openai", "matplotlib", "plotly", "streamlit", "pandas", "mcp"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing, stdout=subprocess.DEVNULL)
            print("✅ Dependencies installed")
        except:
            print(f"❌ Please install manually: pip install {' '.join(missing)}")
            sys.exit(1)

check_and_install_dependencies()

# Now import everything
import requests
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import websockets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# OpenAI and MCP imports (with fallbacks)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "SERVER_HOST": os.getenv("A2A_HOST", "localhost"),
    "SERVER_PORT": int(os.getenv("A2A_PORT", "8000")),
    "DASHBOARD_PORT": int(os.getenv("DASHBOARD_PORT", "8501")),
    "DEBUG": os.getenv("DEBUG", "false").lower() == "true"
}

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Agent:
    """Simple agent representation"""
    id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    last_action: str = "idle"
    reward: float = 0.0
    timestamp: float = 0.0

class FeedbackRequest(BaseModel):
    """RL feedback request"""
    agent_id: str
    action_id: str
    reward: float = Field(..., ge=-1.0, le=1.0)
    context: Dict[str, Any] = {}

# =============================================================================
# A2A COMMUNICATION SERVER
# =============================================================================

class A2ASystem:
    """Complete A2A system in one class"""
    
    def __init__(self):
        self.app = FastAPI(title="RL-A2A System", version="3.0.0")
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.q_tables: Dict[str, np.ndarray] = {}
        self.openai_client = None
        self.running = False
        
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and CONFIG["OPENAI_API_KEY"]:
            self.openai_client = AsyncOpenAI(api_key=CONFIG["OPENAI_API_KEY"])
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all API routes"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/")
        async def root():
            return {
                "system": "RL-A2A",
                "version": "3.0.0",
                "agents": len(self.agents),
                "openai": bool(self.openai_client),
                "endpoints": ["/register", "/agents", "/feedback", "/dashboard", "/ws/{session_id}"]
            }
        
        @self.app.post("/register")
        async def register_agent(agent_id: str):
            """Register new agent"""
            session_id = str(uuid.uuid4())
            
            # Create agent
            agent = Agent(
                id=agent_id,
                position={"x": np.random.uniform(-10, 10), "y": np.random.uniform(-10, 10), "z": 0},
                velocity={"x": 0, "y": 0, "z": 0},
                emotion=np.random.choice(["neutral", "happy", "excited"]),
                timestamp=time.time()
            )
            
            self.agents[agent_id] = agent
            
            # Initialize Q-learning
            self.q_tables[agent_id] = np.random.rand(5, 4) * 0.1
            
            print(f"✅ Agent {agent_id} registered (session: {session_id})")
            return {"agent_id": agent_id, "session_id": session_id, "status": "registered"}
        
        @self.app.get("/agents")
        async def list_agents():
            """List all agents"""
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "position": agent.position,
                        "emotion": agent.emotion,
                        "last_action": agent.last_action,
                        "reward": agent.reward
                    }
                    for agent in self.agents.values()
                ],
                "count": len(self.agents)
            }
        
        @self.app.post("/feedback")
        async def receive_feedback(request: FeedbackRequest):
            """Receive agent feedback"""
            agent_id = request.agent_id
            
            if agent_id in self.agents and agent_id in self.q_tables:
                # Update Q-table (simple)
                state_idx = np.random.randint(0, 5)
                action_idx = np.random.randint(0, 4)
                lr = 0.1
                
                old_q = self.q_tables[agent_id][state_idx, action_idx]
                self.q_tables[agent_id][state_idx, action_idx] = old_q + lr * (request.reward - old_q)
                
                # Update agent reward
                self.agents[agent_id].reward = request.reward
                
                print(f"✅ Feedback: {agent_id} -> {request.reward:.2f}")
                return {"success": True, "message": "Feedback processed"}
            
            return {"success": False, "error": "Agent not found"}
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket for real-time communication"""
            await websocket.accept()
            self.connections[session_id] = websocket
            
            try:
                while True:
                    data = await websocket.receive_bytes()
                    parsed = msgpack.unpackb(data)
                    
                    agent_id = parsed.get("agent_id")
                    if agent_id in self.agents:
                        # Update agent state
                        agent = self.agents[agent_id]
                        agent.position.update(parsed.get("position", {}))
                        agent.velocity.update(parsed.get("velocity", {}))
                        agent.emotion = parsed.get("emotion", agent.emotion)
                        agent.timestamp = time.time()
                        
                        # Get AI action if available
                        action = await self.get_action(agent_id, parsed)
                        
                        # Send response
                        response = {
                            "agent_id": agent_id,
                            "action": action["command"],
                            "message": action["message"],
                            "timestamp": time.time()
                        }
                        await websocket.send(msgpack.packb(response))
                        
                        # Update agent action
                        agent.last_action = action["command"]
            
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if session_id in self.connections:
                    del self.connections[session_id]
        
        @self.app.get("/dashboard")
        async def dashboard_data():
            """Get dashboard data"""
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "position": agent.position,
                        "velocity": agent.velocity,
                        "emotion": agent.emotion,
                        "action": agent.last_action,
                        "reward": agent.reward,
                        "age": time.time() - agent.timestamp
                    }
                    for agent in self.agents.values()
                ],
                "system": {
                    "total_agents": len(self.agents),
                    "active_connections": len(self.connections),
                    "openai_enabled": bool(self.openai_client),
                    "uptime": time.time() - getattr(self, 'start_time', time.time())
                }
            }
    
    async def get_action(self, agent_id: str, observation: Dict) -> Dict[str, str]:
        """Get action for agent (AI or RL)"""
        actions = ["move_forward", "turn_left", "turn_right", "communicate", "observe"]
        
        if self.openai_client:
            try:
                # AI-powered action
                prompt = f"""
                Agent {agent_id} observation: {json.dumps(observation)}
                Choose the best action from: {actions}
                Respond with just the action name.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.3
                )
                
                ai_action = response.choices[0].message.content.strip().lower()
                if ai_action in actions:
                    return {"command": ai_action, "message": f"AI chose: {ai_action}"}
            
            except Exception as e:
                print(f"AI error: {e}")
        
        # Fallback to Q-learning
        if agent_id in self.q_tables:
            state_idx = np.random.randint(0, 5)
            action_idx = np.argmax(self.q_tables[agent_id][state_idx])
            action = actions[action_idx]
        else:
            action = np.random.choice(actions)
        
        return {"command": action, "message": f"RL chose: {action}"}
    
    def simulate_agents(self, count: int = 3):
        """Create demo agents"""
        for i in range(count):
            agent_id = f"demo_agent_{i+1}"
            agent = Agent(
                id=agent_id,
                position={
                    "x": np.random.uniform(-10, 10),
                    "y": np.random.uniform(-10, 10),
                    "z": np.random.uniform(0, 2)
                },
                velocity={
                    "x": np.random.uniform(-2, 2),
                    "y": np.random.uniform(-2, 2),
                    "z": 0
                },
                emotion=np.random.choice(["happy", "neutral", "excited", "sad"]),
                last_action=np.random.choice(["move_forward", "observe", "communicate"]),
                reward=np.random.uniform(-0.5, 1.0),
                timestamp=time.time()
            )
            self.agents[agent_id] = agent
            self.q_tables[agent_id] = np.random.rand(5, 4) * 0.1
        
        print(f"✅ Created {count} demo agents")
    
    async def start_server(self):
        """Start the A2A server"""
        self.start_time = time.time()
        config = uvicorn.Config(
            self.app,
            host=CONFIG["SERVER_HOST"],
            port=CONFIG["SERVER_PORT"],
            log_level="info"
        )
        server = uvicorn.Server(config) 
        print(f"🚀 A2A Server starting on {CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}")
        await server.serve()

# =============================================================================
# VISUALIZATION & DASHBOARD
# =============================================================================

def create_dashboard():
    """Create Streamlit dashboard content"""
    dashboard_code = '''
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import numpy as np

st.set_page_config(page_title="RL-A2A Dashboard", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

SERVER_URL = "http://localhost:8000"

def check_server():
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_agents():
    try:
        response = requests.get(f"{SERVER_URL}/agents", timeout=2)
        return response.json() if response.status_code == 200 else {"agents": [], "count": 0}
    except:
        return {"agents": [], "count": 0}

def get_dashboard_data():
    try:
        response = requests.get(f"{SERVER_URL}/dashboard", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def register_agent(agent_id):
    try:
        response = requests.post(f"{SERVER_URL}/register", params={"agent_id": agent_id})
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_feedback(agent_id, reward):
    try:
        response = requests.post(f"{SERVER_URL}/feedback", json={
            "agent_id": agent_id,
            "action_id": f"manual_{int(time.time())}",
            "reward": reward,
            "context": {"source": "dashboard"}
        })
        return response.status_code == 200
    except:
        return False

def create_3d_plot(agents):
    if not agents:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = {"happy": "#FFD700", "sad": "#4169E1", "neutral": "#808080", 
             "excited": "#FF4500", "angry": "#DC143C"}
    
    fig.add_trace(go.Scatter3d(
        x=[a["position"]["x"] for a in agents],
        y=[a["position"]["y"] for a in agents],
        z=[a["position"]["z"] for a in agents],
        mode='markers+text',
        marker=dict(
            size=12,
            color=[colors.get(a["emotion"], "#808080") for a in agents],
            opacity=0.8,
            line=dict(width=1, color='black')
        ),
        text=[a["id"] for a in agents],
        textposition="top center",
        hovertemplate='%{text}<br>Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>'
    ))
    
    fig.update_layout(
        title="Agent Positions",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            xaxis=dict(range=[-15, 15]), yaxis=dict(range=[-15, 15]), zaxis=dict(range=[-2, 5])
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_metrics_plot(agents):
    if not agents:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Emotions", "Rewards", "Actions", "Activity"],
        specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Emotions
    emotions = [a["emotion"] for a in agents]
    emotion_counts = pd.Series(emotions).value_counts()
    fig.add_trace(go.Pie(values=emotion_counts.values, labels=emotion_counts.index), row=1, col=1)
    
    # Rewards
    fig.add_trace(go.Bar(x=[a["id"] for a in agents], y=[a["reward"] for a in agents]), row=1, col=2)
    
    # Actions
    actions = [a["action"] for a in agents]
    action_counts = pd.Series(actions).value_counts()
    fig.add_trace(go.Bar(x=list(action_counts.index), y=list(action_counts.values)), row=2, col=1)
    
    # Activity (position magnitude)
    activity = [np.sqrt(a["position"]["x"]**2 + a["position"]["y"]**2) for a in agents]
    fig.add_trace(go.Scatter(x=[a["id"] for a in agents], y=activity, mode="markers+lines"), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

# Main Dashboard
st.title("🤖 RL-A2A System Dashboard")

# Server status
server_online = check_server()
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Server Status:** {'🟢 Online' if server_online else '🔴 Offline'}")
with col2:
    if st.button("🔄 Refresh"):
        st.rerun()

if not server_online:
    st.error("⚠️ A2A Server is offline. Start with: `python rla2a.py server`")
    st.stop()

# Get data
dashboard_data = get_dashboard_data()
agents_data = get_agents()
agents = agents_data.get("agents", [])

# Metrics
if dashboard_data:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Agents", dashboard_data["system"]["total_agents"])
    with col2:
        st.metric("Connections", dashboard_data["system"]["active_connections"])
    with col3:
        avg_reward = np.mean([a["reward"] for a in agents]) if agents else 0
        st.metric("Avg Reward", f"{avg_reward:.2f}")
    with col4:
        st.metric("OpenAI", "✅ Enabled" if dashboard_data["system"]["openai_enabled"] else "❌ Disabled")

st.divider()

# Main content
tab1, tab2, tab3 = st.tabs(["🌐 3D View", "📊 Metrics", "⚙️ Controls"])

with tab1:
    st.subheader("Agent Positions")
    fig_3d = create_3d_plot(agents)
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("System Metrics")
    fig_metrics = create_metrics_plot(agents)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    if agents:
        st.subheader("Agent Details")
        df = pd.DataFrame(agents)
        st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("Agent Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Register New Agent**")
        with st.form("register"):
            agent_id = st.text_input("Agent ID")
            if st.form_submit_button("Register"):
                if agent_id:
                    result = register_agent(agent_id)
                    if result:
                        st.success(f"✅ {agent_id} registered!")
                    else:
                        st.error("❌ Registration failed")
    
    with col2:
        st.write("**Send Feedback**")
        with st.form("feedback"):
            fb_agent = st.selectbox("Agent", [a["id"] for a in agents] if agents else [])
            fb_reward = st.slider("Reward", -1.0, 1.0, 0.0, 0.1)
            if st.form_submit_button("Send"):
                if fb_agent:
                    success = send_feedback(fb_agent, fb_reward)
                    if success:
                        st.success(f"✅ Feedback sent to {fb_agent}")
                    else:
                        st.error("❌ Feedback failed")

# Auto-refresh
if st.checkbox("Auto-refresh (every 3s)"):
    time.sleep(3)
    st.rerun()
'''
    
    # Write dashboard to file
    with open("dashboard_temp.py", "w") as f:
        f.write(dashboard_code)
    
    return "dashboard_temp.py"

def start_dashboard():
    """Start Streamlit dashboard"""
    dashboard_file = create_dashboard()
    print(f"🎨 Starting dashboard on port {CONFIG['DASHBOARD_PORT']}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", str(CONFIG["DASHBOARD_PORT"]),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\\n🛑 Dashboard stopped")
    finally:
        # Cleanup
        if os.path.exists(dashboard_file):
            os.remove(dashboard_file)

# =============================================================================
# MCP INTEGRATION
# =============================================================================

def create_mcp_server():
    """Create MCP server if available"""
    if not MCP_AVAILABLE:
        print("❌ MCP not available. Install with: pip install mcp")
        return None
    
    server = Server("rl-a2a")
    
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        return [
            Resource(
                uri="rla2a://system",
                name="RL-A2A System",
                description="Complete system status and information",
                mimeType="application/json"
            )
        ]
    
    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "rla2a://system":
            try:
                response = requests.get(f"http://{CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}/")
                return response.text
            except:
                return json.dumps({"error": "Server not running"})
        raise ValueError(f"Unknown resource: {uri}")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="start_system",
                description="Start the complete RL-A2A system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agents": {"type": "number", "default": 3},
                        "dashboard": {"type": "boolean", "default": True}
                    }
                }
            ),
            Tool(
                name="create_agent",
                description="Create a new agent",
                inputSchema={
                    "type": "object",
                    "properties": {"agent_id": {"type": "string"}},
                    "required": ["agent_id"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        if name == "start_system":
            agents = arguments.get("agents", 3)
            dashboard = arguments.get("dashboard", True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"System configured for {agents} agents",
                    "commands": [
                        f"python rla2a.py server --demo-agents {agents}",
                        "python rla2a.py dashboard" if dashboard else ""
                    ]
                })
            )]
        
        elif name == "create_agent":
            agent_id = arguments.get("agent_id")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "agent_id": agent_id,
                    "message": f"Agent {agent_id} configured"
                })
            )]
        
        raise ValueError(f"Unknown tool: {name}")
    
    return server

async def run_mcp_server():
    """Run MCP server"""
    server = create_mcp_server()
    if not server:
        return
    
    try:
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializeResult(
                    serverName="rl-a2a",
                    serverVersion="3.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except ImportError:
        print("❌ MCP stdio server not available")

# =============================================================================
# SETUP & UTILITIES
# =============================================================================

def setup_environment():
    """Setup environment and config files"""
    print("🔧 Setting up RL-A2A environment...")
    
    # Create .env file
    env_content = f"""# RL-A2A Configuration
OPENAI_API_KEY=your-openai-api-key-here
A2A_HOST=localhost
A2A_PORT=8000
DASHBOARD_PORT=8501
DEBUG=false
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
    
    # Create start scripts
    scripts = {
        "start.py": '''#!/usr/bin/env python3
import subprocess
import sys
import time
import threading

def start_server():
    subprocess.run([sys.executable, "rla2a.py", "server", "--demo-agents", "3"])

def start_dashboard():
    time.sleep(3)  # Wait for server
    subprocess.run([sys.executable, "rla2a.py", "dashboard"])

print("🚀 Starting RL-A2A Complete System...")
print("   Server: http://localhost:8000")
print("   Dashboard: http://localhost:8501")
print("   Press Ctrl+C to stop")

try:
    server_thread = threading.Thread(target=start_server, daemon=True)
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    
    server_thread.start()
    dashboard_thread.start()
    
    server_thread.join()
except KeyboardInterrupt:
    print("\\n👋 Stopped")
''',
        "start_server.py": '''#!/usr/bin/env python3
import subprocess
import sys
subprocess.run([sys.executable, "rla2a.py", "server", "--demo-agents", "3"])
''',
        "start_dashboard.py": '''#!/usr/bin/env python3
import subprocess
import sys
subprocess.run([sys.executable, "rla2a.py", "dashboard"])
'''
    }
    
    for filename, content in scripts.items():
        with open(filename, "w") as f:
            f.write(content)
        if os.name != 'nt':
            os.chmod(filename, 0o755)
    
    print("✅ Created startup scripts")
    
    print("\\n🎉 Setup complete!")
    print("\\n📋 Quick Start:")
    print("1. python start.py           # Complete system")
    print("2. python start_server.py    # Server only")
    print("3. python start_dashboard.py # Dashboard only")
    print("\\n💡 Optional: Set OPENAI_API_KEY in .env for AI features")

def generate_report():
    """Generate a simple HTML report"""
    try:
        response = requests.get(f"http://{CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}/dashboard")
        data = response.json()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>RL-A2A Report</title></head>
        <body>
            <h1>RL-A2A System Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>System Status</h2>
            <ul>
                <li>Total Agents: {data['system']['total_agents']}</li>
                <li>Active Connections: {data['system']['active_connections']}</li>
                <li>OpenAI Enabled: {data['system']['openai_enabled']}</li>
                <li>Uptime: {data['system']['uptime']:.1f}s</li>
            </ul>
            <h2>Agents</h2>
            <table border="1">
                <tr><th>ID</th><th>Position</th><th>Emotion</th><th>Action</th><th>Reward</th></tr>
                {''.join(f'<tr><td>{a["id"]}</td><td>({a["position"]["x"]:.1f}, {a["position"]["y"]:.1f})</td><td>{a["emotion"]}</td><td>{a["action"]}</td><td>{a["reward"]:.2f}</td></tr>' for a in data['agents'])}
            </table>
        </body>
        </html>
        """
        
        with open("rla2a_report.html", "w") as f:
            f.write(html)
        
        print("✅ Report generated: rla2a_report.html")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RL-A2A: Complete Agent Communication System")
    parser.add_argument("command", choices=["server", "dashboard", "setup", "mcp", "report"], 
                       help="Command to run")
    parser.add_argument("--demo-agents", type=int, default=0, help="Create demo agents")
    parser.add_argument("--host", default=CONFIG["SERVER_HOST"], help="Server host")
    parser.add_argument("--port", type=int, default=CONFIG["SERVER_PORT"], help="Server port")
    
    args = parser.parse_args()
    
    # Update config
    CONFIG["SERVER_HOST"] = args.host
    CONFIG["SERVER_PORT"] = args.port
    
    if args.command == "setup":
        setup_environment()
    
    elif args.command == "server":
        system = A2ASystem()
        
        if args.demo_agents > 0:
            system.simulate_agents(args.demo_agents)
        
        # Setup signal handler
        def signal_handler(signum, frame):
            print("\\n🛑 Shutting down server...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await system.start_server()
        except KeyboardInterrupt:
            print("\\n👋 Server stopped")
    
    elif args.command == "dashboard":
        try:
            start_dashboard()
        except KeyboardInterrupt:
            print("\\n👋 Dashboard stopped")
    
    elif args.command == "mcp":
        await run_mcp_server()
    
    elif args.command == "report":
        generate_report()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("RL-A2A: Complete Agent Communication System")
        print("Usage: python rla2a.py {server|dashboard|setup|mcp|report}")
        print("\\nQuick start: python rla2a.py setup")
    else:
        asyncio.run(main())
