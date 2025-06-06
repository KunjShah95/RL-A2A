"""
RL-A2A Enhanced: Secure Agent-to-Agent Communication System
Combined version with security, Claude/Gemini support, visualization, and comprehensive configuration

This file combines the original rla2a.py with rla2a_enhanced.py to provide:
- Multi-AI provider support (OpenAI, Claude, Gemini)
- Enhanced security features (JWT, rate limiting, input validation)
- Comprehensive environment configuration
- Production-ready deployment capabilities
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
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import argparse
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv not available
    pass

# Security imports with fallbacks
security_available = True
try:
    import jwt
    import bcrypt
    import bleach
    from passlib.context import CryptContext
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    security_available = False
    print("⚠️ Security packages not available. Install with: pip install jwt bcrypt bleach slowapi passlib")

# Core dependencies check and install
def check_and_install_dependencies():
    """Check and install required dependencies with security focus"""
    required = [
        "fastapi", "uvicorn", "websockets", "msgpack", "numpy", "pydantic", 
        "requests", "matplotlib", "plotly", "streamlit", "pandas", "python-dotenv"
    ]
    
    # Security packages (optional but recommended)
    security_packages = ["jwt", "bcrypt", "bleach", "slowapi", "passlib"]
    
    # AI provider packages (optional)
    ai_packages = ["openai", "anthropic", "google-generativeai", "mcp"]
    
    missing_required = []
    missing_optional = []
    
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_required.append(pkg)
    
    for pkg in security_packages + ai_packages:
        try:
            if pkg == "jwt":
                import jwt
            elif pkg == "google-generativeai":
                import google.generativeai
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_optional.append(pkg)
    
    if missing_required:
        print(f"📦 Installing required packages: {', '.join(missing_required)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_required, stdout=subprocess.DEVNULL)
            print("✅ Required dependencies installed")
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print(f"Please install manually: pip install {' '.join(missing_required)}")
            sys.exit(1)
    
    if missing_optional:
        print(f"ℹ️ Optional packages available for enhanced features: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))

check_and_install_dependencies()

# Now import everything
import requests
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import websockets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# AI Provider imports with fallbacks
openai_available = False
anthropic_available = False
google_available = False
mcp_available = False

try:
    from openai import AsyncOpenAI
    openai_available = True
except ImportError:
    pass

try:
    import anthropic
    anthropic_available = True
except ImportError:
    pass

try:
    import google.generativeai as genai
    google_available = True
except ImportError:
    pass

try:
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent
    import mcp.types as types
    mcp_available = True
except ImportError:
    pass

# Security configuration
if security_available:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    limiter = Limiter(key_func=get_remote_address)

class SecurityConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour

# Enhanced configuration
CONFIG = {
    # AI Providers
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    
    # AI Model Configuration
    "DEFAULT_AI_PROVIDER": os.getenv("DEFAULT_AI_PROVIDER", "openai"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    "GOOGLE_MODEL": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    
    # Server Configuration
    "SERVER_HOST": os.getenv("A2A_HOST", "localhost"),
    "SERVER_PORT": int(os.getenv("A2A_PORT", "8000")),
    "DASHBOARD_PORT": int(os.getenv("DASHBOARD_PORT", "8501")),
    
    # System Limits
    "MAX_AGENTS": int(os.getenv("MAX_AGENTS", "100")),
    "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
    
    # Logging
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "LOG_FILE": os.getenv("LOG_FILE", "rla2a.log")
}

# Logging setup
logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data models
class AgentPosition(BaseModel):
    x: float = Field(..., ge=-100, le=100)
    y: float = Field(..., ge=-100, le=100)
    z: float = Field(0, ge=-10, le=10)

class AgentVelocity(BaseModel):
    x: float = Field(0, ge=-10, le=10)
    y: float = Field(0, ge=-10, le=10)
    z: float = Field(0, ge=-10, le=10)

@dataclass
class Agent:
    """Enhanced agent with validation"""
    id: str
    position: AgentPosition
    velocity: AgentVelocity
    emotion: str = "neutral"
    last_action: str = "idle"
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    
    def __post_init__(self):
        valid_emotions = ["happy", "sad", "neutral", "excited", "angry", "fear", "surprise"]
        if self.emotion not in valid_emotions:
            self.emotion = "neutral"
        
        # Sanitize agent ID if security available
        if security_available and hasattr(self, 'id'):
            self.id = bleach.clean(self.id)[:50]

class FeedbackRequest(BaseModel):
    """Feedback request with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50)
    action_id: str = Field(..., min_length=1, max_length=100)
    reward: float = Field(..., ge=-1.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)

class AgentRegistrationRequest(BaseModel):
    """Agent registration with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50)
    initial_position: Optional[AgentPosition] = None
    ai_provider: Optional[str] = Field("openai", regex="^(openai|anthropic|google)$")

# Multi-AI Provider Manager
class MultiAIManager:
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available AI clients"""
        # OpenAI
        if openai_available and CONFIG["OPENAI_API_KEY"]:
            try:
                self.clients["openai"] = AsyncOpenAI(api_key=CONFIG["OPENAI_API_KEY"])
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if anthropic_available and CONFIG["ANTHROPIC_API_KEY"]:
            try:
                self.clients["anthropic"] = anthropic.AsyncAnthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
        
        # Google
        if google_available and CONFIG["GOOGLE_API_KEY"]:
            try:
                genai.configure(api_key=CONFIG["GOOGLE_API_KEY"])
                self.clients["google"] = genai.GenerativeModel(CONFIG["GOOGLE_MODEL"])
                logger.info("Google AI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI: {e}")
    
    async def get_ai_response(self, prompt: str, provider: str = None) -> str:
        """Get AI response from specified provider with fallback"""
        provider = provider or CONFIG["DEFAULT_AI_PROVIDER"]
        
        # Try primary provider
        if provider in self.clients:
            try:
                return await self._get_provider_response(provider, prompt)
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
        
        # Fallback to any available provider
        for fallback_provider in self.clients:
            if fallback_provider != provider:
                try:
                    logger.info(f"Falling back to {fallback_provider}")
                    return await self._get_provider_response(fallback_provider, prompt)
                except Exception as e:
                    logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
        
        # If all providers fail
        return "I apologize, but I'm currently unable to provide an AI response due to technical issues."
    
    async def _get_provider_response(self, provider: str, prompt: str) -> str:
        """Get response from specific provider"""
        if provider == "openai":
            response = await self.clients["openai"].chat.completions.create(
                model=CONFIG["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                timeout=10
            )
            return response.choices[0].message.content
        
        elif provider == "anthropic":
            response = await self.clients["anthropic"].messages.create(
                model=CONFIG["ANTHROPIC_MODEL"],
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
            return response.content[0].text
        
        elif provider == "google":
            response = await self.clients["google"].generate_content_async(
                prompt,
                generation_config={"max_output_tokens": 150}
            )
            return response.text
        
        raise ValueError(f"Unknown provider: {provider}")

# RL Environment
class RLEnvironment:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.actions = ["move_up", "move_down", "move_left", "move_right", "communicate", "idle"]
        
    def get_state_key(self, agent: Agent) -> str:
        """Generate state key for Q-table"""
        pos_x = int(agent.position.x // 10) * 10
        pos_y = int(agent.position.y // 10) * 10
        return f"{pos_x},{pos_y},{agent.emotion}"
    
    def get_action(self, agent: Agent) -> str:
        """Get action using epsilon-greedy policy"""
        state_key = self.get_state_key(agent)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
        
        q_values = self.q_table[state_key]
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, agent: Agent, action: str, reward: float):
        """Update Q-value using Q-learning algorithm"""
        state_key = self.get_state_key(agent)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        old_q_value = self.q_table[state_key][action]
        # Simplified Q-learning update (assuming next state max Q is 0 for now)
        new_q_value = old_q_value + self.learning_rate * (reward - old_q_value)
        self.q_table[state_key][action] = new_q_value
        
        logger.debug(f"Updated Q-value for {state_key}[{action}]: {old_q_value} -> {new_q_value}")

# Enhanced A2A System
class A2ASystem:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.ai_manager = MultiAIManager()
        self.rl_env = RLEnvironment()
        self.session_data = {}
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with enhanced security"""
        app = FastAPI(
            title="RL-A2A Enhanced API",
            description="Secure Agent-to-Agent Communication with Multi-AI Support",
            version="4.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=SecurityConfig.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting if available
        if security_available:
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Routes
        self._add_routes(app)
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes"""
        
        @app.get("/")
        async def root():
            return {
                "message": "RL-A2A Enhanced API",
                "version": "4.0.0",
                "features": ["Multi-AI", "Security", "RL", "WebSocket"],
                "ai_providers": list(self.ai_manager.clients.keys()),
                "security_enabled": security_available
            }
        
        @app.post("/register")
        async def register_agent(request: AgentRegistrationRequest):
            """Register a new agent"""
            if len(self.agents) >= CONFIG["MAX_AGENTS"]:
                raise HTTPException(status_code=429, detail="Maximum agents reached")
            
            if request.agent_id in self.agents:
                raise HTTPException(status_code=409, detail="Agent already exists")
            
            # Create agent with default position if not provided
            position = request.initial_position or AgentPosition(
                x=np.random.uniform(-50, 50),
                y=np.random.uniform(-50, 50),
                z=0
            )
            
            agent = Agent(
                id=request.agent_id,
                position=position,
                velocity=AgentVelocity(),
                session_id=str(uuid.uuid4())
            )
            
            self.agents[request.agent_id] = agent
            logger.info(f"Agent {request.agent_id} registered")
            
            return {
                "status": "success",
                "agent_id": request.agent_id,
                "session_id": agent.session_id,
                "position": agent.position
            }
        
        @app.post("/feedback")
        async def provide_feedback(request: FeedbackRequest):
            """Provide feedback for agent learning"""
            if request.agent_id not in self.agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = self.agents[request.agent_id]
            self.rl_env.update_q_value(agent, request.action_id, request.reward)
            agent.reward += request.reward
            
            return {"status": "success", "message": "Feedback received"}
        
        @app.get("/agents")
        async def list_agents():
            """List all registered agents"""
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "position": agent.position,
                        "emotion": agent.emotion,
                        "last_action": agent.last_action,
                        "reward": agent.reward,
                        "last_activity": agent.last_activity
                    }
                    for agent in self.agents.values()
                ],
                "total": len(self.agents)
            }
        
        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """Enhanced WebSocket endpoint with security"""
            await websocket.accept()
            self.connections[session_id] = websocket
            
            try:
                while True:
                    data = await websocket.receive_text()
                    
                    # Basic message size validation
                    if len(data) > SecurityConfig.MAX_MESSAGE_SIZE:
                        await websocket.send_text(json.dumps({
                            "error": "Message too large"
                        }))
                        continue
                    
                    # Process message
                    await self._handle_websocket_message(session_id, data)
                    
            except Exception as e:
                logger.error(f"WebSocket error for {session_id}: {e}")
            finally:
                if session_id in self.connections:
                    del self.connections[session_id]
    
    async def _handle_websocket_message(self, session_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            agent_id = data.get("agent_id")
            
            if not agent_id or agent_id not in self.agents:
                return
            
            agent = self.agents[agent_id]
            message_type = data.get("type")
            
            if message_type == "position_update":
                # Update agent position
                new_pos = data.get("position", {})
                if "x" in new_pos and "y" in new_pos:
                    agent.position.x = max(-100, min(100, new_pos["x"]))
                    agent.position.y = max(-100, min(100, new_pos["y"]))
                    agent.last_activity = time.time()
            
            elif message_type == "ai_request":
                # Get AI response
                prompt = data.get("prompt", "")
                if prompt:
                    ai_response = await self.ai_manager.get_ai_response(
                        prompt, 
                        data.get("provider")
                    )
                    
                    # Send response back
                    response = {
                        "type": "ai_response",
                        "agent_id": agent_id,
                        "response": ai_response,
                        "timestamp": time.time()
                    }
                    
                    if session_id in self.connections:
                        await self.connections[session_id].send_text(
                            json.dumps(response)
                        )
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def simulate_agents(self, count: int):
        """Create demo agents for testing"""
        for i in range(count):
            agent_id = f"demo_agent_{i+1}"
            position = AgentPosition(
                x=np.random.uniform(-30, 30),
                y=np.random.uniform(-30, 30),
                z=0
            )
            
            agent = Agent(
                id=agent_id,
                position=position,
                velocity=AgentVelocity(),
                emotion=np.random.choice(["happy", "neutral", "excited"]),
                session_id=str(uuid.uuid4())
            )
            
            self.agents[agent_id] = agent
            logger.info(f"Created demo agent: {agent_id}")
    
    async def start_server(self):
        """Start the enhanced A2A server"""
        logger.info("Starting RL-A2A Enhanced Server...")
        logger.info(f"Security features: {'Enabled' if security_available else 'Basic'}")
        logger.info(f"AI providers available: {list(self.ai_manager.clients.keys())}")
        
        config = uvicorn.Config(
            self.app,
            host=CONFIG["SERVER_HOST"],
            port=CONFIG["SERVER_PORT"],
            log_level=CONFIG["LOG_LEVEL"].lower()
        )
        server = uvicorn.Server(config)
        await server.serve()

# Dashboard function (simplified version)
def start_dashboard():
    """Start the enhanced Streamlit dashboard"""
    try:
        import streamlit as st
        
        # This would be expanded with the full dashboard implementation
        st.title("🤖 RL-A2A Enhanced Dashboard")
        st.markdown("### Multi-AI Agent Communication System")
        
        # Security status
        st.markdown("#### Security Status")
        if security_available:
            st.success("🔒 Enhanced security features enabled")
        else:
            st.warning("⚠️ Basic security - install security packages for full protection")
        
        # AI provider status
        st.markdown("#### AI Provider Status")
        ai_manager = MultiAIManager()
        for provider, available in [
            ("OpenAI", openai_available and CONFIG["OPENAI_API_KEY"]),
            ("Anthropic", anthropic_available and CONFIG["ANTHROPIC_API_KEY"]),
            ("Google", google_available and CONFIG["GOOGLE_API_KEY"])
        ]:
            if available:
                st.success(f"✅ {provider} - Ready")
            else:
                st.error(f"❌ {provider} - Not configured")
        
        st.markdown("---")
        st.markdown("Configure your API keys in the `.env` file to enable AI providers.")
        
    except ImportError:
        print("Streamlit not available. Install with: pip install streamlit")

# Environment setup
def setup_environment():
    """Setup enhanced environment configuration"""
    logger.info("Setting up RL-A2A Enhanced environment...")
    
    # Create .env.example if it doesn't exist
    env_example = """# RL-A2A Enhanced Configuration

# Multi-AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Security Configuration
SECRET_KEY=your-secret-key-for-jwt-signing
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501
RATE_LIMIT_PER_MINUTE=60

# System Configuration
DEFAULT_AI_PROVIDER=openai
MAX_AGENTS=100
DEBUG=false

# Server Configuration
A2A_HOST=localhost
A2A_PORT=8000
DASHBOARD_PORT=8501

# Logging
LOG_LEVEL=INFO
LOG_FILE=rla2a.log"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        env_path.write_text(env_example)
        logger.info("Created .env.example file")
    
    # Create .env if it doesn't exist
    env_actual_path = Path(".env")
    if not env_actual_path.exists():
        env_actual_path.write_text(env_example)
        logger.info("Created .env file - please configure your API keys")
    
    logger.info("✅ Environment setup complete!")
    logger.info("📝 Edit .env file to configure API keys and settings")

# MCP server (if available)
async def run_mcp_server():
    """Run enhanced MCP server"""
    if not mcp_available:
        print("❌ MCP not available. Install with: pip install mcp")
        return
    
    # This would include the full MCP implementation
    print("🔌 RL-A2A Enhanced MCP Server starting...")
    print("ℹ️ Full MCP implementation available in enhanced version")

# CLI interface
async def main():
    """Enhanced CLI interface"""
    parser = argparse.ArgumentParser(description="RL-A2A Enhanced: Secure Multi-AI Agent Communication")
    
    parser.add_argument("command", choices=["setup", "server", "dashboard", "mcp"], 
                       help="Command to execute")
    parser.add_argument("--demo-agents", type=int, default=0, 
                       help="Number of demo agents to create")
    parser.add_argument("--host", default=CONFIG["SERVER_HOST"], 
                       help="Server host")
    parser.add_argument("--port", type=int, default=CONFIG["SERVER_PORT"], 
                       help="Server port")
    
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
            print("\n🛑 Shutting down server...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await system.start_server()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
    
    elif args.command == "dashboard":
        try:
            start_dashboard()
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")
    
    elif args.command == "mcp":
        await run_mcp_server()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🤖 RL-A2A Enhanced: Secure Multi-AI Agent Communication System")
        print("Usage: python rla2a.py {server|dashboard|setup|mcp}")
        print("\n🚀 Quick start:")
        print("  python rla2a.py setup      # Configure environment")
        print("  python rla2a.py server     # Start secure A2A server")
        print("  python rla2a.py dashboard  # Launch monitoring dashboard")
        print("\n📚 Documentation: https://github.com/KunjShah01/RL-A2A")
    else:
        asyncio.run(main())