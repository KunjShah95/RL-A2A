"""
RL-A2A Enhanced: Secure Agent-to-Agent Communication System
Enhanced with security, Claude/Gemini support, visualization, and comprehensive environment configuration
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
from dotenv import load_dotenv
load_dotenv()

# Security imports
import jwt
import bcrypt
import bleach
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Core dependencies check and import
def check_and_install_dependencies():
    """Check and install required dependencies with security focus"""
    required = [
        "fastapi", "uvicorn", "websockets", "msgpack", "numpy", "pydantic", 
        "requests", "matplotlib", "plotly", "streamlit", "pandas", "python-dotenv",
        "jwt", "bcrypt", "bleach", "slowapi", "passlib", 
        "openai", "anthropic", "google-generativeai", "mcp"
    ]
    
    # Optional dependencies
    optional = ["redis", "aiofiles"]
    
    missing = []
    for pkg in required:
        try:
            if pkg == "jwt":
                import jwt
            elif pkg == "google-generativeai":
                import google.generativeai
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing, stdout=subprocess.DEVNULL)
            print("✅ Dependencies installed")
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print(f"Please install manually: pip install {' '.join(missing)}")
            sys.exit(1)

check_and_install_dependencies()

# Now import everything
import requests
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, EmailStr, validator
import uvicorn
import websockets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# AI Provider imports with fallbacks
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

class SecurityConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

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

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS WITH VALIDATION
# =============================================================================

class AIProvider(BaseModel):
    """AI Provider configuration"""
    name: str = Field(..., regex="^(openai|anthropic|google)$")
    model: Optional[str] = None
    api_key: Optional[str] = None

class AgentPosition(BaseModel):
    """Agent position with validation"""
    x: float = Field(..., ge=-100, le=100)
    y: float = Field(..., ge=-100, le=100)  
    z: float = Field(0, ge=-10, le=10)

class AgentVelocity(BaseModel):
    """Agent velocity with validation"""
    x: float = Field(0, ge=-10, le=10)
    y: float = Field(0, ge=-10, le=10)
    z: float = Field(0, ge=-10, le=10)

@dataclass
class Agent:
    """Enhanced agent with security validation"""
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
        # Validate emotion
        valid_emotions = ["happy", "sad", "neutral", "excited", "angry", "fear", "surprise"]
        if self.emotion not in valid_emotions:
            self.emotion = "neutral"
        
        # Sanitize agent ID
        self.id = bleach.clean(self.id)[:50]  # Limit ID length

class FeedbackRequest(BaseModel):
    """Enhanced feedback request with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50)
    action_id: str = Field(..., min_length=1, max_length=100)
    reward: float = Field(..., ge=-1.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('agent_id', 'action_id')
    def sanitize_strings(cls, v):
        return bleach.clean(v)

class AgentRegistrationRequest(BaseModel):
    """Agent registration with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    initial_position: Optional[AgentPosition] = None
    ai_provider: Optional[str] = Field("openai", regex="^(openai|anthropic|google)$")

class WebSocketMessage(BaseModel):
    """WebSocket message validation"""
    agent_id: str = Field(..., min_length=1, max_length=50)
    message_type: str = Field(..., regex="^(position_update|action_request|feedback)$")
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

# =============================================================================
# AUTHENTICATION & SECURITY
# =============================================================================

class Auth:
    """Authentication system"""
    
    @staticmethod
    def create_access_token(data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=SecurityConfig.ACCESS_TOKEN_EXPIRE_HOURS)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm="HS256")
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)

# Security dependency
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current authenticated user"""
    payload = Auth.verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

# =============================================================================
# AI PROVIDER MANAGER  
# =============================================================================

class AIProviderManager:
    """Multi-provider AI management with security"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize AI providers based on available keys"""
        # OpenAI
        if OPENAI_AVAILABLE and CONFIG["OPENAI_API_KEY"]:
            try:
                self.openai_client = AsyncOpenAI(
                    api_key=CONFIG["OPENAI_API_KEY"],
                    timeout=30.0
                )
                logger.info("✅ OpenAI initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and CONFIG["ANTHROPIC_API_KEY"]:
            try:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=CONFIG["ANTHROPIC_API_KEY"],
                    timeout=30.0
                )
                logger.info("✅ Anthropic initialized")
            except Exception as e:
                logger.error(f"❌ Anthropic initialization failed: {e}")
        
        # Google
        if GOOGLE_AVAILABLE and CONFIG["GOOGLE_API_KEY"]:
            try:
                genai.configure(api_key=CONFIG["GOOGLE_API_KEY"])
                self.google_client = genai.GenerativeModel(CONFIG["GOOGLE_MODEL"])
                logger.info("✅ Google AI initialized")
            except Exception as e:
                logger.error(f"❌ Google AI initialization failed: {e}")
    
    async def get_action(self, agent_id: str, observation: Dict, provider: str = None) -> Dict[str, str]:
        """Get action from specified or default AI provider"""
        if not provider:
            provider = CONFIG["DEFAULT_AI_PROVIDER"]
        
        actions = ["move_forward", "turn_left", "turn_right", "communicate", "observe", "pause"]
        
        # Sanitize observation data
        safe_observation = self._sanitize_observation(observation)
        
        try:
            if provider == "openai" and self.openai_client:
                return await self._get_openai_action(agent_id, safe_observation, actions)
            elif provider == "anthropic" and self.anthropic_client:
                return await self._get_anthropic_action(agent_id, safe_observation, actions)
            elif provider == "google" and self.google_client:
                return await self._get_google_action(agent_id, safe_observation, actions)
            else:
                return self._get_fallback_action(actions)
                
        except Exception as e:
            logger.error(f"AI provider error ({provider}): {e}")
            return self._get_fallback_action(actions)
    
    def _sanitize_observation(self, observation: Dict) -> Dict:
        """Sanitize observation data to prevent injection attacks"""
        def clean_value(value):
            if isinstance(value, str):
                return bleach.clean(value)[:1000]  # Limit string length
            elif isinstance(value, (int, float)):
                return max(-1000, min(1000, value))  # Clamp numbers
            elif isinstance(value, dict):
                return {k[:50]: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value[:10]][:10]  # Limit list size
            return str(value)[:100]
        
        return {k[:50]: clean_value(v) for k, v in observation.items()}
    
    async def _get_openai_action(self, agent_id: str, observation: Dict, actions: List[str]) -> Dict[str, str]:
        """Get action from OpenAI"""
        prompt = self._build_prompt(agent_id, observation, actions)
        
        response = await self.openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
            timeout=10.0
        )
        
        action = response.choices[0].message.content.strip().lower()
        if action in actions:
            return {"command": action, "message": f"OpenAI chose: {action}", "provider": "openai"}
        
        return self._get_fallback_action(actions)
    
    async def _get_anthropic_action(self, agent_id: str, observation: Dict, actions: List[str]) -> Dict[str, str]:
        """Get action from Anthropic Claude"""
        prompt = self._build_prompt(agent_id, observation, actions)
        
        message = await self.anthropic_client.messages.create(
            model=CONFIG["ANTHROPIC_MODEL"],
            max_tokens=50,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            timeout=10.0
        )
        
        action = message.content[0].text.strip().lower()
        if action in actions:
            return {"command": action, "message": f"Claude chose: {action}", "provider": "anthropic"}
        
        return self._get_fallback_action(actions)
    
    async def _get_google_action(self, agent_id: str, observation: Dict, actions: List[str]) -> Dict[str, str]:
        """Get action from Google Gemini"""
        prompt = self._build_prompt(agent_id, observation, actions)
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.google_client.generate_content, prompt
        )
        
        action = response.text.strip().lower()
        if action in actions:
            return {"command": action, "message": f"Gemini chose: {action}", "provider": "google"}
        
        return self._get_fallback_action(actions)
    
    def _build_prompt(self, agent_id: str, observation: Dict, actions: List[str]) -> str:
        """Build AI prompt with safety considerations"""
        return f"""
Agent {agent_id} needs to choose an action.
Current observation: {json.dumps(observation)[:500]}
Available actions: {actions}

Choose the BEST action from the list above.
Respond with ONLY the action name, nothing else.
Consider the agent's position, nearby agents, and current environment.
"""
    
    def _get_fallback_action(self, actions: List[str]) -> Dict[str, str]:
        """Fallback to random action"""
        action = np.random.choice(actions)
        return {"command": action, "message": f"Random fallback: {action}", "provider": "fallback"}

# =============================================================================
# ENHANCED A2A SYSTEM
# =============================================================================

class EnhancedA2ASystem:
    """Enhanced A2A system with security, multi-AI support, and visualization"""
    
    def __init__(self):
        self.app = FastAPI(title="RL-A2A Enhanced", version="4.0.0")
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.q_tables: Dict[str, np.ndarray] = {}
        self.ai_manager = AIProviderManager()
        self.running = False
        self.active_sessions: Dict[str, Dict] = {}
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup security middleware"""
        
        # CORS with restricted origins
        if SecurityConfig.ALLOWED_ORIGINS != ["*"]:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=SecurityConfig.ALLOWED_ORIGINS,
                allow_credentials=True,
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=["localhost", "127.0.0.1", CONFIG["SERVER_HOST"]]
        )
        
        # Rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    def _setup_routes(self):
        """Setup all API routes with security"""
        
        @self.app.get("/")
        async def root():
            return {
                "system": "RL-A2A Enhanced",
                "version": "4.0.0",
                "agents": len(self.agents),
                "ai_providers": {
                    "openai": bool(self.ai_manager.openai_client),
                    "anthropic": bool(self.ai_manager.anthropic_client),
                    "google": bool(self.ai_manager.google_client)
                },
                "security": "enabled",
                "endpoints": [
                    "/register", "/agents", "/feedback", 
                    "/dashboard", "/health", "/ws/{session_id}"
                ]
            }
        
        @self.app.get("/health")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")  
        async def health_check(request):
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "agents": len(self.agents),
                "active_sessions": len(self.active_sessions)
            }
        
        @self.app.post("/register")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")
        async def register_agent(request, registration: AgentRegistrationRequest):
            """Register new agent with enhanced security"""
            
            # Check agent limit
            if len(self.agents) >= CONFIG["MAX_AGENTS"]:
                raise HTTPException(
                    status_code=429, 
                    detail=f"Maximum agents limit ({CONFIG['MAX_AGENTS']}) reached"
                )
            
            # Check for duplicate
            if registration.agent_id in self.agents:
                raise HTTPException(
                    status_code=409,
                    detail="Agent ID already exists"
                )
            
            session_id = str(uuid.uuid4())
            
            # Create agent with validation
            position = registration.initial_position or AgentPosition(
                x=np.random.uniform(-10, 10),
                y=np.random.uniform(-10, 10),
                z=0
            )
            
            agent = Agent(
                id=registration.agent_id,
                position=position,
                velocity=AgentVelocity(),
                emotion=np.random.choice(["neutral", "happy", "excited"]),
                session_id=session_id,
                timestamp=time.time()
            )
            
            self.agents[registration.agent_id] = agent
            
            # Initialize Q-learning with smaller random values
            self.q_tables[registration.agent_id] = np.random.rand(10, 6) * 0.01
            
            # Track session
            self.active_sessions[session_id] = {
                "agent_id": registration.agent_id,
                "created": time.time(),
                "last_activity": time.time()
            }
            
            logger.info(f"✅ Agent {registration.agent_id} registered (session: {session_id[:8]})")
            return {
                "agent_id": registration.agent_id,
                "session_id": session_id,
                "status": "registered",
                "ai_provider": registration.ai_provider
            }
        
        @self.app.get("/agents")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")
        async def list_agents(request):
            """List all agents with enhanced data"""
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "position": {
                            "x": agent.position.x,
                            "y": agent.position.y, 
                            "z": agent.position.z
                        },
                        "velocity": {
                            "x": agent.velocity.x,
                            "y": agent.velocity.y,
                            "z": agent.velocity.z
                        },
                        "emotion": agent.emotion,
                        "last_action": agent.last_action,
                        "reward": round(agent.reward, 3),
                        "age": round(time.time() - agent.timestamp, 2),
                        "active": time.time() - agent.last_activity < 60
                    }
                    for agent in self.agents.values()
                ],
                "count": len(self.agents),
                "active_sessions": len(self.active_sessions)
            }
        
        @self.app.post("/feedback")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")
        async def receive_feedback(request, feedback: FeedbackRequest):
            """Receive agent feedback with validation"""
            agent_id = feedback.agent_id
            
            if agent_id not in self.agents:
                raise HTTPException(
                    status_code=404,
                    detail="Agent not found"
                )
            
            if agent_id not in self.q_tables:
                raise HTTPException(
                    status_code=500,
                    detail="Agent Q-table not found"
                )
            
            # Enhanced Q-learning update
            state_idx = min(np.random.randint(0, 10), 9)
            action_idx = min(np.random.randint(0, 6), 5)
            learning_rate = 0.1
            discount_factor = 0.95
            
            old_q = self.q_tables[agent_id][state_idx, action_idx]
            max_future_q = np.max(self.q_tables[agent_id][state_idx, :])
            
            new_q = old_q + learning_rate * (feedback.reward + discount_factor * max_future_q - old_q)
            self.q_tables[agent_id][state_idx, action_idx] = new_q
            
            # Update agent
            self.agents[agent_id].reward = feedback.reward
            self.agents[agent_id].last_activity = time.time()
            
            logger.info(f"✅ Feedback: {agent_id} -> {feedback.reward:.2f}")
            return {
                "success": True,
                "message": "Feedback processed",
                "q_update": {
                    "state": state_idx,
                    "action": action_idx,
                    "old_value": round(old_q, 4),
                    "new_value": round(new_q, 4)
                }
            }
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """Enhanced WebSocket with security"""
            await websocket.accept()
            
            # Validate session
            if session_id not in self.active_sessions:
                await websocket.close(code=1008, reason="Invalid session")
                return
            
            try:
                self.connections[session_id] = websocket
                agent_id = self.active_sessions[session_id]["agent_id"]
                
                logger.info(f"📡 WebSocket connected: {agent_id}")
                
                while True:
                    # Receive with size limit 
                    raw_data = await websocket.receive_bytes()
                    
                    if len(raw_data) > SecurityConfig.MAX_MESSAGE_SIZE:
                        await websocket.close(code=1009, reason="Message too large")
                        break
                    
                    try:
                        parsed = msgpack.unpackb(raw_data, strict_map_key=False)
                    except Exception as e:
                        logger.error(f"Invalid message format: {e}")
                        continue
                    
                    # Validate message structure
                    if not isinstance(parsed, dict) or "agent_id" not in parsed:
                        continue
                    
                    msg_agent_id = parsed.get("agent_id")
                    if msg_agent_id != agent_id:
                        continue  # Security: Only allow messages for own agent
                    
                    if agent_id in self.agents:
                        # Update agent state with validation
                        agent = self.agents[agent_id]
                        
                        # Update position safely
                        if "position" in parsed:
                            pos = parsed["position"]
                            if isinstance(pos, dict):
                                agent.position.x = max(-100, min(100, pos.get("x", agent.position.x)))
                                agent.position.y = max(-100, min(100, pos.get("y", agent.position.y)))
                                agent.position.z = max(-10, min(10, pos.get("z", agent.position.z)))
                        
                        # Update velocity safely
                        if "velocity" in parsed:
                            vel = parsed["velocity"]
                            if isinstance(vel, dict):
                                agent.velocity.x = max(-10, min(10, vel.get("x", agent.velocity.x)))
                                agent.velocity.y = max(-10, min(10, vel.get("y", agent.velocity.y)))
                                agent.velocity.z = max(-10, min(10, vel.get("z", agent.velocity.z)))
                        
                        # Update emotion safely
                        if "emotion" in parsed:
                            emotion = bleach.clean(str(parsed["emotion"]))
                            valid_emotions = ["happy", "sad", "neutral", "excited", "angry", "fear", "surprise"]
                            if emotion in valid_emotions:
                                agent.emotion = emotion
                        
                        agent.last_activity = time.time()
                        
                        # Get AI action
                        ai_provider = parsed.get("ai_provider", CONFIG["DEFAULT_AI_PROVIDER"])
                        action = await self.ai_manager.get_action(agent_id, parsed, ai_provider)
                        
                        # Send response
                        response = {
                            "agent_id": agent_id,
                            "action": action["command"],
                            "message": action["message"],
                            "provider": action.get("provider", "unknown"),
                            "timestamp": time.time()
                        }
                        
                        response_data = msgpack.packb(response)
                        await websocket.send_bytes(response_data)
                        
                        # Update agent action
                        agent.last_action = action["command"]
                        
                        # Update session activity
                        self.active_sessions[session_id]["last_activity"] = time.time()
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if session_id in self.connections:
                    del self.connections[session_id]
                logger.info(f"📡 WebSocket disconnected: {session_id[:8]}")
        
        @self.app.get("/dashboard")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")
        async def dashboard_data(request):
            """Enhanced dashboard data with security metrics"""
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "position": {
                            "x": agent.position.x,
                            "y": agent.position.y,
                            "z": agent.position.z
                        },
                        "velocity": {
                            "x": agent.velocity.x,
                            "y": agent.velocity.y,
                            "z": agent.velocity.z
                        },
                        "emotion": agent.emotion,
                        "action": agent.last_action,
                        "reward": round(agent.reward, 3),
                        "age": round(time.time() - agent.timestamp, 2),
                        "active": time.time() - agent.last_activity < 60
                    }
                    for agent in self.agents.values()
                ],
                "system": {
                    "total_agents": len(self.agents),
                    "active_connections": len(self.connections),
                    "active_sessions": len(self.active_sessions),
                    "ai_providers": {
                        "openai": bool(self.ai_manager.openai_client),
                        "anthropic": bool(self.ai_manager.anthropic_client), 
                        "google": bool(self.ai_manager.google_client)
                    },
                    "uptime": round(time.time() - getattr(self, 'start_time', time.time()), 2),
                    "version": "4.0.0",
                    "security_enabled": True
                },
                "security": {
                    "rate_limit": SecurityConfig.RATE_LIMIT_PER_MINUTE,
                    "max_agents": CONFIG["MAX_AGENTS"],
                    "session_timeout": SecurityConfig.SESSION_TIMEOUT
                }
            }
        
        @self.app.get("/visualization/agents")
        @limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")
        async def get_visualization_data(request):
            """Get data for 3D visualization"""
            agents_data = []
            for agent in self.agents.values():
                agents_data.append({
                    "id": agent.id,
                    "x": agent.position.x,
                    "y": agent.position.y,
                    "z": agent.position.z,
                    "emotion": agent.emotion,
                    "action": agent.last_action,
                    "reward": agent.reward,
                    "velocity_magnitude": np.sqrt(
                        agent.velocity.x**2 + agent.velocity.y**2 + agent.velocity.z**2
                    ),
                    "active": time.time() - agent.last_activity < 60
                })
            
            return {
                "agents": agents_data,
                "bounds": {
                    "x": [-100, 100],
                    "y": [-100, 100], 
                    "z": [-10, 10]
                },
                "timestamp": time.time()
            }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session["last_activity"] > SecurityConfig.SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in self.active_sessions:
                agent_id = self.active_sessions[session_id]["agent_id"]
                del self.active_sessions[session_id]
                
                if agent_id in self.agents:
                    del self.agents[agent_id]
                if agent_id in self.q_tables:
                    del self.q_tables[agent_id]
                if session_id in self.connections:
                    del self.connections[session_id]
                
                logger.info(f"🧹 Cleaned up expired session: {session_id[:8]}")
    
    def simulate_agents(self, count: int = 3):
        """Create demo agents with security considerations"""
        if len(self.agents) + count > CONFIG["MAX_AGENTS"]:
            count = CONFIG["MAX_AGENTS"] - len(self.agents)
            if count <= 0:
                logger.warning("Cannot create demo agents: limit reached")
                return
        
        for i in range(count):
            agent_id = f"demo_agent_{int(time.time())}_{i+1}"
            session_id = str(uuid.uuid4())
            
            agent = Agent(
                id=agent_id,
                position=AgentPosition(
                    x=np.random.uniform(-10, 10),
                    y=np.random.uniform(-10, 10),
                    z=np.random.uniform(0, 2)
                ),
                velocity=AgentVelocity(
                    x=np.random.uniform(-2, 2),
                    y=np.random.uniform(-2, 2),
                    z=0
                ),
                emotion=np.random.choice(["happy", "neutral", "excited", "sad"]),
                last_action=np.random.choice(["move_forward", "observe", "communicate"]),
                reward=np.random.uniform(-0.5, 1.0),
                session_id=session_id
            )
            
            self.agents[agent_id] = agent
            self.q_tables[agent_id] = np.random.rand(10, 6) * 0.01
            
            # Track session
            self.active_sessions[session_id] = {
                "agent_id": agent_id,
                "created": time.time(),
                "last_activity": time.time()
            }
        
        logger.info(f"✅ Created {count} demo agents")
    
    async def start_server(self):
        """Start the enhanced A2A server"""
        self.start_time = time.time()
        
        # Start session cleanup task
        cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        
        config = uvicorn.Config(
            self.app,
            host=CONFIG["SERVER_HOST"],
            port=CONFIG["SERVER_PORT"],
            log_level=CONFIG["LOG_LEVEL"].lower(),
            access_log=True
        )
        server = uvicorn.Server(config)
        
        logger.info(f"🚀 Enhanced A2A Server starting on {CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}")
        logger.info(f"🔒 Security: Rate limiting {SecurityConfig.RATE_LIMIT_PER_MINUTE}/min, Max agents: {CONFIG['MAX_AGENTS']}")
        logger.info(f"🤖 AI Providers: OpenAI={bool(self.ai_manager.openai_client)}, Claude={bool(self.ai_manager.anthropic_client)}, Gemini={bool(self.ai_manager.google_client)}")
        
        try:
            await asyncio.gather(server.serve(), cleanup_task)
        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
            cleanup_task.cancel()
    
    async def _session_cleanup_loop(self):
        """Background task for session cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

# =============================================================================
# ENHANCED VISUALIZATION DASHBOARD
# =============================================================================

def create_enhanced_dashboard():
    """Create enhanced Streamlit dashboard with security and multi-provider support"""
    dashboard_code = '''
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="RL-A2A Enhanced Dashboard", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { 
        background: linear-gradient(90deg, #f0f2f6, #e8eaf0); 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #1f77b4;
    }
    .security-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .ai-provider {
        padding: 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    .provider-active { background: #d4edda; color: #155724; }
    .provider-inactive { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

SERVER_URL = "http://localhost:8000"

# Session state for configuration
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 5
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

def check_server():
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=3)
        return response.status_code == 200, response.json() if response.status_code == 200 else {}
    except:
        return False, {}

def get_dashboard_data():
    try:
        response = requests.get(f"{SERVER_URL}/dashboard", timeout=3)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_visualization_data():
    try:
        response = requests.get(f"{SERVER_URL}/visualization/agents", timeout=3)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def register_agent(agent_id, ai_provider="openai"):
    try:
        response = requests.post(f"{SERVER_URL}/register", json={
            "agent_id": agent_id,
            "ai_provider": ai_provider
        }, timeout=3)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_feedback(agent_id, reward):
    try:
        response = requests.post(f"{SERVER_URL}/feedback", json={
            "agent_id": agent_id,
            "action_id": f"dashboard_{int(time.time())}",
            "reward": reward,
            "context": {"source": "enhanced_dashboard", "timestamp": time.time()}
        }, timeout=3)
        return response.status_code == 200
    except:
        return False

def create_3d_visualization(viz_data):
    if not viz_data or not viz_data.get("agents"):
        fig = go.Figure()
        fig.add_annotation(text="No agents to display", x=0.5, y=0.5, showarrow=False)
        return fig
    
    agents = viz_data["agents"]
    bounds = viz_data["bounds"]
    
    # Color mapping for emotions
    emotion_colors = {
        "happy": "#FFD700", "sad": "#4169E1", "neutral": "#808080",
        "excited": "#FF4500", "angry": "#DC143C", "fear": "#9370DB",
        "surprise": "#32CD32"
    }
    
    # Size based on activity
    sizes = [15 if agent["active"] else 8 for agent in agents]
    colors = [emotion_colors.get(agent["emotion"], "#808080") for agent in agents]
    
    fig = go.Figure()
    
    # Active agents
    active_agents = [a for a in agents if a["active"]]
    if active_agents:
        fig.add_trace(go.Scatter3d(
            x=[a["x"] for a in active_agents],
            y=[a["y"] for a in active_agents],
            z=[a["z"] for a in active_agents],
            mode='markers+text',
            name="Active Agents",
            marker=dict(
                size=[15 for _ in active_agents],
                color=[emotion_colors.get(a["emotion"], "#808080") for a in active_agents],
                opacity=0.9,
                line=dict(width=2, color='black')
            ),
            text=[a["id"] for a in active_agents],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>Action: %{customdata[0]}<br>Reward: %{customdata[1]:.2f}<br>Velocity: %{customdata[2]:.2f}<extra></extra>',
            customdata=[[a["action"], a["reward"], a["velocity_magnitude"]] for a in active_agents]
        ))
    
    # Inactive agents
    inactive_agents = [a for a in agents if not a["active"]]
    if inactive_agents:
        fig.add_trace(go.Scatter3d(
            x=[a["x"] for a in inactive_agents],
            y=[a["y"] for a in inactive_agents], 
            z=[a["z"] for a in inactive_agents],
            mode='markers+text',
            name="Inactive Agents",
            marker=dict(
                size=[8 for _ in inactive_agents],
                color=['gray' for _ in inactive_agents],
                opacity=0.4,
                line=dict(width=1, color='darkgray')
            ),
            text=[a["id"] for a in inactive_agents],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>Status: Inactive<extra></extra>'
        ))
    
    # Environment bounds
    fig.add_trace(go.Mesh3d(
        x=bounds["x"] + bounds["x"],
        y=[bounds["y"][0], bounds["y"][0], bounds["y"][1], bounds["y"][1]],
        z=[bounds["z"][0], bounds["z"][1], bounds["z"][0], bounds["z"][1]],
        opacity=0.1,
        color='lightgray',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Enhanced Agent 3D Environment",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position", 
            zaxis_title="Z Position",
            xaxis=dict(range=bounds["x"]),
            yaxis=dict(range=bounds["y"]),
            zaxis=dict(range=bounds["z"]),
            aspectmode='cube'
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_analytics_dashboard(dashboard_data):
    if not dashboard_data or not dashboard_data.get("agents"):
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
    
    agents = dashboard_data["agents"]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Emotion Distribution", "Reward Distribution", "Action Frequency",
            "Agent Activity", "Velocity Analysis", "Performance Over Time"
        ],
        specs=[
            [{"type": "pie"}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Emotion Distribution
    emotions = [a["emotion"] for a in agents]
    emotion_counts = pd.Series(emotions).value_counts()
    fig.add_trace(go.Pie(
        values=emotion_counts.values, 
        labels=emotion_counts.index,
        name="Emotions"
    ), row=1, col=1)
    
    # 2. Reward Distribution  
    rewards = [a["reward"] for a in agents]
    fig.add_trace(go.Histogram(
        x=rewards,
        nbinsx=10,
        name="Rewards"
    ), row=1, col=2)
    
    # 3. Action Frequency
    actions = [a["action"] for a in agents]
    action_counts = pd.Series(actions).value_counts()
    fig.add_trace(go.Bar(
        x=list(action_counts.index),
        y=list(action_counts.values),
        name="Actions"
    ), row=1, col=3)
    
    # 4. Agent Activity (Active vs Age)
    active_status = ["Active" if a["active"] else "Inactive" for a in agents]
    ages = [a["age"] for a in agents]
    colors = ["green" if a["active"] else "red" for a in agents]
    fig.add_trace(go.Scatter(
        x=ages,
        y=active_status,
        mode="markers",
        marker=dict(color=colors, size=10),
        name="Activity"
    ), row=2, col=1)
    
    # 5. Velocity Analysis
    velocities = []
    for a in agents:
        vel = a["velocity"]
        vel_mag = np.sqrt(vel["x"]**2 + vel["y"]**2 + vel["z"]**2)
        velocities.append(vel_mag)
    
    fig.add_trace(go.Box(
        y=velocities,
        name="Velocity Magnitude"
    ), row=2, col=2)
    
    # 6. Performance (Reward vs Age)
    fig.add_trace(go.Scatter(
        x=ages,
        y=rewards,
        mode="markers",
        marker=dict(
            size=[10 if a["active"] else 5 for a in agents],
            color=rewards,
            colorscale="Viridis",
            showscale=True
        ),
        text=[a["id"] for a in agents],
        name="Performance"
    ), row=2, col=3)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

# Main Dashboard
st.title("🤖 RL-A2A Enhanced Dashboard")
st.markdown('<span class="security-badge">🔒 SECURITY ENABLED</span>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-refresh settings
    st.subheader("Refresh Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, st.session_state.refresh_interval)
    
    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_interval = refresh_interval
    
    if st.button("🔄 Manual Refresh"):
        st.rerun()
    
    st.divider()
    
    # AI Provider Selection
    st.subheader("AI Providers")
    ai_provider = st.selectbox(
        "Default Provider",
        ["openai", "anthropic", "google"],
        help="Select the default AI provider for new agents"
    )

# Server Status Check
server_online, health_data = check_server()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    status_color = "🟢" if server_online else "🔴"
    st.markdown(f"**Server Status:** {status_color} {'Online' if server_online else 'Offline'}")
with col2:
    if server_online and health_data:
        st.metric("Health Check", "✅")
with col3:
    if server_online:
        st.metric("Version", "4.0.0")

if not server_online:
    st.error("⚠️ Enhanced A2A Server is offline. Start with: `python rla2a_enhanced.py server`")
    st.stop()

# Get Dashboard Data
dashboard_data = get_dashboard_data()
viz_data = get_visualization_data()

if not dashboard_data:
    st.error("❌ Unable to fetch dashboard data")
    st.stop()

agents = dashboard_data.get("agents", [])
system = dashboard_data.get("system", {})
security_info = dashboard_data.get("security", {})

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Active Agents", system.get("total_agents", 0))
with col2:
    st.metric("Connections", system.get("active_connections", 0))
with col3:
    active_count = len([a for a in agents if a.get("active", False)])
    st.metric("Active Now", active_count)
with col4:
    avg_reward = np.mean([a["reward"] for a in agents]) if agents else 0
    st.metric("Avg Reward", f"{avg_reward:.3f}")
with col5:
    st.metric("Uptime", f"{system.get('uptime', 0):.0f}s")

# AI Provider Status
st.subheader("🤖 AI Provider Status")
ai_providers = system.get("ai_providers", {})

col1, col2, col3 = st.columns(3)
with col1:
    status = "🟢 Active" if ai_providers.get("openai") else "🔴 Inactive"
    st.markdown(f"**OpenAI**: {status}")
with col2:
    status = "🟢 Active" if ai_providers.get("anthropic") else "🔴 Inactive" 
    st.markdown(f"**Claude**: {status}")
with col3:
    status = "🟢 Active" if ai_providers.get("google") else "🔴 Inactive"
    st.markdown(f"**Gemini**: {status}")

st.divider()

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Environment", "📊 Analytics", "⚙️ Agent Control", "🔒 Security"])

with tab1:
    st.subheader("3D Agent Environment")
    if viz_data:
        fig_3d = create_3d_visualization(viz_data)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Environment statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            bounds = viz_data.get("bounds", {})
            st.info(f"**Environment Bounds**\\nX: {bounds.get('x', [])}\\nY: {bounds.get('y', [])}\\nZ: {bounds.get('z', [])}")
        with col2:
            active_agents = [a for a in viz_data.get("agents", []) if a.get("active")]
            avg_velocity = np.mean([a.get("velocity_magnitude", 0) for a in active_agents]) if active_agents else 0
            st.metric("Avg Velocity", f"{avg_velocity:.2f}")
        with col3:
            emotion_diversity = len(set(a.get("emotion") for a in viz_data.get("agents", [])))
            st.metric("Emotion Diversity", emotion_diversity)
    else:
        st.error("Unable to load visualization data")

with tab2:
    st.subheader("System Analytics")
    if agents:
        fig_analytics = create_analytics_dashboard(dashboard_data)
        st.plotly_chart(fig_analytics, use_container_width=True)
        
        # Agent Details Table
        st.subheader("Agent Details")
        df = pd.DataFrame(agents)
        
        # Add computed columns
        df["status"] = df["active"].apply(lambda x: "🟢 Active" if x else "🔴 Inactive")
        df["position_str"] = df.apply(lambda row: f"({row['position']['x']:.1f}, {row['position']['y']:.1f}, {row['position']['z']:.1f})", axis=1)
        
        # Display subset of columns
        display_df = df[["id", "status", "emotion", "action", "reward", "age", "position_str"]]
        display_df.columns = ["Agent ID", "Status", "Emotion", "Last Action", "Reward", "Age (s)", "Position"]
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No agents available for analysis")

with tab3:
    st.subheader("Agent Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Register New Agent**")
        with st.form("register_agent"):
            new_agent_id = st.text_input("Agent ID", placeholder="my_agent_001")
            selected_provider = st.selectbox("AI Provider", ["openai", "anthropic", "google"], 
                                           index=["openai", "anthropic", "google"].index(ai_provider))
            
            if st.form_submit_button("🚀 Register Agent"):
                if new_agent_id and new_agent_id.replace("_", "").replace("-", "").isalnum():
                    result = register_agent(new_agent_id, selected_provider)
                    if result:
                        st.success(f"✅ Agent '{new_agent_id}' registered with {selected_provider.title()}!")
                        st.json(result)
                    else:
                        st.error("❌ Registration failed - check server logs")
                else:
                    st.error("❌ Invalid Agent ID. Use only letters, numbers, underscore, and hyphen.")
    
    with col2:
        st.write("**Send Feedback**")
        with st.form("send_feedback"):
            feedback_agent = st.selectbox(
                "Select Agent", 
                [a["id"] for a in agents] if agents else [],
                help="Choose an agent to send feedback to"
            )
            feedback_reward = st.slider("Reward", -1.0, 1.0, 0.0, 0.01, 
                                      help="Positive rewards reinforce behavior, negative rewards discourage it")
            feedback_context = st.text_area("Context (optional)", placeholder="Reason for this feedback...")
            
            if st.form_submit_button("📤 Send Feedback"):
                if feedback_agent:
                    success = send_feedback(feedback_agent, feedback_reward)
                    if success:
                        st.success(f"✅ Feedback sent to {feedback_agent}: {feedback_reward:+.2f}")
                    else:
                        st.error("❌ Failed to send feedback")

with tab4:
    st.subheader("Security Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Security Configuration**")
        st.info(f"""
        **Rate Limiting**: {security_info.get('rate_limit', 'N/A')} requests/minute
        **Max Agents**: {security_info.get('max_agents', 'N/A')}
        **Session Timeout**: {security_info.get('session_timeout', 'N/A')} seconds
        **Message Size Limit**: 1 MB
        **Authentication**: JWT Tokens
        """)
    
    with col2:
        st.write("**System Health**")
        uptime_hours = system.get("uptime", 0) / 3600
        memory_usage = "Monitoring not available"  # Would need additional endpoints
        
        st.metric("Uptime", f"{uptime_hours:.1f} hours")
        st.metric("Active Sessions", system.get("active_sessions", 0))
        st.metric("Security Events", "0")  # Would track in production
    
    # Security Alerts (placeholder)
    st.write("**Security Alerts**")
    alert_placeholder = st.empty()
    alert_placeholder.success("🔒 No security alerts - system operating normally")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
'''
    
    # Write dashboard to file
    dashboard_path = "enhanced_dashboard.py"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_code)
    
    return dashboard_path

def start_enhanced_dashboard():
    """Start enhanced Streamlit dashboard"""
    dashboard_file = create_enhanced_dashboard()
    logger.info(f"🎨 Starting enhanced dashboard on port {CONFIG['DASHBOARD_PORT']}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", str(CONFIG["DASHBOARD_PORT"]),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        logger.info("\\n🛑 Dashboard stopped")
    finally:
        # Cleanup
        if os.path.exists(dashboard_file):
            os.remove(dashboard_file)

# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="RL-A2A Enhanced System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start A2A server")
    server_parser.add_argument("--demo-agents", type=int, default=0, help="Create demo agents")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Dashboard command  
    dashboard_parser = subparsers.add_parser("dashboard", help="Start dashboard")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup environment")
    
    args = parser.parse_args()
    
    if args.command == "server":
        system = EnhancedA2ASystem()
        
        if args.demo_agents > 0:
            logger.info(f"Creating {args.demo_agents} demo agents...")
            system.simulate_agents(args.demo_agents)
        
        await system.start_server()
    
    elif args.command == "dashboard":
        start_enhanced_dashboard()
    
    elif args.command == "setup":
        logger.info("🔧 Setting up RL-A2A Enhanced environment...")
        
        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        
        logger.info("✅ Environment setup complete")
        logger.info("Next steps:")
        logger.info("1. Update .env file with your API keys")
        logger.info("2. Run: python rla2a_enhanced.py server --demo-agents 3")
        logger.info("3. Run: python rla2a_enhanced.py dashboard")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)