try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, WebSocket, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import json
    import uuid
    import msgpack
    import asyncio
    from typing import Dict, List, Optional, Set
    import numpy as np
    import time
    from collections import deque
    print("All imports successful!")
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Please install required packages with:")
    print("pip install fastapi uvicorn websockets msgpack numpy pydantic")
    import sys
    sys.exit(1)

# Create a lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("A2A Agent Communication Server started")
    yield
    # Shutdown code
    print("A2A Agent Communication Server shutting down")

app = FastAPI(lifespan=lifespan)

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active websocket connections
active_connections: Dict[str, WebSocket] = {}
# Store agent session data
agent_sessions: Dict[str, dict] = {}
# Store agent RL models (simple Q-learning implementation)
agent_models: Dict[str, dict] = {}
# Message queues for agent-to-agent communication (replacing Redis)
message_queues: Dict[str, deque] = {}
# Background tasks
background_tasks: Set[asyncio.Task] = set()

# Pydantic models for request validation
class RegistrationRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")

class FeedbackRequest(BaseModel):
    agent_id: str = Field(..., description="Agent providing feedback")
    action_id: str = Field(..., description="Action being evaluated")
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward value between -1.0 and 1.0")
    context: Optional[Dict] = Field(default={}, description="Additional context information")

class ObservationRequest(BaseModel):
    agent_id: str
    session_id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    target_agent_id: Optional[str] = None

# RL Model management
def initialize_agent_model(agent_id: str):
    """Initialize Q-learning model for agent"""
    agent_models[agent_id] = {
        "q_table": np.random.rand(5, 4) * 0.1,  # 5 states, 4 actions
        "epsilon": 0.1,  # exploration rate
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "total_rewards": 0.0,
        "action_count": 0,
        "last_state": None,
        "last_action": None
    }

def get_action(agent_id: str, state: Optional[int] = None) -> int:
    """Get action using epsilon-greedy policy"""
    if agent_id not in agent_models:
        initialize_agent_model(agent_id)
    
    model = agent_models[agent_id]
    
    if state is None:
        state = np.random.randint(0, 5)  # Random state if not provided
    
    # Epsilon-greedy action selection
    if np.random.random() < model["epsilon"]:
        action = np.random.randint(0, 4)  # Random action
    else:
        action = np.argmax(model["q_table"][state])  # Best action
    
    model["last_state"] = state
    model["last_action"] = action
    model["action_count"] += 1
    
    return action

def update_q_table(agent_id: str, reward: float, next_state: Optional[int] = None):
    """Update Q-table based on reward feedback"""
    if agent_id not in agent_models:
        return
    
    model = agent_models[agent_id]
    
    if model["last_state"] is None or model["last_action"] is None:
        return
    
    if next_state is None:
        next_state = np.random.randint(0, 5)
    
    # Q-learning update rule
    old_q = model["q_table"][model["last_state"], model["last_action"]]
    next_max = np.max(model["q_table"][next_state])
    
    new_q = old_q + model["learning_rate"] * (
        reward + model["discount_factor"] * next_max - old_q
    )
    
    model["q_table"][model["last_state"], model["last_action"]] = new_q
    model["total_rewards"] += reward
    
    # Decay epsilon (reduce exploration over time)
    model["epsilon"] = max(0.01, model["epsilon"] * 0.995)

# API Routes
@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "A2A Agent Communication Server",
        "version": "1.0.0",
        "active_agents": len(agent_sessions),
        "active_connections": len(active_connections),
        "endpoints": {
            "register": "/register",
            "websocket": "/ws/{session_id}",
            "feedback": "/feedback",
            "status": "/status",
            "agents": "/agents"
        }
    }

@app.post("/register")
async def register_agent(agent_id: str):
    """Register a new agent and return session ID"""
    session_id = str(uuid.uuid4())
    
    # Store session information
    agent_sessions[session_id] = {
        "agent_id": agent_id,
        "session_id": session_id,
        "created_at": time.time(),
        "last_seen": time.time(),
        "status": "registered"
    }
    
    # Initialize RL model
    initialize_agent_model(agent_id)
    
    # Initialize message queue
    if agent_id not in message_queues:
        message_queues[agent_id] = deque(maxlen=100)
    
    print(f"✅ Agent {agent_id} registered with session {session_id}")
    
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "status": "registered",
        "message": f"Agent {agent_id} successfully registered"
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent communication"""
    await websocket.accept()
    
    # Validate session
    if session_id not in agent_sessions:
        await websocket.close(code=4004, reason="Invalid session ID")
        return
    
    agent_info = agent_sessions[session_id]
    agent_id = agent_info["agent_id"]
    active_connections[session_id] = websocket
    
    print(f"🔄 Agent {agent_id} connected via WebSocket")
    
    try:
        while True:
            # Receive observation from agent
            data = await websocket.receive_bytes()
            observation = msgpack.unpackb(data)
            
            # Update last seen
            agent_sessions[session_id]["last_seen"] = time.time()
            
            # Process observation and get action
            agent_id = observation.get("agent_id", agent_id)
            
            # Simple state mapping from observation
            position = observation.get("position", {})
            emotion = observation.get("emotion", "neutral")
            
            # Map observation to state (simplified)
            state = hash(f"{position.get('x', 0):.1f}_{emotion}") % 5
            
            # Get action using RL
            action_idx = get_action(agent_id, state)
            action_names = ["move_forward", "turn_left", "turn_right", "communicate", "observe"]
            selected_action = action_names[action_idx]
            
            # Create response
            response = {
                "agent_id": agent_id,
                "command": selected_action,
                "message": f"Action selected: {selected_action}",
                "success_probability": float(np.random.random()),
                "timestamp": time.time(),
                "action_id": f"{agent_id}_{int(time.time())}_{action_idx}"
            }
            
            # Check for pending messages
            if agent_id in message_queues and message_queues[agent_id]:
                pending_message = message_queues[agent_id].popleft()
                response["pending_message"] = pending_message
            
            # Send response
            await websocket.send(msgpack.packb(response))
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"❌ WebSocket error for agent {agent_id}: {e}")
    
    finally:
        # Clean up connection
        if session_id in active_connections:
            del active_connections[session_id]
        agent_sessions[session_id]["status"] = "disconnected"
        print(f"🔌 Agent {agent_id} disconnected")

@app.post("/feedback")
async def receive_feedback(request: FeedbackRequest):
    """Receive feedback for reinforcement learning"""
    agent_id = request.agent_id
    
    if agent_id not in agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Update Q-table with reward
    update_q_table(agent_id, request.reward)
    
    print(f"📊 Feedback received - Agent: {agent_id}, Reward: {request.reward:.3f}")
    
    return {
        "success": True,
        "agent_id": agent_id,
        "action_id": request.action_id,
        "reward": request.reward,
        "context": request.context,
        "message": "Feedback processed successfully"
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "server_status": "running",
        "total_agents": len(agent_sessions),
        "active_connections": len(active_connections),
        "total_models": len(agent_models),
        "uptime": time.time(),
        "background_tasks": len(background_tasks)
    }

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = []
    
    for session_id, session_info in agent_sessions.items():
        agent_id = session_info["agent_id"]
        model = agent_models.get(agent_id, {})
        
        agents.append({
            "agent_id": agent_id,
            "session_id": session_id,
            "status": session_info.get("status", "unknown"),
            "connected": session_id in active_connections,
            "created_at": session_info.get("created_at"),
            "last_seen": session_info.get("last_seen"),
            "total_rewards": model.get("total_rewards", 0.0),
            "action_count": model.get("action_count", 0),
            "epsilon": model.get("epsilon", 0.1)
        })
    
    return {
        "agents": agents,
        "total_count": len(agents)
    }

@app.get("/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent"""
    if agent_id not in agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Find session for this agent
    agent_session = None
    for session_id, session_info in agent_sessions.items():
        if session_info["agent_id"] == agent_id:
            agent_session = session_info
            break
    
    model = agent_models[agent_id]
    
    return {
        "agent_id": agent_id,
        "session_info": agent_session,
        "model_stats": {
            "total_rewards": model["total_rewards"],
            "action_count": model["action_count"],
            "epsilon": model["epsilon"],
            "learning_rate": model["learning_rate"],
            "avg_reward": model["total_rewards"] / max(1, model["action_count"])
        },
        "q_table_stats": {
            "shape": model["q_table"].shape,
            "min_value": float(np.min(model["q_table"])),
            "max_value": float(np.max(model["q_table"])),
            "mean_value": float(np.mean(model["q_table"]))
        }
    }

@app.delete("/agents/{agent_id}")
async def remove_agent(agent_id: str):
    """Remove an agent from the system"""
    removed_sessions = []
    
    # Remove all sessions for this agent
    for session_id, session_info in list(agent_sessions.items()):
        if session_info["agent_id"] == agent_id:
            del agent_sessions[session_id]
            removed_sessions.append(session_id)
            
            # Close websocket if connected
            if session_id in active_connections:
                websocket = active_connections[session_id]
                await websocket.close(code=4001, reason="Agent removed")
                del active_connections[session_id]
    
    # Remove model and message queue
    if agent_id in agent_models:
        del agent_models[agent_id]
    
    if agent_id in message_queues:
        del message_queues[agent_id]
    
    print(f"🗑️ Agent {agent_id} removed from system")
    
    return {
        "success": True,
        "agent_id": agent_id,
        "removed_sessions": removed_sessions,
        "message": f"Agent {agent_id} removed successfully"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting A2A Agent Communication Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔄 WebSocket endpoint: ws://localhost:8000/ws/{session_id}")
    print("\n💡 To stop the server, press Ctrl+C")
    
    uvicorn.run(
        "a2a_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )