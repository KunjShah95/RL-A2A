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

class Observation(BaseModel):
    agent_id: str
    session_id: Optional[str] = None
    position: dict
    velocity: dict
    emotion: str
    target_agent_id: Optional[str] = None  # For direct messaging
    
class Action(BaseModel):
    agent_id: str
    command: str
    message: str
    success_probability: float = 1.0
    
class Feedback(BaseModel):
    agent_id: str
    action_id: str
    reward: float
    context: Optional[dict] = None

def get_or_create_agent_model(agent_id: str):
    """Initialize or retrieve RL model for an agent"""
    if agent_id not in agent_models:
        # Simple Q-learning model with states based on emotions
        # and actions based on commands
        states = ["neutral", "afraid", "happy", "angry"]
        actions = ["explore", "retreat", "engage", "observe"]
        
        agent_models[agent_id] = {
            "q_table": np.zeros((len(states), len(actions))),
            "states": {state: i for i, state in enumerate(states)},
            "actions": {action: i for i, action in enumerate(actions)},
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "exploration_rate": 0.2
        }
    return agent_models[agent_id]

def update_q_model(agent_id: str, state: str, action: str, reward: float, next_state: str):
    """Update Q-learning model with reward feedback"""
    model = get_or_create_agent_model(agent_id)
    
    # Get indexes
    state_idx = model["states"].get(state, 0)
    action_idx = model["actions"].get(action, 0)
    next_state_idx = model["states"].get(next_state, 0)
    
    # Q-learning update
    old_value = model["q_table"][state_idx, action_idx]
    next_max = np.max(model["q_table"][next_state_idx])
    
    # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
    model["q_table"][state_idx, action_idx] += model["learning_rate"] * (
        reward + model["discount_factor"] * next_max - old_value
    )

def get_best_action(agent_id: str, state: str):
    """Get best action for state based on Q-learning model"""
    model = get_or_create_agent_model(agent_id)
    
    # Exploration vs exploitation
    if np.random.random() < model["exploration_rate"]:
        # Random action (exploration)
        return np.random.choice(list(model["actions"].keys()))
    
    # Get best action (exploitation)
    state_idx = model["states"].get(state, 0)
    action_idx = np.argmax(model["q_table"][state_idx])
    
    # Map index back to action name
    for action, idx in model["actions"].items():
        if idx == action_idx:
            return action
    
    # Fallback
    return "explore"

def get_message_queue(agent_id: str) -> deque:
    """Get or create message queue for an agent"""
    if agent_id not in message_queues:
        message_queues[agent_id] = deque(maxlen=100)  # Limit queue size
    return message_queues[agent_id]

def queue_message(agent_id: str, message: dict):
    """Add message to agent's queue"""
    queue = get_message_queue(agent_id)
    queue.append(message)
    print(f"[Queue] Message for {agent_id}: {message['message']} ({len(queue)} in queue)")

@app.post("/register")
async def register_agent(agent_id: str):
    """Register an agent and get a session ID"""
    session_id = str(uuid.uuid4())
    agent_sessions[session_id] = {
        "agent_id": agent_id,
        "created_at": time.time(),
        "last_action": None,
        "last_state": None
    }
    return {"session_id": session_id, "agent_id": agent_id}

@app.post("/feedback")
async def process_feedback(feedback: Feedback):
    """Process reward feedback for RL model updates"""
    agent_id = feedback.agent_id
    
    # Find session for this agent
    matching_sessions = [s for s in agent_sessions.values() if s["agent_id"] == agent_id]
    
    if matching_sessions:
        session = matching_sessions[0]
        
        if session["last_action"] and session["last_state"]:
            # Use current state as next state (simplified)
            current_state = feedback.context.get("emotion", "neutral") if feedback.context else "neutral" 
            
            # Update Q-learning model
            update_q_model(
                agent_id,
                session["last_state"],
                session["last_action"],
                feedback.reward,
                current_state
            )
            
            return {"status": "feedback processed", "agent_id": agent_id}
    
    raise HTTPException(status_code=404, detail="Agent not found")

async def process_message_queues(agent_id: str, websocket: WebSocket):
    """Background task to process message queue for an agent"""
    try:
        while True:
            queue = get_message_queue(agent_id)
            
            if queue:
                # Process one message at a time
                message = queue.popleft()
                
                action = Action(
                    agent_id=message.get("sender_id", "system"),
                    command=message.get("command", "notify"),
                    message=message.get("message", "System notification"),
                    success_probability=message.get("success_probability", 1.0)
                )
                
                await websocket.send_bytes(msgpack.packb(action.model_dump(), use_bin_type=True))
                print(f"[Sent Queue] Message to {agent_id} from {action.agent_id}")
            
            # Don't busy-wait
            await asyncio.sleep(0.5)
    except Exception as e:
        print(f"Message queue processing error for {agent_id}: {e}")

@app.websocket("/ws/{session_id}")
async def agent_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in agent_sessions:
        await websocket.close(code=1008, reason="Invalid session ID")
        return
    
    agent_id = agent_sessions[session_id]["agent_id"]
    active_connections[agent_id] = websocket
    print(f"[Server] Agent {agent_id} connected with session {session_id}")
    
    # Start background task for message queue processing
    task = asyncio.create_task(process_message_queues(agent_id, websocket))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    try:
        # Main websocket message loop
        while True:
            data = await websocket.receive_bytes()
            # Use MessagePack for serialization
            obs_dict = msgpack.unpackb(data, raw=False)
            obs = Observation(**obs_dict)
            print(f"[Observation] {obs.agent_id}: {obs.emotion} at {obs.position}")
            
            # Store current state for RL updates
            agent_sessions[session_id]["last_state"] = obs.emotion
            
            if obs.target_agent_id:
                # Direct agent-to-agent communication
                if obs.target_agent_id in active_connections:
                    # Send directly to target agent
                    action = Action(
                        agent_id=obs.agent_id,
                        command="message",
                        message=f"Direct message from {obs.agent_id}"
                    )
                    target_ws = active_connections[obs.target_agent_id]
                    await target_ws.send_bytes(msgpack.packb(action.model_dump(), use_bin_type=True))
                else:
                    # Queue message for later delivery
                    queue_message(obs.target_agent_id, {
                        "sender_id": obs.agent_id,
                        "command": "message",
                        "message": f"Async message from {obs.agent_id}"
                    })
            else:
                # Use RL model to determine best action
                best_action = get_best_action(agent_id, obs.emotion)
                agent_sessions[session_id]["last_action"] = best_action
                
                action = Action(
                    agent_id="system",
                    command=best_action,
                    message=f"{obs.agent_id} should {best_action} now.",
                    success_probability=0.9  # Could be determined by model confidence
                )
                await websocket.send_bytes(msgpack.packb(action.model_dump(), use_bin_type=True))

    except Exception as e:
        print(f"Disconnected or error for {agent_id}: {e}")
    finally:
        if agent_id in active_connections:
            del active_connections[agent_id]