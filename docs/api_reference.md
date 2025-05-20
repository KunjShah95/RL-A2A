# API Reference

This document provides detailed information about the Agent-to-Agent Protocol API, including server endpoints, client methods, data models, and usage examples.

## Table of Contents

- [Server API](#server-api)
  - [Registration Endpoint](#registration-endpoint)
  - [WebSocket Endpoint](#websocket-endpoint)
  - [Feedback Endpoint](#feedback-endpoint)
- [Agent API](#agent-api)
  - [AgentClient Class](#agentclient-class)
  - [Observation Model](#observation-model)
  - [Action Model](#action-model)
  - [Feedback Model](#feedback-model)
- [Reinforcement Learning](#reinforcement-learning)
  - [Q-Learning Implementation](#q-learning-implementation)
  - [Reward System](#reward-system)
- [Error Handling](#error-handling)

## Server API

The A2A server provides several endpoints for agent registration, communication, and feedback processing.

### Registration Endpoint

Registers a new agent with the server and returns a session ID.

- **URL:** `/register`
- **Method:** `POST`
- **Parameters:**
  - `agent_id` (string): Unique identifier for the agent

#### Request Example

```
curl -X POST "http://localhost:8000/register?agent_id=AgentA"
```

#### Response Example

```
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_id": "AgentA"
}
```

#### Status Codes

- `200 OK`: Registration successful
- `400 Bad Request`: Invalid agent ID
- `500 Internal Server Error`: Server error

### WebSocket Endpoint

Establishes a WebSocket connection for real-time agent communication.

- **URL:** `/ws/{session_id}`
- **Protocol:** WebSocket
- **Message Format:** MessagePack serialized objects

#### Connection Example

```
import websockets
import msgpack

async def connect():
    uri = f"ws://localhost:8000/ws/{session_id}"
    async with websockets.connect(uri) as websocket:
        # Send observation
        observation = {
            "agent_id": "AgentA",
            "session_id": session_id,
            "position": {"x": 0, "y": 0},
            "velocity": {"x": 0.5, "y": 0.1},
            "emotion": "neutral"
        }
        await websocket.send(msgpack.packb(observation, use_bin_type=True))
        
        # Receive action
        response_bytes = await websocket.recv()
        action = msgpack.unpackb(response_bytes, raw=False)
        print(f"Received action: {action}")
```

#### Message Schemas

**Observation Schema:**
```
{
  "agent_id": str,
  "session_id": str,
  "position": dict,
  "velocity": dict,
  "emotion": str,
  "target_agent_id": Optional[str]
}
```

**Action Schema:**
```
{
  "agent_id": str,
  "command": str,
  "message": str,
  "success_probability": float
}
```

### Feedback Endpoint

Processes reward feedback for reinforcement learning.

- **URL:** `/feedback`
- **Method:** `POST`
- **Body:**
  ```
  {
    "agent_id": "agent-id-string",
    "action_id": "action-id-string",
    "reward": 0.5,
    "context": {
      "emotion": "happy"
    }
  }
  ```

#### Request Example

```
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "AgentA", "action_id": "123", "reward": 0.5, "context": {"emotion": "happy"}}'
```

#### Response Example

```
{
  "status": "feedback processed",
  "agent_id": "AgentA"
}
```

#### Status Codes

- `200 OK`: Feedback processed successfully
- `404 Not Found`: Agent not found
- `400 Bad Request`: Invalid feedback data
- `500 Internal Server Error`: Server error

## Agent API

### AgentClient Class

The `AgentClient` class provides a high-level interface for agents to interact with the A2A server.

#### Constructor

```
def __init__(self, agent_id: str, server_url: str = "localhost:8000", max_retries: int = 3):
    """
    Initialize agent client with ID and server URL.
    
    Args:
        agent_id: Unique identifier for the agent
        server_url: URL of the A2A server
        max_retries: Maximum number of connection retry attempts
    """
```

#### Methods

**register**

```
async def register(self) -> str:
    """
    Register agent with server and get session ID.
    
    Returns:
        Session ID string
        
    Raises:
        ConnectionError: If unable to connect to the server
    """
```

**send_feedback**

```
async def send_feedback(self, reward: float, context: Optional[Dict] = None):
    """
    Send feedback for reinforcement learning.
    
    Args:
        reward: Numerical reward value (-1.0 to 1.0)
        context: Optional context information
        
    Returns:
        Server response or None if failed
    """
```

**run**

```
async def run(self, iterations: int = 3):
    """
    Run the agent's main loop.
    
    Args:
        iterations: Number of observation-action cycles to perform
    """
```

**analyze_performance**

```
def analyze_performance(self):
    """
    Analyze agent performance based on history.
    
    Returns:
        Dictionary containing performance metrics
    """
```

### Observation Model

The `Observation` class represents an agent's perception of its environment.

```
class Observation(BaseModel):
    agent_id: str
    session_id: Optional[str] = None
    position: dict
    velocity: dict
    emotion: str
    target_agent_id: Optional[str] = None  # For direct messaging
```

### Action Model

The `Action` class represents a command or message sent to an agent.

```
class Action(BaseModel):
    agent_id: str
    command: str
    message: str
    success_probability: float = 1.0
```

### Feedback Model

The `Feedback` class represents reward feedback for reinforcement learning.

```
class Feedback(BaseModel):
    agent_id: str
    action_id: str
    reward: float
    context: Optional[dict] = None
```

## Reinforcement Learning

### Q-Learning Implementation

The A2A server uses a simple Q-learning implementation to learn optimal actions for agents based on their state and received rewards.

```
def update_q_model(agent_id: str, state: str, action: str, reward: float, next_state: str):
    """
    Update Q-learning model with reward feedback.
    
    Args:
        agent_id: Agent identifier
        state: Current state (e.g., emotion)
        action: Action taken
        reward: Reward received
        next_state: Resulting state
    """
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
```

### Reward System

Agents provide feedback to the server based on the appropriateness of actions for their current state. The reward system uses the following heuristics:

- Positive reward (1.0) for appropriate actions (e.g., "retreat" when "afraid")
- Neutral reward (0.5) for partially appropriate actions
- Low reward (0.1) for inappropriate actions

Example reward calculation:

```
def calculate_reward(emotion: str, action: str) -> float:
    if emotion == "afraid" and action == "retreat":
        return 1.0
    elif emotion == "happy" and action == "engage":
        return 1.0
    elif emotion == "angry" and action == "observe":
        return 0.5
    else:
        return 0.1
```

## Error Handling

The A2A protocol implements error handling for various scenarios:

- Connection failures: Exponential backoff retry mechanism
- Invalid session IDs: WebSocket connection closed with code 1008
- Missing agents: 404 Not Found response
- Server errors: 500 Internal Server Error response

Example error handling for connection failures:

```
async def register(self):
    """Register agent with server and get session ID"""
    retries = 0
    while retries < self.max_retries:
        try:
            response = requests.post(
                f"http://{self.server_url}/register",
                params={"agent_id": self.agent_id},
                timeout=5
            )
            data = response.json()
            self.session_id = data["session_id"]
            return self.session_id
        except requests.exceptions.ConnectionError:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            print(f"Server connection failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print(f"Could not connect to server after {self.max_retries} attempts.")
    sys.exit(1)
```
