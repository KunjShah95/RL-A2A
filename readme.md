# Autonomous Agent-to-Agent Protocol (A2A) in Python

<p align="center">
  <img src="docs/assets/a2a_logo.png" alt="A2A Protocol Logo" width="200"/>
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/KunjShah01/AGENT-TO-AGENT-PROTOCOL-/issues"><img src="https://img.shields.io/github/issues/KunjShah01/AGENT-TO-AGENT-PROTOCOL-" alt="Issues"></a>
  <a href="https://github.com/KunjShah01/AGENT-TO-AGENT-PROTOCOL-/stargazers"><img src="https://img.shields.io/github/stars/KunjShah01/AGENT-TO-AGENT-PROTOCOL-" alt="Stars"></a>
</p>

A comprehensive framework, guide, and practical examples for building autonomous agents that communicate, learn, and interact using Python. This repository covers fundamental agent concepts, architectures, implementation details, and a working protocol for agent-to-agent (A2A) communication leveraging modern Python tools and reinforcement learning.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Agent Architectures](#agent-architectures)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Server API](#server-api)
  - [Agent API](#agent-api)
- [Examples](#examples)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [Future Roadmap](#future-roadmap)
- [Changelog](#changelog)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Overview

Autonomous agents are software entities capable of perceiving their environment, reasoning about goals, and acting to achieve them without direct human intervention. This repository explores the principles and implementation of such agents, focusing on agent-to-agent communication and reinforcement learning for adaptive behaviors.

The A2A Protocol provides a standardized way for autonomous agents to:
- Register and establish communication channels
- Exchange observations and actions
- Learn from feedback through reinforcement learning
- Coordinate in multi-agent environments
- Adapt behavior based on experience

## Key Concepts

- **Perception:** Gathering information from the environment (e.g., position, velocity, emotion).
- **Reasoning:** Making decisions, planning, and learning from experience.
- **Action:** Interacting with the environment or other agents.
- **Goals:** Objectives that the agent strives to achieve.
- **Environment:** The context in which agents operate and interact, including other agents.
- **Reinforcement Learning:** Learning optimal behaviors through trial and error with feedback.
- **Agent Communication:** Standardized protocols for agents to exchange information.

## Agent Architectures

The repository explores and demonstrates several agent architectures:

- **Reactive Agents:** Respond directly to sensory input without maintaining internal state.
  - Simple stimulus-response mapping
  - Fast response time
  - Limited reasoning capability

- **Deliberative Agents:** Use planning and reasoning for goal achievement.
  - Maintain internal world model
  - Plan sequences of actions
  - Consider future consequences

- **Hybrid Agents:** Combine reactive and deliberative components.
  - Reactive layer for immediate responses
  - Deliberative layer for planning
  - Coordination between layers

- **Utility-Based Agents:** Make decisions to maximize expected utility.
  - Assign utility values to states
  - Choose actions that maximize utility
  - Balance multiple competing objectives

- **Learning Agents:** Improve performance over time via experience and feedback.
  - Adapt to changing environments
  - Learn from successes and failures
  - Optimize behavior through reinforcement

## System Architecture

This repository implements a practical A2A protocol, combining a FastAPI-based communication server and Python-based agent clients:

```
┌─────────────────┐     WebSocket     ┌─────────────────┐
│                 │◄───Connection────►│                 │
│   Agent Client  │                   │  A2A Server     │
│   (agent_a.py)  │                   │ (a2a_server.py) │
│                 │                   │                 │
└─────────────────┘                   └─────────────────┘
        │                                     │
        │                                     │
        ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│  Observations   │                   │  Reinforcement  │
│  & Feedback     │                   │  Learning       │
└─────────────────┘                   └─────────────────┘
```

## Features

### A2A Communication Server (`a2a_server.py`)

- **Tech Stack:** FastAPI, WebSockets, CORS, Pydantic, NumPy, MessagePack, asyncio
- **Features:**
  - **Agent Registration:** `/register` endpoint for agents to obtain session IDs
  - **WebSocket Endpoint:** `/ws/{session_id}` for real-time agent communication
  - **Reinforcement Learning:** Simple Q-learning model for each agent, learning optimal actions based on rewards
  - **Agent Messaging:** Supports both direct and queued messages between agents
  - **Feedback Processing:** `/feedback` endpoint for agents to send reinforcement rewards

### Agent Implementation Example (`agent_a.py`)

- **Tech Stack:** asyncio, websockets, msgpack, requests, NumPy
- **Features:**
  - **Registration:** Registers itself with the server and obtains a session ID
  - **Observation Loop:** Periodically sends its state (position, velocity, emotion) to the server
  - **Receives Actions:** Waits for server responses indicating which action to take
  - **Sends Feedback:** Provides reward feedback to the server for reinforcement learning
  - **Performance Analysis:** Tracks and analyzes action and reward history

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/KunjShah01/AGENT-TO-AGENT-PROTOCOL-.git
   cd AGENT-TO-AGENT-PROTOCOL-
   ```

2. Install required packages:
   ```bash
   pip install fastapi uvicorn websockets msgpack numpy pydantic requests
   ```

## Quick Start

### Running the Server

1. Start the A2A server:
   ```bash
   uvicorn a2a_server:app --reload
   ```

2. The server will start on `http://localhost:8000` with the following endpoints:
   - `/register` - POST endpoint for agent registration
   - `/ws/{session_id}` - WebSocket endpoint for agent communication
   - `/feedback` - POST endpoint for reinforcement learning feedback

### Running an Agent

1. In a separate terminal, run the example agent:
   ```bash
   python agent_a.py
   ```

2. The agent will:
   - Register with the server
   - Connect via WebSocket
   - Send observations
   - Receive suggested actions
   - Provide reward feedback to improve the server-side RL policy

### Creating Your Own Agent

1. Create a new Python file (e.g., `my_agent.py`)
2. Import the necessary libraries:
   ```python
   import asyncio
   import websockets
   import msgpack
   import requests
   import random
   ```
3. Use the `AgentClient` class as a template or create your own implementation
4. Customize the agent's behavior, state representation, and feedback mechanism

## API Reference

### Server API

#### Registration Endpoint

- **URL:** `/register`
- **Method:** `POST`
- **Parameters:**
  - `agent_id` (string): Unique identifier for the agent
- **Response:**
  ```json
  {
    "session_id": "uuid-string",
    "agent_id": "agent-id-string"
  }
  ```

#### WebSocket Endpoint

- **URL:** `/ws/{session_id}`
- **Protocol:** WebSocket
- **Message Format:** MessagePack serialized objects
- **Observation Schema:**
  ```python
  {
    "agent_id": str,
    "session_id": str,
    "position": dict,
    "velocity": dict,
    "emotion": str,
    "target_agent_id": Optional[str]
  }
  ```
- **Action Schema:**
  ```python
  {
    "agent_id": str,
    "command": str,
    "message": str,
    "success_probability": float
  }
  ```

#### Feedback Endpoint

- **URL:** `/feedback`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "agent_id": "agent-id-string",
    "action_id": "action-id-string",
    "reward": 0.5,
    "context": {
      "emotion": "happy"
    }
  }
  ```

### Agent API

#### AgentClient Class

```python
class AgentClient:
    def __init__(self, agent_id: str, server_url: str = "localhost:8000", max_retries: int = 3):
        """Initialize agent client with ID and server URL"""
        
    async def register(self) -> str:
        """Register agent with server and get session ID"""
        
    async def send_feedback(self, reward: float, context: Optional[Dict] = None):
        """Send feedback for reinforcement learning"""
        
    async def run(self, iterations: int = 3):
        """Run the agent's main loop"""
        
    def analyze_performance(self):
        """Analyze agent performance based on history"""
```

## Examples

### Basic Agent Example

```python
import asyncio
from agent_client import AgentClient

async def main():
    # Create and run agent
    agent = AgentClient("ExampleAgent", "localhost:8000")
    await agent.run(iterations=5)
    
    # Analyze performance
    performance = agent.analyze_performance()
    print(f"\n[{agent.agent_id} Performance]")
    print(performance)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent Interaction

```python
import asyncio
from agent_client import AgentClient

async def run_agents():
    # Create multiple agents
    agent1 = AgentClient("Agent1", "localhost:8000")
    agent2 = AgentClient("Agent2", "localhost:8000")
    
    # Run agents concurrently
    await asyncio.gather(
        agent1.run(iterations=5),
        agent2.run(iterations=5)
    )

if __name__ == "__main__":
    asyncio.run(run_agents())
```

## Directory Structure

```
.
├── a2a_server.py      # The FastAPI-based communication server with agent RL logic
├── agent_a.py         # Example implementation of an autonomous agent
├── README.md          # This documentation file
├── requirements.txt   # Project dependencies
├── docs/              # Additional documentation
│   ├── api/           # Detailed API documentation
│   ├── examples/      # Example code and tutorials
│   └── assets/        # Images and other assets
├── tests/             # Test cases
└── examples/          # Additional example implementations
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository and clone your fork
2. Create a new branch for your feature or bugfix
3. Add your implementation (e.g., new agent, protocol improvement, architecture demo)
4. Update documentation as needed
5. Submit a pull request describing your changes

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Include docstrings for all functions, classes, and modules
- Write unit tests for new functionality
- Update documentation to reflect changes
- Keep pull requests focused on a single change

## Future Roadmap

### Version 0.2.0 (Q3 2025)
- **Expand Agent Architectures:** Add more agent types (belief-desire-intention, multi-agent coordination, etc.)
- **Advanced Reinforcement Learning:** Support for deep RL (DQN, PPO) and more sophisticated reward strategies
- **Multiple Agent Demos:** Simulate multi-agent environments with competitive/cooperative scenarios

### Version 0.3.0 (Q4 2025)
- **Visualization:** Real-time dashboards for monitoring agent interactions and learning
- **Extensible Protocol:** Support for encrypted messages, agent authentication, and richer agent metadata
- **Documentation Expansion:** Full API docs, example notebooks, and tutorials

### Version 1.0.0 (Q1 2026)
- **Benchmarking:** Performance and scalability metrics for agent communication
- **Testing Framework:** Automated tests for server and agent logic
- **Production Readiness:** Stability improvements and performance optimizations

### Long-term Vision
- **Cloud Deployment:** Managed service for A2A protocol
- **Language Agnostic:** Support for multiple programming languages
- **Integration Ecosystem:** Connectors for popular AI and ML frameworks

## Changelog

### v0.1.0 (Current)
- Initial release with basic A2A protocol implementation
- FastAPI-based server with WebSocket support
- Example agent implementation
- Simple Q-learning reinforcement learning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MessagePack](https://msgpack.org/)
- [Reinforcement Learning (Q-learning)](https://en.wikipedia.org/wiki/Q-learning)
- [WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [Autonomous Agents](https://en.wikipedia.org/wiki/Intelligent_agent)

## Contact

For questions, suggestions, or collaboration, feel free to:
- Open an [issue](https://github.com/KunjShah01/AGENT-TO-AGENT-PROTOCOL-/issues)
- Submit a [pull request](https://github.com/KunjShah01/AGENT-TO-AGENT-PROTOCOL-/pulls)
- Contact the maintainer: [Kunj Shah](https://github.com/KunjShah01)
