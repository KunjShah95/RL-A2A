# RL-A2A: Complete Agent-to-Agent Communication System

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
A complete framework for building autonomous agents with OpenAI-powered intelligence, real-time visualization, and multi-agent communication. Features include reinforcement learning, interactive dashboards, and MCP integration - all in a clean, consolidated codebase.
## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Agent Architectures](#agent-architectures)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
## ✨ Key Features

- 🤖 **Multi-Agent Communication**: WebSocket-based real-time agent coordination
- 🧠 **OpenAI Intelligence**: GPT-powered decision making and natural language processing
- 🎨 **3D Visualization**: Real-time interactive dashboards with Plotly and Streamlit
- 📈 **Reinforcement Learning**: Q-learning with intelligent feedback systems
- 🔌 **MCP Integration**: Model Context Protocol support for AI assistant integration
- 🐳 **Easy Deployment**: Single-file system with Docker support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for AI features)

### Installation

```bash
# Clone the repository
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A

# Run setup (installs dependencies and creates config)
python setup.py

# Optional: Set your OpenAI API key
# Edit .env file: OPENAI_API_KEY=your-key-here
```

### Run the System

**Complete System** (Recommended):
```bash
python start_complete.py
# Opens dashboard at http://localhost:8501
```

**Individual Components:**
```bash
# A2A Server only
python start_server.py

# Dashboard only
python start_dashboard.py
```

**Advanced Usage:**
```bash
# All options available
python rl_a2a_system.py --mode complete --agents 5
python rl_a2a_system.py --mode server --host 0.0.0.0 --port 8000
python rl_a2a_system.py --mode mcp  # MCP server
python rl_a2a_system.py --mode demo # Generate demo report
```

## 📁 Project Structure

```
RL-A2A/
├── rl_a2a_system.py     # 🎯 Main system (all features combined)
├── dashboard.py         # 🎨 Streamlit dashboard
├── setup.py            # ⚙️ Simple setup script
├── requirements.txt    # 📦 Dependencies
├── .env               # 🔧 Configuration (created by setup)
├── start_*.py         # 🚀 Startup scripts (created by setup)
└── README.md          # 📚 This file
```

## 🎮 How It Works

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agents     │◄──►│   A2A Server    │◄──►│   Dashboard     │
│ (OpenAI Client) │    │ (WebSocket Hub) │    │ (Streamlit UI)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ GPT Intelligence│    │ Q-Learning RL   │    │ 3D Visualization│
│ Decision Making │    │ Agent Training  │    │ Real-time Plots │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```


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
## 🧠 OpenAI Integration

Agents powered by GPT models for intelligent behavior:

```python
# Set API key in .env file
OPENAI_API_KEY=sk-your-key-here

# Agents automatically get:
# • Situation analysis
# • Strategic decision making
# • Natural language communication
# • Adaptive learning from feedback
```

## 📊 Visualization Dashboard

Interactive Streamlit dashboard featuring:

- **3D Agent Tracking**: Real-time positions with emotion-based colors
- **Performance Metrics**: Reward trends, activity levels, system health
- **Agent Management**: Register new agents, send feedback
- **Data Export**: Download agent data as CSV
- **Auto-Refresh**: Live updates every 1-10 seconds

Access at: http://localhost:8501

## 🔌 MCP Integration

Control the system via AI assistants using Model Context Protocol:

```bash
# Start MCP server
python rl_a2a_system.py --mode mcp

# Configure your MCP client with:
# command: python
# args: ["rl_a2a_system.py", "--mode", "mcp"]
```

Available MCP commands:
- "Start the complete A2A system with 5 agents"
- "Create a new agent called explorer"
- "Generate a visualization report"

## 🛠️ Development

### System Components

All components are in `rl_a2a_system.py`:

1. **A2AServer**: FastAPI-based communication hub
2. **OpenAIAgentBrain**: GPT-powered intelligence
3. **IntelligentAgent**: AI-enhanced agent behavior
4. **PlotlyVisualizer**: Interactive visualization system
5. **MCPServer**: Model Context Protocol integration
6. **RLA2ASystem**: Main orchestrator

### Adding Features

```python
# Extend rl_a2a_system.py
class MyCustomAgent(IntelligentAgent):
    async def custom_behavior(self):
        # Your custom agent logic
        pass
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["python", "rl_a2a_system.py", "--mode", "complete"]
```
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
## MCP Integration

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol that enables AI systems to securely connect with external data sources and tools. By adding MCP support to RL-A2A, you can expose your agent communication protocol as tools that other AI systems can use.

### MCP Server Features

The RL-A2A MCP server provides:

#### Tools
- **register_agent**: Register new agents with the A2A system
- **send_agent_feedback**: Send reinforcement learning feedback for agent actions
- **create_agent_observation**: Create agent observation messages
- **start_a2a_server**: Get commands to start the A2A server

#### Resources
- **a2a://agents**: Information about currently active agents
- **a2a://models**: Reinforcement learning model data and progress
- **a2a://communication**: Agent communication logs and message history

### Quick MCP Setup

1. **Install MCP dependencies:**
   ```bash
   python setup_mcp.py
   ```

2. **Start the A2A server:**
   ```bash
   uvicorn a2a_server:app --reload
   ```

3. **Start the MCP server:**
   ```bash
   python start_mcp_server.py
   ```

4. **Connect MCP clients** (like Claude Desktop, Cline, etc.) using the configuration in `mcp_config.json`

### MCP Configuration

Add this to your MCP client configuration:

```json
{
  "mcpServers": {
    "rl-a2a": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
## OpenAI Integration

### Intelligent Agent Behavior

Enhance your A2A agents with OpenAI-powered intelligence for natural language understanding, strategic decision making, and adaptive learning.

#### Features
- **AI-Powered Decision Making**: Agents use GPT models to analyze situations and make intelligent choices
- **Natural Language Communication**: Agents can communicate in human-readable language
- **Intelligent Feedback**: AI evaluates action outcomes and provides meaningful reward signals
- **Adaptive Learning**: Agents learn and improve behavior over time

#### Quick Start

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Create intelligent agents:**
   ```python
   from openai_integration import IntelligentAgent, OpenAIEnhancedA2ASystem
   
   # Create an intelligent agent
   agent = IntelligentAgent("smart_explorer")
   await agent.register_with_a2a()
   
   # Run intelligent simulation
   system = OpenAIEnhancedA2ASystem()
   await system.run_multi_agent_simulation(num_agents=3, iterations=5)
   ```

3. **Start OpenAI demo:**
   ```bash
   python openai_integration.py
   ```

#### Available Models
- **gpt-4o-mini** (default): Fast and cost-effective
- **gpt-4**: Most capable for complex reasoning
- **gpt-3.5-turbo**: Balanced performance and cost

## Visualization Dashboard

### Real-Time Agent Monitoring

Visualize your agents in real-time with interactive 3D plots, performance metrics, and system analytics.

#### Features
- **3D Agent Tracking**: See agent positions, movements, and emotions in real-time
- **Performance Metrics**: Monitor rewards, learning progress, and system health
- **Interactive Dashboard**: Streamlit-based web interface with live updates
- **Communication Visualization**: Track inter-agent message flows
- **Export Reports**: Generate HTML reports with static visualizations

#### Visualization Types

1. **Streamlit Dashboard** (Recommended)
   ```bash
   streamlit run visualization.py -- streamlit
   # or
   python start_dashboard.py
   ```
   - Real-time updates
   - Interactive controls
   - Multi-panel layout
   - Web-based access at http://localhost:8501

2. **Matplotlib Animation**
   ```bash
   python visualization.py matplotlib
   ```
   - Desktop application
   - Smooth real-time animation
   - Customizable views

3. **Static HTML Reports**
   ```bash
   python visualization.py
   ```
   - Generates `a2a_visualization_report.html`
   - Plotly-powered interactive charts
   - Shareable reports

#### Dashboard Components

- **Agent Positions**: 3D scatter plot with emotion-based colors
- **Velocity Vectors**: Real-time movement direction and speed
- **Reward Tracking**: Learning progress over time
- **Agent Status Table**: Detailed information for each agent
- **Communication Network**: Message flow visualization
- **System Metrics**: Performance and health indicators


### Using MCP Tools

Once connected, you can use MCP commands like:
- "Register a new agent called 'explorer'" 
- "Send positive feedback for agent action XYZ"
- "Show me the current active agents"
- "Create an observation for agent at position (1,2,3)"
- "Create an intelligent agent with OpenAI"
- "Start the visualization dashboard"
- "Generate a visualization report"

### Enhanced MCP Server

The enhanced MCP server (`mcp_server_enhanced.py`) provides additional tools:

#### OpenAI Tools
- **create_intelligent_agent**: Create AI-powered agents
- **run_intelligent_simulation**: Run multi-agent AI simulations

#### Visualization Tools  
- **start_visualization_dashboard**: Launch real-time dashboard
- **generate_visualization_report**: Create HTML reports
- **capture_agent_screenshot**: Take system snapshots

#### System Tools
- **setup_complete_system**: One-command full system setup

### Complete Setup

For the full experience with all features:

```bash
# Complete setup with all features
python setup_full.py

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start everything
python start_a2a.py &          # A2A server
python start_dashboard.py &    # Visualization  
python start_openai_demo.py &  # AI agents
```


    }
  }
}
```

### Using MCP Tools

Once connected, you can use MCP commands like:
- "Register a new agent called 'explorer'" 
- "Send positive feedback for agent action XYZ"
- "Show me the current active agents"
- "Create an observation for agent at position (1,2,3)"



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
