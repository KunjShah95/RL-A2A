# 🤖 RL-A2A Enhanced: Secure Multi-AI Agent Communication System

[![Security](https://img.shields.io/badge/Security-Enhanced-green.svg)](#security) [![AI Support](https://img.shields.io/badge/AI-OpenAI%20|%20Claude%20|%20Gemini-blue.svg)](#ai-providers) [![Version](https://img.shields.io/badge/Version-4.0.0-orange.svg)](#changelog) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

An enhanced, secure Agent-to-Agent (A2A) communication system with reinforcement learning, multi-AI provider support, real-time 3D visualization, and comprehensive security features.

## 🚀 What's New in v4.0 Enhanced

### 🔒 **Security Enhancements**
- **Data Poisoning Protection**: Input validation, sanitization, and size limits
- **JWT Authentication**: Secure token-based authentication system
- **Rate Limiting**: Configurable rate limiting on all endpoints
- **CORS Security**: Configurable allowed origins and trusted hosts
- **Session Management**: Secure session handling with automatic cleanup

### 🤖 **Multi-AI Provider Support**
- **OpenAI**: GPT-4o-mini integration with timeout protection
- **Anthropic**: Claude 3.5 Sonnet support with rate limiting
- **Google**: Gemini 1.5 Flash integration with error handling
- **Fallback System**: Graceful degradation when providers fail

### 📊 **Enhanced Visualization & Monitoring**
- **Real-time 3D Environment**: Interactive agent positioning and tracking
- **Security Dashboard**: Real-time security metrics and alerts
- **Analytics Dashboard**: Comprehensive performance tracking
- **Multi-dimensional Analysis**: Emotion, action, reward, and velocity visualization

## 📺 Project Overview & Objectives

RL-A2A Enhanced democratizes multi-agent system development by providing a comprehensive, production-ready platform that bridges research and real-world applications. Enhanced with enterprise-grade security, multi-AI provider support, and advanced monitoring capabilities.

**Core Objectives:**
- Simplify multi-agent system creation with security-first approach
- Provide real-time agent communication via secure WebSocket architecture  
- Integrate multi-AI provider decision making (OpenAI, Claude, Gemini)
- Deliver comprehensive visualization and security monitoring
- Enable seamless integration with AI assistants through enhanced MCP support

![RL-A2A System Capabilities](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/2f11457e-e527-4a9d-8aa4-e55a883d6aba.png)
![RL-A2A Core Features](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/5da08f4d-c65e-42a2-a5e5-582d02c7fc22.png)

## ✨ Enhanced Features

- 🤖 **Multi-Agent Communication** - Secure WebSocket-based real-time coordination
- 🧠 **Multi-AI Integration** - OpenAI, Claude & Gemini powered intelligent decision making
- 🎨 **Enhanced 3D Visualization** - Interactive Plotly dashboards with security monitoring
- 📈 **Advanced Reinforcement Learning** - Q-learning with adaptive feedback and validation
- 🔒 **Enterprise Security** - JWT authentication, rate limiting, data validation
- 🔌 **Enhanced MCP Support** - Secure Model Context Protocol for AI assistants
- ⚡ **Production Architecture** - Dual approach with Docker support and auto-configuration
- 🗺️ **Environment Management** - Comprehensive .env configuration with security defaults

## 🚀 Enhanced Quick Start

### 1. Enhanced Installation

```bash
# Clone the repository
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A

# Switch to enhanced branch (for latest security features)
git checkout security-fixes-and-enhancements

# Install enhanced dependencies with security packages
pip install -r requirements.txt

# Setup enhanced environment with security defaults
python rla2a_enhanced.py setup
```

### 2. Enhanced Configuration

Copy `.env.example` to `.env` and configure your enhanced settings:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and security configuration:

```bash
# Multi-AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Enhanced Security Configuration
SECRET_KEY=your-secret-key-for-jwt-signing
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501
RATE_LIMIT_PER_MINUTE=60

# System Configuration
DEFAULT_AI_PROVIDER=openai
MAX_AGENTS=100
DEBUG=false
```
uvicorn a2a_server:app --host 0.0.0.0   # Public access

# Multiple agents
python agent_a.py &    # Background agent
python agent_a.py      # Another agent

# Custom agent development
cp agent_a.py my_agent.py
# Edit my_agent.py for custom behavior
```

## 🧠 OpenAI Intelligence

Add your API key for intelligent agent behavior:

```bash
# Edit .env file (created by setup)
OPENAI_API_KEY=sk-your-api-key-here

# Agents automatically get GPT-4o-mini powers:
# • Intelligent situation analysis
# • Strategic decision making
# • Natural language communication
# • Adaptive learning from feedback
```

## 🎨 Visualization Dashboard

Interactive Streamlit dashboard with:

- **🌐 3D Agent Tracking** - Real-time positions with emotion-based colors
- **📊 Performance Metrics** - Rewards, emotions, activity analysis
- **⚙️ Agent Management** - Register agents, send feedback, control system
- **📈 Live Analytics** - Auto-refresh charts and system health
- **💾 Data Export** - Download agent data as CSV

## 🔌 MCP Integration

Control the system via AI assistants (Claude, ChatGPT, etc.):

```bash
python rla2a.py mcp  # Start MCP server
```

**Natural language commands:**
- "Start the RL-A2A system with 5 agents"
- "Create a new agent called explorer"
- "Show me the system status"

📖 **Detailed guide**: [MCP_GUIDE.md](MCP_GUIDE.md)

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["python", "rla2a.py", "server", "--demo-agents", "3"]
```

## 📚 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status |
| `/register?agent_id=X` | POST | Register agent |
| `/agents` | GET | List agents |
| `/feedback` | POST | Send RL feedback |
| `/ws/{session_id}` | WebSocket | Real-time comms |

## 🤝 Contributing

1. **Fork** the repository

```

## Architecture Options

### 🔧 All-in-One (`rla2a.py`) - *Recommended for Beginners*
- ✅ **Setup**: One command installation
- ✅ **Development**: Everything in one place  
- ✅ **Production**: Self-contained deployment
- ✅ **Learning**: Complete system overview
- ⚠️ **Customization**: Requires modifying large file

### ⚡ Modular (`a2a_server.py` + `agent_a.py`) - *Recommended for Development*
- ✅ **Setup**: Manual dependency management
- ✅ **Development**: Easy to extend individual parts
- ✅ **Production**: Scalable components
- ✅ **Learning**: Clear separation of concerns
- ✅ **Deployment**: Docker/container friendly
- ✅ **Customization**: Create new agent files easily


![Architecture Comparison](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/750b6068-343b-40b8-9179-65c5c78feee5.png)

#### Core Endpoints (Available in Both Architectures)
- **GET** `/` - System status
- **POST** `/register?agent_id=X` - Register agent
- **GET** `/agents` - List agents
- **POST** `/feedback` - Send RL feedback
- **WebSocket** `/ws/{session_id}` - Real-time communications
- **GET** `/health` - Health check

#### Additional Modular Endpoints
- **GET** `/agents/{id}` - Detailed agent information

### MCP Integration Tools

**Available Commands:**
- `start_system` - Launch complete system (parameters: `agents`, `dashboard`)
- `create_agent` - Create new agent instance (parameter: `agent_id`)

WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["python", "rla2a.py", "server", "--demo-agents", "3"]
```

### Multi-Container (Modular)
```yaml
# docker-compose.yml
version: '3.8'
services:
  a2a-server:
    build: .
    command: python a2a_server.py
    ports: ["8000:8000"]
    
  agent:
    build: .
    command: python agent_a.py
    depends_on: [a2a-server]
```

## 🔧 Development Guide

### Creating Custom Agents

**Modular approach** (recommended for development):
```python
# Copy and modify existing agent
import agent_a

class MyCustomAgent(agent_a.AgentClient):
    def execute_action(self, action_command: str) -> float:
        # Custom action logic
        return super().execute_action(action_command)
```

**All-in-one approach:**
```python
# Modify rla2a.py directly
class CustomA2ASystem(A2ASystem):
    async def get_action(self, agent_id: str, observation: Dict) -> Dict[str, str]:
        # Custom AI logic
        return await super().get_action(agent_id, observation)
```

### Adding New Features

1. **New Agent Capabilities**: Edit `agent_a.py` or create new agent files
2. **Server Features**: Add endpoints to `a2a_server.py`
3. **MCP Tools**: Extend MCP functionality in `rla2a.py`
4. **Visualization**: Add dashboard features via Streamlit integration

## 🤝 Contributing

1. **Fork** the repository
2. **Choose approach**: Modify `rla2a.py` (all-in-one) or individual files (modular)
3. **Test thoroughly**: Both approaches should work
4. **Update docs**: Keep README and guides current
5. **Submit PR**: Include test instructions

### Development Setup
```bash
# Clone for development
git clone https://github.com/YourUsername/RL-A2A.git
cd RL-A2A

# Test both approaches
python rla2a.py setup            # All-in-one
pip install -r requirements.txt  # Modular

# Run tests
python a2a_server.py &         # Background server
python agent_a.py              # Test agent

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🆘 Support & Community

## 🆘 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/KunjShah01/RL-A2A/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KunjShah01/RL-A2A/discussions)
- 📧 **Contact**: [Kunj Shah](https://github.com/KunjShah01)
- 📖 **Detailed MCP Guide**: [MCP_GUIDE.md](MCP_GUIDE.md)

## 🎯 What Makes This Special?

✅ **Dual Architecture** - Choose complexity level  
✅ **Production Ready** - Used in real applications  
✅ **Educational** - Learn from clean, documented code  
✅ **Extensible** - Add your own agent behaviors  
✅ **AI-Powered** - OpenAI integration out of the box  
✅ **Visual** - Beautiful real-time dashboards  
✅ **Standards** - MCP support for AI ecosystem  

---

**⭐ If this project helps you build amazing multi-agent systems, please give it a star!**

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

## 📈 Performance Visualization

![Multi-Agent Learning Progress](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/96e42532-7c49-47e9-a6ef-4850eb9729e4.png)

*Example of how agents improve their performance over time through reinforcement learning*

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
