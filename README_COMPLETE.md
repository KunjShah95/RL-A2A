# RL-A2A: Complete Agent Communication System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-ready-brightgreen.svg)

## 🎯 Project Aim

RL-A2A aims to democratize multi-agent system development by providing a comprehensive, production-ready platform that bridges the gap between research and real-world applications. The project solves the complexity barrier in agent communication, reinforcement learning integration, and system visualization - enabling developers to focus on agent logic rather than infrastructure.

**Core Objectives:**
- Simplify multi-agent system creation with both beginner-friendly and advanced approaches
- Provide real-time agent communication via WebSocket architecture
- Integrate AI-powered decision making through OpenAI APIs
- Deliver comprehensive visualization and monitoring capabilities
- Enable seamless integration with AI assistants through MCP support

![RL-A2A System Capabilities](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/c78959fd-b966-4e97-8d28-8a158d9bf6d0.png)

![RL-A2A Core Features](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/85c382fd-39a1-408c-9af2-86932736eb52.png)

## ✨ Features

- 🤖 **Multi-Agent Communication** - WebSocket-based real-time coordination
- 🧠 **OpenAI Integration** - GPT-powered intelligent decision making
- 🎨 **3D Visualization** - Interactive Plotly dashboards with Streamlit
- 📈 **Reinforcement Learning** - Q-learning with adaptive feedback
- 🔌 **MCP Support** - Model Context Protocol for AI assistants
- ⚡ **Dual Architecture** - Choose consolidated or modular approach
- 🐳 **Production Ready** - Docker support and auto-configuration

## 🚀 Quick Start

### Option 1: All-in-One (Recommended for Beginners)

```bash
# Clone and setup
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A

# One-time setup (installs dependencies, creates config)
python rla2a.py setup

# Start complete system (server + dashboard + 3 demo agents)
python rla2a.py server --demo-agents 3
```

### Option 2: Modular (Recommended for Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start A2A server
python a2a_server.py

# Terminal 2: Start an agent
python agent_a.py

# Terminal 3 (optional): Start dashboard
python rla2a.py dashboard
```

**Access Points:**
- 🎨 **Dashboard**: http://localhost:8501
- 🔗 **API Documentation**: http://localhost:8000/docs
- 📡 **WebSocket**: ws://localhost:8000/ws/{session_id}

## 📁 Repository Structure

```
RL-A2A/
├── 🎯 rla2a.py           # All-in-one system (complete functionality)
├── 📡 a2a_server.py      # Modular: FastAPI server with RL
├── 🤖 agent_a.py         # Modular: Example agent implementation
├── 📚 MCP_GUIDE.md       # Complete MCP integration guide
├── 📦 requirements.txt   # Dependencies
├── 📋 README.md         # This file
├── ⚙️ .env              # Config (auto-created by setup)
└── 🚀 start.py          # Launcher (auto-created by setup)
```

## 🏗️ Architecture Options

![Architecture Comparison](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/20410dab-5eac-41fc-81f5-6fd222091b86.png)

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

## 🎮 Usage Options

### All-in-One System Commands

```bash
# Complete system with demo agents
python rla2a.py server --demo-agents 5

# Interactive dashboard only
python rla2a.py dashboard

# MCP server for AI assistants
python rla2a.py mcp

# Generate HTML report
python rla2a.py report

# Setup environment
python rla2a.py setup
```

### Modular System Commands

```bash
# Advanced server options
python a2a_server.py                    # Default: localhost:8000
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

## 📈 Performance Visualization

![Multi-Agent Learning Progress](https://agents-storage.nyc3.digitaloceanspaces.com/quickchart/96e42532-7c49-47e9-a6ef-4850eb9729e4.png)

*Example of how agents improve their performance over time through reinforcement learning*

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

## 📚 API Reference

### A2A Server Endpoints

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

## 🐳 Docker Deployment

### All-in-One Container
```dockerfile
FROM python:3.11-slim
COPY . /app
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
python rla2a.py report         # Generate test report
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

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