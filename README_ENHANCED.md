# 🤖 RL-A2A Enhanced: Secure Multi-AI Agent Communication System

[![Security](https://img.shields.io/badge/Security-Enhanced-green.svg)](./SECURITY.md)
[![AI Support](https://img.shields.io/badge/AI-OpenAI%20|%20Claude%20|%20Gemini-blue.svg)](#ai-providers)
[![Version](https://img.shields.io/badge/Version-4.0.0-orange.svg)](#changelog)

An enhanced, secure Agent-to-Agent (A2A) communication system with reinforcement learning, multi-AI provider support, real-time 3D visualization, and comprehensive security features.

## 🚀 New Features in v4.0

### 🔒 **Security Enhancements**
- **Data Poisoning Protection**: Input validation, sanitization, and size limits
- **Authentication**: JWT-based authentication system
- **Rate Limiting**: Configurable rate limiting on all endpoints
- **CORS Security**: Configurable allowed origins
- **Session Management**: Secure session handling with automatic cleanup

### 🤖 **Multi-AI Provider Support**
- **OpenAI**: GPT-4o-mini integration with timeout protection
- **Anthropic**: Claude 3.5 Sonnet support with rate limiting
- **Google**: Gemini 1.5 Flash integration with error handling
- **Fallback System**: Graceful degradation when providers fail

### 📊 **Enhanced Visualization**
- **Real-time 3D Environment**: Interactive agent positioning
- **Analytics Dashboard**: Comprehensive metrics and performance tracking  
- **Security Monitoring**: Real-time security status and alerts
- **Multi-dimensional Analysis**: Emotion, action, reward, and velocity visualization

### ⚙️ **Environment Configuration**
- **Comprehensive .env Support**: All configuration via environment variables
- **API Key Management**: Secure handling of multiple AI provider keys
- **Runtime Configuration**: Dynamic system configuration
- **Production Ready**: Production-grade defaults and security

## 📋 Quick Start

### 1. Installation

```bash
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A
git checkout security-fixes-and-enhancements

# Install dependencies
pip install -r requirements.txt

# Setup environment
python rla2a_enhanced.py setup
```

### 2. Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# AI Provider API Keys
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
```

### 3. Run the System

**Start the A2A Server:**
```bash
python rla2a_enhanced.py server --demo-agents 3
```

**Start the Enhanced Dashboard:**
```bash
python rla2a_enhanced.py dashboard
```

Access the dashboard at `http://localhost:8501`

## 🔧 Usage Examples

### Register an Agent

```python
import requests

response = requests.post("http://localhost:8000/register", json={
    "agent_id": "my_agent_001",
    "ai_provider": "claude",  # or "openai", "google"
    "initial_position": {
        "x": 5.0,
        "y": 3.2,
        "z": 0.0
    }
})
print(response.json())
```

### WebSocket Communication

```python
import asyncio
import websockets
import msgpack

async def agent_communication():
    session_id = "your-session-id-from-registration"
    uri = f"ws://localhost:8000/ws/{session_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send agent update
        message = {
            "agent_id": "my_agent_001",
            "position": {"x": 10.0, "y": 5.0, "z": 1.0},
            "emotion": "excited",
            "ai_provider": "claude"
        }
        
        await websocket.send(msgpack.packb(message))
        
        # Receive AI action
        response = msgpack.unpackb(await websocket.recv())
        print(f"AI Action: {response}")

asyncio.run(agent_communication())
```

### Send Feedback

```python
feedback = requests.post("http://localhost:8000/feedback", json={
    "agent_id": "my_agent_001",
    "action_id": "action_123",
    "reward": 0.8,
    "context": {"source": "user_feedback", "task": "navigation"}
})
```

## 🤖 AI Providers

### OpenAI Integration
```python
# Configuration
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini

# Features
- Fast response times
- Advanced reasoning
- Cost-effective
```

### Anthropic Claude
```python
# Configuration  
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Features
- Exceptional reasoning
- Safety-focused responses
- Long context handling
```

### Google Gemini
```python
# Configuration
GOOGLE_API_KEY=your-google-key
GOOGLE_MODEL=gemini-1.5-flash

# Features
- Fast inference
- Multimodal capabilities
- Cost-effective
```

## 📊 Dashboard Features

### 3D Environment View
- Real-time agent positions
- Emotion-based coloring
- Activity status indicators
- Interactive 3D controls

### Analytics Dashboard  
- Emotion distribution analysis
- Reward performance tracking
- Action frequency monitoring
- Velocity and movement patterns

### Agent Control Panel
- Register new agents
- Send feedback rewards
- Monitor agent status
- Configure AI providers

### Security Dashboard
- Rate limiting status
- Active session monitoring
- Security event tracking
- System health metrics

## 🔒 Security Features

### Input Validation
- Pydantic model validation
- Data sanitization with bleach
- Size limits on all inputs
- Type safety enforcement

### Authentication
- JWT token authentication
- Session management
- Secure password hashing
- Configurable token expiry

### Network Security
- CORS configuration
- Trusted host middleware
- Rate limiting
- DoS protection

### Monitoring
- Security event logging
- Failed authentication tracking
- Rate limit monitoring
- System health alerts

See [SECURITY.md](./SECURITY.md) for complete security documentation.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   A2A Server    │    │  AI Providers   │
│  (Streamlit)    │◄──►│   (FastAPI)     │◄──►│ OpenAI/Claude/  │
│                 │    │                 │    │    Gemini       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │
         │              ┌─────────────────┐
         └─────────────►│   WebSocket     │
                        │  Connections    │
                        └─────────────────┘
```

## 🔧 API Reference

### Core Endpoints

| Method | Endpoint | Description | Security |
|--------|----------|-------------|----------|
| GET | `/` | System information | Public |
| GET | `/health` | Health check | Rate limited |
| POST | `/register` | Register agent | Rate limited |
| GET | `/agents` | List agents | Rate limited |
| POST | `/feedback` | Send feedback | Rate limited |
| WS | `/ws/{session_id}` | Agent communication | Session validated |
| GET | `/dashboard` | Dashboard data | Rate limited |

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/visualization/agents` | 3D visualization data |
| POST | `/auth/login` | User authentication |
| GET | `/security/status` | Security metrics |

## 📈 Performance & Scalability

### System Limits
- **Maximum Agents**: 100 (configurable)
- **Rate Limit**: 60 requests/minute (configurable)  
- **Message Size**: 1MB maximum
- **Session Timeout**: 1 hour (configurable)

### Optimization Features
- Automatic session cleanup
- Connection pooling
- Efficient data serialization
- Background task management

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Security tests
python tests/security_tests.py

# Load testing
python tests/load_tests.py --agents 50 --duration 300
```

## 🚀 Production Deployment

### Docker Support
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "rla2a_enhanced.py", "server"]
```

### Environment Variables
```bash
# Production configuration
A2A_HOST=0.0.0.0
A2A_PORT=8000
DEBUG=false
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://yourdomain.com
```

### Health Checks
```bash
curl http://localhost:8000/health
```

## 📚 Documentation

- [Security Guide](./SECURITY.md) - Comprehensive security documentation
- [API Documentation](./docs/API.md) - Detailed API reference  
- [AI Provider Guide](./docs/AI_PROVIDERS.md) - AI integration guide
- [Deployment Guide](./docs/DEPLOYMENT.md) - Production deployment
- [Contributing Guide](./CONTRIBUTING.md) - Development guidelines

## 🤝 Contributing

1. **Security First**: Review [SECURITY.md](./SECURITY.md) before contributing
2. **Fork & Branch**: Create feature branches for new work
3. **Test Coverage**: Include tests for all new features
4. **Documentation**: Update docs for any API changes
5. **Pull Request**: Submit PRs with detailed descriptions

## 🔄 Changelog

### v4.0.0 - Enhanced Security & Multi-AI Support
- ✅ Comprehensive security enhancements
- ✅ Multi-AI provider support (OpenAI, Claude, Gemini)  
- ✅ Enhanced 3D visualization dashboard
- ✅ Environment configuration system
- ✅ Production-ready deployment
- ✅ Comprehensive documentation

### v3.0.0 - Original Consolidated Version
- Basic A2A communication
- OpenAI integration only
- Simple dashboard
- Limited security features

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/KunjShah01/RL-A2A/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KunjShah01/RL-A2A/discussions)
- **Security**: See [SECURITY.md](./SECURITY.md) for reporting vulnerabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT integration and API
- **Anthropic** for Claude AI capabilities  
- **Google** for Gemini AI support
- **FastAPI** for the robust web framework
- **Streamlit** for the interactive dashboard
- **Community** for feedback and contributions

---

**Built with ❤️ for secure, intelligent agent communication**

[![Built with Python](https://img.shields.io/badge/Built%20with-Python-blue.svg)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)