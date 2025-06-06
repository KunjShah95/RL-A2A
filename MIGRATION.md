# 🔄 Migration Guide: v3.0 → v4.0 Enhanced

This guide helps you migrate from the original RL-A2A system to the enhanced version with security fixes and multi-AI support.

## 🚨 Breaking Changes

### 1. **Environment Configuration**
**OLD (v3.0):**
```python
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "SERVER_HOST": os.getenv("A2A_HOST", "localhost"),
    "SERVER_PORT": int(os.getenv("A2A_PORT", "8000")),
}
```

**NEW (v4.0):**
```bash
# .env file required
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key  
GOOGLE_API_KEY=your-google-key
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:8501
RATE_LIMIT_PER_MINUTE=60
```

### 2. **Agent Registration**
**OLD:**
```python
POST /register?agent_id=my_agent
```

**NEW:**
```python
POST /register
{
    "agent_id": "my_agent",
    "ai_provider": "openai",  # New field
    "initial_position": {     # New field
        "x": 0.0,
        "y": 0.0, 
        "z": 0.0
    }
}
```

### 3. **WebSocket Messages**
**OLD:**
```python
{
    "agent_id": "my_agent",
    "position": {"x": 1, "y": 2},
    "emotion": "happy"
}
```

**NEW:**
```python
{
    "agent_id": "my_agent",
    "position": {"x": 1.0, "y": 2.0, "z": 0.0},  # z required
    "velocity": {"x": 0.1, "y": 0.2, "z": 0.0},  # New field
    "emotion": "happy",
    "ai_provider": "claude"  # New field
}
```

## 📋 Migration Steps

### Step 1: Update Dependencies

```bash
# Install new requirements  
pip install -r requirements.txt

# New dependencies include:
# - anthropic (Claude support)
# - google-generativeai (Gemini support)  
# - pyjwt (Authentication)
# - bcrypt (Password hashing)
# - slowapi (Rate limiting)
# - bleach (Input sanitization)
```

### Step 2: Environment Setup

1. **Create .env file:**
```bash
cp .env.example .env
```

2. **Add your API keys:**
```bash
# Required for existing OpenAI functionality
OPENAI_API_KEY=sk-your-openai-key-here

# Optional new providers
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-ai-key

# Security settings
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

3. **Configure security (optional):**
```bash
# Restrict CORS in production
ALLOWED_ORIGINS=https://yourdomain.com,https://dashboard.yourdomain.com

# Adjust rate limiting
RATE_LIMIT_PER_MINUTE=100

# Set agent limits
MAX_AGENTS=50
```

### Step 3: Update Client Code

#### Agent Registration
**OLD Code:**
```python
import requests

response = requests.post(
    "http://localhost:8000/register",
    params={"agent_id": "my_agent"}
)
```

**NEW Code:**
```python
import requests

response = requests.post(
    "http://localhost:8000/register", 
    json={
        "agent_id": "my_agent",
        "ai_provider": "openai",  # or "anthropic", "google"
        "initial_position": {"x": 0, "y": 0, "z": 0}  # optional
    }
)
```

#### WebSocket Communication
**OLD Code:**
```python
message = {
    "agent_id": "my_agent",
    "position": {"x": 5, "y": 3},
    "emotion": "happy"
}
await websocket.send(msgpack.packb(message))
```

**NEW Code:**
```python
message = {
    "agent_id": "my_agent", 
    "position": {"x": 5.0, "y": 3.0, "z": 0.0},  # z coordinate required
    "velocity": {"x": 0.1, "y": -0.2, "z": 0.0},  # new field
    "emotion": "happy",
    "ai_provider": "claude"  # specify provider per message
}
await websocket.send(msgpack.packb(message))
```

#### Error Handling
**NEW:** Add proper error handling for rate limits and validation:
```python
try:
    response = requests.post(endpoint, json=data)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        print("Rate limit exceeded, wait before retrying")
    elif e.response.status_code == 422:
        print(f"Validation error: {e.response.json()}")
    else:
        print(f"HTTP error: {e}")
```

### Step 4: Update Dashboard Usage

**OLD:** Basic dashboard
```bash
python rla2a.py dashboard
```

**NEW:** Enhanced dashboard with security
```bash
python rla2a_enhanced.py dashboard
```

New dashboard features:
- 3D agent visualization
- Multi-AI provider status
- Security monitoring
- Real-time analytics
- Agent control panel

## 🔄 Gradual Migration Strategy

### Phase 1: Basic Migration
1. Keep existing code running
2. Set up new environment with `.env`
3. Test with demo agents
4. Verify basic functionality

### Phase 2: Feature Integration  
1. Add new AI providers gradually
2. Update client code for enhanced features
3. Implement security measures
4. Test rate limiting and validation

### Phase 3: Full Migration
1. Switch all agents to new system
2. Enable all security features
3. Monitor performance and security
4. Decommission old system

## 🔧 Code Examples

### Multi-AI Provider Usage
```python
# Different agents using different AI providers
agents = [
    {"id": "explorer_1", "provider": "openai"},
    {"id": "analyzer_1", "provider": "claude"}, 
    {"id": "creative_1", "provider": "google"}
]

for agent in agents:
    response = requests.post("/register", json={
        "agent_id": agent["id"],
        "ai_provider": agent["provider"]
    })
```

### Security-Aware Client
```python
import time
import requests
from functools import wraps

def rate_limited(max_calls_per_minute=60):
    min_interval = 60.0 / max_calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limited(max_calls_per_minute=50)  # Stay under server limit
def safe_api_call(endpoint, data):
    return requests.post(f"http://localhost:8000{endpoint}", json=data)
```

### Enhanced Agent Class
```python
class EnhancedAgent:
    def __init__(self, agent_id, ai_provider="openai"):
        self.agent_id = agent_id
        self.ai_provider = ai_provider
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.emotion = "neutral"
        self.session_id = None
        
    def register(self):
        response = requests.post("/register", json={
            "agent_id": self.agent_id,
            "ai_provider": self.ai_provider,
            "initial_position": self.position
        })
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            return True
        return False
    
    def update_position(self, x, y, z=None):
        self.position["x"] = max(-100, min(100, x))  # Bounds checking
        self.position["y"] = max(-100, min(100, y))
        if z is not None:
            self.position["z"] = max(-10, min(10, z))
```

## ⚠️ Common Migration Issues

### Issue 1: CORS Errors
**Problem:** Dashboard can't connect to server
**Solution:** 
```bash
# Add dashboard URL to allowed origins
ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000
```

### Issue 2: Rate Limiting
**Problem:** "Too Many Requests" errors
**Solution:**
```python
# Implement client-side rate limiting
# Or increase server limit:
RATE_LIMIT_PER_MINUTE=120
```

### Issue 3: Validation Errors
**Problem:** 422 validation errors
**Solution:**
```python
# Ensure all required fields are present and valid
message = {
    "agent_id": "valid_agent_123",  # alphanumeric + underscore/hyphen only
    "position": {"x": 1.0, "y": 2.0, "z": 0.0},  # all coordinates required
    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},  # within [-10, 10] range
    "emotion": "happy"  # must be valid emotion from list
}
```

### Issue 4: Authentication Errors
**Problem:** 401 Unauthorized (if authentication enabled)
**Solution:**
```python
# Get token and use in headers
token = get_auth_token()  # Your auth implementation
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(endpoint, json=data, headers=headers)
```

## 🧪 Testing Your Migration

### 1. Basic Functionality Test
```bash
# Start server
python rla2a_enhanced.py server --demo-agents 3

# Test health endpoint
curl http://localhost:8000/health

# Check agents
curl http://localhost:8000/agents
```

### 2. AI Provider Test
```python
# Test each AI provider
providers = ["openai", "anthropic", "google"]
for provider in providers:
    response = requests.post("/register", json={
        "agent_id": f"test_{provider}",
        "ai_provider": provider
    })
    print(f"{provider}: {'✅' if response.ok else '❌'}")
```

### 3. Security Test
```bash
# Test rate limiting
for i in range(70):  # Exceed limit 
    curl http://localhost:8000/health
# Should see 429 error
```

### 4. Dashboard Test
```bash
# Start enhanced dashboard
python rla2a_enhanced.py dashboard

# Open http://localhost:8501
# Verify all features work
```

## 📚 Additional Resources

- [Security Guide](./SECURITY.md) - Complete security documentation
- [API Reference](./docs/API.md) - Updated API documentation
- [Enhanced README](./README_ENHANCED.md) - Full feature overview
- [Troubleshooting](./docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🎯 Migration Checklist

- [ ] Install new dependencies
- [ ] Create and configure .env file  
- [ ] Update agent registration code
- [ ] Add z-coordinate to positions
- [ ] Include velocity in WebSocket messages
- [ ] Add error handling for validation
- [ ] Test with new AI providers
- [ ] Verify security features
- [ ] Update dashboard integration
- [ ] Monitor performance and security
- [ ] Document any custom changes

## 🆘 Getting Help

If you encounter issues during migration:

1. **Check logs** for specific error messages
2. **Review security documentation** for configuration issues
3. **Test with demo agents** to verify setup
4. **Create GitHub issue** with migration details if needed

Remember: The enhanced system is backward compatible for basic operations, so you can migrate gradually while keeping existing functionality working.