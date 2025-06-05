# MCP Integration Guide for RL-A2A

This guide explains how to add Model Context Protocol (MCP) support to your RL-A2A agent communication system.

## What is MCP?

Model Context Protocol (MCP) is an open protocol that enables AI assistants to securely connect with external data sources and tools. By adding MCP support to RL-A2A, you can:

- Expose your agent communication tools to AI assistants
- Enable natural language interaction with your agent system
- Integrate with MCP-compatible clients like Claude Desktop, Cline, etc.
- Create a bridge between conversational AI and your agent infrastructure

## Architecture

```
MCP Client (Claude, Cline, etc.)
        ↓ (MCP Protocol)
MCP Server (mcp_server.py)
        ↓ (HTTP/WebSocket)
A2A Server (a2a_server.py)
        ↓ (WebSocket)
Agents (agent_a.py, etc.)
```

## Installation Steps

### 1. Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A

# Run the setup script
python rla2a.py setup

# Test the installation
python rla2a.py report
```

### 2. Manual Setup

```bash
# Install MCP dependencies
pip install mcp>=1.0.0
pip install -r requirements.txt

# Make start script executable (Unix/Mac)
chmod +x start_mcp_server.py
```

## Usage

### Starting the Servers

1. **Start A2A Server:**
   ```bash
   python a2a_server.py
   ```

2. **Start MCP Server:**
   ```bash
   python rla2a.py mcp
   ```

### Connecting MCP Clients

Add this configuration to your MCP client:

```json
{
  "mcpServers": {
    "rl-a2a": {
      "command": "python",
      "args": ["rla2a.py", "mcp"],
      "env": {
        "A2A_SERVER_URL": "http://localhost:8000"
      }
    }
  }
}
```

## Available Tools

### 1. start_system
**Description:** Start the complete RL-A2A system

**Parameters:**
- `agents` (number): Number of demo agents to create (default: 3)
- `dashboard` (boolean): Whether to start dashboard (default: true)

**Example Usage:**
"Start the RL-A2A system with 5 agents"

### 2. create_agent
**Description:** Create a new agent in the system

**Parameters:**
- `agent_id` (string): Unique identifier for the agent

**Example Usage:**
"Create a new agent called 'explorer'"

## Available Resources

### 1. rla2a://system
Complete RL-A2A system information and status

## Example Interactions

Once your MCP client is connected, you can use natural language commands:

1. **"Start the RL-A2A system with 5 agents"**
   - Uses `start_system` tool
   - Returns system configuration and commands

2. **"Show me information about the system"**
   - Reads `rla2a://system` resource
   - Returns current system status

3. **"Create a new agent called scout"**
   - Uses `create_agent` tool
   - Configures new agent for creation

## Troubleshooting

### Common Issues

1. **"MCP module not found"**
   ```bash
   pip install mcp>=1.0.0
   ```

2. **"A2A server connection failed"**
   - Ensure A2A server is running on correct port
   - Check firewall settings
   - Verify server URL in configuration

3. **"Permission denied on start script"**
   ```bash
   chmod +x start_mcp_server.py
   ```

4. **"Invalid MCP configuration"**
   - Verify JSON syntax in configuration
   - Check file paths in configuration
   - Ensure Python executable is in PATH

### Debug Mode

Run with debug logging:
```bash
export MCP_DEBUG=1
python rla2a.py mcp
```

### Testing

Run the test suite to verify installation:
```bash
python rla2a.py setup
python rla2a.py report
```

## Advanced Configuration

### Custom Server URL
Set environment variable:
```bash
export A2A_SERVER_URL="http://your-server:8000"
```

### Multiple Environments
Create environment-specific configs:
```json
{
  "mcpServers": {
    "rl-a2a-dev": {
      "command": "python",
      "args": ["rla2a.py", "mcp"],
      "env": {
        "A2A_SERVER_URL": "http://localhost:8000"
      }
    },
    "rl-a2a-prod": {
      "command": "python", 
      "args": ["rla2a.py", "mcp"],
      "env": {
        "A2A_SERVER_URL": "https://your-prod-server.com"
      }
    }
  }
}
```

## Next Steps

1. **Extend Tools:** Add more MCP tools for your specific use cases
2. **Custom Resources:** Create additional resources for monitoring
3. **Authentication:** Add security features for production use
4. **Scaling:** Implement load balancing for multiple A2A servers
5. **Integration:** Connect with other MCP servers and tools

## Support

For questions or issues:
- Open an issue on GitHub
- Check the rla2a.py file for diagnostic information
- Review A2A server logs for connection issues

## Contributing

Contributions to improve MCP integration are welcome:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Note:** This MCP integration is designed to be a starting point. You can extend it with additional tools, resources, and functionality specific to your agent communication needs.