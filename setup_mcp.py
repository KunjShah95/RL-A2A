#!/usr/bin/env python3
"""
Setup script for RL-A2A MCP Server
"""

import subprocess
import sys
import os
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version} is compatible")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing MCP dependencies...")
    
    try:
        # Install MCP first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp>=1.0.0"])
        print("✅ MCP installed successfully")
        
        # Install other requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-mcp.txt"])
        print("✅ All dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def create_start_script():
    """Create a start script for the MCP server"""
    script_content = f"""#!/usr/bin/env python3
import subprocess
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start the MCP server
if __name__ == "__main__":
    subprocess.run([sys.executable, "mcp_server.py"])
"""
    
    with open("start_mcp_server.py", "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_mcp_server.py", 0o755)
    
    print("✅ Start script created: start_mcp_server.py")

def verify_setup():
    """Verify the MCP setup"""
    print("🔍 Verifying setup...")
    
    # Check if required files exist
    required_files = [
        "mcp_server.py",
        "mcp_config.json", 
        "requirements-mcp.txt",
        "a2a_server.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
        print(f"✅ Found: {file}")
    
    # Test import of MCP
    try:
        import mcp
        print("✅ MCP module can be imported")
    except ImportError:
        print("❌ Cannot import MCP module")
        return False
    
    return True

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 RL-A2A MCP Server Setup Complete!")
    print("="*60)
    print("\nHow to use:")
    print("1. Start the A2A server:")
    print("   uvicorn a2a_server:app --reload")
    print("\n2. In another terminal, start the MCP server:")
    print("   python start_mcp_server.py")
    print("\n3. Connect MCP clients to use A2A tools:")
    print("   - register_agent: Register new agents")
    print("   - send_agent_feedback: Send RL feedback") 
    print("   - create_agent_observation: Create agent observations")
    print("   - start_a2a_server: Get server start commands")
    print("\n4. View resources:")
    print("   - a2a://agents: Active agents info")
    print("   - a2a://models: RL models data") 
    print("   - a2a://communication: Communication logs")
    print("\n📚 For more details, check the documentation in readme.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Setting up RL-A2A MCP Server...")
    
    check_python_version()
    install_dependencies()
    create_start_script()
    
    if verify_setup():
        print_usage_instructions()
    else:
        print("❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()