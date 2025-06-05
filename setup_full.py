#!/usr/bin/env python3
"""
Complete setup script for RL-A2A with OpenAI and Visualization support
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🚀" + "="*60 + "🚀")
    print("    RL-A2A Complete Setup: OpenAI + Visualization + MCP")
    print("🚀" + "="*60 + "🚀")

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} is compatible")

def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install main dependencies
        print("   Installing core dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-full.txt"
        ], stdout=subprocess.DEVNULL)
        
        print("✅ All dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Try running: pip install -r requirements-full.txt")
        sys.exit(1)

def setup_environment():
    """Setup environment configuration"""
    print("\\n🔧 Setting up environment...")
    
    # Create .env file template
    env_content = \"\"\"# RL-A2A Configuration
# OpenAI API Key (required for AI features)
OPENAI_API_KEY=your-openai-api-key-here

# A2A Server Configuration
A2A_SERVER_URL=http://localhost:8000
A2A_SERVER_HOST=localhost
A2A_SERVER_PORT=8000

# Visualization Configuration
STREAMLIT_SERVER_PORT=8501
DASHBOARD_REFRESH_INTERVAL=3

# MCP Configuration
MCP_SERVER_NAME=rl-a2a-mcp-server
MCP_SERVER_VERSION=1.0.0

# Development Settings
DEBUG=false
LOG_LEVEL=info
\"\"\"
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
        print("   🔑 Don't forget to add your OpenAI API key!")
    else:
        print("✅ .env file already exists")

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\\n📜 Creating startup scripts...")
    
    # A2A Server startup script
    a2a_script = \"\"\"#!/usr/bin/env python3
import subprocess
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

host = os.getenv('A2A_SERVER_HOST', 'localhost')
port = os.getenv('A2A_SERVER_PORT', '8000')

print(f"🚀 Starting A2A Server on {host}:{port}")
subprocess.run([
    sys.executable, "-m", "uvicorn", 
    "a2a_server:app", "--reload", 
    "--host", host, 
    "--port", port
])
\"\"\"
    
    with open("start_a2a.py", "w") as f:
        f.write(a2a_script)
    
    # Dashboard startup script
    dashboard_script = \"\"\"#!/usr/bin/env python3
import subprocess
import sys
import os

# Load environment variables  
from dotenv import load_dotenv
load_dotenv()

port = os.getenv('STREAMLIT_SERVER_PORT', '8501')

print(f"🎨 Starting Visualization Dashboard on port {port}")
print("   Open your browser to: http://localhost:{port}")

subprocess.run([
    sys.executable, "-m", "streamlit", "run", 
    "visualization.py", "--server.port", port,
    "--", "streamlit"
])
\"\"\"
    
    with open("start_dashboard.py", "w") as f:
        f.write(dashboard_script)
    
    # OpenAI demo script
    demo_script = \"\"\"#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from openai_integration import demo_openai_integration

# Load environment variables
load_dotenv()

print("🧠 Starting OpenAI Integration Demo")
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Please set OPENAI_API_KEY in .env file")
    exit(1)

asyncio.run(demo_openai_integration())
\"\"\"
    
    with open("start_openai_demo.py", "w") as f:
        f.write(demo_script)
    
    # Make scripts executable on Unix
    if os.name != 'nt':
        for script in ["start_a2a.py", "start_dashboard.py", "start_openai_demo.py"]:
            os.chmod(script, 0o755)
    
    print("✅ Created startup scripts:")
    print("   - start_a2a.py: Start A2A server")
    print("   - start_dashboard.py: Start visualization dashboard")  
    print("   - start_openai_demo.py: Run OpenAI demo")

def create_docker_config():
    """Create Docker configuration for easy deployment"""
    print("\\n🐳 Creating Docker configuration...")
    
    dockerfile = \"\"\"FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-full.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Environment variables
ENV A2A_SERVER_HOST=0.0.0.0
ENV A2A_SERVER_PORT=8000
ENV STREAMLIT_SERVER_PORT=8501

# Default command
CMD ["python", "start_a2a.py"]
\"\"\"
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    docker_compose = \"\"\"version: '3.8'

services:
  a2a-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - A2A_SERVER_HOST=0.0.0.0
      - A2A_SERVER_PORT=8000
    command: python start_a2a.py

  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - A2A_SERVER_URL=http://a2a-server:8000
      - STREAMLIT_SERVER_PORT=8501
    depends_on:
      - a2a-server
    command: python start_dashboard.py

volumes:
  a2a_data:
\"\"\"
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Created Docker configuration files")

def verify_installation():
    """Verify the installation"""
    print("\\n🔍 Verifying installation...")
    
    # Check required files
    required_files = [
        "a2a_server.py",
        "openai_integration.py", 
        "visualization.py",
        "mcp_server.py",
        ".env",
        "start_a2a.py",
        "start_dashboard.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    # Test imports
    try:
        import openai
        import matplotlib
        import plotly
        import streamlit
        import mcp
        print("✅ All Python modules can be imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def print_usage_instructions():
    """Print comprehensive usage instructions"""
    print("\\n" + "🎉" + "="*60 + "🎉")
    print("           RL-A2A Setup Complete!")
    print("🎉" + "="*60 + "🎉")
    
    print("\\n📋 Next Steps:")
    print("-" * 30)
    
    print("\\n1. 🔑 Configure OpenAI API Key:")
    print("   Edit .env file and add your OpenAI API key:")
    print("   OPENAI_API_KEY=your-actual-api-key-here")
    
    print("\\n2. 🚀 Start the A2A Server:")
    print("   python start_a2a.py")
    print("   or: uvicorn a2a_server:app --reload")
    
    print("\\n3. 🎨 Launch Visualization Dashboard:")
    print("   python start_dashboard.py")
    print("   Then open: http://localhost:8501")
    
    print("\\n4. 🧠 Try OpenAI Integration:")
    print("   python start_openai_demo.py")
    
    print("\\n5. 🔌 Set up MCP (optional):")
    print("   python start_mcp_server.py")
    print("   Configure your MCP client with mcp_config.json")
    
    print("\\n6. 🐳 Docker Deployment (optional):")
    print("   docker-compose up")
    
    print("\\n📖 Documentation:")
    print("-" * 20)
    print("   • README.md - Main documentation")
    print("   • MCP_GUIDE.md - MCP integration guide")
    print("   • Run 'python test_mcp.py' to test MCP setup")
    
    print("\\n🛠️  Available Tools:")
    print("-" * 20)
    print("   • OpenAI Integration: Intelligent agent behavior")
    print("   • Real-time Visualization: 3D plots, dashboards")  
    print("   • MCP Server: AI assistant integration")
    print("   • WebSocket Communication: Real-time agent coordination")
    print("   • Reinforcement Learning: Agent improvement over time")
    
    print("\\n🆘 Need Help?")
    print("-" * 15)
    print("   • Check the logs if servers don't start")
    print("   • Verify .env configuration")
    print("   • Run 'pip install -r requirements-full.txt' if imports fail")
    print("   • Open GitHub issues for bugs/questions")
    
    print("\\n🚀 Happy agent building!")
    print("="*64)

def main():
    """Main setup function"""
    print_banner()
    
    try:
        check_python_version()
        install_dependencies()
        setup_environment()
        create_startup_scripts()
        create_docker_config()
        
        if verify_installation():
            print_usage_instructions()
        else:
            print("❌ Installation verification failed")
            print("   Please check the errors above and re-run setup")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()