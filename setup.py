#!/usr/bin/env python3
"""
Simple setup script for RL-A2A System
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    print("🚀" + "="*50 + "🚀")
    print("    RL-A2A System Setup")
    print("🚀" + "="*50 + "🚀")

def install_dependencies():
    """Install required dependencies"""
    requirements = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0", 
        "websockets>=12.0",
        "msgpack>=1.0.7",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "openai>=1.12.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0", 
        "streamlit>=1.29.0",
        "pandas>=2.1.0",
        "mcp>=1.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + requirements, stdout=subprocess.DEVNULL)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Try manually: pip install fastapi uvicorn streamlit plotly openai mcp")
        return False

def create_env_file():
    """Create environment configuration"""
    env_content = """# RL-A2A Configuration
OPENAI_API_KEY=your-openai-api-key-here
A2A_SERVER_URL=http://localhost:8000
A2A_SERVER_HOST=localhost
A2A_SERVER_PORT=8000
STREAMLIT_SERVER_PORT=8501
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def create_start_scripts():
    """Create startup scripts"""
    
    # Server start script
    server_script = """#!/usr/bin/env python3
import subprocess
import sys

print("🚀 Starting RL-A2A Server...")
subprocess.run([sys.executable, "rl_a2a_system.py", "--mode", "server"])
"""
    
    # Dashboard start script
    dashboard_script = """#!/usr/bin/env python3  
import subprocess
import sys

print("🎨 Starting Dashboard...")
subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501"])
"""
    
    # Complete system script
    complete_script = """#!/usr/bin/env python3
import subprocess
import sys

print("🚀 Starting Complete RL-A2A System...")
subprocess.run([sys.executable, "rl_a2a_system.py", "--mode", "complete", "--agents", "3"])
"""
    
    scripts = {
        "start_server.py": server_script,
        "start_dashboard.py": dashboard_script,
        "start_complete.py": complete_script
    }
    
    for filename, content in scripts.items():
        with open(filename, "w") as f:
            f.write(content)
        
        # Make executable on Unix
        if os.name != 'nt':
            os.chmod(filename, 0o755)
    
    print("✅ Created startup scripts")

def print_usage():
    """Print usage instructions"""
    print("\\n" + "🎉" + "="*50 + "🎉")
    print("    Setup Complete!")
    print("🎉" + "="*50 + "🎉")
    
    print("\\n📋 Quick Start:")
    print("1. Set your OpenAI API key in .env file (optional)")
    print("2. Choose how to run:")
    print("")
    print("   🖥️  Complete System (Recommended):")
    print("      python start_complete.py")
    print("")
    print("   🔧 Individual Components:")
    print("      python start_server.py     # A2A Server only")
    print("      python start_dashboard.py  # Dashboard only")
    print("")
    print("   📊 Dashboard URL: http://localhost:8501")
    print("   🔗 A2A Server: http://localhost:8000")
    print("")
    print("📚 Features:")
    print("• Multi-agent communication")
    print("• OpenAI-powered intelligence")  
    print("• Real-time 3D visualization")
    print("• Interactive dashboard")
    print("• MCP integration")
    print("")
    print("🆘 Need help? Check the README.md")

def main():
    print_header()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} OK")
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create configuration
    create_env_file()
    create_start_scripts()
    
    # Print usage
    print_usage()

if __name__ == "__main__":
    main()