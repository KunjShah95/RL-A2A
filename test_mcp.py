#!/usr/bin/env python3
"""
Test script for RL-A2A MCP Server functionality
"""

import asyncio
import json
import subprocess
import time
import sys
import signal
from typing import Optional

class MCPTester:
    def __init__(self):
        self.a2a_process: Optional[subprocess.Popen] = None
        self.mcp_process: Optional[subprocess.Popen] = None
    
    async def start_a2a_server(self):
        """Start the A2A server for testing"""
        print("🚀 Starting A2A server...")
        try:
            self.a2a_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "a2a_server:app", "--reload", "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a bit for server to start
            await asyncio.sleep(3)
            
            # Check if server is running
            if self.a2a_process.poll() is None:
                print("✅ A2A server started successfully")
                return True
            else:
                print("❌ Failed to start A2A server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting A2A server: {e}")
            return False
    
    def test_mcp_server_import(self):
        """Test if MCP server can be imported"""
        print("🧪 Testing MCP server import...")
        try:
            import mcp_server
            print("✅ MCP server module imported successfully")
            return True
        except ImportError as e:
            print(f"❌ Failed to import MCP server: {e}")
            return False
    
    def test_dependencies(self):
        """Test if all required dependencies are available"""
        print("🧪 Testing dependencies...")
        
        required_modules = [
            "mcp",
            "fastapi", 
            "uvicorn",
            "msgpack",
            "numpy",
            "pydantic",
            "requests"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError:
                print(f"❌ {module} - MISSING")
                missing_modules.append(module)
        
        if missing_modules:
            print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
            print("Run: python setup_mcp.py")
            return False
        else:
            print("✅ All dependencies available")
            return True
    
    def test_config_files(self):
        """Test if configuration files are valid"""
        print("🧪 Testing configuration files...")
        
        try:
            # Test MCP config
            with open("mcp_config.json", "r") as f:
                config = json.load(f)
            
            # Check required keys
            required_keys = ["mcpServers", "server", "tools", "resources"]
            for key in required_keys:
                if key not in config:
                    print(f"❌ Missing key in mcp_config.json: {key}")
                    return False
            
            print("✅ mcp_config.json is valid")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in mcp_config.json: {e}")
            return False
        except FileNotFoundError:
            print("❌ mcp_config.json not found")
            return False
    
    async def test_a2a_registration(self):
        """Test A2A agent registration"""
        print("🧪 Testing A2A agent registration...")
        
        try:
            import requests
            
            # Test registration
            response = requests.post(
                "http://localhost:8000/register",
                params={"agent_id": "test_agent"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "session_id" in data and "agent_id" in data:
                    print("✅ A2A agent registration works")
                    return True
                else:
                    print("❌ Invalid response format")
                    return False
            else:
                print(f"❌ Registration failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Registration test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes"""
        print("🧹 Cleaning up...")
        
        if self.a2a_process:
            try:
                self.a2a_process.terminate()
                self.a2a_process.wait(timeout=5)
                print("✅ A2A server stopped")
            except subprocess.TimeoutExpired:
                self.a2a_process.kill()
                print("⚠️ A2A server force-killed")
        
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
                print("✅ MCP server stopped")
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
                print("⚠️ MCP server force-killed")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\\n🛑 Interrupted, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 RL-A2A MCP Server Test Suite")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        test_results = []
        
        # Test 1: Dependencies
        test_results.append(self.test_dependencies())
        
        # Test 2: Config files
        test_results.append(self.test_config_files())
        
        # Test 3: MCP server import
        test_results.append(self.test_mcp_server_import())
        
        # Test 4: Start A2A server
        if await self.start_a2a_server():
            test_results.append(True)
            
            # Test 5: A2A registration
            await asyncio.sleep(2)  # Let server fully start
            test_results.append(await self.test_a2a_registration())
        else:
            test_results.append(False)
            test_results.append(False)
        
        # Summary
        print("\\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)
        
        passed = sum(test_results)
        total = len(test_results)
        
        test_names = [
            "Dependencies Check",
            "Configuration Files", 
            "MCP Server Import",
            "A2A Server Start",
            "Agent Registration"
        ]
        
        for i, (name, result) in enumerate(zip(test_names, test_results)):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{i+1}. {name}: {status}")
        
        print(f"\\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! MCP setup is working correctly.")
            print("\\nNext steps:")
            print("1. Start A2A server: uvicorn a2a_server:app --reload")
            print("2. Start MCP server: python start_mcp_server.py")
            print("3. Connect your MCP client using mcp_config.json")
        else:
            print(f"❌ {total - passed} test(s) failed. Please fix the issues above.")
        
        # Cleanup
        self.cleanup()
        
        return passed == total

def main():
    """Main test function"""
    tester = MCPTester()
    
    try:
        result = asyncio.run(tester.run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        tester.cleanup()
        print("\\n🛑 Tests interrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()