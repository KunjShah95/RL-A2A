import asyncio
import websockets
import msgpack
import json
import random
import uuid
import requests
import numpy as np
import time
import sys
from typing import Dict, Optional

class AgentClient:
    def __init__(self, agent_id: str, server_url: str = "localhost:8000", max_retries: int = 3):
        self.agent_id = agent_id
        self.server_url = server_url
        self.session_id = None
        self.websocket = None
        self.state_history = []
        self.action_history = []
        self.last_action_id = None
        self.max_retries = max_retries
        
    async def register(self):
        """Register agent with server and get session ID"""
        retries = 0
        while retries < self.max_retries:
            try:
                print(f"[{self.agent_id}] Attempting to connect to server at http://{self.server_url}/register")
                response = requests.post(
                    f"http://{self.server_url}/register",
                    params={"agent_id": self.agent_id},
                    timeout=5  # Add timeout to avoid hanging
                )
                data = response.json()
                self.session_id = data["session_id"]
                print(f"[{self.agent_id}] Registered with session ID: {self.session_id}")
                return self.session_id
            except requests.exceptions.ConnectionError:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                print(f"[{self.agent_id}] Server connection failed. Retrying in {wait_time} seconds... ({retries}/{self.max_retries})")
                time.sleep(wait_time)
        
        print(f"[{self.agent_id}] Could not connect to server after {self.max_retries} attempts.")
        print(f"[{self.agent_id}] Please ensure the server is running with: uvicorn a2a_server:app --reload")
        sys.exit(1)
    
    async def send_feedback(self, reward: float, context: Optional[Dict] = None):
        """Send feedback for reinforcement learning"""
        if not self.last_action_id:
            print(f"[{self.agent_id}] No action to provide feedback for")
            return
        
        try:    
            response = requests.post(
                f"http://{self.server_url}/feedback",
                json={
                    "agent_id": self.agent_id,
                    "action_id": self.last_action_id,
                    "reward": reward,
                    "context": context or {}
                },
                timeout=5
            )
            print(f"[{self.agent_id}] Feedback sent: {reward}")
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"[{self.agent_id}] Failed to send feedback, server connection failed")
            return None
        
    async def run(self, iterations: int = 3):
        """Run the agent's main loop"""
        if not self.session_id:
            await self.register()
            
        uri = f"ws://{self.server_url}/ws/{self.session_id}"
        print(f"[{self.agent_id}] Connecting to WebSocket at {uri}")
        
        try:
            async with websockets.connect(uri) as ws:
                self.websocket = ws
                for i in range(iterations):
                    # Generate random position and emotion
                    position = {"x": i, "y": i+1}
                    velocity = {"x": 0.5, "y": 0.1}
                    emotion = random.choice(["neutral", "afraid", "happy", "angry"])
                    
                    # Record the current state
                    self.state_history.append({
                        "position": position,
                        "emotion": emotion,
                        "iteration": i
                    })
                    
                    # Create observation
                    obs = {
                        "agent_id": self.agent_id,
                        "session_id": self.session_id,
                        "position": position,
                        "velocity": velocity,
                        "emotion": emotion,
                        # Uncommment to send a direct message to another agent
                        # "target_agent_id": "AgentB" 
                    }
                    
                    # Send observation using MessagePack
                    await ws.send(msgpack.packb(obs, use_bin_type=True))
                    print(f"[{self.agent_id} Sent]", obs)
                    
                    # Receive action
                    response_bytes = await ws.recv()
                    response = msgpack.unpackb(response_bytes, raw=False)
                    self.last_action_id = str(uuid.uuid4())  # Generate action ID for feedback
                    self.action_history.append({
                        "action_id": self.last_action_id,
                        "command": response["command"],
                        "message": response["message"],
                        "probability": response.get("success_probability", 1.0)
                    })
                    print(f"[{self.agent_id} Received Action]", response)
                    
                    # Provide feedback based on action and state
                    # This is a simple heuristic - would be based on domain-specific logic
                    reward = 0.0
                    if emotion == "afraid" and response["command"] == "retreat":
                        reward = 1.0
                    elif emotion == "happy" and response["command"] == "engage":
                        reward = 1.0
                    elif emotion == "angry" and response["command"] == "observe":
                        reward = 0.5
                    else:
                        reward = 0.1
                    
                    await self.send_feedback(reward, {"emotion": emotion})
                    
                    await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosedError:
            print(f"[{self.agent_id}] WebSocket connection closed unexpectedly")
        except ConnectionRefusedError:
            print(f"[{self.agent_id}] Could not connect to WebSocket server at {uri}")
            print(f"[{self.agent_id}] Please ensure the server is running with: uvicorn a2a_server:app --reload")
                
    def analyze_performance(self):
        """Analyze agent performance based on history"""
        if not self.action_history:
            return "No actions recorded yet"
            
        total_reward = 0
        for i, action in enumerate(self.action_history):
            if i < len(self.state_history):
                state = self.state_history[i]
                
                # Simple reward calculation based on state-action appropriateness
                if state["emotion"] == "afraid" and action["command"] == "retreat":
                    reward = 1.0
                elif state["emotion"] == "happy" and action["command"] == "engage":
                    reward = 1.0
                else:
                    reward = 0.1
                    
                total_reward += reward
        
        if len(self.action_history) > 0:        
            avg_reward = total_reward / len(self.action_history)
            return {
                "total_actions": len(self.action_history),
                "average_reward": avg_reward,
                "action_distribution": self._count_actions()
            }
        else:
            return {
                "total_actions": 0,
                "average_reward": 0,
                "action_distribution": {}
            }
    
    def _count_actions(self):
        """Count distribution of actions"""
        counts = {}
        for action in self.action_history:
            cmd = action["command"]
            counts[cmd] = counts.get(cmd, 0) + 1
        return counts
                

async def main():
    # Create and run agent
    agent = AgentClient("AgentA", "localhost:8000")
    await agent.run(iterations=5)
    
    # Analyze performance
    performance = agent.analyze_performance()
    print(f"\n[{agent.agent_id} Performance]")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    asyncio.run(main())