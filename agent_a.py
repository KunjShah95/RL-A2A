import asyncio
import websockets
import msgpack
import json
import random
import uuid
import requests
import numpy as np  # Removed unused import
import time
import sys
import os
from typing import Dict, Optional
from openai import OpenAI
import gym
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io


class AgentClient:
    def __init__(
        self,
        agent_id: str,
        server_url: str = "localhost:8000",
        max_retries: int = 3,
        gym_env: str = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        AgentClient integrates with OpenAI SDK for LLM-based action selection and can interact with OpenAI Gym environments.

        :param agent_id: Unique identifier for the agent
        :param server_url: URL of the server to connect to
        :param max_retries: Maximum number of retries for server connection
        :param gym_env: Name of the OpenAI Gym environment to use
        :param openai_api_key: Optionally provide the OpenAI API key directly. If not provided, will use OPENAI_API_KEY from environment.
        """
        self.agent_id = agent_id
        self.server_url = server_url
        self.session_id = None
        self.websocket = None
        self.state_history = []
        self.action_history = []
        self.last_action_id = None
        self.max_retries = max_retries

        # --- OpenAI SDK integration ---
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable or pass openai_api_key to AgentClient."
            )
        self.client = OpenAI(api_key=api_key)

        # --- Gym integration ---
        self.gym_env_name = gym_env
        self.env = gym.make(gym_env) if gym_env else None

    def get_action_from_llm(self, observation):
        prompt = f"Given the observation {observation}, suggest an action for an autonomous agent."
        response = self.client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def run_gym_episode(self, max_steps: int = 100):
        """Run a single episode in the OpenAI Gym environment using LLM for action selection."""
        if not self.env:
            print("No Gym environment specified.")
            return
        obs = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            action_str = self.get_action_from_llm(str(obs))
            try:
                action = int(action_str)  # For discrete action spaces
            except Exception:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.env.render()
            if done:
                print(
                    f"Episode finished after {step + 1} steps. Total reward: {total_reward}"
                )
                break
        self.env.close()

    async def register(self):
        """Register agent with server and get session ID"""
        retries = 0
        while retries < self.max_retries:
            try:
                print(
                    f"[{self.agent_id}] Attempting to connect to server at http://{self.server_url}/register"
                )
                response = requests.post(
                    f"http://{self.server_url}/register",
                    params={"agent_id": self.agent_id},
                    timeout=5,  # Add timeout to avoid hanging
                )
                data = response.json()
                self.session_id = data["session_id"]
                print(
                    f"[{self.agent_id}] Registered with session ID: {self.session_id}"
                )
                return self.session_id
            except requests.exceptions.ConnectionError:
                retries += 1
                wait_time = 2**retries  # Exponential backoff
                print(
                    f"[{self.agent_id}] Server connection failed. Retrying in {wait_time} seconds... ({retries}/{self.max_retries})"
                )
                time.sleep(wait_time)

        print(
            f"[{self.agent_id}] Could not connect to server after {self.max_retries} attempts."
        )
        print(
            f"[{self.agent_id}] Please ensure the server is running with: uvicorn a2a_server:app --reload"
        )
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
                    "context": context or {},
                },
                timeout=5,
            )
            print(f"[{self.agent_id}] Feedback sent: {reward}")
            return response.json()
        except requests.exceptions.ConnectionError:
            print(
                f"[{self.agent_id}] Failed to send feedback, server connection failed"
            )
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
                    position = {"x": i, "y": i + 1}
                    velocity = {"x": 0.5, "y": 0.1}
                    emotion = random.choice(["neutral", "afraid", "happy", "angry"])

                    # Record the current state
                    self.state_history.append(
                        {"position": position, "emotion": emotion, "iteration": i}
                    )

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
                    self.last_action_id = str(
                        uuid.uuid4()
                    )  # Generate action ID for feedback
                    self.action_history.append(
                        {
                            "action_id": self.last_action_id,
                            "command": response["command"],
                            "message": response["message"],
                            "probability": response.get("success_probability", 1.0),
                        }
                    )
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
            print(
                f"[{self.agent_id}] Please ensure the server is running with: uvicorn a2a_server:app --reload"
            )

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
                "action_distribution": self._count_actions(),
            }
        else:
            return {"total_actions": 0, "average_reward": 0, "action_distribution": {}}

    def _count_actions(self):
        """Count distribution of actions"""
        counts = {}
        for action in self.action_history:
            cmd = action["command"]
            counts[cmd] = counts.get(cmd, 0) + 1
        return counts

    def plot_performance(self):
        """Plot the agent's reward over time using matplotlib."""
        rewards = []
        for i, action in enumerate(self.action_history):
            if i < len(self.state_history):
                state = self.state_history[i]
                if state["emotion"] == "afraid" and action["command"] == "retreat":
                    rewards.append(1.0)
                elif state["emotion"] == "happy" and action["command"] == "engage":
                    rewards.append(1.0)
                elif state["emotion"] == "angry" and action["command"] == "observe":
                    rewards.append(0.5)
                else:
                    rewards.append(0.1)
        if not rewards:
            print("No rewards to plot.")
            return
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title(f"Agent {self.agent_id} Reward Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# --- Flask frontend for visualization ---
app = Flask(__name__)
agent_instance = None  # Will be set in main


@app.route("/plot.png")
def plot_png():
    if agent_instance is None:
        return "Agent not initialized", 400
    rewards = []
    for i, action in enumerate(agent_instance.action_history):
        if i < len(agent_instance.state_history):
            state = agent_instance.state_history[i]
            if state["emotion"] == "afraid" and action["command"] == "retreat":
                rewards.append(1.0)
            elif state["emotion"] == "happy" and action["command"] == "engage":
                rewards.append(1.0)
            elif state["emotion"] == "angry" and action["command"] == "observe":
                rewards.append(0.5)
            else:
                rewards.append(0.1)
    if not rewards:
        return "No rewards to plot", 400
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Agent {agent_instance.agent_id} Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype="image/png")


@app.route("/")
def index():
    return """<html><body><h1>Agent Reward Visualization</h1><img src="/plot.png" alt="Reward Plot"></body></html>"""


async def main():
    global agent_instance
    agent_instance = AgentClient("AgentA", "localhost:8000", gym_env="CartPole-v1")
    await agent_instance.run(iterations=5)
    agent_instance.run_gym_episode(max_steps=200)
    performance = agent_instance.analyze_performance()
    print(f"\n[{agent_instance.agent_id} Performance]")
    print(json.dumps(performance, indent=2))
    # Show matplotlib plot in a window
    agent_instance.plot_performance()
    # Start Flask app for web visualization (optional)
    print("\nVisit http://localhost:5000 to view the reward plot in your browser.")
    from threading import Thread

    Thread(target=app.run, kwargs={"debug": False, "use_reloader": False}).start()


async def main_multi_agents():
    """Run multiple agents in parallel and visualize their performance."""
    from threading import Thread
    import matplotlib.pyplot as plt

    agent_ids = ["AgentA", "AgentB", "AgentC"]
    agents = [
        AgentClient(agent_id, "localhost:8000", gym_env="CartPole-v1")
        for agent_id in agent_ids
    ]
    performances = {}

    async def run_agent(agent):
        await agent.run(iterations=5)
        agent.run_gym_episode(max_steps=100)
        performances[agent.agent_id] = agent.analyze_performance()
        print(f"\n[{agent.agent_id} Performance]")
        print(json.dumps(performances[agent.agent_id], indent=2))

    # Run all agents concurrently
    await asyncio.gather(*(run_agent(agent) for agent in agents))

    # Plot all agents' rewards on the same graph
    plt.figure(figsize=(10, 5))
    for agent in agents:
        rewards = []
        for i, action in enumerate(agent.action_history):
            if i < len(agent.state_history):
                state = agent.state_history[i]
                if state["emotion"] == "afraid" and action["command"] == "retreat":
                    rewards.append(1.0)
                elif state["emotion"] == "happy" and action["command"] == "engage":
                    rewards.append(1.0)
                elif state["emotion"] == "angry" and action["command"] == "observe":
                    rewards.append(0.5)
                else:
                    rewards.append(0.1)
        plt.plot(rewards, marker="o", label=agent.agent_id)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Multiple Agents Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally, start Flask app for one of the agents
    global agent_instance
    agent_instance = agents[0]
    print(
        "\nVisit http://localhost:5000 to view the reward plot for AgentA in your browser."
    )
    Thread(target=app.run, kwargs={"debug": False, "use_reloader": False}).start()


if __name__ == "__main__":
    # To run a single agent, use: asyncio.run(main())
    # To run multiple agents, use: asyncio.run(main_multi_agents())
    asyncio.run(main_multi_agents())
