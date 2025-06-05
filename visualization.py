#!/usr/bin/env python3
"""
Visualization System for RL-A2A Agent Communication Protocol
Provides real-time dashboards, graphs, and monitoring capabilities
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pandas as pd
import streamlit as st
import requests
from datetime import datetime, timedelta
import threading
import queue
import websockets
import msgpack

@dataclass
class AgentState:
    """Agent state for visualization"""
    agent_id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    timestamp: float
    session_id: str = ""
    action: str = ""
    reward: float = 0.0

@dataclass
class Communication:
    """Communication event for visualization"""
    sender: str
    receiver: str
    message: str
    timestamp: float
    message_type: str = "direct"

class DataCollector:
    """Collects data from A2A system for visualization"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.agents: Dict[str, AgentState] = {}
        self.communications: List[Communication] = []
        self.rewards: List[Tuple[str, float, float]] = []  # agent_id, reward, timestamp
        self.is_collecting = False
        self.data_queue = queue.Queue()
    
    async def collect_from_websocket(self, session_id: str):
        """Collect real-time data via WebSocket"""
        try:
            uri = f"ws://localhost:8000/ws/{session_id}"
            async with websockets.connect(uri) as websocket:
                print(f"📡 Connected to WebSocket for data collection")
                
                while self.is_collecting:
                    try:
                        # Send periodic ping to keep connection alive
                        ping_data = {
                            "agent_id": "visualizer",
                            "session_id": session_id,
                            "position": {"x": 0, "y": 0, "z": 0},
                            "velocity": {"x": 0, "y": 0, "z": 0},
                            "emotion": "neutral"
                        }
                        await websocket.send(msgpack.packb(ping_data))
                        
                        # Receive data with timeout
                        data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        parsed_data = msgpack.unpackb(data)
                        self.data_queue.put(parsed_data)
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"WebSocket error: {e}")
                        break
                        
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
    
    def start_collection(self):
        """Start data collection"""
        self.is_collecting = True
        
        # Register visualizer to get session ID
        try:
            response = requests.post(
                f"{self.server_url}/register",
                params={"agent_id": "data_visualizer"}
            )
            response.raise_for_status()
            session_data = response.json()
            session_id = session_data["session_id"]
            
            # Start WebSocket collection in background
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=lambda: loop.run_until_complete(
                    self.collect_from_websocket(session_id)
                )
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"Failed to start data collection: {e}")
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
    
    def update_agent_state(self, data: Dict):
        """Update agent state from received data"""
        if "agent_id" in data:
            agent_state = AgentState(
                agent_id=data["agent_id"],
                position=data.get("position", {"x": 0, "y": 0, "z": 0}),
                velocity=data.get("velocity", {"x": 0, "y": 0, "z": 0}),
                emotion=data.get("emotion", "neutral"),
                timestamp=time.time(),
                session_id=data.get("session_id", ""),
                action=data.get("command", ""),
                reward=data.get("reward", 0.0)
            )
            self.agents[data["agent_id"]] = agent_state
    
    def process_queued_data(self):
        """Process data from queue"""
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.update_agent_state(data)
            except queue.Empty:
                break

class PlotlyVisualizer:
    """Advanced visualizations using Plotly"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.fig_3d = None
        self.fig_metrics = None
    
    def create_3d_agent_plot(self) -> go.Figure:
        """Create 3D visualization of agent positions"""
        fig = go.Figure()
        
        # Process current data
        self.data_collector.process_queued_data()
        
        if self.data_collector.agents:
            # Extract agent data
            agent_ids = list(self.data_collector.agents.keys())
            positions = [self.data_collector.agents[aid].position for aid in agent_ids]
            emotions = [self.data_collector.agents[aid].emotion for aid in agent_ids]
            
            # Color mapping for emotions
            emotion_colors = {
                "happy": "gold",
                "sad": "blue", 
                "neutral": "gray",
                "excited": "orange",
                "angry": "red"
            }
            
            colors = [emotion_colors.get(emotion, "gray") for emotion in emotions]
            
            # Create 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=[pos["x"] for pos in positions],
                y=[pos["y"] for pos in positions],
                z=[pos["z"] for pos in positions],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                text=agent_ids,
                textposition="top center",
                name='Agents',
                hovertemplate='<b>%{text}</b><br>' +
                             'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                             'Emotion: ' + '<br>'.join(emotions)
            ))
            
            # Add velocity vectors
            for i, (agent_id, pos) in enumerate(zip(agent_ids, positions)):
                vel = self.data_collector.agents[agent_id].velocity
                if any(abs(v) > 0.1 for v in vel.values()):  # Only show significant velocities
                    fig.add_trace(go.Scatter3d(
                        x=[pos["x"], pos["x"] + vel["x"]],
                        y=[pos["y"], pos["y"] + vel["y"]], 
                        z=[pos["z"], pos["z"] + vel["z"]],
                        mode='lines',
                        line=dict(color='red', width=4),
                        name=f'{agent_id} velocity',
                        showlegend=False
                    ))
        
        # Update layout
        fig.update_layout(
            title="Agent Positions and Velocities",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position",
                bgcolor="rgba(0,0,0,0.1)"
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_metrics_dashboard(self) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Agent Emotions Over Time",
                "Reward Distribution", 
                "Communication Network",
                "Agent Activity"
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        self.data_collector.process_queued_data()
        
        if self.data_collector.agents:
            agents = list(self.data_collector.agents.values())
            
            # 1. Emotions timeline (placeholder - would need historical data)
            emotions = [agent.emotion for agent in agents]
            emotion_counts = pd.Series(emotions).value_counts()
            
            fig.add_trace(
                go.Pie(
                    values=emotion_counts.values,
                    labels=emotion_counts.index,
                    name="Emotions"
                ),
                row=1, col=2
            )
            
            # 2. Rewards over time
            if self.data_collector.rewards:
                reward_df = pd.DataFrame(
                    self.data_collector.rewards,
                    columns=["agent_id", "reward", "timestamp"]
                )
                for agent_id in reward_df["agent_id"].unique():
                    agent_rewards = reward_df[reward_df["agent_id"] == agent_id]
                    fig.add_trace(
                        go.Scatter(
                            x=agent_rewards["timestamp"],
                            y=agent_rewards["reward"],
                            mode="lines+markers",
                            name=f"{agent_id} rewards",
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
            
            # 3. Communication network (simplified)
            agent_ids = [agent.agent_id for agent in agents]
            activity_scores = [abs(agent.velocity["x"]) + abs(agent.velocity["y"]) + abs(agent.velocity["z"]) for agent in agents]
            
            fig.add_trace(
                go.Bar(
                    x=agent_ids,
                    y=activity_scores,
                    name="Activity Level",
                    marker_color="lightblue"
                ),
                row=2, col=2
            )
            
            # 4. Agent positions (2D projection)
            fig.add_trace(
                go.Scatter(
                    x=[agent.position["x"] for agent in agents],
                    y=[agent.position["y"] for agent in agents],
                    mode="markers+text",
                    text=agent_ids,
                    textposition="top center",
                    marker=dict(size=10, opacity=0.7),
                    name="Positions (X-Y)"
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="RL-A2A System Metrics Dashboard",
            showlegend=True
        )
        
        return fig
    
    def save_html_report(self, filename: str = "a2a_report.html"):
        """Generate and save HTML report"""
        # Create visualizations
        fig_3d = self.create_3d_agent_plot()
        fig_metrics = self.create_metrics_dashboard()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL-A2A System Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .visualization {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RL-A2A System Visualization Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Active Agents: {len(self.data_collector.agents)}</p>
            </div>
            
            <div class="visualization" id="plot-3d"></div>
            <div class="visualization" id="plot-metrics"></div>
            
            <script>
                {pyo.plot(fig_3d, output_type='div', include_plotlyjs=False)};
                {pyo.plot(fig_metrics, output_type='div', include_plotlyjs=False)};
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Report saved to {filename}")

class StreamlitDashboard:
    """Real-time Streamlit dashboard"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
    
    def run_dashboard(self):
        """Run Streamlit dashboard"""
        st.set_page_config(
            page_title="RL-A2A Dashboard",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 RL-A2A Agent Communication Dashboard")
        
        # Sidebar controls
        st.sidebar.title("Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 3)
        
        if st.sidebar.button("Start Data Collection"):
            self.data_collector.start_collection()
            st.sidebar.success("Data collection started!")
        
        if st.sidebar.button("Stop Data Collection"):
            self.data_collector.stop_collection()
            st.sidebar.info("Data collection stopped!")
        
        # Main content
        visualizer = PlotlyVisualizer(self.data_collector)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Agent Positions (3D)")
            fig_3d = visualizer.create_3d_agent_plot()
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.subheader("System Metrics")
            fig_metrics = visualizer.create_metrics_dashboard()
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Agent status table
        st.subheader("Agent Status")
        if self.data_collector.agents:
            agent_data = []
            for agent_state in self.data_collector.agents.values():
                agent_data.append({
                    "Agent ID": agent_state.agent_id,
                    "Position": f"({agent_state.position['x']:.2f}, {agent_state.position['y']:.2f}, {agent_state.position['z']:.2f})",
                    "Emotion": agent_state.emotion,
                    "Last Action": agent_state.action,
                    "Last Update": datetime.fromtimestamp(agent_state.timestamp).strftime("%H:%M:%S")
                })
            
            df = pd.DataFrame(agent_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No agent data available. Start data collection to see live agent information.")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

class MatplotlibAnimator:
    """Real-time animation using Matplotlib"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.agent_plots = {}
        
    def animate(self, frame):
        """Animation function for matplotlib"""
        self.ax.clear()
        self.data_collector.process_queued_data()
        
        if self.data_collector.agents:
            # Plot agents
            for agent_state in self.data_collector.agents.values():
                x, y = agent_state.position["x"], agent_state.position["y"]
                
                # Color based on emotion
                color_map = {
                    "happy": "yellow",
                    "sad": "blue",
                    "neutral": "gray", 
                    "excited": "orange",
                    "angry": "red"
                }
                color = color_map.get(agent_state.emotion, "gray")
                
                # Plot agent
                circle = Circle((x, y), 0.5, color=color, alpha=0.7)
                self.ax.add_patch(circle)
                
                # Add label
                self.ax.text(x, y+0.7, agent_state.agent_id, 
                           ha='center', va='bottom', fontsize=8)
                
                # Add velocity vector
                vx, vy = agent_state.velocity["x"], agent_state.velocity["y"]
                if abs(vx) > 0.1 or abs(vy) > 0.1:
                    self.ax.arrow(x, y, vx, vy, head_width=0.2, 
                                head_length=0.2, fc='red', ec='red')
        
        # Set axis properties
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"RL-A2A Agents (Frame {frame})")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        
        # Add legend
        legend_elements = []
        emotions = set(agent.emotion for agent in self.data_collector.agents.values())
        for emotion in emotions:
            color = {"happy": "yellow", "sad": "blue", "neutral": "gray", 
                    "excited": "orange", "angry": "red"}.get(emotion, "gray")
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=emotion))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right')
    
    def start_animation(self, interval=1000):
        """Start the matplotlib animation"""
        ani = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, blit=False
        )
        plt.show()
        return ani

def create_visualization_dashboard():
    """Create and run the main visualization dashboard"""
    print("🎨 Starting RL-A2A Visualization Dashboard")
    print("=" * 50)
    
    # Create data collector
    data_collector = DataCollector()
    
    # Create Streamlit dashboard
    dashboard = StreamlitDashboard(data_collector)
    
    print("🚀 Starting Streamlit dashboard...")
    print("   Open your browser to: http://localhost:8501")
    print("   Use Ctrl+C to stop")
    
    # This would typically be run with: streamlit run visualization.py
    dashboard.run_dashboard()

def demo_matplotlib_animation():
    """Demonstrate matplotlib animation"""
    print("🎬 Starting Matplotlib Animation Demo")
    
    data_collector = DataCollector()
    data_collector.start_collection()
    
    animator = MatplotlibAnimator(data_collector)
    
    try:
        print("Starting animation... Close the plot window to stop.")
        ani = animator.start_animation(interval=1000)  # Update every second
    except KeyboardInterrupt:
        print("\\nAnimation stopped by user")
    finally:
        data_collector.stop_collection()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        create_visualization_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "matplotlib":
        demo_matplotlib_animation()
    else:
        # Generate static report
        print("📊 Generating static visualization report...")
        data_collector = DataCollector()
        data_collector.start_collection()
        
        # Collect some data
        time.sleep(5)
        
        visualizer = PlotlyVisualizer(data_collector)
        visualizer.save_html_report("a2a_visualization_report.html")
        
        data_collector.stop_collection()
        print("✅ Report generated! Open a2a_visualization_report.html in your browser.")