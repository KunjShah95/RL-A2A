#!/usr/bin/env python3
"""
RL-A2A Streamlit Dashboard
Real-time visualization and control interface
"""

import streamlit as st
import asyncio
import time
import json
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
import threading

# Page config
st.set_page_config(
    page_title="RL-A2A Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .success { color: #28a745; }
    .warning { color: #ffc107; }
    .error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Configuration
A2A_SERVER_URL = "http://localhost:8000"

class DashboardState:
    """Dashboard state management"""
    
    def __init__(self):
        if 'agents' not in st.session_state:
            st.session_state.agents = {}
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rewards' not in st.session_state:
            st.session_state.rewards = []
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True

def check_server_status():
    """Check if A2A server is running"""
    try:
        response = requests.get(f"{A2A_SERVER_URL}/docs", timeout=2)
        return response.status_code == 200
    except:
        return False

def register_agent(agent_id: str):
    """Register a new agent"""
    try:
        response = requests.post(f"{A2A_SERVER_URL}/register", params={"agent_id": agent_id})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def send_feedback(agent_id: str, action_id: str, reward: float):
    """Send feedback to agent"""
    try:
        response = requests.post(f"{A2A_SERVER_URL}/feedback", json={
            "agent_id": agent_id,
            "action_id": action_id,
            "reward": reward,
            "context": {"source": "dashboard"}
        })
        return response.status_code == 200
    except:
        return False

def simulate_agent_data(num_agents: int = 3):
    """Simulate agent data for demo"""
    agent_data = {}
    
    for i in range(num_agents):
        agent_id = f"agent_{i+1}"
        agent_data[agent_id] = {
            "agent_id": agent_id,
            "position": {
                "x": np.random.uniform(-10, 10),
                "y": np.random.uniform(-10, 10),
                "z": np.random.uniform(0, 5)
            },
            "velocity": {
                "x": np.random.uniform(-2, 2),
                "y": np.random.uniform(-2, 2),
                "z": np.random.uniform(-1, 1)
            },
            "emotion": np.random.choice(["happy", "neutral", "excited", "sad", "angry"]),
            "last_action": np.random.choice(["move_forward", "turn_left", "turn_right", "communicate", "observe"]),
            "reward": np.random.uniform(-1, 1),
            "timestamp": time.time(),
            "status": "active"
        }
    
    return agent_data

def create_3d_plot(agents_data: Dict):
    """Create 3D agent visualization"""
    if not agents_data:
        return go.Figure()
    
    agents = list(agents_data.values())
    
    emotion_colors = {
        "happy": "#FFD700",      # Gold
        "sad": "#4169E1",        # Royal Blue
        "neutral": "#808080",    # Gray
        "excited": "#FF4500",    # Orange Red
        "angry": "#DC143C"       # Crimson
    }
    
    fig = go.Figure()
    
    # Add agents
    fig.add_trace(go.Scatter3d(
        x=[agent["position"]["x"] for agent in agents],
        y=[agent["position"]["y"] for agent in agents],
        z=[agent["position"]["z"] for agent in agents],
        mode='markers+text',
        marker=dict(
            size=15,
            color=[emotion_colors.get(agent["emotion"], "#808080") for agent in agents],
            opacity=0.8,
            line=dict(width=2, color='rgba(0,0,0,0.8)')
        ),
        text=[agent["agent_id"] for agent in agents],
        textposition="top center",
        name='Agents',
        hovertemplate='<b>%{text}</b><br>' +
                     'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                     '<extra></extra>'
    ))
    
    # Add velocity vectors
    for agent in agents:
        pos = agent["position"]
        vel = agent["velocity"]
        
        if abs(vel["x"]) > 0.1 or abs(vel["y"]) > 0.1 or abs(vel["z"]) > 0.1:
            fig.add_trace(go.Scatter3d(
                x=[pos["x"], pos["x"] + vel["x"]*2],
                y=[pos["y"], pos["y"] + vel["y"]*2],
                z=[pos["z"], pos["z"] + vel["z"]*2],
                mode='lines',
                line=dict(color='red', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="Agent Positions & Velocities",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
            bgcolor="rgba(240,240,240,0.1)",
            xaxis=dict(range=[-15, 15]),
            yaxis=dict(range=[-15, 15]),
            zaxis=dict(range=[-2, 8])
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_metrics_dashboard(agents_data: Dict, rewards_data: List):
    """Create comprehensive metrics dashboard"""
    if not agents_data:
        return go.Figure()
    
    agents = list(agents_data.values())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Emotion Distribution", "Agent Activity", "Reward Trends", "System Health"],
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "indicator"}]
        ]
    )
    
    # 1. Emotion distribution
    emotions = [agent["emotion"] for agent in agents]
    emotion_counts = pd.Series(emotions).value_counts()
    
    fig.add_trace(
        go.Pie(
            values=emotion_counts.values,
            labels=emotion_counts.index,
            name="Emotions",
            marker=dict(colors=["#FFD700", "#4169E1", "#808080", "#FF4500", "#DC143C"])
        ),
        row=1, col=1
    )
    
    # 2. Agent activity
    activity_scores = []
    agent_ids = []
    for agent in agents:
        vel = agent["velocity"]
        activity = abs(vel["x"]) + abs(vel["y"]) + abs(vel["z"])
        activity_scores.append(activity)
        agent_ids.append(agent["agent_id"])
    
    fig.add_trace(
        go.Bar(
            x=agent_ids,
            y=activity_scores,
            name="Activity",
            marker_color="lightblue"
        ),
        row=1, col=2
    )
    
    # 3. Reward trends (simulated)
    if rewards_data:
        times = [r[2] for r in rewards_data[-20:]]  # Last 20 rewards
        rewards = [r[1] for r in rewards_data[-20:]]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=rewards,
                mode="lines+markers",
                name="Rewards",
                line=dict(color="green", width=2)
            ),
            row=2, col=1
        )
    
    # 4. System health indicator
    health_score = len([a for a in agents if a["status"] == "active"]) / max(len(agents), 1) * 100
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health %"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

def main():
    """Main dashboard function"""
    # Initialize dashboard state
    dashboard_state = DashboardState()
    
    # Header
    st.title("🤖 RL-A2A System Dashboard")
    st.markdown("Real-time monitoring and control for Agent-to-Agent communication")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Server status
        server_online = check_server_status()
        status_color = "🟢" if server_online else "🔴"
        st.markdown(f"**Server Status:** {status_color} {'Online' if server_online else 'Offline'}")
        
        if not server_online:
            st.warning("A2A Server is offline. Start it with: `python rl_a2a_system.py --mode server`")
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)
        
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
        
        st.divider()
        
        # Agent Management
        st.subheader("Agent Management")
        
        # Register new agent
        with st.form("register_agent"):
            new_agent_id = st.text_input("Agent ID", placeholder="e.g., explorer_001")
            register_btn = st.form_submit_button("Register Agent")
            
            if register_btn and new_agent_id:
                if server_online:
                    result = register_agent(new_agent_id)
                    if result:
                        st.success(f"✅ Agent {new_agent_id} registered!")
                        st.json(result)
                    else:
                        st.error("❌ Registration failed")
                else:
                    st.error("❌ Server offline")
        
        # Send feedback
        st.subheader("Send Feedback")
        with st.form("send_feedback"):
            fb_agent_id = st.text_input("Agent ID", key="feedback_agent")
            fb_action_id = st.text_input("Action ID", value=f"action_{int(time.time())}")
            fb_reward = st.slider("Reward", -1.0, 1.0, 0.0, 0.1)
            feedback_btn = st.form_submit_button("Send Feedback")
            
            if feedback_btn and fb_agent_id:
                if server_online:
                    success = send_feedback(fb_agent_id, fb_action_id, fb_reward)
                    if success:
                        st.success("✅ Feedback sent!")
                        # Add to rewards history
                        st.session_state.rewards.append((fb_agent_id, fb_reward, time.time()))
                    else:
                        st.error("❌ Feedback failed")
                else:
                    st.error("❌ Server offline")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Simulate or fetch real agent data
    if server_online:
        # In a real implementation, this would fetch from the A2A server
        agents_data = simulate_agent_data()
    else:
        agents_data = simulate_agent_data()  # Demo data
    
    # Update session state
    st.session_state.agents = agents_data
    
    # Metrics
    with col1:
        st.metric("Active Agents", len(agents_data), delta=1)
    
    with col2:
        avg_reward = np.mean([r[1] for r in st.session_state.rewards[-10:]]) if st.session_state.rewards else 0
        st.metric("Avg Reward", f"{avg_reward:.3f}", delta=f"{avg_reward-0.1:.3f}")
    
    with col3:
        happy_agents = len([a for a in agents_data.values() if a["emotion"] == "happy"])
        st.metric("Happy Agents", happy_agents, delta=0)
    
    with col4:
        total_messages = len(st.session_state.messages)
        st.metric("Messages", total_messages, delta=1)
    
    st.divider()
    
    # Main visualizations
    tab1, tab2, tab3 = st.tabs(["🌐 3D View", "📊 Metrics", "📋 Agent Details"])
    
    with tab1:
        st.subheader("Agent Positions & Movement")
        fig_3d = create_3d_plot(agents_data)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Emotion legend
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Emotion Colors:**")
            st.markdown("🟡 Happy • 🔵 Sad • ⚫ Neutral • 🟠 Excited • 🔴 Angry")
        
        with col2:
            st.markdown("**Red vectors** show agent velocity and direction")
    
    with tab2:
        st.subheader("System Metrics & Analytics")
        fig_metrics = create_metrics_dashboard(agents_data, st.session_state.rewards)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with tab3:
        st.subheader("Agent Status Table")
        
        if agents_data:
            # Create detailed agent table
            agent_rows = []
            for agent in agents_data.values():
                agent_rows.append({
                    "Agent ID": agent["agent_id"],
                    "Position": f"({agent['position']['x']:.1f}, {agent['position']['y']:.1f}, {agent['position']['z']:.1f})",
                    "Velocity": f"({agent['velocity']['x']:.1f}, {agent['velocity']['y']:.1f}, {agent['velocity']['z']:.1f})",
                    "Emotion": agent["emotion"],
                    "Last Action": agent["last_action"],
                    "Reward": f"{agent['reward']:.3f}",
                    "Status": agent["status"]
                })
            
            df = pd.DataFrame(agent_rows)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"agents_data_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            st.info("No agent data available")
    
    # Footer
    st.divider()
    with st.expander("📘 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Start A2A server: `python rl_a2a_system.py --mode server`
        2. Register agents using the sidebar
        3. Monitor real-time positions in 3D view
        4. Send feedback to improve agent learning
        5. View metrics and analytics
        
        **Features:**
        - Real-time 3D agent tracking
        - Interactive metrics dashboard
        - Agent registration and feedback
        - Data export capabilities
        
        **Tips:**
        - Enable auto-refresh for live updates
        - Use positive rewards (0.5-1.0) for good behavior
        - Use negative rewards (-1.0 to -0.5) for poor behavior
        """)
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()