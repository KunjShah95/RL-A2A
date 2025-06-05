#!/usr/bin/env python3
"""
OpenAI SDK Integration for RL-A2A Agent Communication Protocol
Provides intelligent agent behavior and natural language processing capabilities
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from openai import AsyncOpenAI
import numpy as np
import msgpack
import requests

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are an intelligent autonomous agent in a multi-agent communication system. 
Your role is to:
1. Analyze observations from other agents
2. Generate strategic decisions based on current state
3. Communicate effectively with other agents
4. Learn from feedback and adapt behavior
5. Coordinate with other agents to achieve objectives

Always respond in JSON format with clear action recommendations."""

@dataclass
class AgentContext:
    """Context information for agent decision making"""
    agent_id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    emotion: str
    observations: List[Dict]
    recent_actions: List[str]
    feedback_history: List[float]

class OpenAIAgentBrain:
    """OpenAI-powered agent intelligence system"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = model
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    async def analyze_situation(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze current situation and generate intelligent response"""
        
        # Prepare context for OpenAI
        situation_prompt = f"""
        Current Agent Status:
        - Agent ID: {context.agent_id}
        - Position: {context.position}
        - Velocity: {context.velocity}
        - Emotion: {context.emotion}
        - Recent Actions: {context.recent_actions[-5:] if context.recent_actions else []}
        - Feedback History: {context.feedback_history[-5:] if context.feedback_history else []}
        
        Observations from Environment:
        {json.dumps(context.observations, indent=2)}
        
        Based on this information, provide:
        1. Situation analysis
        2. Recommended action
        3. Communication strategy
        4. Risk assessment
        5. Learning opportunities
        
        Respond in JSON format.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": situation_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Store conversation history
            agent_id = context.agent_id
            if agent_id not in self.conversation_history:
                self.conversation_history[agent_id] = []
            
            self.conversation_history[agent_id].append({
                "timestamp": asyncio.get_event_loop().time(),
                "context": context.__dict__,
                "analysis": analysis
            })
            
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "situation_analysis": "Unable to process with OpenAI",
                "recommended_action": "continue_previous_behavior",
                "communication_strategy": "maintain_current_state",
                "risk_assessment": "unknown",
                "learning_opportunities": []
            }
    
    async def generate_agent_message(self, sender_id: str, target_id: str, context: str) -> Dict[str, Any]:
        """Generate intelligent inter-agent communication"""
        
        message_prompt = f"""
        Generate a message for agent-to-agent communication:
        - From: {sender_id}
        - To: {target_id}
        - Context: {context}
        
        The message should be:
        1. Clear and actionable
        2. Contextually relevant
        3. Conducive to coordination
        4. Efficient in communication
        
        Provide the message content and metadata in JSON format.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "message": f"Hello {target_id}, this is {sender_id}. Sharing current status.",
                "priority": "normal",
                "action_required": False,
                "error": str(e)
            }
    
    async def evaluate_feedback(self, agent_id: str, action: str, outcome: Dict) -> float:
        """Use OpenAI to evaluate action outcomes and generate reward signals"""
        
        evaluation_prompt = f"""
        Evaluate the performance of an agent action:
        - Agent: {agent_id}
        - Action: {action}
        - Outcome: {json.dumps(outcome)}
        
        Provide a reward score between -1.0 and 1.0 where:
        - 1.0 = Excellent performance, optimal action
        - 0.5 = Good performance, beneficial action
        - 0.0 = Neutral performance, no significant impact
        - -0.5 = Poor performance, suboptimal action
        - -1.0 = Terrible performance, harmful action
        
        Also provide reasoning for the score.
        
        Respond in JSON format with 'reward' and 'reasoning' fields.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": evaluation_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return float(result.get("reward", 0.0))
            
        except Exception as e:
            # Fallback to simple heuristic
            return 0.0 if "error" in str(outcome).lower() else 0.1

class IntelligentAgent:
    """Enhanced agent with OpenAI-powered intelligence"""
    
    def __init__(self, agent_id: str, openai_api_key: Optional[str] = None):
        self.agent_id = agent_id
        self.brain = OpenAIAgentBrain(api_key=openai_api_key)
        self.context = AgentContext(
            agent_id=agent_id,
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            velocity={"x": 0.0, "y": 0.0, "z": 0.0},
            emotion="neutral",
            observations=[],
            recent_actions=[],
            feedback_history=[]
        )
        self.session_id: Optional[str] = None
        self.a2a_server_url = "http://localhost:8000"
    
    async def register_with_a2a(self) -> bool:
        """Register with A2A server"""
        try:
            response = requests.post(
                f"{self.a2a_server_url}/register",
                params={"agent_id": self.agent_id}
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            print(f"✅ Agent {self.agent_id} registered with session {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False
    
    async def think_and_act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and generate intelligent response"""
        
        # Update context
        self.context.observations.append(observation)
        if len(self.context.observations) > 10:
            self.context.observations = self.context.observations[-10:]
        
        # Get AI analysis
        analysis = await self.brain.analyze_situation(self.context)
        
        # Generate action based on analysis
        action = {
            "agent_id": self.agent_id,
            "action": analysis.get("recommended_action", "observe"),
            "message": analysis.get("communication_strategy", ""),
            "confidence": analysis.get("confidence", 0.5),
            "reasoning": analysis.get("situation_analysis", ""),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Update recent actions
        self.context.recent_actions.append(action["action"])
        if len(self.context.recent_actions) > 20:
            self.context.recent_actions = self.context.recent_actions[-20:]
        
        return action
    
    async def send_feedback(self, action_id: str, outcome: Dict) -> bool:
        """Send intelligent feedback based on outcome evaluation"""
        
        # Use AI to evaluate the outcome
        reward = await self.brain.evaluate_feedback(self.agent_id, action_id, outcome)
        
        # Update feedback history
        self.context.feedback_history.append(reward)
        if len(self.context.feedback_history) > 50:
            self.context.feedback_history = self.context.feedback_history[-50:]
        
        try:
            response = requests.post(
                f"{self.a2a_server_url}/feedback",
                json={
                    "agent_id": self.agent_id,
                    "action_id": action_id,
                    "reward": reward,
                    "context": {
                        "outcome": outcome,
                        "ai_evaluation": True,
                        "emotion": self.context.emotion
                    }
                }
            )
            response.raise_for_status()
            print(f"✅ Feedback sent: {reward:.2f} for action {action_id}")
            return True
        except Exception as e:
            print(f"❌ Feedback failed: {e}")
            return False
    
    async def communicate_with_agent(self, target_agent_id: str, context: str) -> Dict[str, Any]:
        """Generate and send intelligent inter-agent message"""
        message_data = await self.brain.generate_agent_message(
            self.agent_id, target_agent_id, context
        )
        return message_data

class OpenAIEnhancedA2ASystem:
    """A2A system enhanced with OpenAI capabilities"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.brain = OpenAIAgentBrain(api_key=openai_api_key)
        self.agents: Dict[str, IntelligentAgent] = {}
    
    def create_agent(self, agent_id: str) -> IntelligentAgent:
        """Create a new intelligent agent"""
        agent = IntelligentAgent(agent_id, self.brain.client.api_key)
        self.agents[agent_id] = agent
        return agent
    
    async def run_multi_agent_simulation(self, num_agents: int = 3, iterations: int = 5):
        """Run a simulation with multiple intelligent agents"""
        
        print(f"🤖 Starting simulation with {num_agents} AI-powered agents...")
        
        # Create agents
        agents = []
        for i in range(num_agents):
            agent_id = f"ai_agent_{i+1}"
            agent = self.create_agent(agent_id)
            agents.append(agent)
        
        # Register all agents
        registration_tasks = [agent.register_with_a2a() for agent in agents]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        successful_agents = [agent for agent, success in zip(agents, results) if success]
        print(f"✅ {len(successful_agents)} agents registered successfully")
        
        # Run simulation
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            # Generate observations for each agent
            tasks = []
            for agent in successful_agents:
                observation = {
                    "environment": f"iteration_{iteration}",
                    "other_agents": [a.agent_id for a in successful_agents if a != agent],
                    "timestamp": asyncio.get_event_loop().time(),
                    "iteration": iteration
                }
                tasks.append(agent.think_and_act(observation))
            
            # Process all agents concurrently
            actions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Display results
            for agent, action in zip(successful_agents, actions):
                if isinstance(action, dict):
                    print(f"  {agent.agent_id}: {action['action']} (confidence: {action['confidence']:.2f})")
                    print(f"    Reasoning: {action['reasoning'][:100]}...")
            
            # Send feedback
            feedback_tasks = []
            for agent, action in zip(successful_agents, actions):
                if isinstance(action, dict):
                    outcome = {"success": True, "iteration": iteration}
                    feedback_tasks.append(
                        agent.send_feedback(f"action_{iteration}", outcome)
                    )
            
            await asyncio.gather(*feedback_tasks, return_exceptions=True)
            await asyncio.sleep(1)  # Brief pause between iterations
        
        print("✅ Simulation completed!")
        return successful_agents

# Example usage and testing
async def demo_openai_integration():
    """Demonstrate OpenAI integration capabilities"""
    
    print("🧠 RL-A2A OpenAI Integration Demo")
    print("=" * 50)
    
    # Check API key
    if not OPENAI_API_KEY:
        print("⚠️  Set OPENAI_API_KEY environment variable for full functionality")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create enhanced system
    system = OpenAIEnhancedA2ASystem()
    
    # Run simulation
    await system.run_multi_agent_simulation(num_agents=2, iterations=3)

if __name__ == "__main__":
    asyncio.run(demo_openai_integration())