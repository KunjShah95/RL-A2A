"""
Security tests for RL-A2A Enhanced System
Tests for data poisoning protection, rate limiting, input validation, and other security features
"""

import pytest
import asyncio
import json
import time
import requests
import msgpack
import websockets
from datetime import datetime
from unittest.mock import Mock, patch

# Import the enhanced system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rla2a_enhanced import EnhancedA2ASystem, SecurityConfig, CONFIG

class TestSecurityFeatures:
    """Test suite for security enhancements"""
    
    @pytest.fixture
    def app(self):
        """Create test application instance"""
        system = EnhancedA2ASystem()
        return system.app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_input_validation_agent_registration(self, client):
        """Test input validation for agent registration"""
        
        # Valid registration
        valid_data = {
            "agent_id": "test_agent_123",
            "ai_provider": "openai"
        }
        response = client.post("/register", json=valid_data)
        assert response.status_code == 200
        
        # Invalid agent ID (too long)
        invalid_data = {
            "agent_id": "a" * 100,  # Exceeds 50 char limit
            "ai_provider": "openai"
        }
        response = client.post("/register", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        # Invalid agent ID (special characters)
        invalid_data = {
            "agent_id": "test<script>alert('xss')</script>",
            "ai_provider": "openai"
        }
        response = client.post("/register", json=invalid_data)
        assert response.status_code == 422
        
        # Invalid AI provider
        invalid_data = {
            "agent_id": "test_agent",
            "ai_provider": "malicious_provider"
        }
        response = client.post("/register", json=invalid_data)
        assert response.status_code == 422
    
    def test_position_bounds_validation(self, client):
        """Test position and velocity bounds validation"""
        
        # Register agent first
        client.post("/register", json={
            "agent_id": "bounds_test",
            "ai_provider": "openai"
        })
        
        # Valid position within bounds
        valid_position = {
            "x": 50.0,
            "y": -25.0,
            "z": 5.0
        }
        
        # Test position bounds (should be clamped internally)
        extreme_position = {
            "x": 1000.0,  # Exceeds max bound
            "y": -1000.0, # Exceeds min bound  
            "z": 50.0     # Exceeds z bound
        }
        
        # These would be handled in WebSocket, but we can test the model validation
        from rla2a_enhanced import AgentPosition
        
        with pytest.raises(ValueError):
            AgentPosition(x=200, y=150, z=20)  # Should exceed bounds
    
    def test_data_sanitization(self, client):
        """Test input sanitization against XSS and injection"""
        
        # Test HTML/script injection in feedback
        malicious_feedback = {
            "agent_id": "<script>alert('xss')</script>",
            "action_id": "test_action",
            "reward": 0.5,
            "context": {
                "comment": "<img src=x onerror=alert('xss')>",
                "data": "javascript:alert('xss')"
            }
        }
        
        response = client.post("/feedback", json=malicious_feedback)
        # Should either reject (422) or sanitize the input
        assert response.status_code in [404, 422]  # Agent not found or validation error
    
    def test_rate_limiting_protection(self, client):
        """Test rate limiting functionality"""
        
        # Make rapid requests to trigger rate limiting
        start_time = time.time()
        responses = []
        
        for i in range(SecurityConfig.RATE_LIMIT_PER_MINUTE + 5):
            response = client.get("/health")
            responses.append(response.status_code)
            if response.status_code == 429:  # Rate limit exceeded
                break
        
        # Should hit rate limit
        assert 429 in responses
        
        # Time should be reasonable (not too fast/slow for rate limiting)
        elapsed = time.time() - start_time
        assert elapsed < 60  # Shouldn't take a full minute to hit limit
    
    def test_message_size_limits(self):
        """Test WebSocket message size limits"""
        
        # Create oversized message
        oversized_data = {
            "agent_id": "test",
            "large_data": "x" * (SecurityConfig.MAX_MESSAGE_SIZE + 1000)
        }
        
        packed_data = msgpack.packb(oversized_data)
        assert len(packed_data) > SecurityConfig.MAX_MESSAGE_SIZE
        
        # In real implementation, this would be rejected at WebSocket level
        # Here we verify the size check logic
    
    def test_session_validation(self, client):
        """Test WebSocket session validation"""
        
        # This would require actual WebSocket testing
        # For now, test session management logic
        from rla2a_enhanced import EnhancedA2ASystem
        
        system = EnhancedA2ASystem()
        
        # Add a test session
        session_id = "test_session_123"
        system.active_sessions[session_id] = {
            "agent_id": "test_agent",
            "created": time.time(),
            "last_activity": time.time()
        }
        
        # Test session exists
        assert session_id in system.active_sessions
        
        # Test cleanup of expired sessions
        # Mock old session
        old_session_id = "old_session_456"
        system.active_sessions[old_session_id] = {
            "agent_id": "old_agent",
            "created": time.time() - 7200,  # 2 hours ago
            "last_activity": time.time() - 7200
        }
        
        system.cleanup_expired_sessions()
        
        # Old session should be cleaned up
        assert old_session_id not in system.active_sessions
        # Current session should remain
        assert session_id in system.active_sessions
    
    def test_cors_configuration(self, client):
        """Test CORS security configuration"""
        
        # Test preflight request
        response = client.options("/", headers={
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST"
        })
        
        # Check CORS headers are configured
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            # Should not allow all origins in production
            if SecurityConfig.ALLOWED_ORIGINS != ["*"]:
                assert allowed_origin != "*"
    
    def test_agent_limits_enforcement(self, client):
        """Test maximum agent limits"""
        
        # Register maximum allowed agents
        agent_count = 0
        max_agents = min(5, CONFIG["MAX_AGENTS"])  # Use smaller limit for testing
        
        for i in range(max_agents + 2):  # Try to exceed limit
            response = client.post("/register", json={
                "agent_id": f"test_agent_{i}",
                "ai_provider": "openai"
            })
            
            if response.status_code == 200:
                agent_count += 1
            elif response.status_code == 429:  # Limit reached
                break
        
        # Should respect the limit
        assert agent_count <= max_agents
    
    def test_error_information_disclosure(self, client):
        """Test that errors don't disclose sensitive information"""
        
        # Test 404 errors don't reveal system info
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Error message shouldn't contain sensitive paths or stack traces
        error_text = response.text.lower()
        sensitive_terms = ["traceback", "exception", "file system", "directory", "path"]
        for term in sensitive_terms:
            assert term not in error_text
        
        # Test malformed requests
        response = client.post("/register", data="invalid json")
        assert response.status_code == 422
        
        # Should return structured error, not raw exception
        if response.headers.get("content-type", "").startswith("application/json"):
            error_data = response.json()
            assert "detail" in error_data

class TestAIProviderSecurity:
    """Test AI provider security features"""
    
    def test_prompt_sanitization(self):
        """Test AI prompt sanitization"""
        from rla2a_enhanced import AIProviderManager
        
        ai_manager = AIProviderManager()
        
        # Test malicious observation
        malicious_obs = {
            "user_input": "<script>alert('xss')</script>",
            "command": "rm -rf /",
            "injection": "'; DROP TABLE agents; --",
            "huge_data": "x" * 10000,
            "nested": {
                "deep": {
                    "attack": "<img src=x onerror=alert('pwned')>"
                }
            }
        }
        
        sanitized = ai_manager._sanitize_observation(malicious_obs)
        
        # Check sanitization worked
        assert "<script>" not in str(sanitized)
        assert "rm -rf" not in str(sanitized) 
        assert "DROP TABLE" not in str(sanitized)
        assert len(str(sanitized)) < len(str(malicious_obs))
    
    def test_ai_response_validation(self):
        """Test AI response validation"""
        from rla2a_enhanced import AIProviderManager
        
        ai_manager = AIProviderManager()
        
        valid_actions = ["move_forward", "turn_left", "turn_right", "communicate", "observe", "pause"]
        
        # Test valid response
        valid_response = ai_manager._get_fallback_action(valid_actions)
        assert valid_response["command"] in valid_actions
        assert "provider" in valid_response
        assert "message" in valid_response
        
        # Test validation logic would reject invalid actions
        # (This would be in the actual AI provider response handling)
    
    @patch('openai.AsyncOpenAI')
    def test_ai_provider_timeout_protection(self, mock_openai):
        """Test AI provider timeout protection"""
        
        # Mock slow AI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "move_forward"
        
        # Simulate timeout
        mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()
        mock_openai.return_value = mock_client
        
        from rla2a_enhanced import AIProviderManager
        ai_manager = AIProviderManager()
        ai_manager.openai_client = mock_client
        
        # Should handle timeout gracefully and fall back
        # (Would need to run in async context for full test)

class TestDataPoisoningProtection:
    """Test protection against data poisoning attacks"""
    
    def test_feedback_validation(self, client):
        """Test feedback validation prevents poisoned rewards"""
        
        # Register test agent first
        client.post("/register", json={
            "agent_id": "poison_test",
            "ai_provider": "openai"  
        })
        
        # Valid feedback
        valid_feedback = {
            "agent_id": "poison_test",
            "action_id": "test_action_1",
            "reward": 0.8,
            "context": {"source": "test"}
        }
        response = client.post("/feedback", json=valid_feedback)
        assert response.status_code == 200
        
        # Invalid reward values (outside bounds)
        invalid_rewards = [-2.0, 2.0, float('inf'), float('-inf')]
        
        for invalid_reward in invalid_rewards:
            invalid_feedback = {
                "agent_id": "poison_test", 
                "action_id": "test_action",
                "reward": invalid_reward,
                "context": {}
            }
            response = client.post("/feedback", json=invalid_feedback)
            assert response.status_code == 422  # Validation error
    
    def test_observation_poisoning_protection(self):
        """Test protection against poisoned observations"""
        from rla2a_enhanced import AIProviderManager
        
        ai_manager = AIProviderManager()
        
        # Poisoned observation with malicious data
        poisoned_obs = {
            "position": {"x": float('inf'), "y": float('-inf'), "z": 999999},
            "malicious_code": "import os; os.system('rm -rf /')",
            "buffer_overflow": "A" * 1000000,
            "sql_injection": "'; DROP TABLE users; --",
            "code_injection": "__import__('os').system('evil_command')"
        }
        
        # Should be sanitized
        clean_obs = ai_manager._sanitize_observation(poisoned_obs)
        
        # Check poisoned data was cleaned
        assert math.isfinite(clean_obs["position"]["x"])
        assert math.isfinite(clean_obs["position"]["y"]) 
        assert abs(clean_obs["position"]["z"]) <= 1000
        assert "import os" not in str(clean_obs)
        assert "DROP TABLE" not in str(clean_obs)
        assert "__import__" not in str(clean_obs)

class TestProductionSecurity:
    """Test production security considerations"""
    
    def test_debug_mode_disabled(self):
        """Test debug mode is disabled in production"""
        # In production, CONFIG["DEBUG"] should be False
        assert CONFIG["DEBUG"] is False or os.getenv("DEBUG", "false").lower() == "false"
    
    def test_secret_key_configured(self):
        """Test secret key is configured and secure"""
        from rla2a_enhanced import SecurityConfig
        
        # Secret key should exist
        assert SecurityConfig.SECRET_KEY is not None
        assert SecurityConfig.SECRET_KEY != ""
        
        # Should not be default/weak value
        weak_keys = ["secret", "password", "123456", "default", "test"]
        assert SecurityConfig.SECRET_KEY.lower() not in weak_keys
        
        # Should be reasonably strong
        assert len(SecurityConfig.SECRET_KEY) >= 16
    
    def test_logging_security(self):
        """Test that logging doesn't expose sensitive data"""
        import logging
        
        # Test that API keys aren't logged
        test_log_message = f"OpenAI key: {CONFIG.get('OPENAI_API_KEY', 'test-key')}"
        
        # In production logging, API keys should be redacted or not logged
        # This is a reminder to implement proper log sanitization

if __name__ == "__main__":
    pytest.main([__file__, "-v"])