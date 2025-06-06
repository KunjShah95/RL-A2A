# 🚀 Production Deployment Guide

This guide covers deploying the RL-A2A Enhanced system in production environments with proper security configurations.

## 🏗️ Deployment Options

### 1. Docker Deployment (Recommended)

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN adduser --disabled-password --gecos '' appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "rla2a_enhanced.py", "server"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  rla2a-server:
    build: .
    container_name: rla2a-enhanced
    ports:
      - "8000:8000"
    environment:
      - A2A_HOST=0.0.0.0
      - A2A_PORT=8000
      - DEBUG=false
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - rla2a-network

  rla2a-dashboard:
    build: .
    container_name: rla2a-dashboard
    ports:
      - "8501:8501"
    environment:
      - DASHBOARD_PORT=8501
    depends_on:
      - rla2a-server
    command: ["python", "rla2a_enhanced.py", "dashboard"]
    restart: unless-stopped
    networks:
      - rla2a-network

  redis:
    image: redis:7-alpine
    container_name: rla2a-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - rla2a-network

  nginx:
    image: nginx:alpine
    container_name: rla2a-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - rla2a-server
      - rla2a-dashboard
    restart: unless-stopped
    networks:
      - rla2a-network

networks:
  rla2a-network:
    driver: bridge

volumes:
  redis_data:
```

### 2. Kubernetes Deployment

#### k8s-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rla2a-enhanced
  labels:
    app: rla2a-enhanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rla2a-enhanced
  template:
    metadata:
      labels:
        app: rla2a-enhanced
    spec:
      containers:
      - name: rla2a-server
        image: rla2a-enhanced:4.0.0
        ports:
        - containerPort: 8000
        env:
        - name: A2A_HOST
          value: "0.0.0.0"
        - name: DEBUG
          value: "false"
        envFrom:
        - secretRef:
            name: rla2a-secrets
        - configMapRef:
            name: rla2a-config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: rla2a-service
spec:
  selector:
    app: rla2a-enhanced
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rla2a-config
data:
  A2A_PORT: "8000"
  RATE_LIMIT_PER_MINUTE: "100"
  MAX_AGENTS: "1000"
  LOG_LEVEL: "INFO"
---
apiVersion: v1
kind: Secret
metadata:
  name: rla2a-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
  GOOGLE_API_KEY: "your-google-key"
  SECRET_KEY: "your-jwt-secret"
```

### 3. Cloud Platform Deployment

#### AWS ECS/Fargate
```json
{
  "family": "rla2a-enhanced",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "rla2a-server",
      "image": "your-registry/rla2a-enhanced:4.0.0",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "A2A_HOST", "value": "0.0.0.0"},
        {"name": "DEBUG", "value": "false"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:rla2a/openai"},
        {"name": "SECRET_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:rla2a/jwt"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rla2a-enhanced",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

## 🔒 Production Security Configuration

### Environment Variables
```bash
# Production .env
# Server Configuration
A2A_HOST=0.0.0.0
A2A_PORT=8000
DEBUG=false

# Security
SECRET_KEY=your-super-secret-jwt-key-min-32-chars
ALLOWED_ORIGINS=https://yourdomain.com,https://dashboard.yourdomain.com
RATE_LIMIT_PER_MINUTE=100
MAX_AGENTS=1000
SESSION_TIMEOUT=3600

# AI Providers
OPENAI_API_KEY=sk-your-production-openai-key
ANTHROPIC_API_KEY=sk-ant-your-production-anthropic-key
GOOGLE_API_KEY=your-production-google-key
DEFAULT_AI_PROVIDER=openai

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/app/logs/rla2a.log

# Performance
MAX_MESSAGE_SIZE=1048576
```

### Nginx Configuration
```nginx
upstream rla2a_backend {
    least_conn;
    server rla2a-server:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # API endpoints
    location /api/ {
        proxy_pass http://rla2a_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Dashboard
    location / {
        proxy_pass http://rla2a-dashboard:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check endpoint
    location = /health {
        access_log off;
        proxy_pass http://rla2a_backend/health;
    }
}
```

## 📊 Monitoring & Observability

### Prometheus Metrics (Optional Enhancement)
```python
# Add to rla2a_enhanced.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter('rla2a_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('rla2a_request_duration_seconds', 'Request duration')
ACTIVE_AGENTS = Gauge('rla2a_active_agents', 'Number of active agents')
ACTIVE_SESSIONS = Gauge('rla2a_active_sessions', 'Number of active sessions')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration
```python
# Production logging setup
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/rla2a.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'WARNING',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Health Check Endpoints
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "agents": len(system.agents),
        "sessions": len(system.active_sessions)
    }

@app.get("/ready")
async def readiness_check():
    # Check AI providers
    providers_healthy = {
        "openai": bool(system.ai_manager.openai_client),
        "anthropic": bool(system.ai_manager.anthropic_client),
        "google": bool(system.ai_manager.google_client)
    }
    
    if not any(providers_healthy.values()):
        raise HTTPException(status_code=503, detail="No AI providers available")
    
    return {"status": "ready", "providers": providers_healthy}
```

## 🔧 Performance Tuning

### Production Settings
```python
# Uvicorn production configuration
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,  # CPU cores
    access_log=True,
    log_level="info",
    loop="uvloop",  # Faster event loop
    http="httptools",  # Faster HTTP parser
    limit_concurrency=1000,
    limit_max_requests=10000,
    timeout_keep_alive=30
)
```

### Resource Limits
```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## 🚨 Production Checklist

### Security Checklist
- [ ] All API keys stored in secure secret management
- [ ] HTTPS enabled with valid SSL certificates
- [ ] CORS configured for specific domains only
- [ ] Rate limiting enabled and configured
- [ ] Input validation active on all endpoints
- [ ] Session timeouts configured
- [ ] Debug mode disabled
- [ ] Security headers configured in reverse proxy
- [ ] Regular security updates scheduled

### Reliability Checklist
- [ ] Health checks configured
- [ ] Monitoring and alerting set up
- [ ] Log aggregation configured
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Load balancing configured
- [ ] Auto-scaling rules defined
- [ ] Circuit breakers implemented

### Performance Checklist
- [ ] Resource limits configured
- [ ] Connection pooling enabled
- [ ] Async/await used throughout
- [ ] Database connections optimized
- [ ] Caching strategy implemented
- [ ] Static assets served by CDN
- [ ] Compression enabled

### Compliance Checklist
- [ ] Data retention policies defined
- [ ] Privacy policies implemented
- [ ] Audit logging enabled
- [ ] User consent mechanisms
- [ ] Data encryption at rest
- [ ] GDPR/CCPA compliance reviewed

## 🔄 Deployment Process

### Blue-Green Deployment
```bash
#!/bin/bash
# deploy.sh

# Build new version
docker build -t rla2a-enhanced:$NEW_VERSION .

# Deploy to green environment
kubectl set image deployment/rla2a-enhanced-green rla2a-server=rla2a-enhanced:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/rla2a-enhanced-green

# Run health checks
curl -f http://green.yourdomain.com/health

# Switch traffic
kubectl patch service rla2a-service -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for issues
sleep 300

# If successful, update blue environment
kubectl set image deployment/rla2a-enhanced-blue rla2a-server=rla2a-enhanced:$NEW_VERSION
```

### Rolling Updates
```bash
# Kubernetes rolling update
kubectl set image deployment/rla2a-enhanced rla2a-server=rla2a-enhanced:4.0.0

# Monitor rollout
kubectl rollout status deployment/rla2a-enhanced

# Rollback if needed
kubectl rollout undo deployment/rla2a-enhanced
```

## 📈 Scaling Guidelines

### Horizontal Scaling
- **Small**: 1-2 replicas, 100 agents
- **Medium**: 3-5 replicas, 500 agents
- **Large**: 5+ replicas, 1000+ agents

### Vertical Scaling
- **CPU**: Scale based on AI provider usage
- **Memory**: Scale based on agent count and session data
- **Network**: Scale based on WebSocket connections

### Database Scaling
- Use Redis for session storage at scale
- Implement database sharding for large deployments
- Consider read replicas for analytics

## 🆘 Troubleshooting

### Common Issues
1. **High Memory Usage**: Increase session cleanup frequency
2. **Slow AI Responses**: Implement provider fallbacks
3. **WebSocket Disconnections**: Check network load balancer settings
4. **Rate Limiting**: Adjust limits based on usage patterns

### Debug Commands
```bash
# Check container logs
docker logs rla2a-enhanced

# Check resource usage
kubectl top pods

# Test endpoints
curl -f http://localhost:8000/health
curl -f http://localhost:8000/metrics

# Database connections
redis-cli ping
```

This deployment guide ensures your RL-A2A Enhanced system runs securely and efficiently in production environments.