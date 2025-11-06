# Deployment Guide

## Overview

This comprehensive deployment guide covers various deployment scenarios for the Day Trading Orchestrator, from development environments to production-grade cloud deployments. Choose the approach that best fits your needs and infrastructure requirements.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Development Environment](#development-environment)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [High Availability Setup](#high-availability-setup)
8. [Security Configuration](#security-configuration)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Backup and Recovery](#backup-and-recovery)
11. [Troubleshooting](#troubleshooting)

## Deployment Overview

### Architecture Patterns

The Day Trading Orchestrator can be deployed using multiple architectural patterns:

#### 1. Monolithic Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                     Single Node                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Frontend   │  │   Backend    │  │   Database   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```
**Best for**: Small deployments, development, testing

#### 2. Microservices Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                           │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  Strategy    │    Data      │   Broker     │     Risk        │
│   Service    │   Service    │  Service     │   Service       │
└──────────────┴──────────────┴──────────────┴─────────────────┘
```
**Best for**: Medium to large deployments, scalability requirements

#### 3. Cloud-Native Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                  Cloud Load Balancer                        │
├─────────────────────────────────────────────────────────────┤
│  Auto-scaling Groups         │  Managed Database           │
│  ┌─────┐ ┌─────┐ ┌─────┐     │  ┌──────┐ ┌──────┐           │
│  │ App │ │ App │ │ App │     │  │ DB   │ │Cache │           │
│  └─────┘ └─────┘ └─────┘     │  └──────┘ └──────┘           │
└─────────────────────────────────────────────────────────────┘
```
**Best for**: Production deployments, high availability, global scale

### Deployment Decision Matrix

| Requirement | Development | Local | Docker | Kubernetes | Cloud |
|-------------|-------------|-------|---------|------------|--------|
| Setup Time | 5 minutes | 15 minutes | 10 minutes | 45 minutes | 30 minutes |
| Scalability | ❌ | ❌ | ✅ | ✅✅ | ✅✅ |
| High Availability | ❌ | ❌ | ❌ | ✅ | ✅✅ |
| Cost | Free | Low | Low | Medium | Variable |
| Management | Simple | Simple | Medium | Complex | Medium |
| Use Case | Development | Testing | Small Prod | Medium Prod | Enterprise |

## Development Environment

### Prerequisites

```bash
# System requirements
- Python 3.9+
- Node.js 16+
- Git
- 8GB RAM minimum
- 20GB disk space
```

### Quick Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/trading-orchestrator/core.git
cd core

# Install Python dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install frontend dependencies
cd frontend
npm install

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
cd ..
python scripts/init_db.py

# Start development servers
python scripts/dev_setup.py
```

### Development Setup Script

```bash
#!/bin/bash
# scripts/dev_setup.sh

set -e

echo "🚀 Setting up Day Trading Orchestrator Development Environment"

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python version
if ! command -v python3.9 &> /dev/null; then
    echo "❌ Python 3.9+ is required"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required"
    exit 1
fi

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup frontend
echo "⚛️  Setting up frontend..."
cd frontend
npm install
cd ..

# Create environment file
echo "⚙️  Creating environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Setup database
echo "🗄️  Setting up database..."
python scripts/init_db.py

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{logs,cache,backups,uploads}
mkdir -p config/strategies

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'python main.py --dev' to start development servers"
```

### Development Configuration

```yaml
# config/development.yaml
environment: development
debug: true
log_level: DEBUG

# Database
database:
  type: sqlite
  path: data/dev_database.db
  echo: true

# Redis (optional for development)
redis:
  enabled: false
  host: localhost
  port: 6379

# Security
security:
  secret_key: dev-secret-key-change-in-production
  jwt_expiration: 3600
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"

# API settings
api:
  host: 0.0.0.0
  port: 8000
  workers: 1
  reload: true

# Frontend settings
frontend:
  host: 0.0.0.0
  port: 3000
  dev_mode: true
  hot_reload: true

# Broker settings (development)
brokers:
  alpaca:
    enabled: true
    paper_trading: true
    base_url: https://paper-api.alpaca.markets
  
  binance:
    enabled: false  # Disable for development

# Logging
logging:
  level: DEBUG
  format: detailed
  file: data/logs/development.log
  max_size: 10MB
  backup_count: 5
```

### Development Commands

```bash
# Start all services in development mode
python main.py --dev

# Start individual services
python main.py --backend-only
python main.py --frontend-only
python main.py --worker-only

# Database operations
python scripts/db_create.py
python scripts/db_migrate.py
python scripts/db_seed.py
python scripts/db_reset.py

# Testing
python -m pytest
python -m pytest --cov=.
python -m pytest tests/integration/

# Code quality
black .
flake8 .
mypy .
isort .
```

## Local Deployment

### System Requirements

```bash
# Minimum requirements
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB SSD
- OS: Ubuntu 20.04+ / CentOS 8+ / macOS 11+ / Windows 10+
- Network: 100 Mbps
```

### Manual Installation

```bash
#!/bin/bash
# scripts/install_production.sh

set -e

echo "🚀 Installing Day Trading Orchestrator for Production"

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.9 \
    python3.9-venv \
    python3-pip \
    nodejs \
    npm \
    git \
    curl \
    wget \
    unzip \
    postgresql-client \
    redis-tools \
    nginx \
    supervisor

# Create application user
sudo useradd -r -s /bin/false trading-app || true
sudo mkdir -p /opt/trading-orchestrator
sudo chown trading-app:trading-app /opt/trading-orchestrator

# Setup application directory
sudo -u trading-app git clone https://github.com/trading-orchestrator/core.git /opt/trading-orchestrator
cd /opt/trading-orchestrator

# Create virtual environment
sudo -u trading-app python3.9 -m venv venv
sudo -u trading-app ./venv/bin/pip install --upgrade pip
sudo -u trading-app ./venv/bin/pip install -r requirements.txt

# Setup frontend
sudo -u trading-app npm install
sudo -u trading-app npm run build

# Create configuration
sudo -u trading-app cp config/production.example.yaml config/production.yaml
sudo -u trading-app mkdir -p data/{logs,cache,backups}

# Setup database
sudo -u postgres createdb trading_orchestrator || true
sudo -u trading-app ./venv/bin/python scripts/init_db.py

# Setup nginx
sudo cp scripts/nginx.conf /etc/nginx/sites-available/trading-orchestrator
sudo ln -sf /etc/nginx/sites-available/trading-orchestrator /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Setup supervisor
sudo cp scripts/supervisor.conf /etc/supervisor/conf.d/trading-orchestrator.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start trading-orchestrator

echo "✅ Installation complete!"
echo "🌐 Access the application at: http://your-server-ip"
```

### Local Production Configuration

```yaml
# config/production.yaml
environment: production
debug: false
log_level: INFO

# Database
database:
  type: postgresql
  host: localhost
  port: 5432
  name: trading_orchestrator
  user: trading_user
  password: secure_password_here
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30

# Redis
redis:
  host: localhost
  port: 6379
  password: redis_password_here
  db: 0
  max_connections: 50

# Security
security:
  secret_key: CHANGE_THIS_SECRET_KEY_IN_PRODUCTION
  jwt_expiration: 1800
  cors_origins:
    - "https://yourdomain.com"
  rate_limiting:
    enabled: true
    requests_per_minute: 100

# API
api:
  host: 127.0.0.1
  port: 8000
  workers: 4
  max_requests: 1000
  max_requests_jitter: 50
  timeout: 30

# Frontend
frontend:
  static_path: frontend/dist
  cache_control: "public, max-age=31536000"

# SSL/TLS
ssl:
  enabled: true
  cert_path: /etc/ssl/certs/trading-orchestrator.crt
  key_path: /etc/ssl/private/trading-orchestrator.key

# Broker connections
brokers:
  alpaca:
    enabled: true
    paper_trading: false
    base_url: https://api.alpaca.markets
    rate_limit: 200  # requests per minute
  
  binance:
    enabled: true
    api_url: https://api.binance.com
    rate_limit: 1200  # requests per minute

# Performance
performance:
  enable_caching: true
  cache_ttl: 300
  enable_compression: true
  max_request_size: 16MB

# Monitoring
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_endpoint: /health
```

### Service Configuration

```ini
# /etc/supervisor/conf.d/trading-orchestrator.conf
[program:trading-orchestrator]
command=/opt/trading-orchestrator/venv/bin/python main.py --production
directory=/opt/trading-orchestrator
user=trading-app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/trading-orchestrator.err.log
stdout_logfile=/var/log/supervisor/trading-orchestrator.out.log
environment=PATH="/opt/trading-orchestrator/venv/bin",PYTHONPATH="/opt/trading-orchestrator"

[program:trading-worker]
command=/opt/trading-orchestrator/venv/bin/python worker.py
directory=/opt/trading-orchestrator
user=trading-app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/trading-orchestrator-worker.err.log
stdout_logfile=/var/log/supervisor/trading-orchestrator-worker.out.log

[program:trading-scheduler]
command=/opt/trading-orchestrator/venv/bin/python scheduler.py
directory=/opt/trading-orchestrator
user=trading-app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/trading-orchestrator-scheduler.err.log
stdout_logfile=/var/log/supervisor/trading-orchestrator-scheduler.out.log
```

```nginx
# /etc/nginx/sites-available/trading-orchestrator
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/trading-orchestrator.crt;
    ssl_certificate_key /etc/ssl/private/trading-orchestrator.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;

    # Proxy to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Serve static files
    location / {
        root /opt/trading-orchestrator/frontend/dist;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Metrics endpoint (restricted)
    location /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://127.0.0.1:9090;
    }
}
```

## Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: trading_orchestrator
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: secure_password_here
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_orchestrator"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:6-alpine
    command: redis-server --requirepass redis_password_here
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # InfluxDB for time-series data
  influxdb:
    image: influxdb:2.0
    environment:
      INFLUXDB_DB: market_data
      INFLUXDB_ADMIN_USER: admin
      INFLUXDB_ADMIN_PASSWORD: influx_password_here
      INFLUXDB_USER: trading_user
      INFLUXDB_USER_PASSWORD: user_password_here
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"
    restart: unless-stopped

  # Application
  app:
    build: .
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://trading_user:secure_password_here@postgres:5432/trading_orchestrator
      - REDIS_URL=redis://:redis_password_here@redis:6379/0
      - INFLUXDB_URL=http://influxdb:8086
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # Worker
  worker:
    build: .
    command: python worker.py
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://trading_user:secure_password_here@postgres:5432/trading_orchestrator
      - REDIS_URL=redis://:redis_password_here@redis:6379/0
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      replicas: 2

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    ports:
      - "3000:80"
    restart: unless-stopped

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - app
      - frontend
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin_password_here
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  influxdb_data:
  prometheus_data:
  grafana_data:
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
```

### Docker Commands

```bash
# Build and start services
docker-compose up -d --build

# View logs
docker-compose logs -f app
docker-compose logs -f worker

# Scale workers
docker-compose up -d --scale worker=3

# Stop services
docker-compose down

# Update services
docker-compose pull
docker-compose up -d

# Database operations
docker-compose exec app python scripts/db_migrate.py
docker-compose exec app python scripts/db_backup.py

# Monitoring
docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: '3.8'

services:
  app:
    image: trading-orchestrator:latest
    environment:
      - DATABASE_URL=postgresql://trading_user:secure_password_here@postgres:5432/trading_orchestrator
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
      placement:
        constraints:
          - node.role == worker
    networks:
      - app-network
    secrets:
      - db_password
      - redis_password

  worker:
    image: trading-orchestrator:latest
    command: python worker.py
    environment:
      - DATABASE_URL=postgresql://trading_user:secure_password_here@postgres:5432/trading_orchestrator
    deploy:
      replicas: 5
      restart_policy:
        condition: on-failure
    networks:
      - app-network
    secrets:
      - db_password
      - redis_password

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager
    networks:
      - app-network

networks:
  app-network:
    driver: overlay

secrets:
  db_password:
    external: true
  redis_password:
    external: true
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-orchestrator
  labels:
    name: trading-orchestrator

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-orchestrator-config
  namespace: trading-orchestrator
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_TYPE: "postgresql"
  REDIS_ENABLED: "true"
  INFLUXDB_ENABLED: "true"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-orchestrator-secrets
  namespace: trading-orchestrator
type: Opaque
stringData:
  DATABASE_PASSWORD: "secure-database-password"
  REDIS_PASSWORD: "secure-redis-password"
  JWT_SECRET: "your-jwt-secret-key"
  API_SECRET_KEY: "your-api-secret-key"

---
# k8s/postgres.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: trading-orchestrator
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: trading-orchestrator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: trading_orchestrator
        - name: POSTGRES_USER
          value: trading_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: DATABASE_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: trading-orchestrator
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
# k8s/redis.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: trading-orchestrator
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: trading-orchestrator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6-alpine
        command: ["redis-server"]
        args: ["--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: REDIS_PASSWORD
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: trading-orchestrator
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

### Application Deployment

```yaml
# k8s/app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-orchestrator
  template:
    metadata:
      labels:
        app: trading-orchestrator
    spec:
      containers:
      - name: app
        image: trading-orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://trading_user:$(DATABASE_PASSWORD)@postgres:5432/trading_orchestrator"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379/0"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: JWT_SECRET
        envFrom:
        - configMapRef:
            name: trading-orchestrator-config
        - secretRef:
            name: trading-orchestrator-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: trading-orchestrator-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: trading-orchestrator-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: trading-orchestrator-logs-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
spec:
  selector:
    app: trading-orchestrator
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# k8s/worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator-worker
  namespace: trading-orchestrator
spec:
  replicas: 5
  selector:
    matchLabels:
      app: trading-orchestrator-worker
  template:
    metadata:
      labels:
        app: trading-orchestrator-worker
    spec:
      containers:
      - name: worker
        image: trading-orchestrator:latest
        command: ["python", "worker.py"]
        env:
        - name: DATABASE_URL
          value: "postgresql://trading_user:$(DATABASE_PASSWORD)@postgres:5432/trading_orchestrator"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379/0"
        envFrom:
        - configMapRef:
            name: trading-orchestrator-config
        - secretRef:
            name: trading-orchestrator-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - yourdomain.com
    secretName: trading-orchestrator-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trading-orchestrator
            port:
              number: 8000
```

### Kubernetes Commands

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n trading-orchestrator
kubectl get services -n trading-orchestrator
kubectl get ingress -n trading-orchestrator

# View logs
kubectl logs -f deployment/trading-orchestrator -n trading-orchestrator
kubectl logs -f deployment/trading-orchestrator-worker -n trading-orchestrator

# Scale application
kubectl scale deployment trading-orchestrator --replicas=5 -n trading-orchestrator
kubectl scale deployment trading-orchestrator-worker --replicas=10 -n trading-orchestrator

# Update deployment
kubectl set image deployment/trading-orchestrator app=trading-orchestrator:v1.1.0 -n trading-orchestrator

# Rollback deployment
kubectl rollout undo deployment/trading-orchestrator -n trading-orchestrator

# Check resource usage
kubectl top pods -n trading-orchestrator
kubectl top nodes

# Port forwarding for development
kubectl port-forward service/trading-orchestrator 8000:8000 -n trading-orchestrator
```

## Cloud Deployment

### AWS Deployment

#### Infrastructure as Code (Terraform)

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "trading-orchestrator"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Environment = var.environment
    Project     = "trading-orchestrator"
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "trading-orchestrator-db"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type         = "gp2"
  storage_encrypted    = true
  
  db_name  = "trading_orchestrator"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = var.db_backup_retention
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "trading-orchestrator-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null
  
  tags = {
    Environment = var.environment
    Project     = "trading-orchestrator"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "trading-orchestrator-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "trading-orchestrator-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "trading-orchestrator"
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "trading-orchestrator-redis"
  description                  = "Redis cluster for trading orchestrator"
  
  node_type                    = var.redis_node_type
  port                         = 6379
  parameter_group_name         = "default.redis6.x"
  
  num_cache_clusters           = var.redis_num_cache_clusters
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = var.redis_auth_token
  
  tags = {
    Environment = var.environment
    Project     = "trading-orchestrator"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "trading-orchestrator"
  cluster_version = "1.27"
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  
  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  node_groups = {
    main = {
      desired_capacity = var.eks_desired_capacity
      max_capacity     = var.eks_max_capacity
      min_capacity     = var.eks_min_capacity
      
      instance_types = [var.eks_instance_type]
      
      k8s_labels = {
        Environment = var.environment
        Project     = "trading-orchestrator"
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "trading-orchestrator"
  }
}
```

#### Variables and Configuration

```hcl
# terraform/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Database configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Initial database storage"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum database storage"
  type        = number
  default     = 100
}

variable "db_backup_retention" {
  description = "Database backup retention period"
  type        = number
  default     = 7
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "trading_user"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Redis configuration
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters"
  type        = number
  default     = 2
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
}

# EKS configuration
variable "eks_instance_type" {
  description = "EKS worker node instance type"
  type        = string
  default     = "t3.medium"
}

variable "eks_desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "eks_min_capacity" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "eks_max_capacity" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}
```

### Azure Deployment

```yaml
# azure-pipelines.yml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistryServiceConnection: 'your-acr-connection'
  imageRepository: 'trading-orchestrator'
  containerRegistry: 'yourregistry.azurecr.io'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.9'
      displayName: 'Use Python 3.9'
    
    - script: |
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
      displayName: 'Install dependencies'
    
    - script: |
        source venv/bin/activate
        pytest --cov=. --cov-report=xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: '**/test-*.xml'
        testRunTitle: 'Publish test results'
    
    - task: Docker@2
      displayName: Build and push Docker image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: Dockerfile
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy to AKS
  dependsOn: Build
  jobs:
  - deployment: Deploy
    displayName: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: Deploy to Kubernetes
            inputs:
              action: deploy
              manifests: |
                $(Pipeline.Workspace)/manifests/deployment.yml
                $(Pipeline.Workspace)/manifests/service.yml
                $(Pipeline.Workspace)/manifests/ingress.yml
              containers: |
                $(containerRegistry)/$(imageRepository):$(tag)
```

### Google Cloud Deployment

```yaml
# cloudbuild.yaml
steps:
# Build the application
- name: 'python:3.9'
  entrypoint: 'bash'
  args:
  - '-c'
  - |
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m pytest

# Build the Docker image
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/trading-orchestrator:$BUILD_ID', '.']

# Push the Docker image
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/trading-orchestrator:$BUILD_ID']

# Deploy to Cloud Run
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'run'
  - 'deploy'
  - 'trading-orchestrator'
  - '--image'
  - 'gcr.io/$PROJECT_ID/trading-orchestrator:$BUILD_ID'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
  - '--allow-unauthenticated'

# Deploy database (Cloud SQL)
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'sql'
  - 'instances'
  - 'create'
  - 'trading-orchestrator-db'
  - '--database-version=POSTGRES_13'
  - '--tier=db-custom-2-7680'
  - '--region=us-central1'
  - '--storage-auto-increase'
  - '--storage-type=SSD'

# Create database and user
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'sql'
  - 'databases'
  - 'create'
  - 'INSTANCE=trading-orchestrator-db'
  - 'DATABASE=trading_orchestrator'

- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'sql'
  - 'users'
  - 'create'
  - 'INSTANCE=trading-orchestrator-db'
  - 'USERNAME=trading_user'
  - '--password=secure_password_here'

images:
- 'gcr.io/$PROJECT_ID/trading-orchestrator:$BUILD_ID'
```

## High Availability Setup

### Load Balancing Configuration

```nginx
# nginx-load-balancer.conf
upstream trading_backend {
    least_conn;
    
    server app1.internal:8000 weight=3 max_fails=3 fail_timeout=30s;
    server app2.internal:8000 weight=3 max_fails=3 fail_timeout=30s;
    server app3.internal:8000 weight=3 max_fails=3 fail_timeout=30s;
    
    keepalive 32;
}

upstream trading_websocket {
    ip_hash;
    
    server app1.internal:8000;
    server app2.internal:8000;
    server app3.internal:8000;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/trading-orchestrator.crt;
    ssl_certificate_key /etc/ssl/private/trading-orchestrator.key;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://trading_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        access_log off;
    }
    
    # Main API endpoints
    location /api/ {
        proxy_pass http://trading_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # WebSocket endpoints
    location /ws {
        proxy_pass http://trading_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket timeouts
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
    
    # Static files with caching
    location /static/ {
        root /var/www/trading-orchestrator;
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # Gzip static files
        gzip_static on;
    }
    
    # Rate limiting
    location /api/auth/ {
        limit_req zone=login burst=5 nodelay;
        proxy_pass http://trading_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

# Health check server
server {
    listen 80;
    server_name health.yourdomain.com;
    
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    location /ready {
        access_log off;
        return 200 "ready\n";
        add_header Content-Type text/plain;
    }
}
```

### Database Clustering

```sql
-- PostgreSQL streaming replication setup
-- Primary server configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 32
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s

-- Primary server permissions (pg_hba.conf)
host replication replicator 192.168.1.0/24 md5
host all all 192.168.1.0/24 md5

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'replication_password';

-- Backup primary database
pg_basebackup -h primary-host -U replicator -D /var/lib/postgresql/standby -Ft -z -P -R

-- Standby server configuration (postgresql.conf)
hot_standby = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s

-- Standby server connection (primary_conninfo)
primary_conninfo = 'host=primary-host port=5432 user=replicator password=replication_password'
```

### Monitoring and Health Checks

```yaml
# monitoring/health-check.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: health-check-config
data:
  check_database.py: |
    #!/usr/bin/env python3
    import psycopg2
    import redis
    import sys
    import time
    
    def check_postgres():
        try:
            conn = psycopg2.connect(
                host='postgres',
                database='trading_orchestrator',
                user='trading_user',
                password='secure_password_here'
            )
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.fetchone()
            conn.close()
            return True, "Database connection OK"
        except Exception as e:
            return False, f"Database connection failed: {e}"
    
    def check_redis():
        try:
            r = redis.Redis(host='redis', port=6379, password='redis_password_here')
            r.ping()
            return True, "Redis connection OK"
        except Exception as e:
            return False, f"Redis connection failed: {e}"
    
    def main():
        checks = [
            ("Database", check_postgres),
            ("Redis", check_redis),
        ]
        
        all_healthy = True
        for name, check_func in checks:
            healthy, message = check_func()
            print(f"{name}: {message}")
            if not healthy:
                all_healthy = False
        
        if all_healthy:
            sys.exit(0)
        else:
            sys.exit(1)
    
    if __name__ == "__main__":
        main()

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: health-check
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: health-check
            image: python:3.9-alpine
            command: ["python", "/scripts/check_database.py"]
            volumeMounts:
            - name: scripts
              mountPath: /scripts
          volumes:
          - name: scripts
            configMap:
              name: health-check-config
          restartPolicy: OnFailure
```

## Security Configuration

### SSL/TLS Setup

```bash
#!/bin/bash
# scripts/setup-ssl.sh

set -e

DOMAIN=${1:-yourdomain.com}
EMAIL=${2:-admin@yourdomain.com}

echo "Setting up SSL certificate for $DOMAIN"

# Install Certbot
sudo apt update
sudo apt install -y certbot python3-certbot-nginx

# Stop nginx temporarily
sudo systemctl stop nginx

# Obtain certificate
sudo certbot certonly \
    --standalone \
    --agree-tos \
    --no-eff-email \
    --email $EMAIL \
    -d $DOMAIN \
    -d www.$DOMAIN

# Setup automatic renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -

# Setup nginx with SSL
sudo cp nginx-ssl.conf /etc/nginx/sites-available/trading-orchestrator
sudo ln -sf /etc/nginx/sites-available/trading-orchestrator /etc/nginx/sites-enabled/

# Start nginx
sudo systemctl start nginx

echo "SSL certificate setup complete!"
echo "Certificate location: /etc/letsencrypt/live/$DOMAIN/"
```

### Security Headers

```nginx
# security-headers.conf
# Add security headers to all responses

add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' wss: https:; media-src 'self'; object-src 'none'; child-src 'self'; worker-src 'self';" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Remove server signature
server_tokens off;

# Disable server signature on PHP
server_signature off;

# Limit request methods
if ($request_method !~ ^(GET|HEAD|POST)$ ) {
    return 444;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;

# IP whitelist for admin endpoints
location /admin {
    allow 192.168.1.0/24;  # Internal network
    allow 10.0.0.0/8;      # Internal network
    deny all;               # Deny all others
    
    proxy_pass http://trading_backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# API rate limiting
location /api/ {
    limit_req zone=api burst=20 nodelay;
    proxy_pass http://trading_backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# Login rate limiting
location /api/auth/login {
    limit_req zone=login burst=3 nodelay;
    proxy_pass http://trading_backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### Firewall Configuration

```bash
#!/bin/bash
# scripts/setup-firewall.sh

set -e

echo "Configuring firewall rules..."

# Reset UFW
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if needed)
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal database connections (if needed)
sudo ufw allow from 10.0.0.0/8 to any port 5432

# Allow internal Redis connections (if needed)
sudo ufw allow from 10.0.0.0/8 to any port 6379

# Allow monitoring (if applicable)
sudo ufw allow from 192.168.1.0/24 to any port 9090

# Enable firewall
sudo ufw --force enable

echo "Firewall configuration complete!"
sudo ufw status verbose
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'trading-orchestrator'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

# Alert rules
# monitoring/alert_rules.yml
groups:
- name: trading-orchestrator
  rules:
  - alert: ApplicationDown
    expr: up{job="trading-orchestrator"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Trading Orchestrator application is down"
      description: "Trading Orchestrator application has been down for more than 1 minute"
      
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests per second"
      
  - alert: DatabaseConnectionFailure
    expr: pg_up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "PostgreSQL database is not responding"
      
  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"
      
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 80%"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Trading Orchestrator Overview",
    "tags": ["trading", "orchestrator"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{ method }} {{ status }}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "Median"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error rate"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "trading_orchestrator_active_connections",
            "legendFormat": "Active connections"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### Log Aggregation

```yaml
# fluentd-config.yml
<source>
  @type tail
  @id nginx_access
  path /var/log/nginx/access.log
  pos_file /var/log/fluentd-nginx-access.log.pos
  tag nginx.access
  format /^(?<remote_addr>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^ ]*) +\S*)?" (?<status>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^ ]*)" "(?<agent>[^ ]*)")?$/
  time_format %d/%b/%Y:%H:%M:%S %z
</source>

<source>
  @type tail
  @id nginx_error
  path /var/log/nginx/error.log
  pos_file /var/log/fluentd-nginx-error.log.pos
  tag nginx.error
  format /^(?<timestamp>[^ ]* \[?[^\]]*\]?) (?<level>[^ ]*) (?<message>.*)$/
</source>

<source>
  @type tail
  @id app_logs
  path /opt/trading-orchestrator/logs/app.log
  pos_file /var/log/fluentd-app.log.pos
  tag app.logs
  format json
</source>

<filter nginx.**>
  @type parser
  key_name message
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<match nginx.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name nginx-logs
  type_name _doc
</match>

<match app.logs>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name app-logs
  type_name _doc
</match>
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup.sh

set -e

BACKUP_DIR="/var/backups/trading-orchestrator"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

echo "Starting backup at $(date)"

# Database backup
echo "Backing up database..."
pg_dump -h localhost -U trading_user -d trading_orchestrator \
    --clean --create --verbose \
    | gzip > $BACKUP_DIR/database_$DATE.sql.gz

# Redis backup
echo "Backing up Redis..."
redis-cli --rdb - > $BACKUP_DIR/redis_$DATE.rdb

# Application data backup
echo "Backing up application data..."
tar -czf $BACKUP_DIR/app_data_$DATE.tar.gz \
    /opt/trading-orchestrator/data \
    /opt/trading-orchestrator/config

# Configuration backup
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    /etc/nginx/sites-available/trading-orchestrator \
    /etc/supervisor/conf.d/trading-orchestrator.conf

# Upload to cloud storage (AWS S3 example)
echo "Uploading to cloud storage..."
aws s3 cp $BACKUP_DIR/database_$DATE.sql.gz s3://trading-orchestrator-backups/database/
aws s3 cp $BACKUP_DIR/redis_$DATE.rdb s3://trading-orchestrator-backups/redis/
aws s3 cp $BACKUP_DIR/app_data_$DATE.tar.gz s3://trading-orchestrator-backups/app-data/
aws s3 cp $BACKUP_DIR/config_$DATE.tar.gz s3://trading-orchestrator-backups/config/

# Clean up old local backups
echo "Cleaning up old local backups..."
find $BACKUP_DIR -name "*.gz" -o -name "*.tar.gz" -o -name "*.rdb" \
    -type f -mtime +$RETENTION_DAYS -delete

# Clean up old cloud backups
echo "Cleaning up old cloud backups..."
aws s3 ls s3://trading-orchestrator-backups/database/ | while read -r line; do
    createDate=`echo $line|awk {'print $1" "$2'}`
    createDate=`date -d"$createDate" +%s`
    olderThan=`date -d"-$RETENTION_DAYS days" +%s`
    if [[ $createDate -lt $olderThan ]]; then
        fileName=`echo $line|awk {'print $4'}`
        echo "Deleting old backup: $fileName"
        aws s3 rm s3://trading-orchestrator-backups/database/$fileName
    fi
done

echo "Backup completed at $(date)"
```

### Recovery Procedures

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1
RESTORE_TYPE=${2:-"database"} # database, redis, app_data, config

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file> [database|redis|app_data|config]"
    exit 1
fi

echo "Starting restore from $BACKUP_FILE (type: $RESTORE_TYPE)"

case $RESTORE_TYPE in
    database)
        echo "Restoring database..."
        gunzip -c $BACKUP_FILE | psql -h localhost -U trading_user
        ;;
    redis)
        echo "Restoring Redis..."
        redis-cli FLUSHALL
        redis-cli --pipe < $BACKUP_FILE
        ;;
    app_data)
        echo "Restoring application data..."
        tar -xzf $BACKUP_FILE -C /
        chown -R trading-app:trading-app /opt/trading-orchestrator/data
        ;;
    config)
        echo "Restoring configuration..."
        tar -xzf $BACKUP_FILE -C /
        chown -R root:root /etc/nginx/sites-available/trading-orchestrator
        chown -R root:root /etc/supervisor/conf.d/trading-orchestrator.conf
        systemctl reload nginx
        supervisorctl reload
        ;;
    *)
        echo "Invalid restore type: $RESTORE_TYPE"
        echo "Valid types: database, redis, app_data, config"
        exit 1
        ;;
esac

echo "Restore completed successfully!"
```

### Backup Schedule

```bash
#!/bin/bash
# Add to crontab for automated backups

# Daily database backup at 2 AM
0 2 * * * /opt/trading-orchestrator/scripts/backup.sh database

# Weekly full backup at 1 AM on Sunday
0 1 * * 0 /opt/trading-orchestrator/scripts/backup.sh full

# Monthly archive backup at 12 AM on 1st of month
0 0 1 * * /opt/trading-orchestrator/scripts/backup.sh archive
```

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Start

```bash
# Check application logs
tail -f /opt/trading-orchestrator/logs/app.log

# Check system logs
journalctl -u trading-orchestrator -f

# Check database connection
psql -h localhost -U trading_user -d trading_orchestrator -c "SELECT 1;"

# Check Redis connection
redis-cli ping

# Verify configuration
python -c "import yaml; yaml.safe_load(open('config/production.yaml'))"
```

#### High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check Python memory usage
python -c "import psutil; print(f'Process memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB')"

# Enable memory profiling
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2
```

#### Database Performance Issues

```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Check database size
SELECT pg_size_pretty(pg_database_size('trading_orchestrator'));

-- Vacuum and analyze
VACUUM ANALYZE;
```

#### SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in /etc/ssl/certs/trading-orchestrator.crt -text -noout

# Test certificate
curl -I https://yourdomain.com

# Renew certificate
sudo certbot renew --dry-run

# Check certificate expiration
echo | openssl s_client -servername yourdomain.com -connect yourdomain.com:443 2>/dev/null | openssl x509 -noout -dates
```

### Performance Monitoring

```bash
# System performance
top -p $(pgrep -f "python.*main.py")
iostat 1
netstat -i
df -h

# Application performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Database performance
psql -h localhost -U trading_user -d trading_orchestrator -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
"

# Redis performance
redis-cli --latency-history
redis-cli info stats
```

### Emergency Procedures

#### Quick System Recovery

```bash
#!/bin/bash
# scripts/emergency-recovery.sh

set -e

echo "Starting emergency recovery procedures..."

# Stop all services
systemctl stop nginx
supervisorctl stop all

# Restore latest backup
BACKUP_FILE=$(ls -t /var/backups/trading-orchestrator/database_*.sql.gz | head -1)
if [ -n "$BACKUP_FILE" ]; then
    echo "Restoring database from $BACKUP_FILE"
    gunzip -c $BACKUP_FILE | psql -h localhost -U trading_user
else
    echo "No database backup found!"
fi

# Restart services
systemctl start nginx
supervisorctl start all

# Verify services
sleep 10
curl -f http://localhost/health || echo "Health check failed!"

echo "Emergency recovery completed."
```

#### Graceful Shutdown

```bash
#!/bin/bash
# scripts/graceful-shutdown.sh

echo "Starting graceful shutdown..."

# Stop accepting new connections
systemctl stop nginx

# Wait for active connections to close (max 30 seconds)
for i in {1..30}; do
    CONNECTIONS=$(netstat -an | grep :8000 | grep ESTABLISHED | wc -l)
    if [ $CONNECTIONS -eq 0 ]; then
        echo "All connections closed."
        break
    fi
    echo "Waiting for $CONNECTIONS connections to close... ($i/30)"
    sleep 1
done

# Stop application
supervisorctl stop all

echo "Graceful shutdown completed."
```

## Conclusion

This deployment guide provides comprehensive coverage for deploying the Day Trading Orchestrator across various environments. Key takeaways:

### Deployment Checklist

- [ ] Choose appropriate deployment model
- [ ] Set up security measures (SSL, firewall, etc.)
- [ ] Configure monitoring and logging
- [ ] Implement backup and recovery procedures
- [ ] Test deployment in staging environment
- [ ] Set up high availability if required
- [ ] Document deployment procedures
- [ ] Train operations team

### Support Resources

- **Documentation**: Comprehensive guides and references
- **Community**: User forums and discussions
- **Professional Services**: Expert deployment assistance
- **24/7 Support**: Critical issue resolution

### Next Steps

1. **Choose Deployment Model**: Select based on your requirements
2. **Follow Deployment Guide**: Step-by-step instructions
3. **Test Thoroughly**: Verify all functionality works
4. **Monitor Performance**: Set up comprehensive monitoring
5. **Plan for Scale**: Consider growth requirements

---

**Need Deployment Assistance?** Contact our professional services team for expert deployment support and optimization.