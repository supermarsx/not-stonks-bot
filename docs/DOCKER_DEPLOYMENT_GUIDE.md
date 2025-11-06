# Docker Containerization Guide
## Day Trading Orchestrator - Complete Docker Setup

<div align="center">

![Docker](https://img.shields.io/badge/Docker-v1.0.0-blue?style=for-the-badge&logo=docker)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-v2.0+-blue.svg)](https://docs.docker.com/compose/)
[![Multi-stage Builds](https://img.shields.io/badge/Multi--stage%20Builds-Optimized-green.svg)](https://docs.docker.com/develop/build/multistage-build/)

**Production-Ready Docker Deployment**

[🚀 Quick Start](#-quick-start) • [📦 Multi-Stage Builds](#️-multi-stage-builds) • [🐳 Docker Compose](#-docker-compose) • [☁️ Cloud Deployment](#️-cloud-deployment) • [🔧 Configuration](#️-configuration)

</div>

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Multi-Stage Builds](#️-multi-stage-builds)
3. [Docker Compose Setup](#-docker-compose)
4. [Environment-Specific Configurations](#️-environment-specific-configurations)
5. [Security Best Practices](#-security-best-practices)
6. [Performance Optimization](#️-performance-optimization)
7. [Production Deployment](#-production-deployment)
8. [Cloud Deployment](#️-cloud-deployment)
9. [Monitoring and Logging](#️-monitoring-and-logging)
10. [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Development with Docker

```bash
# Clone repository
git clone https://github.com/trading-orchestrator/day-trading-orchestrator.git
cd day-trading-orchestrator

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Access services
# Web UI: http://localhost:3000
# API: http://localhost:8000
# Terminal: docker-compose exec app bash
```

### Production with Docker

```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker-compose.prod.yml ps

# View metrics
docker stats
```

## 📦 Multi-Stage Builds

### Application Dockerfile

Create optimized multi-stage Docker builds:

```dockerfile
# Dockerfile - Production optimized
# syntax=docker/dockerfile:1.4

# ============================================================================
# Stage 1: Dependencies (Base Image)
# ============================================================================
FROM python:3.11-slim as dependencies

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        gcc \
        g++ \
        libc6-dev \
        libpq-dev \
        curl \
        wget \
        git \
        && rm -rf /var/cache/apt

# Set working directory
WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy and install Python dependencies
COPY trading_orchestrator/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Builder (Optional for compiled extensions)
# ============================================================================
FROM dependencies as builder

# Install build dependencies for compiled packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/cache/apt

# Copy source code for compilation
COPY . .

# Build wheel packages (for packages that need compilation)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels \
    TA-Lib==0.4.25 psycopg2-binary==2.9.9

# ============================================================================
# Stage 3: Application (Final Runtime Image)
# ============================================================================
FROM python:3.11-slim as application

# Install runtime dependencies only
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        libpq5 \
        curl \
        && rm -rf /var/cache/apt \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Install wheels from builder stage (if any)
COPY --from=builder --chown=appuser:appuser /app/wheels /tmp/wheels
RUN if [ -d /tmp/wheels ]; then pip install --no-cache-dir /tmp/wheels/*.whl; fi

# Create necessary directories with proper permissions
RUN mkdir -p data/{logs,cache,backups,uploads} \
    config/strategies \
    models/local \
    ssl/certs \
    && chown -R appuser:appuser /app/data /app/config /app/models /app/ssl

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Default command
CMD ["python", "main.py"]
```

### Frontend Dockerfile

```dockerfile
# frontend/Dockerfile - Production optimized
# syntax=docker/dockerfile:1.4

# ============================================================================
# Stage 1: Build (Node.js with dependencies)
# ============================================================================
FROM node:18-alpine as builder

# Install dependencies
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Build application
RUN npm run build

# ============================================================================
# Stage 2: Runtime (Nginx server)
# ============================================================================
FROM nginx:alpine as runtime

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost || exit 1

# Expose port
EXPOSE 80

# Default command
CMD ["nginx", "-g", "daemon off;"]
```

### Worker Dockerfile

```dockerfile
# worker/Dockerfile - For background workers
FROM python:3.11-slim as base

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
        gcc \
        g++ \
        libpq-dev \
        curl \
        && rm -rf /var/cache/apt

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY trading_orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p data/logs && chown -R appuser:appuser /app/data
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port (for health checks)
EXPOSE 8000

# Default command for worker
CMD ["python", "worker.py"]
```

## 🐳 Docker Compose Setup

### Development Environment

```yaml
# docker-compose.dev.yml - Development setup
version: '3.8'

services:
  # Database (SQLite for development)
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: application
    container_name: trading-orchestrator-dev
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
    ports:
      - "8000:8000"
      - "3000:3000"  # Frontend
    networks:
      - dev-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Worker service
  worker:
    build:
      context: .
      dockerfile: worker/Dockerfile
    container_name: trading-worker-dev
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      app:
        condition: service_healthy
    networks:
      - dev-network
    restart: unless-stopped

  # Redis (optional for caching)
  redis:
    image: redis:7-alpine
    container_name: redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data_dev:/data
    networks:
      - dev-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 3s
      retries: 3

  # Frontend (Development)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    container_name: frontend-dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    networks:
      - dev-network
    restart: unless-stopped
    environment:
      - NODE_ENV=development
      - VITE_API_URL=http://localhost:8000

  # Monitoring (Development)
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-dev
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.dev.yml:/etc/prometheus/prometheus.yml
      - prometheus_data_dev:/prometheus
    networks:
      - dev-network
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-dev
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data_dev:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - dev-network
    restart: unless-stopped

volumes:
  redis_data_dev:
  prometheus_data_dev:
  grafana_data_dev:

networks:
  dev-network:
    driver: bridge
```

### Production Environment

```yaml
# docker-compose.prod.yml - Production setup
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: trading-postgres-prod
    environment:
      - POSTGRES_DB=trading_orchestrator
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data_prod:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_orchestrator"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    secrets:
      - db_password

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: trading-redis-prod
    command: redis-server --requirepass $(cat /run/secrets/redis_password) --appendonly yes
    volumes:
      - redis_data_prod:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "$(cat /run/secrets/redis_password)", "ping"]
      interval: 30s
      timeout: 3s
      retries: 3
    secrets:
      - redis_password

  # Application Service
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: application
    container_name: trading-app-prod
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://trading_user:$(cat /run/secrets/db_password)@postgres:5432/trading_orchestrator
      - REDIS_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ssl_certs_prod:/etc/ssl/certs
    ports:
      - "8000:8000"
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    secrets:
      - db_password
      - redis_password
      - jwt_secret
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # Background Worker
  worker:
    build:
      context: .
      dockerfile: worker/Dockerfile
    container_name: trading-worker-prod
    command: python worker.py
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://trading_user:$(cat /run/secrets/db_password)@postgres:5432/trading_orchestrator
      - REDIS_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://app:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      app:
        condition: service_healthy
    secrets:
      - db_password
      - redis_password
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  # Frontend (Nginx)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: runtime
    container_name: trading-frontend-prod
    volumes:
      - ./nginx/prod.conf:/etc/nginx/nginx.conf:ro
      - ssl_certs_prod:/etc/ssl/certs:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost"]
      interval: 30s
      timeout: 3s
      retries: 3
    depends_on:
      - app

  # Load Balancer (HAProxy)
  load-balancer:
    image: haproxy:2.8-alpine
    container_name: trading-lb-prod
    volumes:
      - ./haproxy/prod.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - ssl_certs_prod:/etc/ssl/certs:ro
    ports:
      - "8080:80"
      - "8443:443"
    networks:
      - prod-network
    restart: unless-stopped
    depends_on:
      - app
      - frontend
    deploy:
      replicas: 1

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus-prod
    volumes:
      - ./monitoring/prometheus.prod.yml:/etc/prometheus/prometheus.yml
      - prometheus_data_prod:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - prod-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana-prod
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data_prod:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - prod-network
    restart: unless-stopped
    secrets:
      - grafana_password
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  # Log Aggregation
  fluentd:
    image: fluent/fluentd:v1.16-debian-1
    container_name: fluentd-prod
    volumes:
      - ./monitoring/fluentd/conf:/fluentd/etc
      - ./logs:/var/log/trading-orchestrator
    networks:
      - prod-network
    restart: unless-stopped
    depends_on:
      - app
      - worker

volumes:
  postgres_data_prod:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/trading-orchestrator/data/postgres

  redis_data_prod:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/trading-orchestrator/data/redis

  prometheus_data_prod:
  grafana_data_prod:
  ssl_certs_prod:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/trading-orchestrator/ssl

networks:
  prod-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: trading-prod
    ipam:
      config:
        - subnet: 172.20.0.0/16

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
```

## ⚙️ Environment-Specific Configurations

### Development Configuration

```yaml
# docker-compose.dev.yml - Additional development services
version: '3.8'

services:
  # Database seed data
  database-seed:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading-seed-dev
    command: python -c "from trading_orchestrator.database import seed_database; seed_database()"
    environment:
      - DATABASE_URL=sqlite:///data/trading_orchestrator.db
    volumes:
      - ./data:/app/data
    depends_on:
      - app
    networks:
      - dev-network

  # Mailhog for email testing
  mailhog:
    image: mailhog/mailhog:latest
    container_name: mailhog-dev
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    networks:
      - dev-network

  # LocalStack for AWS services testing
  localstack:
    image: localstack/localstack:latest
    container_name: localstack-dev
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,sns,sqs
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - "./localstack:/tmp/localstack"
    networks:
      - dev-network

  # PostgreSQL for development
  postgres-dev:
    image: postgres:15-alpine
    container_name: postgres-dev
    environment:
      - POSTGRES_DB=trading_orchestrator_dev
      - POSTGRES_USER=dev_user
      - POSTGRES_PASSWORD=dev_password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init-db.dev.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5433:5432"
    networks:
      - dev-network

volumes:
  postgres_dev_data:
```

### Staging Configuration

```yaml
# docker-compose.staging.yml - Staging environment
version: '3.8'

services:
  app-staging:
    build:
      context: .
      dockerfile: Dockerfile
      target: application
    container_name: trading-app-staging
    environment:
      - ENVIRONMENT=staging
      - DEBUG=false
      - DATABASE_URL=postgresql://trading_user:$(cat /run/secrets/db_password)@postgres-staging:5432/trading_orchestrator_staging
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1.5G
          cpus: '0.75'
    secrets:
      - db_password_staging

  postgres-staging:
    image: postgres:15-alpine
    container_name: postgres-staging
    environment:
      - POSTGRES_DB=trading_orchestrator_staging
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password_staging
    volumes:
      - postgres_staging_data:/var/lib/postgresql/data
    networks:
      - staging-network
    secrets:
      - db_password_staging

volumes:
  postgres_staging_data:

networks:
  staging-network:
    driver: bridge

secrets:
  db_password_staging:
    file: ./secrets/staging_db_password.txt
```

## 🔒 Security Best Practices

### Dockerfile Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim as base

# Create non-root user immediately
RUN groupadd -r --gid=1000 appuser && \
    useradd -r --uid=1000 --gid=appuser --home-dir=/app --shell=/sbin/nologin --comment="Trading App User" appuser

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        && \
    apt-get purge -y --auto-remove \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (better caching)
COPY trading_orchestrator/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=appuser:appuser . .

# Set secure permissions
RUN chmod -R 750 /app && \
    chmod -R 640 /app/config/* /app/.env

# Security: Remove unnecessary packages
RUN apt-get autoremove -y && \
    apt-get autoclean

# Security: Don't run as root
USER appuser

# Security: Read-only filesystem
# COPY --chown=appuser:appuser data /tmp/data
# VOLUME ["/tmp/data"]

# Security: No shell access
# SHELL ["/bin/false"]

# Health check with security
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

EXPOSE 8000

CMD ["python", "main.py"]
```

### Docker Compose Security

```yaml
# Security-enhanced docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    container_name: trading-app
    # Security: Drop all capabilities
    cap_drop:
      - ALL
    cap_add:
      - SETUID
      - SETGID
    
    # Security: Read-only root filesystem
    read_only: true
    
    # Security: User namespace
    user: "1000:1000"
    
    # Security: Security options
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    
    # Security: Limit resources
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    
    # Security: Temporary filesystem for writable areas
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/tmp:noexec,nosuid,size=50m
    
    # Security: Environment variables
    environment:
      - ENVIRONMENT=production
      - PYTHONUNBUFFERED=1
    
    # Security: No host networking
    networks:
      - app-network
    
    # Security: Don't auto-restart in infinite loop
    restart: "no"

networks:
  app-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: trading-app
    ipam:
      config:
        - subnet: 172.21.0.0/16
          gateway: 172.21.0.1
```

### Container Security Scanning

```bash
#!/bin/bash
# scripts/security-scan.sh

echo "🔍 Running security scans..."

# Trivy vulnerability scanner
echo "📊 Running Trivy scan..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/tmp aquasec/trivy image trading-orchestrator:latest

# Docker Scout
echo "🔍 Running Docker Scout..."
docker scout cves trading-orchestrator:latest

# Dockle (CIS Docker Benchmark)
echo "🛡️ Running Dockle security audit..."
docker run --rm -i hadolint/dockle < Dockerfile

# Hadolint (Dockerfile linting)
echo "📝 Linting Dockerfile..."
docker run --rm -i hadolint/hadolint < Dockerfile

echo "✅ Security scanning completed!"
```

## ⚡ Performance Optimization

### Multi-Stage Build Optimization

```dockerfile
# Multi-stage build with optimization
# syntax=docker/dockerfile:1.4

# Base image with common dependencies
FROM python:3.11-slim as base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Virtual environment
FROM base as venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Dependencies with caching
FROM venv as dependencies
WORKDIR /app
COPY trading_orchestrator/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Application build
FROM dependencies as builder
# Copy and compile any native extensions
COPY . .
RUN python setup.py build_ext --inplace || true

# Runtime image
FROM base as runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
        && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from dependencies
COPY --from=dependencies --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "main.py"]
```

### Docker Compose Performance

```yaml
# Performance-optimized docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    deploy:
      replicas: 2
      placement:
        constraints: [node.role == worker]
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        max_failure_ratio: 0.1
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    
    # Performance: Use overlay network
    networks:
      - app-overlay
    
    # Performance: Resource limits
    mem_limit: 2g
    mem_reservation: 1g
    cpus: 1.0
    cpu_shares: 512
    
    # Performance: Logging
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  worker:
    build: .
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

volumes:
  app_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/trading-orchestrator/data

networks:
  app-overlay:
    driver: overlay
    attachable: true
    ipam:
      config:
        - subnet: 172.22.0.0/16
```

### Container Performance Tuning

```bash
#!/bin/bash
# scripts/optimize-container.sh

echo "⚡ Optimizing container performance..."

# Set container limits
docker run --memory=2g \
           --cpus=1.0 \
           --memory-reservation=1g \
           --memory-swap=2g \
           --cpu-shares=512 \
           trading-orchestrator:latest

# Enable caching
docker run --mount type=cache,target=/root/.cache/pip \
           trading-orchestrator:latest

# Use buildkit for faster builds
export DOCKER_BUILDKIT=1
docker build --cache-from trading-orchestrator:latest \
             --target application \
             -t trading-orchestrator:latest .

# Multi-stage build with buildkit
docker build --target dependencies \
             --cache-from trading-orchestrator:dependencies \
             -t trading-orchestrator:dependencies \
             .

echo "✅ Container optimization completed!"
```

## 🌩️ Production Deployment

### Production Scripts

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

ENVIRONMENT=${1:-production}
TAG=${2:-latest}

echo "🚀 Deploying to production environment..."

# Build optimized images
echo "📦 Building production images..."
DOCKER_BUILDKIT=1 docker build \
    --target application \
    --build-arg ENVIRONMENT=production \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from trading-orchestrator:latest \
    --tag trading-orchestrator:${TAG} \
    .

# Build worker image
DOCKER_BUILDKIT=1 docker build \
    --target runtime \
    --build-arg ENVIRONMENT=production \
    --tag trading-orchestrator-worker:${TAG} \
    -f worker/Dockerfile .

# Security scan
echo "🔍 Running security scans..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image trading-orchestrator:${TAG}

# Push to registry (if using)
if [ -n "$REGISTRY_URL" ]; then
    echo "📤 Pushing to registry..."
    docker tag trading-orchestrator:${TAG} ${REGISTRY_URL}/trading-orchestrator:${TAG}
    docker push ${REGISTRY_URL}/trading-orchestrator:${TAG}
fi

# Deploy with zero downtime
echo "🔄 Deploying with zero downtime..."
docker-compose -f docker-compose.prod.yml pull app worker
docker-compose -f docker-compose.prod.yml up -d --remove-orphans

# Health check
echo "✅ Running health checks..."
sleep 30
docker-compose -f docker-compose.prod.yml exec app curl -f http://localhost:8000/health

echo "🎉 Production deployment completed!"
```

### Blue-Green Deployment

```yaml
# docker-compose.blue-green.yml
version: '3.8'

services:
  app-blue:
    build: .
    environment:
      - ENVIRONMENT=production
      - APP_VERSION=blue
    ports:
      - "8001:8000"
    networks:
      - bg-network
    deploy:
      replicas: 2
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.trading-app.rule=Host(`trading.yourdomain.com`)"
      - "traefik.http.routers.trading-app.entrypoints=websecure"

  app-green:
    build: .
    environment:
      - ENVIRONMENT=production
      - APP_VERSION=green
    ports:
      - "8002:8000"
    networks:
      - bg-network
    deploy:
      replicas: 2
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.trading-app-green.rule=Host(`trading.yourdomain.com`)"
      - "traefik.http.routers.trading-app-green.entrypoints=websecure"

networks:
  bg-network:
    driver: bridge

# Traffic splitting with Traefik
# Use Traefik to split traffic between blue and green deployments
```

### Health Check and Monitoring

```bash
#!/bin/bash
# scripts/monitor-deployment.sh

echo "📊 Monitoring deployment..."

# Check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null; then
            echo "✅ $service is healthy"
            return 0
        fi
        echo "⏳ $service health check failed (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service health check failed after $max_attempts attempts"
    return 1
}

# Monitor services
check_service "Trading Orchestrator" "http://localhost:8000/health"
check_service "Frontend" "http://localhost:3000"
check_service "Database" "postgresql://localhost:5432"

# Check resource usage
echo "💻 Resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check logs for errors
echo "📋 Recent errors:"
docker-compose logs --tail=100 app | grep -i error || echo "No errors found"

echo "✅ Monitoring completed!"
```

## ☁️ Cloud Deployment

### AWS ECS Deployment

```json
{
  "family": "trading-orchestrator",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "trading-orchestrator",
      "image": "your-account.dkr.ecr.region.amazonaws.com/trading-orchestrator:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:ssm:region:account:parameter/trading-orchestrator/database-url"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:ssm:region:account:parameter/trading-orchestrator/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-orchestrator",
          "awslogs-region": "us-east-1",
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

### Google Cloud Run Deployment

```yaml
# cloudrun-deploy.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: trading-orchestrator
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/cpu-throttling: "false"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/your-project/trading-orchestrator:latest
        ports:
        - name: http1
          containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: openai-key
        resources:
          limits:
            cpu: 2
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          timeoutSeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          timeoutSeconds: 10
          periodSeconds: 15
```

### Azure Container Instances

```yaml
# azure-deploy.yaml
apiVersion: 2021-03-01
location: eastus
name: trading-orchestrator
properties:
  containers:
  - name: trading-orchestrator
    properties:
      image: your-registry.azurecr.io/trading-orchestrator:latest
      resources:
        requests:
          cpu: 2.0
          memoryInGb: 4.0
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: "postgresql://..."
      volumeMounts:
      - name: config-volume
        mountPath: /app/config
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - port: 8000
      protocol: tcp
  volumes:
  - name: config-volume
    azureFile:
      share: trading-config
      storageAccountName: yourstorageaccount
      storageAccountKey: yourkey
tags:
  project: trading-orchestrator
  environment: production
```

## 🔧 Configuration Management

### Environment Variables Management

```bash
#!/bin/bash
# scripts/manage-env.sh

ENV=${1:-development}
ACTION=${2:-apply}

case $ACTION in
  apply)
    echo "🔧 Applying environment: $ENV"
    docker-compose -f docker-compose.$ENV.yml up -d
    ;;
  update)
    echo "🔄 Updating environment: $ENV"
    docker-compose -f docker-compose.$ENV.yml pull
    docker-compose -f docker-compose.$ENV.yml up -d --force-recreate
    ;;
  backup)
    echo "💾 Backing up environment: $ENV"
    docker-compose -f docker-compose.$ENV.yml exec app python scripts/backup.py
    ;;
  restore)
    echo "📥 Restoring environment: $ENV"
    docker-compose -f docker-compose.$ENV.yml exec app python scripts/restore.py
    ;;
  *)
    echo "Usage: $0 [environment] [action]"
    echo "Environments: development, staging, production"
    echo "Actions: apply, update, backup, restore"
    exit 1
    ;;
esac
```

### Secrets Management

```bash
#!/bin/bash
# scripts/generate-secrets.sh

echo "🔐 Generating secure secrets..."

# Generate database password
openssl rand -base64 32 > secrets/db_password.txt
chmod 600 secrets/db_password.txt

# Generate Redis password
openssl rand -base64 32 > secrets/redis_password.txt
chmod 600 secrets/redis_password.txt

# Generate JWT secret
openssl rand -base64 64 > secrets/jwt_secret.txt
chmod 600 secrets/jwt_secret.txt

# Generate API keys
python -c "
import secrets
print('OpenAI Key:', secrets.token_urlsafe(32))
print('Anthropic Key:', secrets.token_urlsafe(32))
print('App Secret:', secrets.token_urlsafe(64))
" > secrets/api_keys.txt

echo "✅ Secrets generated successfully!"
echo "📁 Location: ./secrets/"
```

## 📊 Monitoring and Logging

### Comprehensive Monitoring

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3001:3000"
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    networks:
      - monitoring

  caddy:
    image: caddy:latest
    container_name: caddy
    ports:
      - "8080:80"
      - "8443:443"
    volumes:
      - ./monitoring/caddy/Caddyfile:/etc/caddy/Caddyfile
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

## ❓ Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check container logs
docker-compose logs app

# Debug container
docker-compose run --rm app /bin/bash

# Check permissions
docker-compose exec app ls -la /app

# Check environment variables
docker-compose exec app env | grep -E "(DATABASE|API)"
```

#### Database Connection Issues

```bash
# Test database connection
docker-compose exec app python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://trading_user:password@postgres:5432/trading_orchestrator')
connection = engine.connect()
print('Database connection successful')
connection.close()
"

# Check database logs
docker-compose logs postgres

# Test network connectivity
docker-compose exec app nc -zv postgres 5432
```

#### Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Increase memory limits
docker-compose -f docker-compose.prod.yml up -d --scale app=1

# Use swap
docker run --memory-swap=4g your-image
```

#### Performance Issues

```bash
# Monitor performance
docker stats

# Check container health
docker ps

# Profile application
docker-compose exec app python -m cProfile -o profile.prof main.py
```

### Debugging Tools

```bash
# Useful debugging commands
docker-compose exec app bash
docker-compose exec worker python -c "import trading_orchestrator; print('OK')"
docker inspect trading-orchestrator-container
docker network ls
docker volume ls
```

---

**Need Help?**

- 📖 [Documentation](docs/)
- 🐛 [Troubleshooting Guide](docs/troubleshooting.md)
- 💬 [Docker Support](https://docs.docker.com/)

<div align="center">

**Docker Deployment Made Easy! 🐳**

Made with ❤️ by the Trading Orchestrator Team

</div>
