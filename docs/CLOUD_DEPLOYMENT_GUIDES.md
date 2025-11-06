# Cloud Deployment Guides
## Day Trading Orchestrator - Multi-Cloud Deployment

<div align="center">

![Cloud Deployment](https://img.shields.io/badge/Cloud%20Deployment-v1.0.0-blue?style=for-the-badge&logo=cloud)
[![AWS](https://img.shields.io/badge/AWS-Ready-orange.svg)](https://aws.amazon.com/)
[![GCP](https://img.shields.io/badge/GCP-Ready-blue.svg)](https://cloud.google.com/)
[![Azure](https://img.shields.io/badge/Azure-Ready-blue.svg)](https://azure.microsoft.com/)
[![DigitalOcean](https://img.shields.io/badge/DigitalOcean-Ready-blue.svg)](https://www.digitalocean.com/)

**Enterprise-Grade Cloud Deployments**

[☁️ AWS](#-aws-deployment) • [🔵 GCP](#-gcp-deployment) • [🟣 Azure](#-azure-deployment) • [🌊 DigitalOcean](#-digitalocean-deployment)

</div>

## 📋 Table of Contents

1. [Deployment Overview](#-deployment-overview)
2. [AWS Deployment](#-aws-deployment)
3. [GCP Deployment](#-gcp-deployment)
4. [Azure Deployment](#-azure-deployment)
5. [DigitalOcean Deployment](#-digitalocean-deployment)
6. [Comparison Matrix](#-comparison-matrix)
7. [Cost Optimization](#-cost-optimization)
8. [Security Best Practices](#-security-best-practices)
9. [Monitoring Setup](#-monitoring-setup)
10. [Disaster Recovery](#-disaster-recovery)

## 🚀 Deployment Overview

### Architecture Patterns

#### Single Cloud Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Provider                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Load       │  │  Application│  │  Database   │         │
│  │  Balancer   │  │  Servers    │  │  Cluster    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

#### Multi-Region Deployment
```
┌─────────────────┐           ┌─────────────────┐
│   Region A      │           │   Region B      │
├─────────────────┤           ├─────────────────┤
│  ┌───────────┐  │           │  ┌───────────┐  │
│  │Load       │  │           │  │Load       │  │
│  │Balancer A │  │◄──────────┤  │Balancer B │  │
│  └───────────┘  │  Global   │  └───────────┘  │
│  ┌───────────┐  │  Route 53 │  ┌───────────┐  │
│  │App        │  │           │  │App        │  │
│  │Cluster A  │  │           │  │Cluster B  │  │
│  └───────────┘  │           │  └───────────┘  │
│  ┌───────────┐  │           │  ┌───────────┐  │
│  │Database A │  │           │  │Database B │  │
│  │(Primary)  │  │           │  │(Replica)  │  │
│  └───────────┘  │           │  └───────────┘  │
└─────────────────┘           └─────────────────┘
```

#### Hybrid Cloud Deployment
```
┌─────────────────┐           ┌─────────────────┐
│    On-Premise   │           │   Cloud         │
├─────────────────┤           ├─────────────────┤
│  ┌───────────┐  │  VPN/     │  ┌───────────┐  │
│  │Legacy     │  │  Direct   │  │Cloud      │  │
│  │Systems    │  │  Connect  │  │Services   │  │
│  └───────────┘  │◄──────────┤  └───────────┘  │
│  ┌───────────┐  │           │  ┌───────────┐  │
│  │Database   │  │           │  │App        │  │
│  │Server     │  │           │  │Cluster    │  │
│  └───────────┘  │           │  └───────────┘  │
└─────────────────┘           └─────────────────┘
```

## ☁️ AWS Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          AWS VPC                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Public    │  │  Private    │  │  Database   │         │
│  │  Subnet     │  │  Subnet     │  │  Subnet     │         │
│  │  (AZ-a)     │  │  (AZ-a)     │  │  (AZ-a)     │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │Load         │  │ECS          │  │RDS          │         │
│  │Balancer     │  │Tasks        │  │PostgreSQL   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Public    │  │  Private    │  │  Database   │         │
│  │  Subnet     │  │  Subnet     │  │  Subnet     │         │
│  │  (AZ-b)     │  │  (AZ-b)     │  │  (AZ-b)     │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │Bastion      │  │ECS          │  │RDS          │         │
│  │Host         │  │Tasks        │  │Read Replica │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### AWS Services Used

| Service | Purpose | Alternatives |
|---------|---------|--------------|
| **ECS/EKS** | Container orchestration | EKS, App Runner, Fargate |
| **RDS** | Managed PostgreSQL | Aurora, DocumentDB, DynamoDB |
| **ElastiCache** | Redis caching | MemoryDB, ElastiCache Serverless |
| **S3** | Object storage | EFS, FSx, EBS |
| **ALB** | Load balancing | NLB, Gateway LB |
| **Route 53** | DNS management | CloudFlare, Azure DNS |
| **CloudWatch** | Monitoring | DataDog, New Relic |
| **IAM** | Security | AWS STS, Cognito |
| **VPC** | Network isolation | Direct Connect, VPN |
| **Secrets Manager** | Secret storage | Systems Manager, Parameter Store |

### Terraform Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket = "trading-orchestrator-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = var.aws_region
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "trading-orchestrator"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-${var.environment}"
  cidr = var.vpc_cidr
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  database_subnets = var.database_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  enable_s3_endpoint = true
  enable_dynamodb_endpoint = true
  
  # NAT Gateway settings
  single_nat_gateway = var.environment == "development"
  
  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role = true
  create_flow_log_cloudwatch_log_group = true
  
  tags = {
    Type = "VPC"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = "1.28"
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true
  
  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  # EKS Add-ons
  cluster_addons = {
    vpc-cni = {
      most_recent = true
    }
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  node_groups = {
    main = {
      desired_capacity = var.eks_desired_capacity
      max_capacity     = var.eks_max_capacity
      min_capacity     = var.eks_min_capacity
      
      instance_types = [var.eks_instance_type]
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup  = "main"
      }
      
      taints = var.eks_node_taints
      
      update_config = {
        max_unavailable_percentage = 25
      }
      
      tags = {
        Environment = var.environment
        NodeGroup  = "main"
      }
    }
    
    spot = {
      desired_capacity = var.eks_spot_capacity
      max_capacity     = var.eks_spot_max_capacity
      min_capacity     = var.eks_spot_min_capacity
      
      instance_types = [var.eks_spot_instance_type]
      
      capacity_type  = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup  = "spot"
      }
      
      tags = {
        Environment = var.environment
        NodeGroup  = "spot"
        CapacityType = "SPOT"
      }
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type         = "gp3"
  storage_encrypted    = true
  
  # Storage performance
  iops                  = var.db_iops
  performance_insights_enabled = var.db_performance_insights
  monitoring_interval   = 60
  monitoring_role_arn   = aws_iam_role.rds_monitoring.arn
  
  # Backup settings
  backup_retention_period = var.db_backup_retention
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Database settings
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  # Final snapshot for production
  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"
  final_snapshot_identifier = var.environment == "production" ? 
    "${var.project_name}-${var.environment}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null
  
  tags = {
    Environment = var.environment
    Component   = "database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-${var.environment}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Component   = "redis"
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "${var.project_name}-${var.environment}-redis"
  description                  = "Redis cluster for ${var.project_name}"
  
  node_type                    = var.redis_node_type
  port                         = 6379
  parameter_group_name         = "default.redis7.x"
  
  num_cache_clusters           = var.redis_num_cache_clusters
  automatic_failover_enabled   = var.environment != "development"
  multi_az_enabled            = var.environment != "development"
  auto_minor_version_upgrade  = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = var.redis_auth_token
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format      = "text"
    log_type        = "slow-log"
  }
  
  tags = {
    Environment = var.environment
    Component   = "redis"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "app_storage" {
  bucket = "${var.project_name}-${var.environment}-storage-${random_string.suffix.result}"
}

resource "aws_s3_bucket_versioning" "app_storage" {
  bucket = aws_s3_bucket.app_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_storage" {
  bucket = aws_s3_bucket.app_storage.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = var.environment == "production"
  
  tags = {
    Environment = var.environment
    Component   = "load-balancer"
  }
}

# Security Groups
resource "aws_security_group" "eks" {
  name_prefix = "${var.project_name}-${var.environment}-eks-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Component   = "eks"
  }
}

# CloudWatch
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${module.eks.cluster_name}/application"
  retention_in_days = var.cloudwatch_retention_days
  
  tags = {
    Environment = var.environment
    Component   = "application"
  }
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/${var.project_name}/redis"
  retention_in_days = var.cloudwatch_retention_days
  
  tags = {
    Environment = var.environment
    Component   = "redis"
  }
}

# Secrets Manager
resource "aws_secretsmanager_secret" "database" {
  name                    = "${var.project_name}/${var.environment}/database"
  description             = "Database credentials for ${var.project_name}"
  recovery_window_in_days = 7
  kms_key_id             = aws_kms_key.secrets.arn
  
  tags = {
    Environment = var.environment
    Component   = "secrets"
  }
}

resource "aws_secretsmanager_secret_version" "database" {
  secret_id = aws_secretsmanager_secret.database.id
  secret_string = jsonencode({
    username = var.db_username
    password = var.db_password
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
  })
}
```

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-orchestrator
  labels:
    name: trading-orchestrator
    environment: production
  annotations:
    description: "Trading Orchestrator Application Namespace"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-orchestrator-secrets
  namespace: trading-orchestrator
  labels:
    app: trading-orchestrator
type: Opaque
data:
  # Base64 encoded values (use proper base64 encoding for production)
  database-url: cG9zdGdyZXNxbDovL3RyYWRpbmdfdXNlcjpwYXNzd29yZEByeWRzLmFtYXpvbmF3cy5jb206NTQzMi90cmFkaW5nX29yY2hlc3RyYXRvcg==
  redis-password: c2VjdXJlX3JlZGlzX3Bhc3N3b3Jk
  jwt-secret: eW91cl9qd3Rfc2VjcmV0X2tleV9oZXJl
  openai-key: c2steW91cl9vcGVuYWlfYXBpX2tleQ==
  alpaca-key: eW91cl9hbHBhY2FfYXBpX2tleQ==
  alpaca-secret: eW91cl9hbHBhY2FfZ3NlY3JldF9rZXk=

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-orchestrator-config
  namespace: trading-orchestrator
  labels:
    app: trading-orchestrator
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  AWS_REGION: "us-east-1"
  DATABASE_TYPE: "postgresql"
  REDIS_ENABLED: "true"
  MONITORING_ENABLED: "true"
  health-check-interval: "30"
  request-timeout: "30"
  max-workers: "10"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
  labels:
    app: trading-orchestrator
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: trading-orchestrator
  template:
    metadata:
      labels:
        app: trading-orchestrator
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: trading-orchestrator
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: app
        image: public.ecr.aws/your-account/trading-orchestrator:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 8001
          protocol: TCP
        
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: database-url
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: redis-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: jwt-secret
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: trading-orchestrator-secrets
              key: openai-key
        
        envFrom:
        - configMapRef:
            name: trading-orchestrator-config
        
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
            port: http
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          successThreshold: 1
        
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
        
        startupProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
          successThreshold: 1
        
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        - name: tmp-volume
          mountPath: /tmp
          readOnly: false
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      
      volumes:
      - name: config-volume
        configMap:
          name: trading-orchestrator-config
      - name: logs-volume
        emptyDir: {}
      - name: tmp-volume
        emptyDir: {}
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - trading-orchestrator
              topologyKey: kubernetes.io/hostname
      
      tolerations:
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
      - key: "node.kubernetes.io/unreachable"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
  labels:
    app: trading-orchestrator
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  - port: 443
    targetPort: http
    protocol: TCP
    name: https
  selector:
    app: trading-orchestrator

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-orchestrator-hpa
  namespace: trading-orchestrator
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-orchestrator
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
  labels:
    app: trading-orchestrator
  annotations:
    kubernetes.io/ingress.class: "alb"
    alb.ingress.kubernetes.io/scheme: "internet-facing"
    alb.ingress.kubernetes.io/target-type: "ip"
    alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"
    alb.ingress.kubernetes.io/ssl-policy: "ELBSecurityPolicy-TLS-1-2-2017-01"
    alb.ingress.kubernetes.io/backend-protocol: "HTTP"
    alb.ingress.kubernetes.io/healthcheck-path: "/health"
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "30"
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: "5"
    alb.ingress.kubernetes.io/healthy-threshold-count: "2"
    alb.ingress.kubernetes.io/unhealthy-threshold-count: "3"
spec:
  tls:
  - hosts:
    - api.trading-orchestrator.com
    secretName: trading-orchestrator-tls
  rules:
  - host: api.trading-orchestrator.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trading-orchestrator
            port:
              number: 80
```

### AWS CDK Deployment

```typescript
// cdk/bin/trading-orchestrator.ts
#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { TradingOrchestratorStack } from '../lib/trading-orchestrator-stack';

const app = new cdk.App();

new TradingOrchestratorStack(app, 'TradingOrchestratorStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  tags: {
    Environment: 'production',
    Project: 'trading-orchestrator',
    Owner: 'devops-team',
  },
});
```

```typescript
// cdk/lib/trading-orchestrator-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

export class TradingOrchestratorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC
    const vpc = new ec2.Vpc(this, 'TradingOrchestratorVPC', {
      maxAzs: 3,
      natGateways: 2,
      vpcName: 'trading-orchestrator-vpc',
    });

    // RDS Subnet Group
    const dbSubnetGroup = new rds.SubnetGroup(this, 'DatabaseSubnetGroup', {
      vpc,
      description: 'Subnet group for Trading Orchestrator database',
    });

    // Secrets
    const dbSecret = new secretsmanager.Secret(this, 'DatabaseSecret', {
      secretName: '/trading-orchestrator/database',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          username: 'trading_user',
        }),
        generateStringKey: 'password',
        excludeCharacters: '"@/\\',
      },
    });

    // RDS Database
    const database = new rds.DatabaseInstance(this, 'TradingOrchestratorDB', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.V15,
      }),
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.R6G,
        ec2.InstanceSize.XLARGE
      ),
      credentials: rds.Credentials.fromSecret(dbSecret),
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      },
      allocatedStorage: 100,
      maxAllocatedStorage: 1000,
      storageEncrypted: true,
      backupRetention: cdk.Duration.days(7),
      monitoringInterval: cdk.Duration.minutes(60),
      enablePerformanceInsights: true,
      deletionProtection: true,
    });

    // ECR Repository
    const ecrRepository = new ecr.Repository(this, 'TradingOrchestratorECR', {
      repositoryName: 'trading-orchestrator',
      imageScanOnPush: true,
      lifecycleRules: [
        {
          maxImageCount: 10,
          description: 'Keep only the last 10 images',
        },
      ],
    });

    // ECS Cluster
    const cluster = new ecs.Cluster(this, 'TradingOrchestratorCluster', {
      clusterName: 'trading-orchestrator-cluster',
      vpc,
      containerInsights: true,
    });

    // Task Definition
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TradingOrchestratorTask', {
      cpu: 1024,
      memory: 2048,
      runtimePlatform: {
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
      },
    });

    // Container
    const container = taskDefinition.addContainer('TradingOrchestratorContainer', {
      image: ecs.ContainerImage.fromEcrRepository(ecrRepository),
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'trading-orchestrator',
        logRetention: 30,
      }),
      environment: {
        ENVIRONMENT: 'production',
        DATABASE_URL: database.dbInstanceEndpointAddress,
      },
      secrets: {
        DATABASE_PASSWORD: ecs.Secret.fromSecretsManager(dbSecret, 'password'),
      },
      healthCheck: {
        command: ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
      },
    });

    container.addPortMappings({
      containerPort: 8000,
      protocol: ecs.Protocol.TCP,
    });

    // ECS Service
    const service = new ecs.FargateService(this, 'TradingOrchestratorService', {
      cluster,
      taskDefinition,
      desiredCount: 3,
      enableExecuteCommand: true,
      deploymentConfiguration: {
        maximumPercent: 200,
        minimumHealthyPercent: 100,
      },
      circuitBreaker: {
        rollback: true,
      },
    });

    // Application Load Balancer
    const alb = new elbv2.ApplicationLoadBalancer(this, 'TradingOrchestratorALB', {
      vpc,
      internetFacing: true,
      loadBalancerName: 'trading-orchestrator-alb',
    });

    const listener = alb.addListener('HTTPSListener', {
      port: 443,
      protocol: elbv2.ApplicationProtocol.HTTPS,
    });

    // Security Group for ALB
    alb.connections.allowFromAnyIpv4(ec2.Port.tcp(443));
    alb.connections.allowToAnyIpv4(ec2.Port.tcp(8000));

    // Target Group
    const targetGroup = listener.addTargets('ECSTargetGroup', {
      port: 8000,
      targets: [service],
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
      },
    });

    // Auto Scaling
    const scaling = service.autoScaleTaskCount({
      minCapacity: 3,
      maxCapacity: 10,
    });

    scaling.scaleOnCpuUtilization('CPUScaling', {
      targetUtilizationPercent: 70,
    });

    scaling.scaleOnMemoryUtilization('MemoryScaling', {
      targetUtilizationPercent: 80,
    });

    // Outputs
    new cdk.CfnOutput(this, 'LoadBalancerDNS', {
      value: alb.loadBalancerDnsName,
      description: 'Load Balancer DNS Name',
    });

    new cdk.CfnOutput(this, 'DatabaseEndpoint', {
      value: database.dbInstanceEndpointAddress,
      description: 'Database Endpoint',
    });

    new cdk.CfnOutput(this, 'ECRRepository', {
      value: ecrRepository.repositoryUri,
      description: 'ECR Repository URI',
    });
  }
}
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy-aws.sh

set -e

ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}

echo "🚀 Deploying Trading Orchestrator to AWS"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"

# Configure AWS CLI
aws configure set region $REGION

# Build and push Docker image
echo "📦 Building Docker image..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build -t trading-orchestrator:latest .
docker tag trading-orchestrator:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/trading-orchestrator:latest

echo "📤 Pushing to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/trading-orchestrator:latest

# Deploy infrastructure with Terraform
echo "🏗️ Deploying infrastructure..."
cd terraform

terraform init \
  -backend-config="bucket=trading-orchestrator-terraform-state" \
  -backend-config="region=$REGION"

terraform plan \
  -var="environment=$ENVIRONMENT" \
  -var="aws_region=$REGION" \
  -var="owner=$USER" \
  -out=tfplan

terraform apply tfplan

# Deploy to Kubernetes
echo "☸️ Deploying to EKS..."
aws eks update-kubeconfig --region $REGION --name trading-orchestrator-production

kubectl apply -f k8s/

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/trading-orchestrator -n trading-orchestrator

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n trading-orchestrator
kubectl get services -n trading-orchestrator

# Get service endpoint
LOAD_BALANCER_DNS=$(kubectl get service trading-orchestrator -n trading-orchestrator -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "🎉 Deployment completed!"
echo "Service URL: https://$LOAD_BALANCER_DNS"
echo "Health Check: https://$LOAD_BALANCER_DNS/health"
```

## 🔵 GCP Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GCP VPC Network                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Public    │  │  Private    │  │  Private    │         │
│  │  Subnet     │  │  Subnet     │  │  Subnet     │         │
│  │  (us-central1)│ │ (us-central1)│ │ (us-central1)│        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │Global       │  │GKE          │  │Cloud SQL    │         │
│  │Load         │  │Cluster      │  │PostgreSQL   │         │
│  │Balancer     │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Terraform Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Random string for uniqueness
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "sql-component.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy        = false
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-${var.environment}"
  auto_create_subnetworks = false
  mtu                     = 1460
  routing_mode           = "REGIONAL"

  labels = {
    environment = var.environment
    project     = var.project_name
  }
}

# Subnets
resource "google_compute_subnetwork" "private" {
  name          = "${var.project_name}-${var.environment}-private"
  ip_cidr_range = var.private_subnet_cidr
  region        = var.region
  network       = google_compute_network.main.id

  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = var.services_secondary_cidr
  }

  secondary_ip_range {
    range_name    = "pod-range"
    ip_cidr_range = var.pods_secondary_cidr
  }

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }

  private_ip_google_access = true
}

resource "google_compute_subnetwork" "public" {
  name          = "${var.project_name}-${var.environment}-public"
  ip_cidr_range = var.public_subnet_cidr
  region        = var.region
  network       = google_compute_network.main.id
}

# Cloud Router
resource "google_compute_router" "main" {
  name    = "${var.project_name}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.main.id

  bgp {
    asn = 64514
  }
}

# NAT Gateway
resource "google_compute_router_nat" "main" {
  name                               = "${var.project_name}-${var.environment}-nat"
  router                            = google_compute_router.main.name
  region                            = var.region
  nat_ip_allocate_option           = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = "${var.project_name}-${var.environment}"
  location = var.region

  # Use regional cluster for high availability
  location = var.region
  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.private.id

  # Enable network policy
  network_policy {
    enabled = true
  }

  # Addon configuration
  addons_config {
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    vertical_pod_autoscaling {
      enabled = true
      cpu_manager_policy = "steady-state"
      memory_manager_policy = "None"
    }
    dns_cache_config {
      enabled = true
    }
  }

  # Binary authorization
  binary_authorization {
    enabled = true
  }

  # Security
  enable_shielded_nodes = true

  # Network config
  ip_allocation_policy {
    cluster_secondary_range_name  = "pod-range"
    services_secondary_range_name = "services-range"
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "SYSTEM_EVENTS",
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "SYSTEM_EVENTS",
    ]
    managed_prometheus {
      enabled = true
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Node configuration
  default_node_pool {
    name       = "default-node-pool"
    location   = var.zone
    node_count = var.gke_initial_node_count

    node_config {
      preemptible  = true
      machine_type = var.gke_machine_type
      disk_size_gb = var.gke_disk_size
      disk_type    = "pd-ssd"

      oauth_scopes = [
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/devstorage.read_write",
        "https://www.googleapis.com/auth/cloud-platform",
      ]

      shielded_instance_config {
        enable_secure_boot          = true
        enable_integrity_monitoring = true
      }

      tags = ["trading-orchestrator", "gke"]
    }

    autoscaling {
      min_node_count = var.gke_min_nodes
      max_node_count = var.gke_max_nodes
    }

    max_pods_per_node = 110
    placement_policy {
      placement_type = "COMPACT"
    }
  }

  # Security
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_global_access_config {
      enabled = true
    }
    master_ipv4_cidr_block = "172.16.0.0/28"
  }

  # Cost optimization
  cost_management_config {
    enabled = true
  }

  labels = {
    environment = var.environment
    project     = var.project_name
  }

  resource_labels = {
    environment = var.environment
    project     = var.project_name
  }
}

# Cloud SQL Instance
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region
  deletion_protection = var.environment == "production"

  settings {
    tier = var.db_tier

    backup_configuration {
      enabled    = true
      start_time = "03:00"
      location   = var.region
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      require_ssl     = true
    }

    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }

    database_flags {
      name  = "log_statement"
      value = "all"
    }

    maintenance_window {
      day          = 7
      hour         = 3
    }

    user_labels = {
      environment = var.environment
      project     = var.project_name
    }

    disk_type = "PD_SSD"
    disk_size = var.db_disk_size
    disk_autoresize = true
    disk_autoresize_limit = var.db_disk_autoresize_limit

    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    connector_enforcement = "REQUIRED"
  }

  encryption_key_name = google_kms_crypto_key.sql_key.id
  deletion_protection = var.environment == "production"

  labels = {
    environment = var.environment
    project     = var.project_name
  }
}

# Cloud SQL Database
resource "google_sql_database" "main" {
  name     = "trading_orchestrator"
  instance = google_sql_database_instance.main.name
}

# Secret Manager
resource "google_secret_manager_secret" "database" {
  for_each = toset(["username", "password"])

  secret_id = "${var.project_name}-${var.environment}-database-${each.value}"

  replication {
    user_managed {
      replicas {
        location         = var.region
        customer_managed_encryption {
          kms_key_name = google_kms_crypto_key.secrets_key.id
        }
      }
    }
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_secret_manager_secret_version" "database" {
  for_each = toset({
    username = var.db_username
    password = var.db_password
  })

  secret      = google_secret_manager_secret.database[each.key].id
  secret_data = each.value
}

# Cloud Storage
resource "google_storage_bucket" "main" {
  name                        = "${var.project_name}-${var.environment}-${random_string.suffix.result}"
  location                    = var.region
  force_destroy              = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment = var.environment
    project     = var.project_name
  }
}

# Artifact Registry
resource "google_artifact_registry_repository" "main" {
  location = var.region
  repository_id = "${var.project_name}-${var.environment}"
  description = "Container images for Trading Orchestrator"
  format = "DOCKER"
}

# Load Balancer
resource "google_compute_global_address" "lb" {
  name = "${var.project_name}-${var.environment}-lb"
}

resource "google_compute_managed_ssl_certificate" "main" {
  name = "${var.project_name}-${var.environment}-cert"

  managed {
    domains = [var.domain]
  }
}

resource "google_compute_url_map" "main" {
  name            = "${var.project_name}-${var.environment}-url-map"
  default_service = google_compute_backend_service.main.id
}

resource "google_compute_target_https_proxy" "main" {
  name             = "${var.project_name}-${var.environment}-https-proxy"
  url_map          = google_compute_url_map.main.id
  ssl_certificates = [google_compute_managed_ssl_certificate.main.id]
}

resource "google_compute_global_forwarding_rule" "main" {
  name                  = "${var.project_name}-${var.environment}-forwarding-rule"
  target                = google_compute_target_https_proxy.main.id
  port_range           = "443"
  ip_address           = google_compute_global_address.lb.address
}

# Backend Service
resource "google_compute_backend_service" "main" {
  name                  = "${var.project_name}-${var.environment}-backend"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  timeout_sec           = 30
  health_checks         = [google_compute_health_check.main.id]
  
  session_affinity {
    affinity_type = "CLIENT_IP"
    ttl_sec      = 300
  }
}

# Health Check
resource "google_compute_health_check" "main" {
  name                = "${var.project_name}-${var.environment}-health-check"
  timeout_sec         = 5
  check_interval_sec  = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 80
    request_path = "/health"
  }
}
```

### Cloud Run Deployment

```yaml
# cloudrun-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: trading-orchestrator
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
    run.googleapis.com/cpu-throttling: "false"
    autoscaling.knative.dev/maxScale: "50"
    run.googleapis.com/memory: "2Gi"
    run.googleapis.com/cpu: "2"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "50"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/ingress: all
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      serviceAccountName: trading-orchestrator-sa
      containers:
      - image: gcr.io/PROJECT_ID/trading-orchestrator:latest
        ports:
        - name: http1
          containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: GOOGLE_CLOUD_PROJECT
          value: PROJECT_ID
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: openai-key
        resources:
          limits:
            cpu: 2
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 1Gi
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          timeoutSeconds: 30
          periodSeconds: 15
          failureThreshold: 10
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

## 🟣 Azure Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Virtual Network                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Public    │  │  Private    │  │  Database   │         │
│  │  Subnet     │  │  Subnet     │  │  Subnet     │         │
│  │  (East US)  │  │  (East US)  │  │  (East US)  │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │Application  │  │AKS          │  │PostgreSQL   │         │
│  │Gateway      │  │Cluster      │  │Flexible     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Terraform Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "terraformstatesa"
    container_name       = "terraform-state"
    key                  = "trading-orchestrator/terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
  skip_provider_registration = true
}

# Random string for uniqueness
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location

  tags = {
    environment = var.environment
    project     = var.project_name
    managed_by  = "terraform"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = [var.vnet_address_space]
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  dns_servers = var.dns_servers

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Subnets
resource "azurerm_subnet" "public" {
  name                 = "${var.project_name}-${var.environment}-public"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = var.public_subnet_address_prefixes
  service_endpoints    = ["Microsoft.KeyVault", "Microsoft.Storage"]
}

resource "azurerm_subnet" "private" {
  name                 = "${var.project_name}-${var.environment}-private"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = var.private_subnet_address_prefixes
  service_endpoints    = ["Microsoft.KeyVault", "Microsoft.Storage"]

  delegation {
    name = "aks-delegation"
    service_delegation {
      name = "Microsoft.ContainerService/managedClusters"
    }
  }
}

resource "azurerm_subnet" "database" {
  name                 = "${var.project_name}-${var.environment}-database"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = var.database_subnet_address_prefixes
  service_endpoints    = ["Microsoft.KeyVault", "Microsoft.Storage"]
}

# Network Security Groups
resource "azurerm_network_security_group" "public" {
  name                = "${var.project_name}-${var.environment}-public-nsg"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

resource "azurerm_network_security_group" "private" {
  name                = "${var.project_name}-${var.environment}-private-nsg"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowAKS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "VirtualNetwork"
    destination_address_prefix = "VirtualNetwork"
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Azure Kubernetes Service
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-${var.environment}-aks"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}${var.environment}"

  default_node_pool {
    name                = "default"
    node_count          = var.aks_node_count
    vm_size             = var.aks_vm_size
    os_disk_size_gb     = var.aks_os_disk_size_gb
    type                = "VirtualMachineScaleSets"
    availability_zones  = [1, 2, 3]
    enable_auto_scaling = true
    min_count           = var.aks_min_nodes
    max_count           = var.aks_max_nodes
    node_labels = {
      workload = "trading-orchestrator"
    }
    tags = {
      workload = "trading-orchestrator"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "azure"
    network_policy     = "azure"
    service_cidr       = var.aks_service_cidr
    dns_service_ip     = var.aks_dns_service_ip
    docker_bridge_cidr = var.aks_docker_bridge_cidr
  }

  addons_profile {
    oms_agent {
      enabled                    = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
    }
    azure_policy {
      enabled = true
    }
    key_vault_secrets_provider {
      enabled = true
    }
  }

  azure_active_directory_role_based_access_control {
    managed                      = true
    azure_rbac_enabled          = true
    admin_group_object_ids       = var.aad_admin_group_object_ids
    tenant_id                   = data.azurerm_client_config.current.tenant_id
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Azure Database for PostgreSQL
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.project_name}-${var.environment}-pg"
  location               = azurerm_resource_group.main.location
  resource_group_name    = azurerm_resource_group.main.name
  delegated_subnet_id    = azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  storage_tier           = "P30"
  storage_size_gb        = var.pg_storage_size_gb
  sku_name              = var.pg_sku_name
  backup_retention_days = var.pg_backup_retention_days
  geo_redundant_backup  = var.environment == "production" ? "Enabled" : "Disabled"

  admin_username = var.db_username
  admin_password = var.db_password

  lifecycle {
    ignore_changes = [password]
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "trading_orchestrator"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = "postgresql.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${var.project_name}-${var.environment}-postgres-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  resource_group_name   = azurerm_resource_group.main.name
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${var.project_name}${var.environment}acr"
  resource_group_name = azurerm_resource_group.main.location
  location           = azurerm_resource_group.main.location
  sku                = "Basic"
  admin_enabled      = false

  network_rule_set {
    default_action = "Deny"
    ip_rule {
      action   = "Allow"
      ip_range = var.acr_ip_allowlist
    }
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Azure Application Gateway
resource "azurerm_application_gateway" "main" {
  name                = "${var.project_name}-${var.environment}-appgw"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = var.appgw_capacity
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.public.id
  }

  frontend_port {
    name = "frontend-port"
    port = 80
  }

  frontend_port {
    name = "frontend-port-https"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.main.id
  }

  backend_address_pool {
    name  = "backend-pool"
    fqdns = [var.backend_fqdn]
  }

  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Enabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 30
    probe_name            = "backend-probe"
  }

  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "frontend-port"
    protocol                       = "Http"
  }

  probe {
    name                = "backend-probe"
    host                = "localhost"
    path                = "/health"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    protocol            = "Http"
  }

  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "backend-http-settings"
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Public IP for Application Gateway
resource "azurerm_public_ip" "main" {
  name                = "${var.project_name}-${var.environment}-pip"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                = "Standard"
}

# Azure Key Vault
resource "azurerm_key_vault" "main" {
  name                = "${var.project_name}-${var.environment}-kv"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id          = data.azurerm_client_config.current.tenant_id
  sku_name           = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled = var.environment == "production"

  network_acls {
    default_action             = "Deny"
    bypass                     = "AzureServices"
    ip_rules                   = var.key_vault_ip_allowlist
    virtual_network_subnet_ids = [azurerm_subnet.private.id]
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "database_url" {
  name         = "database-url"
  value        = "postgresql://${var.db_username}:${var.db_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/trading_orchestrator"
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.current]
}

resource "azurerm_key_vault_secret" "redis_url" {
  name         = "redis-url"
  value        = "redis://:${var.redis_password}@${var.redis_host}:6379/0"
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.current]
}
```

## 🌊 DigitalOcean Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 DigitalOcean VPC                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Droplet   │  │  Kubernetes │  │  Managed    │         │
│  │   (LB)      │  │   Cluster   │  │ PostgreSQL  │         │
│  │             │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Terraform Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Random string for uniqueness
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# VPC
resource "digitalocean_vpc" "main" {
  name     = "${var.project_name}-${var.environment}-vpc"
  region   = var.region
  ip_range = var.vpc_cidr
}

# Droplets
resource "digitalocean_droplet" "load_balancer" {
  count  = 1
  name   = "${var.project_name}-${var.environment}-lb-${count.index}"
  image  = "ubuntu-22-04-x64"
  size   = var.lb_droplet_size
  region = var.region
  vpc_uuid = digitalocean_vpc.main.id

  ssh_keys = var.ssh_fingerprints

  tags = ["load-balancer", var.project_name]

  provisioner "remote-exec" {
    inline = [
      "apt-get update",
      "apt-get install -y nginx",
      "systemctl enable nginx",
    ]
  }

  connection {
    type        = "ssh"
    host        = self.ipv4_address
    user        = "root"
    private_key = file(var.ssh_private_key_path)
  }
}

# Managed Database
resource "digitalocean_database_cluster" "main" {
  name       = "${var.project_name}-${var.environment}-db"
  engine     = "pg"
  version    = "15"
  size       = var.db_size
  region     = var.region
  num_nodes  = var.db_num_nodes
  tags       = [var.project_name]

  vpc_uuid = digitalocean_vpc.main.id
  maintenance_window {
    day  = "sun"
    hour = "02:00"
  }

  lifecycle {
    ignore_changes = [password]
  }
}

resource "digitalocean_database_db" "main" {
  cluster_id = digitalocean_database_cluster.main.id
  name       = "trading_orchestrator"
}

resource "digitalocean_database_user" "main" {
  cluster_id = digitalocean_database_cluster.main.id
  name       = var.db_username
  mysql_auth_plugin = "mysql_native_password"
}

# Kubernetes Cluster
resource "digitalocean_kubernetes_cluster" "main" {
  name     = "${var.project_name}-${var.environment}-cluster"
  region   = var.region
  version  = var.k8s_version
  vpc_uuid = digitalocean_vpc.main.id

  node_pool {
    name       = "default-node-pool"
    size       = var.k8s_node_pool_size
    auto_scale = true
    min_nodes  = var.k8s_min_nodes
    max_nodes  = var.k8s_max_nodes
    node_count = var.k8s_initial_nodes

    labels = {
      pool = "default"
    }

    tags = [var.project_name, "node-pool"]
  }

  maintenance_policy {
    start_time = "04:00"
    day        = "any"
  }

  auto_upgrade = true

  tags = [var.project_name, "production"]
}

# Container Registry
resource "digitalocean_container_registry" "main" {
  name = "${var.project_name}-${var.environment}-registry"
  subscription_tier_slug = "starter"
}

resource "digitalocean_container_registry_docker_credentials" "main" {
  registry_name = digitalocean_container_registry.main.name
  read_write    = true
}

# Spaces (S3-compatible storage)
resource "digitalocean_spaces_bucket" "main" {
  name   = "${var.project_name}-${var.environment}-storage"
  region = var.region
  acl    = "private"

  versioning {
    enabled = true
  }

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }

  lifecycle_rule {
    enabled = true
    noncurrent_version_expiration {
      days = 30
    }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# Firewall
resource "digitalocean_firewall" "kubernetes" {
  name = "${var.project_name}-${var.environment}-k8s-firewall"
  tags = [digitalocean_kubernetes_cluster.main.tags[0]]

  inbound_rule {
    protocol = "tcp"
    ports    = "22"
    sources {
      addresses = ["${var.admin_ip}/32"]
    }
  }

  inbound_rule {
    protocol = "tcp"
    ports    = "6443"
    sources {
      addresses = ["${var.admin_ip}/32"]
    }
  }

  outbound_rule {
    protocol         = "tcp"
    ports            = "all"
    destination_addresses = ["0.0.0.0/0"]
  }

  outbound_rule {
    protocol = "icmp"
    destination_addresses = ["0.0.0.0/0"]
  }
}

# Load Balancer
resource "digitalocean_loadbalancer" "public" {
  name   = "${var.project_name}-${var.environment}-lb"
  region = var.region
  size   = var.lb_size

  forwarding_rule {
    entry_port     = 80
    entry_protocol = "http"

    target_port     = 80
    target_protocol = "http"
    proxy_protocol  = "off"
  }

  forwarding_rule {
    entry_port     = 443
    entry_protocol = "https"

    target_port     = 80
    target_protocol = "http"
    proxy_protocol  = "off"
    tls_passthrough = true
  }

  health_check {
    protocol               = "http"
    port                   = 80
    path                   = "/health"
    check_interval_seconds = 10
    response_timeout_seconds = 5
    healthy_threshold      = 2
    unhealthy_threshold    = 3
  }

  sticky_sessions {
    type = "cookies"
  }

  redirect_http_to_https = true

  tags = [var.project_name]
}

# DNS Records
resource "digitalocean_domain" "main" {
  name = var.domain
}

resource "digitalocean_record" "root" {
  domain = digitalocean_domain.main.name
  type   = "A"
  name   = "@"
  value  = digitalocean_loadbalancer.public.ipv4_address
}

resource "digitalocean_record" "www" {
  domain = digitalocean_domain.main.name
  type   = "A"
  name   = "www"
  value  = digitalocean_loadbalancer.public.ipv4_address
}

resource "digitalocean_record" "api" {
  domain = digitalocean_domain.main.name
  type   = "CNAME"
  name   = "api"
  value  = digitalocean_loadbalancer.public.fqdn
}
```

### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator
  labels:
    app: trading-orchestrator
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-orchestrator
  template:
    metadata:
      labels:
        app: trading-orchestrator
        version: v1.0.0
    spec:
      containers:
      - name: app
        image: registry.digitalocean.com/trading-orchestrator-registry/trading-orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
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
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: trading-orchestrator-service
spec:
  selector:
    app: trading-orchestrator
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-orchestrator-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - trading-orchestrator.yourdomain.com
    secretName: trading-orchestrator-tls
  rules:
  - host: trading-orchestrator.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trading-orchestrator-service
            port:
              number: 80
```

## 📊 Comparison Matrix

| Feature | AWS | GCP | Azure | DigitalOcean |
|---------|-----|-----|-------|--------------|
| **Setup Time** | 45-60 min | 30-45 min | 40-55 min | 20-30 min |
| **Complexity** | High | Medium | High | Low |
| **Cost** | Variable | Variable | Variable | Predictable |
| **Scalability** | Excellent | Excellent | Good | Good |
| **Managed Services** | Extensive | Extensive | Extensive | Limited |
| **Free Tier** | 12 months | 12 months | 12 months | $0 |
| **Documentation** | Excellent | Excellent | Good | Limited |
| **Community** | Largest | Growing | Large | Medium |
| **Enterprise Features** | Advanced | Advanced | Advanced | Basic |
| **Multi-Cloud** | Best | Good | Good | N/A |

### Cost Comparison (Monthly Estimate)

```
Small Deployment (2 vCPU, 4GB RAM, 50GB Storage):
├── AWS (t3.medium + RDS db.t3.micro): ~$150/month
├── GCP (e2-medium + Cloud SQL db-f1-micro): ~$120/month
├── Azure (B2s + Azure Database): ~$130/month
└── DigitalOcean (Basic Droplet + Managed DB): ~$90/month

Medium Deployment (4 vCPU, 8GB RAM, 100GB Storage):
├── AWS (t3.large + RDS db.t3.small): ~$300/month
├── GCP (e2-standard-2 + Cloud SQL db-n1-standard-2): ~$280/month
├── Azure (B2s + Azure Database): ~$260/month
└── DigitalOcean (Regular Droplet + Managed DB): ~$200/month

Large Deployment (8 vCPU, 16GB RAM, 200GB Storage):
├── AWS (t3.xlarge + RDS db.r5.large): ~$600/month
├── GCP (e2-standard-4 + Cloud SQL db-n1-standard-4): ~$550/month
├── Azure (D2s v3 + Azure Database): ~$520/month
└── DigitalOcean (Professional Droplet + Managed DB): ~$400/month
```

## 💰 Cost Optimization

### AWS Cost Optimization

```hcl
# Spot Instances for non-critical workloads
resource "aws_launch_configuration" "spot" {
  name          = "${var.project_name}-spot-launch-config"
  image_id      = var.ami_id
  instance_type = var.spot_instance_type
  spot_price    = var.spot_price

  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y amazon-ssm-agent
              EOF

  security_groups = [aws_security_group.spot.id]
  iam_instance_profile = aws_iam_instance_profile.spot.name

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "spot" {
  name                = "${var.project_name}-spot-asg"
  launch_configuration = aws_launch_configuration.spot.name
  min_size            = var.spot_min_size
  max_size            = var.spot_max_size
  desired_capacity    = var.spot_desired_capacity
  vpc_zone_identifier = module.vpc.private_subnets

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-spot-instance"
    propagate_at_launch = true
  }
}

# Reserved Instances for predictable workloads
resource "aws_db_instance_reserved" "production" {
  offering_id         = data.aws_db_reserved_instance_offering.production.id
  db_instance_class   = var.db_instance_class
  reserved_db_instance_name = "${var.project_name}-production-ri"
  multi_az            = true
}
```

### GCP Cost Optimization

```hcl
# Preemptible nodes for non-critical workloads
resource "google_container_node_pool" "preemptible" {
  name       = "preemptible-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.id
  node_count = 2

  node_config {
    preemptible  = true
    machine_type = var.preemptible_machine_type
    disk_size_gb = var.node_disk_size
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 10
  }
}
```

### Azure Cost Optimization

```hcl
# Spot VMs for development workloads
resource "azurerm_linux_virtual_machine_scale_set" "spot" {
  name                = "${var.project_name}-spot-vmss"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku               = var.spot_vm_sku
  instances         = var.spot_instance_count
  admin_username    = var.admin_username
  admin_password    = var.admin_password

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  os_disk {
    storage_account_type = "Standard_LRS"
    caching             = "ReadWrite"
    disk_size_gb        = 30
  }

  spot_instance {
    priority        = "Spot"
    eviction_policy = "Delete"
    max_price       = -1
  }

  network_interface {
    name                    = "nic"
    primary                 = true
    network_security_group_id = azurerm_network_security_group.private.id

    ip_configuration {
      name      = "internal"
      subnet_id = azurerm_subnet.private.id
      primary   = true
    }
  }

  tags = {
    environment = var.environment
    project     = var.project_name
  }
}
```

## 🔒 Security Best Practices

### Common Security Measures

1. **Network Security**
   - Use private subnets for application workloads
   - Implement network segmentation
   - Use Security Groups/NSGs effectively
   - Enable VPC flow logs

2. **Data Security**
   - Encrypt data at rest and in transit
   - Use managed key services
   - Implement proper backup strategies
   - Regular security audits

3. **Access Control**
   - Use IAM roles over static credentials
   - Implement least privilege principle
   - Enable MFA for all accounts
   - Regular access reviews

4. **Monitoring and Compliance**
   - Enable comprehensive logging
   - Set up security monitoring
   - Implement SIEM solutions
   - Regular vulnerability assessments

### Security Checklist

```yaml
# security-checklist.md
## Network Security
- [ ] Private subnets for application layer
- [ ] Security groups/NSGs properly configured
- [ ] VPC flow logs enabled
- [ ] DDoS protection enabled
- [ ] WAF configured for web applications

## Data Security
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enabled
- [ ] Key management service configured
- [ ] Database access logging enabled
- [ ] Regular backup encryption

## Access Control
- [ ] IAM roles configured
- [ ] Multi-factor authentication enabled
- [ ] Least privilege principle applied
- [ ] Service accounts properly configured
- [ ] Regular access reviews scheduled

## Monitoring
- [ ] CloudTrail/Audit logging enabled
- [ ] Security event monitoring configured
- [ ] Alerting rules established
- [ ] Incident response plan documented
- [ ] Regular penetration testing scheduled

## Compliance
- [ ] SOC 2 compliance requirements met
- [ ] GDPR compliance for EU data
- [ ] PCI DSS if handling payment data
- [ ] Industry-specific regulations addressed
- [ ] Regular compliance audits scheduled
```

## 📊 Monitoring Setup

### AWS CloudWatch Setup

```hcl
# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.main.arn_suffix],
            [".", "RequestCount", ".", "."],
            [".", "HTTPCode_Target_2XX_Count", ".", "."],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Application Load Balancer Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.main.identifier],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Database Metrics"
          period  = 300
        }
      }
    ]
  })
}
```

### GCP Cloud Monitoring Setup

```hcl
# Cloud Monitoring Alert
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "High Error Rate - Trading Orchestrator"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Error rate above 5%"

    condition_threshold {
      filter          = "resource.type=\"gke_container\" AND resource.labels.cluster_name=\"${google_container_cluster.primary.name}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "86400s"
  }

  combiner = "OR"

  enabled = true
}
```

## 🔄 Disaster Recovery

### Multi-Region Deployment

```yaml
# Multi-region deployment configuration
# Primary region: us-east-1
# Secondary region: us-west-2

apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-region-config
  namespace: trading-orchestrator
data:
  primary-region: "us-east-1"
  secondary-region: "us-west-2"
  failover-strategy: "manual" # or "automatic"
  health-check-url: "https://api.trading-orchestrator.com/health"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator-secondary
  namespace: trading-orchestrator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-orchestrator
      region: "us-west-2"
  template:
    metadata:
      labels:
        app: trading-orchestrator
        region: "us-west-2"
    spec:
      containers:
      - name: app
        image: trading-orchestrator:latest
        env:
        - name: REGION
          value: "us-west-2"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secondary-url
              key: url
```

### Backup and Restore Procedures

```bash
#!/bin/bash
# scripts/backup-dr.sh

set -e

BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="trading-orchestrator-backups"

# Database backup
echo "📦 Backing up primary database..."
aws rds create-db-snapshot \
    --db-instance-identifier trading-orchestrator-prod \
    --db-snapshot-identifier "trading-orchestrator-backup-${BACKUP_TIMESTAMP}"

# Application data backup
echo "📦 Backing up application data..."
aws s3 sync /opt/trading-orchestrator/data/ \
    "s3://${BACKUP_BUCKET}/data/${BACKUP_TIMESTAMP}/" \
    --server-side-encryption AES256

# Configuration backup
echo "📦 Backing up configuration..."
tar -czf "config-backup-${BACKUP_TIMESTAMP}.tar.gz" \
    /opt/trading-orchestrator/config/
aws s3 cp "config-backup-${BACKUP_TIMESTAMP}.tar.gz" \
    "s3://${BACKUP_BUCKET}/config/" \
    --server-side-encryption AES256

# Cross-region replication
echo "🔄 Setting up cross-region replication..."
aws s3 sync "s3://${BACKUP_BUCKET}/" \
    "s3://trading-orchestrator-backups-us-west-2/" \
    --source-region us-east-1 \
    --region us-west-2 \
    --server-side-encryption AES256

echo "✅ Backup completed: ${BACKUP_TIMESTAMP}"
```

### Failover Script

```bash
#!/bin/bash
# scripts/failover.sh

set -e

SECONDARY_REGION="us-west-2"
PRIMARY_REGION="us-east-1"

echo "🔄 Starting failover procedure..."

# Update DNS to secondary region
echo "📡 Updating DNS records..."
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch file://failover-dns.json

# Scale up secondary region
echo "⚡ Scaling up secondary region..."
aws ecs update-service \
    --cluster trading-orchestrator-secondary \
    --service trading-orchestrator \
    --desired-count 5 \
    --region ${SECONDARY_REGION}

# Verify secondary region health
echo "🔍 Verifying secondary region health..."
for i in {1..30}; do
    if curl -f https://trading-orchestrator-secondary.com/health; then
        echo "✅ Secondary region is healthy"
        break
    fi
    echo "⏳ Waiting for secondary region... ($i/30)"
    sleep 10
done

# Disable primary region
echo "🚫 Disabling primary region..."
aws ecs update-service \
    --cluster trading-orchestrator-primary \
    --service trading-orchestrator \
    --desired-count 0 \
    --region ${PRIMARY_REGION}

echo "✅ Failover completed!"
```

## 🎯 Conclusion

This comprehensive cloud deployment guide provides enterprise-grade deployment options across all major cloud providers. Choose the provider that best fits your:

- **Budget requirements**: DigitalOcean for cost-effective deployments
- **Enterprise needs**: AWS or GCP for comprehensive managed services
- **Microsoft ecosystem**: Azure for seamless integration
- **Compliance requirements**: All providers offer compliance certifications

**Next Steps:**
1. Choose your cloud provider
2. Follow the provider-specific deployment guide
3. Set up monitoring and alerting
4. Implement disaster recovery procedures
5. Conduct security audits
6. Plan for cost optimization

---

<div align="center">

**Deploy Everywhere, Trade Anywhere! ☁️**

Made with ❤️ by the Trading Orchestrator Team

</div>
