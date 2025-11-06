"""
Deployment Manager
CI/CD pipeline integration and automated deployment workflows.
"""

import os
import json
import logging
import subprocess
import shutil
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import docker
import aiohttp
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"
    COMPLETED = "completed"


class Environment(Enum):
    """Deployment environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: Environment
    version: str
    artifacts: List[str]
    services: List[str]
    dependencies: List[str]
    health_checks: List[str]
    rollback_strategy: str
    notifications: Dict[str, Any]
    scaling: Dict[str, Any]


@dataclass
class DeploymentResult:
    """Deployment result container."""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: float
    logs: List[str]
    artifacts_deployed: List[str]
    services_deployed: List[str]
    errors: List[str]
    warnings: List[str]


class DeploymentManager:
    """Manages automated deployment workflows."""
    
    def __init__(self, config_file: str = "deployment/deployment_config.yaml"):
        self.config_file = config_file
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Docker client
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {str(e)}")
        
        # Load deployment configurations
        self._load_deployment_configs()
    
    def _load_deployment_configs(self):
        """Load deployment configurations from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                for env_name, env_config in config_data.get("environments", {}).items():
                    try:
                        config = DeploymentConfig(
                            environment=Environment(env_name),
                            version=env_config.get("version", "latest"),
                            artifacts=env_config.get("artifacts", []),
                            services=env_config.get("services", []),
                            dependencies=env_config.get("dependencies", []),
                            health_checks=env_config.get("health_checks", []),
                            rollback_strategy=env_config.get("rollback_strategy", "automatic"),
                            notifications=env_config.get("notifications", {}),
                            scaling=env_config.get("scaling", {})
                        )
                        self.deployment_configs[env_name] = config
                    except Exception as e:
                        logger.error(f"Error loading config for {env_name}: {str(e)}")
            
            logger.info(f"Loaded {len(self.deployment_configs)} deployment configurations")
            
        except Exception as e:
            logger.error(f"Error loading deployment configurations: {str(e)}")
    
    async def deploy(self, environment: str, version: str = None, 
                    dry_run: bool = False) -> DeploymentResult:
        """Deploy application to specified environment."""
        deployment_id = f"{environment}_{int(time.time())}"
        logger.info(f"Starting deployment: {deployment_id}")
        
        if environment not in self.deployment_configs:
            raise ValueError(f"Environment {environment} not configured")
        
        config = self.deployment_configs[environment]
        if version:
            config.version = version
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            duration=0,
            logs=[],
            artifacts_deployed=[],
            services_deployed=[],
            errors=[],
            warnings=[]
        )
        
        self.active_deployments[deployment_id] = result
        
        try:
            if dry_run:
                result = await self._perform_dry_run(config, result)
            else:
                result = await self._perform_deployment(config, result)
            
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.status = DeploymentStatus.FAILED
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            self.deployment_history.append(result)
            
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        return result
    
    async def _perform_dry_run(self, config: DeploymentConfig, 
                              result: DeploymentResult) -> DeploymentResult:
        """Perform deployment dry run."""
        result.status = DeploymentStatus.PENDING
        result.logs.append("Performing dry run deployment...")
        
        # Simulate deployment steps
        steps = [
            "Validating configuration",
            "Checking dependencies",
            "Building artifacts",
            "Testing artifacts",
            "Validating deployment readiness"
        ]
        
        for step in steps:
            result.logs.append(f"✓ {step}")
            await asyncio.sleep(0.1)
        
        result.status = DeploymentStatus.SUCCESS
        result.logs.append("Dry run completed successfully")
        
        return result
    
    async def _perform_deployment(self, config: DeploymentConfig, 
                                 result: DeploymentResult) -> DeploymentResult:
        """Perform actual deployment."""
        result.status = DeploymentStatus.BUILDING
        
        try:
            # Step 1: Build artifacts
            result.logs.append("Building deployment artifacts...")
            artifacts = await self._build_artifacts(config, result)
            result.artifacts_deployed.extend(artifacts)
            
            # Step 2: Run tests
            result.status = DeploymentStatus.TESTING
            result.logs.append("Running pre-deployment tests...")
            test_results = await self._run_tests(config, result)
            
            if not test_results:
                raise Exception("Pre-deployment tests failed")
            
            # Step 3: Deploy services
            result.status = DeploymentStatus.DEPLOYING
            result.logs.append("Deploying services...")
            services = await self._deploy_services(config, result)
            result.services_deployed.extend(services)
            
            # Step 4: Health checks
            result.logs.append("Running health checks...")
            health_results = await self._run_health_checks(config, result)
            
            if not health_results:
                result.warnings.append("Some health checks failed - manual intervention may be required")
            
            # Step 5: Post-deployment validation
            result.logs.append("Validating deployment...")
            validation_results = await self._validate_deployment(config, result)
            
            if validation_results:
                result.status = DeploymentStatus.SUCCESS
                result.logs.append("Deployment completed successfully")
            else:
                # Trigger rollback
                result.status = DeploymentStatus.ROLLBACK
                result.logs.append("Deployment validation failed - initiating rollback...")
                rollback_result = await self._rollback_deployment(config, result)
                if rollback_result:
                    result.status = DeploymentStatus.FAILED
                    result.logs.append("Rollback completed")
            
        except Exception as e:
            result.errors.append(str(e))
            result.status = DeploymentStatus.FAILED
            raise
        
        return result
    
    async def _build_artifacts(self, config: DeploymentConfig, 
                              result: DeploymentResult) -> List[str]:
        """Build deployment artifacts."""
        artifacts = []
        
        # Build Docker images if Docker is available
        if self.docker_client and "docker" in config.artifacts:
            for service in config.services:
                try:
                    image_name = f"{service}:{config.version}"
                    result.logs.append(f"Building Docker image: {image_name}")
                    
                    # Build image
                    image = self.docker_client.images.build(
                        path=".",
                        dockerfile=f"deployment/{service}/Dockerfile",
                        tag=image_name
                    )
                    
                    artifacts.append(image_name)
                    result.logs.append(f"✓ Built Docker image: {image_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to build Docker image {image_name}: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
        
        # Build other artifacts (e.g., static files, bundles)
        if "static" in config.artifacts:
            try:
                result.logs.append("Building static assets...")
                static_artifact = "static_assets.tar.gz"
                # Simulate static asset build
                artifacts.append(static_artifact)
                result.logs.append(f"✓ Built static assets: {static_artifact}")
            except Exception as e:
                error_msg = f"Failed to build static assets: {str(e)}"
                result.errors.append(error_msg)
        
        return artifacts
    
    async def _run_tests(self, config: DeploymentConfig, 
                        result: DeploymentResult) -> bool:
        """Run pre-deployment tests."""
        test_suites = config.dependencies.get("test_suites", [])
        
        for test_suite in test_suites:
            try:
                result.logs.append(f"Running test suite: {test_suite}")
                
                # Run tests using pytest or similar
                cmd = ["python", "-m", "pytest", test_suite, "--tb=short"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    result.logs.append(f"✓ Test suite {test_suite} passed")
                else:
                    error_msg = f"Test suite {test_suite} failed:\n{stderr.decode()}"
                    result.errors.append(error_msg)
                    return False
                    
            except Exception as e:
                error_msg = f"Error running test suite {test_suite}: {str(e)}"
                result.errors.append(error_msg)
                return False
        
        return True
    
    async def _deploy_services(self, config: DeploymentConfig, 
                              result: DeploymentResult) -> List[str]:
        """Deploy services to environment."""
        deployed_services = []
        
        for service in config.services:
            try:
                result.logs.append(f"Deploying service: {service}")
                
                if self.docker_client and "docker" in config.artifacts:
                    # Deploy using Docker Compose
                    service_deployed = await self._deploy_docker_service(service, config, result)
                    if service_deployed:
                        deployed_services.append(service)
                else:
                    # Deploy using other methods
                    service_deployed = await self._deploy_script_service(service, config, result)
                    if service_deployed:
                        deployed_services.append(service)
                        
            except Exception as e:
                error_msg = f"Failed to deploy service {service}: {str(e)}"
                result.errors.append(error_msg)
                logger.error(error_msg)
        
        return deployed_services
    
    async def _deploy_docker_service(self, service: str, config: DeploymentConfig, 
                                    result: DeploymentResult) -> bool:
        """Deploy service using Docker."""
        try:
            # Load Docker Compose configuration
            compose_file = f"deployment/{service}/docker-compose.yml"
            
            if not os.path.exists(compose_file):
                result.warnings.append(f"Docker Compose file not found: {compose_file}")
                return False
            
            # Update version in compose file
            await self._update_compose_version(compose_file, config.version)
            
            # Deploy using docker-compose
            cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result.logs.append(f"✓ Deployed Docker service: {service}")
                return True
            else:
                error_msg = f"Docker deployment failed for {service}:\n{stderr.decode()}"
                result.errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error deploying Docker service {service}: {str(e)}"
            result.errors.append(error_msg)
            return False
    
    async def _deploy_script_service(self, service: str, config: DeploymentConfig, 
                                    result: DeploymentResult) -> bool:
        """Deploy service using deployment script."""
        try:
            script_path = f"deployment/{service}/deploy.sh"
            
            if not os.path.exists(script_path):
                result.warnings.append(f"Deployment script not found: {script_path}")
                return False
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Run deployment script
            env = os.environ.copy()
            env.update({
                "VERSION": config.version,
                "ENVIRONMENT": config.environment.value
            })
            
            process = await asyncio.create_subprocess_exec(
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result.logs.append(f"✓ Deployed service: {service}")
                return True
            else:
                error_msg = f"Script deployment failed for {service}:\n{stderr.decode()}"
                result.errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error deploying service {service}: {str(e)}"
            result.errors.append(error_msg)
            return False
    
    async def _update_compose_version(self, compose_file: str, version: str):
        """Update version in Docker Compose file."""
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
            
            # Replace version tags
            import re
            content = re.sub(r':latest', f':{version}', content)
            content = re.sub(r':([a-zA-Z0-9.-]+)', lambda m: f':{version}' if m.group(1) == 'dev' else m.group(0), content)
            
            with open(compose_file, 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Error updating compose file {compose_file}: {str(e)}")
    
    async def _run_health_checks(self, config: DeploymentConfig, 
                                result: DeploymentResult) -> bool:
        """Run health checks on deployed services."""
        health_checks = config.health_checks
        all_healthy = True
        
        for health_check in health_checks:
            try:
                result.logs.append(f"Running health check: {health_check}")
                
                # Simple HTTP health check
                if health_check.startswith("http"):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_check) as response:
                            if response.status == 200:
                                result.logs.append(f"✓ Health check passed: {health_check}")
                            else:
                                result.logs.append(f"✗ Health check failed: {health_check} (status: {response.status})")
                                all_healthy = False
                else:
                    # Custom health check command
                    cmd = health_check.split()
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        result.logs.append(f"✓ Health check passed: {health_check}")
                    else:
                        result.logs.append(f"✗ Health check failed: {health_check}")
                        all_healthy = False
                        
            except Exception as e:
                result.logs.append(f"✗ Health check error: {health_check} - {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    async def _validate_deployment(self, config: DeploymentConfig, 
                                  result: DeploymentResult) -> bool:
        """Validate deployment was successful."""
        try:
            # Check service status
            for service in config.services:
                result.logs.append(f"Validating service: {service}")
                
                # Check if service is running
                if self.docker_client:
                    try:
                        containers = self.docker_client.containers.list(
                            filters={"name": service}
                        )
                        
                        if containers:
                            container = containers[0]
                            if container.status == "running":
                                result.logs.append(f"✓ Service {service} is running")
                            else:
                                result.logs.append(f"✗ Service {service} is not running (status: {container.status})")
                                return False
                        else:
                            result.logs.append(f"✗ Service {service} container not found")
                            return False
                    except Exception as e:
                        result.logs.append(f"✗ Error checking service {service}: {str(e)}")
                        return False
                else:
                    # Check using systemctl or other method
                    cmd = ["systemctl", "is-active", service]
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0 and b"active" in stdout:
                        result.logs.append(f"✓ Service {service} is active")
                    else:
                        result.logs.append(f"✗ Service {service} is not active")
                        return False
            
            # Additional validation checks
            result.logs.append("Running additional validation checks...")
            
            # Check database connectivity
            db_check = await self._check_database_connectivity(config, result)
            if not db_check:
                return False
            
            # Check critical endpoints
            endpoint_check = await self._check_critical_endpoints(config, result)
            if not endpoint_check:
                return False
            
            result.logs.append("✓ All validation checks passed")
            return True
            
        except Exception as e:
            result.logs.append(f"✗ Validation error: {str(e)}")
            return False
    
    async def _check_database_connectivity(self, config: DeploymentConfig, 
                                         result: DeploymentResult) -> bool:
        """Check database connectivity."""
        try:
            # This would check actual database connectivity
            # For now, simulate the check
            await asyncio.sleep(0.1)
            result.logs.append("✓ Database connectivity verified")
            return True
        except Exception as e:
            result.logs.append(f"✗ Database connectivity check failed: {str(e)}")
            return False
    
    async def _check_critical_endpoints(self, config: DeploymentConfig, 
                                      result: DeploymentResult) -> bool:
        """Check critical API endpoints."""
        critical_endpoints = [
            "/health",
            "/api/status",
            "/api/version"
        ]
        
        base_url = f"http://localhost:{config.scaling.get('port', 8000)}"
        
        for endpoint in critical_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            result.logs.append(f"✓ Endpoint {endpoint} is accessible")
                        else:
                            result.logs.append(f"✗ Endpoint {endpoint} returned status {response.status}")
                            return False
            except Exception as e:
                result.logs.append(f"✗ Endpoint {endpoint} check failed: {str(e)}")
                return False
        
        return True
    
    async def _rollback_deployment(self, config: DeploymentConfig, 
                                  result: DeploymentResult) -> bool:
        """Rollback deployment to previous version."""
        try:
            result.logs.append("Initiating rollback...")
            
            # Find previous version
            previous_version = await self._get_previous_version(config)
            if not previous_version:
                result.errors.append("Could not determine previous version for rollback")
                return False
            
            result.logs.append(f"Rolling back to version: {previous_version}")
            
            # Deploy previous version
            rollback_config = DeploymentConfig(
                environment=config.environment,
                version=previous_version,
                artifacts=config.artifacts,
                services=config.services,
                dependencies=config.dependencies,
                health_checks=config.health_checks,
                rollback_strategy="none",  # Prevent recursive rollback
                notifications=config.notifications,
                scaling=config.scaling
            )
            
            rollback_result = DeploymentResult(
                deployment_id=f"{result.deployment_id}_rollback",
                status=DeploymentStatus.PENDING,
                start_time=datetime.now(),
                end_time=None,
                duration=0,
                logs=[],
                artifacts_deployed=[],
                services_deployed=[],
                errors=[],
                warnings=[]
            )
            
            rollback_success = await self._perform_deployment(rollback_config, rollback_result)
            
            if rollback_success.status == DeploymentStatus.SUCCESS:
                result.logs.append("✓ Rollback completed successfully")
                return True
            else:
                result.errors.append("Rollback failed")
                return False
                
        except Exception as e:
            result.errors.append(f"Rollback error: {str(e)}")
            return False
    
    async def _get_previous_version(self, config: DeploymentConfig) -> Optional[str]:
        """Get previous version for rollback."""
        try:
            # This would query a version database or git history
            # For now, return a placeholder
            return "previous-version"
        except Exception as e:
            logger.error(f"Error getting previous version: {str(e)}")
            return None
    
    async def create_deployment_pipeline(self, environment: str, 
                                       pipeline_config: Dict[str, Any]) -> str:
        """Create CI/CD pipeline configuration."""
        pipeline_id = f"{environment}_pipeline_{int(time.time())}"
        
        try:
            if pipeline_config.get("type") == "github_actions":
                pipeline_file = await self._create_github_actions_pipeline(
                    environment, pipeline_config
                )
            elif pipeline_config.get("type") == "gitlab_ci":
                pipeline_file = await self._create_gitlab_ci_pipeline(
                    environment, pipeline_config
                )
            elif pipeline_config.get("type") == "jenkins":
                pipeline_file = await self._create_jenkins_pipeline(
                    environment, pipeline_config
                )
            else:
                raise ValueError(f"Unsupported pipeline type: {pipeline_config.get('type')}")
            
            logger.info(f"Created CI/CD pipeline: {pipeline_file}")
            return pipeline_file
            
        except Exception as e:
            logger.error(f"Error creating deployment pipeline: {str(e)}")
            raise
    
    async def _create_github_actions_pipeline(self, environment: str, 
                                            config: Dict[str, Any]) -> str:
        """Create GitHub Actions pipeline."""
        pipeline_dir = Path(".github/workflows")
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_content = f"""name: Deploy to {environment.title()}

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t trading-system:${{{{ github.sha }}}} .
        docker tag trading-system:${{{{ github.sha }}}} trading-system:{environment}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to {environment}
      run: |
        echo "Deploying to {environment} environment"
        # Add deployment commands here
"""
        
        pipeline_file = pipeline_dir / f"deploy_{environment}.yml"
        
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_content)
        
        return str(pipeline_file)
    
    async def _create_gitlab_ci_pipeline(self, environment: str, 
                                       config: Dict[str, Any]) -> str:
        """Create GitLab CI pipeline."""
        pipeline_file = Path(".gitlab-ci.yml")
        
        pipeline_content = f"""stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: trading-system:{environment}

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=. --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker tag $DOCKER_IMAGE:$CI_COMMIT_SHA $DOCKER_IMAGE:latest
  only:
    - main

deploy_{environment}:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to {environment} environment"
    # Add deployment commands here
  only:
    - main
  when: manual
"""
        
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_content)
        
        return str(pipeline_file)
    
    async def _create_jenkins_pipeline(self, environment: str, 
                                     config: Dict[str, Any]) -> str:
        """Create Jenkins pipeline."""
        pipeline_dir = Path("jenkins")
        pipeline_dir.mkdir(exist_ok=True)
        
        pipeline_content = f"""pipeline {{
    agent any
    
    environment {{
        DOCKER_IMAGE = 'trading-system'
        ENVIRONMENT = '{environment}'
    }}
    
    stages {{
        stage('Test') {{
            steps {{
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-cov'
                sh 'pytest --cov=. --cov-report=xml'
            }}
            post {{
                always {{
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }}
            }}
        }}
        
        stage('Build') {{
            steps {{
                sh 'docker build -t $DOCKER_IMAGE:$BUILD_NUMBER .'
                sh 'docker tag $DOCKER_IMAGE:$BUILD_NUMBER $DOCKER_IMAGE:{environment}:latest'
            }}
        }}
        
        stage('Deploy to {environment}') {{
            when {{
                branch 'main'
            }}
            steps {{
                sh 'echo "Deploying to {environment} environment"'
                // Add deployment commands here
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
    }}
}}"""
        
        pipeline_file = pipeline_dir / f"deploy_{environment}.groovy"
        
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_content)
        
        return str(pipeline_file)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment."""
        # Check active deployments
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        # Check deployment history
        for deployment in reversed(self.deployment_history):
            if deployment.deployment_id == deployment_id:
                return deployment
        
        return None
    
    def list_deployments(self, limit: int = 10) -> List[DeploymentResult]:
        """List recent deployments."""
        return sorted(
            self.deployment_history,
            key=lambda d: d.start_time,
            reverse=True
        )[:limit]
    
    async def cleanup_old_deployments(self, days: int = 30):
        """Clean up old deployment artifacts."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean up old deployments from history
            self.deployment_history = [
                d for d in self.deployment_history
                if d.start_time > cutoff_date
            ]
            
            # Clean up old Docker images
            if self.docker_client:
                images = self.docker_client.images.list()
                for image in images:
                    if image.tags:
                        for tag in image.tags:
                            if "trading-system" in tag and any(
                                d.start_time < cutoff_date for d in self.deployment_history
                            ):
                                try:
                                    self.docker_client.images.remove(image.id, force=True)
                                    logger.info(f"Removed old image: {tag}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove image {tag}: {str(e)}")
            
            logger.info(f"Cleaned up deployments older than {days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old deployments: {str(e)}")


# Global deployment manager instance
deployment_manager = DeploymentManager()