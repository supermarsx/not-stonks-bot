#!/usr/bin/env python3
"""
Provider System Installation and Setup Script

This script helps set up the expanded LLM provider system by:
1. Installing required dependencies
2. Setting up configuration templates
3. Testing provider connections
4. Configuring local providers (Ollama, LocalAI, vLLM)
5. Creating example configurations
"""

import asyncio
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the trading_orchestrator to the Python path
sys.path.append('/workspace/trading_orchestrator')

from ai.providers import ProviderFactory
from loguru import logger


class ProviderSystemInstaller:
    """Installs and configures the expanded LLM provider system"""
    
    def __init__(self):
        self.workspace_root = Path('/workspace/trading_orchestrator')
        self.config_dir = self.workspace_root / 'config'
        self.providers_dir = self.workspace_root / 'ai' / 'providers'
        self.example_configs_dir = self.workspace_root / 'ai' / 'providers' / 'examples' / 'configs'
        
        self.installed_dependencies = []
        self.created_configs = []
        self.test_results = {}
        
    async def run_installation(self) -> Dict[str, Any]:
        """Run complete installation process"""
        print("🚀 Starting Provider System Installation")
        print("=" * 50)
        
        results = {
            'installation_start': self.get_timestamp(),
            'steps': {},
            'overall_status': 'pending',
            'summary': {}
        }
        
        try:
            # Step 1: Install Dependencies
            results['steps']['dependencies'] = await self.install_dependencies()
            
            # Step 2: Setup Configuration Directories
            results['steps']['directories'] = await self.setup_directories()
            
            # Step 3: Create Configuration Templates
            results['steps']['templates'] = await self.create_config_templates()
            
            # Step 4: Setup Local Providers (optional)
            results['steps']['local_providers'] = await self.setup_local_providers()
            
            # Step 5: Test Provider System
            results['steps']['testing'] = await self.test_provider_system()
            
            # Step 6: Create Example Applications
            results['steps']['examples'] = await self.create_example_applications()
            
            # Calculate overall status
            results['overall_status'] = self.calculate_installation_status(results['steps'])
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            results['overall_status'] = 'error'
            results['error'] = str(e)
            
        results['installation_end'] = self.get_timestamp()
        results['installation_duration'] = str(self.parse_timestamp(results['installation_end']) - self.parse_timestamp(results['installation_start']))
        
        # Generate summary
        results['summary'] = self.generate_installation_summary(results['steps'])
        
        return results
        
    async def install_dependencies(self) -> Dict[str, Any]:
        """Install required Python dependencies"""
        print("\n📦 Installing Dependencies")
        print("-" * 30)
        
        step_result = {
            'status': 'running',
            'installed_packages': [],
            'failed_packages': [],
            'details': {}
        }
        
        # Core dependencies
        core_packages = [
            'aiohttp>=3.8.0',
            'aiofiles>=22.0.0',
            'asyncio-throttle>=1.0.2',
            'tenacity>=8.0.0'
        ]
        
        # Optional provider dependencies
        optional_packages = {
            'openai': 'openai>=1.0.0',
            'anthropic': 'anthropic>=0.7.0',
            'requests': 'requests>=2.28.0'
        }
        
        # Install core dependencies
        for package in core_packages:
            success = await self.install_package(package)
            if success:
                step_result['installed_packages'].append(package)
            else:
                step_result['failed_packages'].append(package)
                
        # Try to install optional dependencies
        for provider, package in optional_packages.items():
            try:
                success = await self.install_package(package)
                if success:
                    step_result['installed_packages'].append(package)
                    self.installed_dependencies.append(provider)
                else:
                    step_result['failed_packages'].append(package)
            except Exception as e:
                logger.warning(f"Could not install {package}: {e}")
                step_result['failed_packages'].append(package)
                
        # Check which providers are now available
        available_providers = self.check_provider_availability()
        step_result['details']['available_providers'] = available_providers
        step_result['details']['core_installed'] = len(step_result['installed_packages'])
        step_result['details']['optional_installed'] = len([p for p in self.installed_dependencies if p in optional_packages.keys()])
        
        if step_result['failed_packages']:
            step_result['status'] = 'completed_with_warnings'
            print(f"⚠️  Some packages failed to install: {', '.join(step_result['failed_packages'])}")
        else:
            step_result['status'] = 'completed'
            print("✅ All dependencies installed successfully")
            
        return step_result
        
    async def install_package(self, package: str) -> bool:
        """Install a single package"""
        try:
            print(f"  Installing {package}...")
            
            # Use pip to install
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package, '--quiet'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {package} installed successfully")
                return True
            else:
                print(f"  ❌ Failed to install {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout installing {package}")
            return False
        except Exception as e:
            print(f"  💥 Error installing {package}: {e}")
            return False
            
    def check_provider_availability(self) -> List[str]:
        """Check which providers are available based on installed packages"""
        available = []
        
        try:
            import openai
            available.append('openai')
        except ImportError:
            pass
            
        try:
            import anthropic
            available.append('anthropic')
        except ImportError:
            pass
            
        try:
            import aiohttp
            available.append('openai_compatible')
        except ImportError:
            pass
            
        # Local providers don't require special packages
        available.extend(['ollama', 'localai', 'vllm'])
        
        return available
        
    async def setup_directories(self) -> Dict[str, Any]:
        """Setup required directories"""
        print("\n📁 Setting up Configuration Directories")
        print("-" * 40)
        
        step_result = {
            'status': 'completed',
            'created_directories': [],
            'existing_directories': []
        }
        
        directories = [
            self.config_dir,
            self.example_configs_dir,
            self.workspace_root / 'ai' / 'providers' / 'logs',
            self.workspace_root / 'ai' / 'providers' / 'cache',
            self.workspace_root / 'ai' / 'providers' / 'test_results'
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                
                if directory.exists():
                    step_result['created_directories'].append(str(directory))
                    print(f"  📁 {directory}")
                else:
                    print(f"  ❌ Failed to create {directory}")
                    
            except Exception as e:
                print(f"  💥 Error creating {directory}: {e}")
                step_result['status'] = 'error'
                return step_result
                
        return step_result
        
    async def create_config_templates(self) -> Dict[str, Any]:
        """Create configuration templates"""
        print("\n📄 Creating Configuration Templates")
        print("-" * 35)
        
        step_result = {
            'status': 'completed',
            'created_files': [],
            'templates': {}
        }
        
        # Template configurations
        templates = self.create_config_templates_data()
        
        for template_name, config_data in templates.items():
            file_path = self.example_configs_dir / f"{template_name}.json"
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                    
                step_result['created_files'].append(str(file_path))
                step_result['templates'][template_name] = config_data
                print(f"  📄 {file_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to create {file_path.name}: {e}")
                step_result['status'] = 'error'
                
        return step_result
        
    def create_config_templates_data(self) -> Dict[str, Dict]:
        """Create configuration template data"""
        return {
            'development': {
                'description': 'Development configuration with local providers',
                'providers': [
                    {
                        'name': 'ollama_local',
                        'provider_type': 'ollama',
                        'base_url': 'http://localhost:11434',
                        'priority': 1,
                        'rate_limit': {
                            'requests_per_minute': 60,
                            'tokens_per_minute': 100000,
                            'burst_limit': 10
                        },
                        'health_check_interval': 60,
                        'enable_caching': True
                    },
                    {
                        'name': 'openai_backup',
                        'provider_type': 'openai',
                        'api_key': 'env:OPENAI_API_KEY',
                        'priority': 2,
                        'rate_limit': {
                            'requests_per_minute': 1000,
                            'tokens_per_minute': 100000,
                            'burst_limit': 50
                        },
                        'health_check_interval': 30,
                        'enable_caching': True
                    }
                ],
                'settings': {
                    'health_check_interval': 60,
                    'global_timeout': 30.0,
                    'failover_strategy': 'health_based',
                    'enable_circuit_breaker': True
                }
            },
            
            'production': {
                'description': 'Production configuration with multiple cloud providers',
                'providers': [
                    {
                        'name': 'openai_primary',
                        'provider_type': 'openai',
                        'api_key': 'env:OPENAI_API_KEY',
                        'priority': 1,
                        'rate_limit': {
                            'requests_per_minute': 5000,
                            'tokens_per_minute': 2000000,
                            'burst_limit': 100
                        },
                        'health_check_interval': 30,
                        'enable_caching': True,
                        'extra_config': {
                            'organization': 'env:OPENAI_ORG_ID'
                        }
                    },
                    {
                        'name': 'anthropic_primary',
                        'provider_type': 'anthropic',
                        'api_key': 'env:ANTHROPIC_API_KEY',
                        'priority': 2,
                        'rate_limit': {
                            'requests_per_minute': 1000,
                            'tokens_per_minute': 400000,
                            'burst_limit': 20
                        },
                        'health_check_interval': 30,
                        'enable_caching': True
                    },
                    {
                        'name': 'openrouter_backup',
                        'provider_type': 'openai_compatible',
                        'base_url': 'https://openrouter.ai/api/v1',
                        'api_key': 'env:OPENROUTER_API_KEY',
                        'priority': 3,
                        'rate_limit': {
                            'requests_per_minute': 100,
                            'tokens_per_minute': 100000,
                            'burst_limit': 10
                        },
                        'health_check_interval': 60,
                        'extra_config': {
                            'referer': 'your-app-name'
                        }
                    },
                    {
                        'name': 'ollama_local',
                        'provider_type': 'ollama',
                        'base_url': 'http://localhost:11434',
                        'priority': 4,
                        'rate_limit': {
                            'requests_per_minute': 60,
                            'tokens_per_minute': 100000,
                            'burst_limit': 10
                        },
                        'health_check_interval': 120
                    }
                ],
                'settings': {
                    'health_check_interval': 30,
                    'global_timeout': 30.0,
                    'failover_strategy': 'health_based',
                    'enable_circuit_breaker': True,
                    'circuit_breaker_failure_threshold': 5,
                    'circuit_breaker_recovery_timeout': 60
                }
            },
            
            'research': {
                'description': 'Research configuration with local high-performance providers',
                'providers': [
                    {
                        'name': 'vllm_high_perf',
                        'provider_type': 'vllm',
                        'base_url': 'http://localhost:8000',
                        'priority': 1,
                        'rate_limit': {
                            'requests_per_minute': 200,
                            'tokens_per_minute': 500000,
                            'burst_limit': 20
                        },
                        'health_check_interval': 60,
                        'timeout': 60.0
                    },
                    {
                        'name': 'localai_custom',
                        'provider_type': 'localai',
                        'base_url': 'http://localhost:8080',
                        'priority': 2,
                        'rate_limit': {
                            'requests_per_minute': 120,
                            'tokens_per_minute': 200000,
                            'burst_limit': 15
                        },
                        'health_check_interval': 60,
                        'timeout': 45.0
                    }
                ],
                'settings': {
                    'health_check_interval': 60,
                    'global_timeout': 60.0,
                    'failover_strategy': 'priority_based',
                    'enable_circuit_breaker': True
                }
            },
            
            'minimal': {
                'description': 'Minimal configuration with single provider',
                'providers': [
                    {
                        'name': 'openai_minimal',
                        'provider_type': 'openai',
                        'api_key': 'env:OPENAI_API_KEY',
                        'priority': 1,
                        'rate_limit': {
                            'requests_per_minute': 100,
                            'tokens_per_minute': 10000,
                            'burst_limit': 10
                        },
                        'health_check_interval': 60
                    }
                ],
                'settings': {
                    'health_check_interval': 60,
                    'global_timeout': 30.0,
                    'failover_strategy': 'priority_based',
                    'enable_circuit_breaker': False
                }
            }
        }
        
    async def setup_local_providers(self) -> Dict[str, Any]:
        """Setup instructions for local providers"""
        print("\n🐳 Local Provider Setup Instructions")
        print("-" * 35)
        
        step_result = {
            'status': 'completed',
            'instructions': {},
            'scripts_created': []
        }
        
        # Create setup scripts for local providers
        scripts = self.create_local_provider_scripts()
        
        scripts_dir = self.workspace_root / 'scripts' / 'local_providers'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        for script_name, script_content in scripts.items():
            script_path = scripts_dir / script_name
            
            try:
                with open(script_path, 'w') as f:
                    f.write(script_content)
                    
                # Make executable
                os.chmod(script_path, 0o755)
                
                step_result['scripts_created'].append(str(script_path))
                step_result['instructions'][script_name] = f"Created: {script_path}"
                print(f"  📜 {script_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to create {script_path.name}: {e}")
                
        return step_result
        
    def create_local_provider_scripts(self) -> Dict[str, str]:
        """Create setup scripts for local providers"""
        
        # Ollama setup script
        ollama_setup = """#!/bin/bash
# Ollama Setup Script
# This script installs and configures Ollama for local LLM serving

echo "🦙 Setting up Ollama..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Run Ollama container
echo "🐳 Starting Ollama container..."
docker run -d \\
    --name ollama \\
    -p 11434:11434 \\
    -v ollama:/root/.ollama \\
    --restart unless-stopped \\
    ollama/ollama

echo "⏳ Waiting for Ollama to start..."
sleep 10

# Install a default model (Llama 3)
echo "📦 Installing default model (llama3)..."
docker exec ollama ollama pull llama3

echo "✅ Ollama setup complete!"
echo "🌐 Ollama is now running at http://localhost:11434"
echo "📋 To check status: docker ps"
echo "📋 To stop: docker stop ollama"
echo "📋 To view logs: docker logs ollama"
"""
        
        # LocalAI setup script
        localai_setup = """#!/bin/bash
# LocalAI Setup Script
# This script sets up LocalAI for OpenAI-compatible local LLM serving

echo "🤖 Setting up LocalAI..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Create LocalAI configuration directory
mkdir -p localai-configs

# Create docker-compose.yml for LocalAI
cat > localai-configs/docker-compose.yml << 'EOF'
version: '3.8'

services:
  localai:
    image: mudler/localai:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./configs:/configs
    environment:
      - THREADS=4
      - MODELS=/models
      - REBUILD=true
    restart: unless-stopped
EOF

# Create models directory
mkdir -p localai-configs/models

echo "🐳 Starting LocalAI..."
cd localai-configs
docker-compose up -d

echo "⏳ Waiting for LocalAI to start..."
sleep 15

echo "✅ LocalAI setup complete!"
echo "🌐 LocalAI is now running at http://localhost:8080"
echo "📋 Available models: curl http://localhost:8080/v1/models"
echo "📋 To check status: docker-compose ps"
echo "📋 To view logs: docker-compose logs -f"
"""
        
        # vLLM setup script
        vllm_setup = """#!/bin/bash
# vLLM Setup Script
# This script sets up vLLM for high-performance local LLM serving

echo "⚡ Setting up vLLM..."

# Check if Python and pip are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install vLLM
echo "📦 Installing vLLM..."
pip install vllm

# Create startup script
cat > start_vllm.sh << 'EOF'
#!/bin/bash
# vLLM Startup Script
# Usage: ./start_vllm.sh [model_name]

MODEL=${1:-"meta-llama/Llama-2-7b-chat-hf"}
PORT=${2:-8000}

echo "⚡ Starting vLLM with model: $MODEL on port: $PORT"
echo "🌐 Server will be available at http://localhost:$PORT"

python -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL" \\
    --port "$PORT" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 4096
EOF

chmod +x start_vllm.sh

echo "✅ vLLM setup complete!"
echo "📜 Start script created: start_vllm.sh"
echo "🚀 To start vLLM: ./start_vllm.sh"
echo "📖 Usage: ./start_vllm.sh [model_name] [port]"
echo "💡 Example: ./start_vllm.sh meta-llama/Llama-2-7b-chat-hf 8000"
"""
        
        return {
            'setup_ollama.sh': ollama_setup,
            'setup_localai.sh': localai_setup,
            'setup_vllm.sh': vllm_setup
        }
        
    async def test_provider_system(self) -> Dict[str, Any]:
        """Test the provider system installation"""
        print("\n🧪 Testing Provider System")
        print("-" * 25)
        
        step_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Test 1: Import provider modules
            try:
                from ai.providers import ProviderFactory, ProviderConfig
                step_result['tests'].append({
                    'name': 'provider_imports',
                    'status': 'passed',
                    'message': 'All provider modules imported successfully'
                })
            except Exception as e:
                step_result['tests'].append({
                    'name': 'provider_imports',
                    'status': 'failed',
                    'message': f'Failed to import provider modules: {e}'
                })
                
            # Test 2: Create provider factory
            try:
                factory = ProviderFactory()
                available_types = factory.get_available_provider_types()
                step_result['tests'].append({
                    'name': 'factory_creation',
                    'status': 'passed',
                    'message': f'Factory created with {len(available_types)} provider types',
                    'details': {'available_types': available_types}
                })
            except Exception as e:
                step_result['tests'].append({
                    'name': 'factory_creation',
                    'status': 'failed',
                    'message': f'Failed to create factory: {e}'
                })
                
            # Test 3: Configuration loading
            config_files = list(self.example_configs_dir.glob('*.json'))
            step_result['tests'].append({
                'name': 'config_loading',
                'status': 'passed' if config_files else 'failed',
                'message': f'Found {len(config_files)} configuration templates',
                'details': {'config_files': [f.name for f in config_files]}
            })
            
            # Test 4: Quick provider test
            test_config = {
                'name': 'test_provider',
                'provider_type': 'openai',
                'api_key': 'test-key',
                'priority': 1
            }
            
            try:
                from ai.providers.testing_suite import quick_test_provider
                test_result = await quick_test_provider(test_config)
                
                step_result['tests'].append({
                    'name': 'quick_provider_test',
                    'status': 'passed',
                    'message': 'Quick provider test function works',
                    'details': {'test_status': test_result.get('status', 'unknown')}
                })
            except Exception as e:
                step_result['tests'].append({
                    'name': 'quick_provider_test',
                    'status': 'warning',
                    'message': f'Quick provider test had issues (expected for test key): {e}'
                })
                
            step_result['status'] = 'completed'
            
        except Exception as e:
            step_result['status'] = 'error'
            step_result['error'] = str(e)
            
        return step_result
        
    async def create_example_applications(self) -> Dict[str, Any]:
        """Create example applications demonstrating usage"""
        print("\n📝 Creating Example Applications")
        print("-" * 32)
        
        step_result = {
            'status': 'completed',
            'created_files': [],
            'examples': {}
        }
        
        examples_dir = self.workspace_root / 'ai' / 'providers' / 'examples' / 'applications'
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        examples = self.create_example_applications_data()
        
        for example_name, example_content in examples.items():
            example_path = examples_dir / f"{example_name}.py"
            
            try:
                with open(example_path, 'w') as f:
                    f.write(example_content)
                    
                step_result['created_files'].append(str(example_path))
                step_result['examples'][example_name] = example_path.name
                print(f"  📝 {example_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to create {example_path.name}: {e}")
                step_result['status'] = 'error'
                
        return step_result
        
    def create_example_applications_data(self) -> Dict[str, str]:
        """Create example application code"""
        
        # Basic usage example
        basic_example = '''"""
Basic Provider Usage Example

This example demonstrates basic usage of the LLM provider system.
"""

import asyncio
from ai.providers import ProviderFactory, ProviderConfig

async def main():
    # Create provider factory
    factory = ProviderFactory()
    
    # Configure OpenAI provider
    config = {
        'name': 'openai_example',
        'provider_type': 'openai',
        'api_key': 'your-api-key-here',
        'priority': 1
    }
    
    try:
        # Create provider
        provider = await factory.create_from_config_dict(config)
        
        if provider:
            # Test connection
            health = await provider.health_check()
            print(f"Provider status: {health.is_healthy}")
            
            # List models
            models = await provider.list_models()
            print(f"Available models: {len(models)}")
            
            # Make a completion request
            response = await provider.generate_completion(
                messages=[{"role": "user", "content": "Hello, world!"}],
                model="gpt-3.5-turbo"
            )
            
            print(f"Response: {response.get('content', 'No response')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Enhanced AI manager example
        enhanced_example = '''"""
Enhanced AI Manager Example

This example demonstrates the enhanced AI models manager with failover.
"""

import asyncio
from ai.models.enhanced_ai_models_manager import EnhancedAIModelsManager

async def main():
    # Configure multiple providers
    config = {
        'providers': [
            {
                'name': 'openai_primary',
                'provider_type': 'openai',
                'api_key': 'your-openai-key',
                'priority': 1
            },
            {
                'name': 'ollama_backup',
                'provider_type': 'ollama',
                'base_url': 'http://localhost:11434',
                'priority': 2
            }
        ],
        'global_timeout': 30.0
    }
    
    # Initialize AI manager
    ai_manager = EnhancedAIModelsManager(config)
    await ai_manager.initialize()
    
    # Register a tool
    async def get_weather(location: str):
        return {"location": location, "temperature": "20°C"}
    
    ai_manager.register_tool("get_weather", get_weather)
    
    # Make request with automatic failover
    try:
        response = await ai_manager.generate_completion(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            task_type="quick_decision",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information"
                }
            }]
        )
        
        print(f"Response: {response.get('content', 'No response')}")
        print(f"Provider used: {response.get('provider', 'Unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Get usage statistics
    stats = ai_manager.get_usage_stats()
    print(f"Usage stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        return {
            'basic_usage': basic_example,
            'enhanced_ai_manager': enhanced_example
        }
        
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def parse_timestamp(self, timestamp: str):
        """Parse timestamp string"""
        from datetime import datetime
        return datetime.fromisoformat(timestamp)
        
    def calculate_installation_status(self, steps: Dict[str, Any]) -> str:
        """Calculate overall installation status"""
        statuses = []
        
        for step_name, step_result in steps.items():
            status = step_result.get('status', 'unknown')
            if status == 'error':
                return 'error'
            elif status in ['completed', 'completed_with_warnings']:
                statuses.append(status)
            else:
                return 'pending'
                
        if all(status == 'completed' for status in statuses):
            return 'completed'
        else:
            return 'completed_with_warnings'
            
    def generate_installation_summary(self, steps: Dict[str, Any]) -> Dict[str, Any]:
        """Generate installation summary"""
        summary = {
            'steps_completed': 0,
            'steps_with_warnings': 0,
            'dependencies_installed': 0,
            'configs_created': 0,
            'examples_created': 0
        }
        
        for step_name, step_result in steps.items():
            status = step_result.get('status', 'unknown')
            
            if status == 'completed':
                summary['steps_completed'] += 1
            elif status == 'completed_with_warnings':
                summary['steps_completed'] += 1
                summary['steps_with_warnings'] += 1
                
        # Extract counts from steps
        if 'dependencies' in steps:
            deps = steps['dependencies']
            summary['dependencies_installed'] = len(deps.get('installed_packages', []))
            
        if 'templates' in steps:
            templates = steps['templates']
            summary['configs_created'] = len(templates.get('created_files', []))
            
        if 'examples' in steps:
            examples = steps['examples']
            summary['examples_created'] = len(examples.get('created_files', []))
            
        return summary


async def main():
    """Run provider system installation"""
    
    # Setup logging
    logger.add("installation.log", level="INFO")
    
    print("🚀 LLM Provider System Installation")
    print("=" * 50)
    print("This script will install and configure the expanded LLM provider system")
    print("including support for OpenAI, Anthropic, Ollama, LocalAI, vLLM, and more.")
    print()
    
    installer = ProviderSystemInstaller()
    
    try:
        # Run installation
        results = await installer.run_installation()
        
        # Display results
        print("\n📊 Installation Results:")
        print("=" * 30)
        
        overall_status = results.get('overall_status', 'unknown')
        status_emoji = {
            'completed': '✅',
            'completed_with_warnings': '⚠️',
            'error': '❌',
            'pending': '🔄'
        }.get(overall_status, '❓')
        
        print(f"{status_emoji} Overall Status: {overall_status.upper()}")
        
        # Step results
        for step_name, step_result in results.get('steps', {}).items():
            step_status = step_result.get('status', 'unknown')
            status_emoji = {
                'completed': '✅',
                'completed_with_warnings': '⚠️',
                'error': '❌',
                'running': '🔄'
            }.get(step_status, '❓')
            
            print(f"{status_emoji} {step_name.replace('_', ' ').title()}: {step_status}")
            
        # Summary
        summary = results.get('summary', {})
        print(f"\n📈 Summary:")
        print(f"   Steps completed: {summary.get('steps_completed', 0)}")
        print(f"   Dependencies installed: {summary.get('dependencies_installed', 0)}")
        print(f"   Configs created: {summary.get('configs_created', 0)}")
        print(f"   Examples created: {summary.get('examples_created', 0)}")
        
        if results.get('error'):
            print(f"\n⚠️ Error: {results['error']}")
        
        # Next steps
        print(f"\n📋 Next Steps:")
        print("   1. Copy a configuration template from ai/providers/examples/configs/")
        print("   2. Add your API keys to environment variables")
        print("   3. Setup local providers using scripts in scripts/local_providers/")
        print("   4. Test the system with: python -m ai.providers.validate_system")
        print("   5. Check example applications in ai/providers/examples/applications/")
        
        # Final status
        if overall_status in ['completed', 'completed_with_warnings']:
            print(f"\n🎉 Provider System Installation {'Completed!' if overall_status == 'completed' else 'Completed with Warnings!'}")
            return 0
        else:
            print(f"\n❌ Provider System Installation Failed")
            return 1
            
    except Exception as e:
        logger.error(f"Installation script failed: {e}")
        print(f"\n💥 Installation script failed: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)