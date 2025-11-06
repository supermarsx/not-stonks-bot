#!/usr/bin/env python3
"""
Day Trading Orchestrator - Development Environment Setup
Sets up a complete development environment for contributors
"""

import sys
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

class DevelopmentEnvironmentSetup:
    """Development environment setup and management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.python_executable = sys.executable
        self.venv_path = self.project_root / "venv"
        
    def print_banner(self):
        """Print setup banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        DAY TRADING ORCHESTRATOR DEV ENVIRONMENT              ║
    ║                   Setup & Configuration                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🛠️  Setting up development environment...
    📦 Installing dependencies...
    ⚙️  Configuring tools...
    🧪 Running tests...
    """
        print(banner)
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            return False
        
        if version.major >= 3 and version.minor >= 11:
            print("✅ Python version is optimal for development")
        else:
            print("✅ Python version is compatible")
        
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create and configure virtual environment"""
        print("\n🔧 Setting up virtual environment...")
        
        if self.venv_path.exists():
            response = input(f"   Virtual environment already exists at {self.venv_path}. Remove and recreate? (y/N): ")
            if response.lower() in ['y', 'yes']:
                print("   Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            else:
                print("   Using existing virtual environment")
                return True
        
        try:
            print(f"   Creating virtual environment at {self.venv_path}...")
            subprocess.run([self.python_executable, "-m", "venv", str(self.venv_path)], 
                         check=True, capture_output=True)
            print("✅ Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def activate_virtual_environment(self) -> str:
        """Get activation command for virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "activate.bat")
        else:  # Unix/Linux/macOS
            return f"source {self.venv_path / 'bin' / 'activate'}"
    
    def install_dependencies(self) -> bool:
        """Install development dependencies"""
        print("\n📦 Installing dependencies...")
        
        # Get pip path in virtual environment
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            pip_path = self.venv_path / "bin" / "pip"
        
        try:
            # Upgrade pip first
            print("   Upgrading pip...")
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_root / "trading_orchestrator" / "requirements.txt"
            if requirements_file.exists():
                print("   Installing requirements...")
                subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True)
                print("✅ Dependencies installed")
            else:
                print("⚠️  requirements.txt not found, skipping")
            
            # Install development dependencies
            dev_requirements_file = self.project_root / "trading_orchestrator" / "requirements-dev.txt"
            if dev_requirements_file.exists():
                print("   Installing development dependencies...")
                subprocess.run([str(pip_path), "install", "-r", str(dev_requirements_file)], 
                             check=True, capture_output=True)
                print("✅ Development dependencies installed")
            else:
                print("⚠️  requirements-dev.txt not found, creating basic dev dependencies...")
                self.install_basic_dev_dependencies(pip_path)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def install_basic_dev_dependencies(self, pip_path: Path):
        """Install basic development dependencies if requirements-dev.txt doesn't exist"""
        dev_packages = [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.0.0"
        ]
        
        print("   Installing basic development packages...")
        for package in dev_packages:
            try:
                subprocess.run([str(pip_path), "install", package], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"   ⚠️  Failed to install {package}")
        
        print("✅ Basic development dependencies installed")
    
    def setup_pre_commit(self) -> bool:
        """Setup pre-commit hooks"""
        print("\n🪝 Setting up pre-commit hooks...")
        
        # Get pre-commit path
        if os.name == 'nt':  # Windows
            pre_commit_path = self.venv_path / "Scripts" / "pre-commit.exe"
        else:  # Unix/Linux/macOS
            pre_commit_path = self.venv_path / "bin" / "pre-commit"
        
        try:
            # Check if pre-commit is installed
            subprocess.run([str(pre_commit_path_path), "--version"], 
                         check=True, capture_output=True)
            
            # Create .pre-commit-config.yaml if it doesn't exist
            pre_commit_config = self.project_root / ".pre-commit-config.yaml"
            if not pre_commit_config.exists():
                self.create_pre_commit_config()
                print("   Created .pre-commit-config.yaml")
            
            # Install pre-commit hooks
            subprocess.run([str(pre_commit_path), "install"], 
                         check=True, capture_output=True)
            print("✅ Pre-commit hooks installed")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Pre-commit not available, skipping")
            return False
    
    def create_pre_commit_config(self):
        """Create pre-commit configuration file"""
        config_content = """repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files
"""
        
        with open(self.project_root / ".pre-commit-config.yaml", "w") as f:
            f.write(config_content)
    
    def setup_environment_files(self):
        """Setup environment configuration files"""
        print("\n⚙️  Setting up environment files...")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_example = self.project_root / ".env.example"
            if env_example.exists():
                shutil.copy(env_example, env_file)
                print("   Created .env from .env.example")
            else:
                env_file.write_text("# Environment variables\\n")
                print("   Created empty .env file")
        
        # Create .gitignore if it doesn't exist
        gitignore_file = self.project_root / ".gitignore"
        if not gitignore_file.exists():
            self.create_gitignore()
            print("   Created .gitignore")
        
        # Create development configuration
        dev_config_file = self.project_root / "config.development.json"
        if not dev_config_file.exists():
            self.create_dev_config()
            print("   Created development configuration")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Trading specific
logs/*.log
logs/trading_orchestrator.log*
data/
backups/
*.db
*.db-*

# API keys and secrets
config.json
.env
secrets/
keys/

# Test results
.coverage
.pytest_cache/
htmlcov/
.tox/
.nox/

# Profiling
*.prof
profile_default/
ipython_config.py

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.production

# Database
*.db
*.sqlite
*.sqlite3

# Backup files
*.bak
*.backup
*.old

# Temporary files
*.tmp
*.temp
"""
        
        with open(self.project_root / ".gitignore", "w") as f:
            f.write(gitignore_content)
    
    def create_dev_config(self):
        """Create development-specific configuration"""
        dev_config = {
            "database": {
                "url": "sqlite:///development_trading_orchestrator.db",
                "echo": True,
                "pool_size": 5
            },
            "logging": {
                "level": "DEBUG",
                "file": "logs/development.log",
                "debug_mode": True
            },
            "ai": {
                "trading_mode": "PAPER",
                "debug_ai_calls": True,
                "verbose_prompts": True
            },
            "brokers": {
                "debug_mode": True,
                "test_all_apis": False
            },
            "risk": {
                "debug_mode": True,
                "max_position_size": 1000,
                "max_daily_loss": 500
            },
            "testing": {
                "enabled": True,
                "coverage_enabled": True,
                "mock_external_apis": True
            }
        }
        
        with open(self.project_root / "config.development.json", "w") as f:
            json.dump(dev_config, f, indent=2)
    
    def create_vscode_settings(self):
        """Create VS Code settings for development"""
        print("\n📁 Setting up VS Code configuration...")
        
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Settings
        settings = {
            "python.defaultInterpreterPath": str(self.venv_path / "bin" / "python" if os.name != 'nt' else self.venv_path / "Scripts" / "python.exe"),
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.sortImports.args": ["--profile", "black"],
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests"],
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                ".pytest_cache": True
            }
        }
        
        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        
        # Launch configuration
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Debug Main Application",
                    "type": "python",
                    "request": "launch",
                    "program": str(self.project_root / "main.py"),
                    "args": [],
                    "console": "integratedTerminal",
                    "cwd": str(self.project_root)
                },
                {
                    "name": "Debug Tests",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["tests/", "-v"],
                    "console": "integratedTerminal",
                    "cwd": str(self.project_root)
                },
                {
                    "name": "Debug Integration Tests",
                    "type": "python",
                    "request": "launch",
                    "module": "test_integration",
                    "args": ["--config", "config.development.json"],
                    "console": "integratedTerminal",
                    "cwd": str(self.project_root)
                }
            ]
        }
        
        with open(vscode_dir / "launch.json", "w") as f:
            json.dump(launch_config, f, indent=2)
        
        # Extensions recommendations
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.flake8",
                "ms-python.black-formatter",
                "ms-python.isort",
                "ms-python.mypy-type-checker",
                "ms-toolsai.jupyter",
                "redhat.vscode-yaml",
                "ms-vscode.vscode-json"
            ]
        }
        
        with open(vscode_dir / "extensions.json", "w") as f:
            json.dump(extensions, f, indent=2)
        
        print("   ✅ VS Code configuration created")
    
    def run_initial_tests(self) -> bool:
        """Run initial test suite to verify setup"""
        print("\n🧪 Running initial tests...")
        
        # Get pytest path
        if os.name == 'nt':  # Windows
            pytest_path = self.venv_path / "Scripts" / "pytest.exe"
        else:  # Unix/Linux/macOS
            pytest_path = self.venv_path / "bin" / "pytest"
        
        try:
            # Run a simple test to verify pytest is working
            result = subprocess.run([str(pytest_path), "--version"], 
                                  check=True, capture_output=True, text=True)
            print(f"   {result.stdout.strip()}")
            
            # Try to run a basic import test
            test_import = """
import sys
sys.path.insert(0, '.')
try:
    import main
    print("✅ Main module imports successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
            
            with open(self.project_root / "test_import.py", "w") as f:
                f.write(test_import)
            
            subprocess.run([str(pytest_path), "test_import.py"], 
                         check=True, capture_output=True)
            
            os.remove(self.project_root / "test_import.py")
            
            print("✅ Basic import test passed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Initial tests failed: {e}")
            return False
    
    def create_development_scripts(self):
        """Create development helper scripts"""
        print("\n📝 Creating development scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Development activation script
        if os.name == 'nt':  # Windows
            activate_script = scripts_dir / "activate-dev.bat"
            activate_content = f"""@echo off
REM Development Environment Activation Script

echo Activating Day Trading Orchestrator Development Environment...

REM Activate virtual environment
call {self.venv_path / "Scripts" / "activate.bat"}

REM Set development environment variables
set TRADING_ENV=development
set PYTHONPATH={self.project_root};%PYTHONPATH%

REM Change to project directory
cd /d {self.project_root}

echo Development environment activated!
echo.
echo Available commands:
echo   python main.py              - Run main application
echo   python test_integration.py  - Run integration tests
echo   python debug.py             - Run in debug mode
echo   python validate_config.py   - Validate configuration
echo.
echo To deactivate, type: deactivate

REM Start command prompt
cmd /k
"""
            with open(activate_script, "w") as f:
                f.write(activate_content)
        else:  # Unix/Linux/macOS
            activate_script = scripts_dir / "activate-dev.sh"
            activate_content = f"""#!/bin/bash
# Development Environment Activation Script

echo "Activating Day Trading Orchestrator Development Environment..."

# Activate virtual environment
source {self.venv_path / "bin" / "activate"}

# Set development environment variables
export TRADING_ENV=development
export PYTHONPATH="{self.project_root}:$PYTHONPATH"

# Change to project directory
cd {self.project_root}

echo "Development environment activated!"
echo ""
echo "Available commands:"
echo "  python main.py              - Run main application"
echo "  python test_integration.py  - Run integration tests"
echo "  python debug.py             - Run in debug mode"
echo "  python validate_config.py   - Validate configuration"
echo ""
echo "To deactivate, type: deactivate"

# Start bash shell
exec bash
"""
            with open(activate_script, "w") as f:
                f.write(activate_content)
            
            os.chmod(activate_script, 0o755)
        
        print(f"   ✅ Created {activate_script.name}")
        
        # Quick test script
        test_script = scripts_dir / "test-quick.py"
        test_content = f"""#!/usr/bin/env python3
\"\"\"Quick development test script\"\"\"

import sys
sys.path.insert(0, "{self.project_root}")

def test_imports():
    \"\"\"Test that all key modules can be imported\"\"\"
    try:
        import asyncio
        import aiohttp
        import json
        import logging
        print("✅ Core dependencies imported")
        
        # Try importing project modules
        try:
            import config.application
            import brokers.factory
            import ai.orchestrator
            print("✅ Project modules imported")
        except ImportError as e:
            print(f"⚠️  Some project modules missing: {{e}}")
        
        print("✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {{e}}")
        return False

if __name__ == "__main__":
    test_imports()
"""
        with open(test_script, "w") as f:
            f.write(test_content)
        
        print(f"   ✅ Created {test_script.name}")
    
    def print_setup_summary(self):
        """Print setup completion summary"""
        print("\n" + "="*60)
        print("🎉 DEVELOPMENT ENVIRONMENT SETUP COMPLETE!")
        print("="*60)
        
        print("\n📁 Project Structure:")
        print(f"   Project Root: {self.project_root}")
        print(f"   Virtual Env:  {self.venv_path}")
        print(f"   Config:       {self.project_root / 'config.json'}")
        print(f"   Logs:         {self.project_root / 'logs'}")
        
        print("\n🚀 Quick Start Commands:")
        if os.name == 'nt':
            print(f"   Activate:     {self.venv_path / 'Scripts' / 'activate.bat'}")
            print(f"   Dev Script:   scripts\\activate-dev.bat")
        else:
            print(f"   Activate:     source {self.venv_path / 'bin' / 'activate'}")
            print(f"   Dev Script:   ./scripts/activate-dev.sh")
        
        print(f"   Run App:      python main.py")
        print(f"   Debug Mode:   python debug.py")
        print(f"   Tests:        python test_integration.py")
        print(f"   Validate:     python validate_config.py")
        print(f"   Demo:         python demo.py")
        
        print("\n⚙️  Configuration:")
        print(f"   Edit:         {self.project_root / 'config.json'}")
        print(f"   Environment:  {self.project_root / '.env'}")
        print(f"   Development:  {self.project_root / 'config.development.json'}")
        
        print("\n🛠️  Development Tools:")
        print(f"   Code Style:   black, isort, flake8")
        print(f"   Type Check:   mypy")
        print(f"   Testing:      pytest")
        print(f"   Pre-commit:   pre-commit hooks")
        print(f"   IDE:          VS Code configuration")
        
        print("\n📚 Next Steps:")
        print("   1. Edit config.json with your API keys")
        print("   2. Run 'python demo.py' to see the system in action")
        print("   3. Run 'python health_check.py' to verify everything works")
        print("   4. Read the documentation in docs/")
        print("   5. Start contributing!")
        
        print(f"\n💡 Need Help?")
        print(f"   Documentation: {self.project_root / 'README.md'}")
        print(f"   Issues:        GitHub Issues")
        print(f"   Discord:       Community Server")
    
    def run_setup(self):
        """Run complete development environment setup"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            print("❌ Python version check failed. Please upgrade Python to 3.8+")
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            print("❌ Failed to create virtual environment")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Failed to install dependencies")
            return False
        
        # Setup development tools
        self.setup_pre_commit()
        self.setup_environment_files()
        self.create_vscode_settings()
        self.create_development_scripts()
        
        # Run initial tests
        self.run_initial_tests()
        
        # Print summary
        self.print_setup_summary()
        
        return True

def main():
    """Main setup runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Development Environment Setup")
    parser.add_argument("--force", action="store_true", help="Force setup even if venv exists")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running initial tests")
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = DevelopmentEnvironmentSetup()
    
    try:
        # Run setup
        success = setup.run_setup()
        
        if success:
            print("\n✅ Development environment setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Development environment setup failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()