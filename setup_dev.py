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
            
            # Install basic development packages
            dev_packages = [
                "pytest>=7.0.0",
                "pytest-asyncio>=0.21.0",
                "black>=23.0.0",
                "isort>=5.12.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0"
            ]
            
            print("   Installing development packages...")
            for package in dev_packages:
                try:
                    subprocess.run([str(pip_path), "install", package], 
                                 check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    print(f"   ⚠️  Failed to install {package}")
            
            print("✅ Development dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def print_setup_summary(self):
        """Print setup completion summary"""
        print("\n" + "="*60)
        print("🎉 DEVELOPMENT ENVIRONMENT SETUP COMPLETE!")
        print("="*60)
        
        print("\n🚀 Quick Start Commands:")
        if os.name == 'nt':
            print(f"   Activate:     {self.venv_path / 'Scripts' / 'activate.bat'}")
        else:
            print(f"   Activate:     source {self.venv_path / 'bin' / 'activate'}")
        
        print(f"   Run App:      python main.py")
        print(f"   Debug Mode:   python debug.py")
        print(f"   Demo:         python demo.py")
        
        print("\n📚 Next Steps:")
        print("   1. Run 'python demo.py' to see the system in action")
        print("   2. Read the documentation in docs/")
        print("   3. Start contributing!")
    
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
        
        # Print summary
        self.print_setup_summary()
        
        return True

def main():
    """Main setup runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Development Environment Setup")
    parser.add_argument("--force", action="store_true", help="Force setup even if venv exists")
    
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
        sys.exit(1)

if __name__ == "__main__":
    main()