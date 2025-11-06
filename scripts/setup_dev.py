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
        
        print("✅ Development environment setup completed successfully!")
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