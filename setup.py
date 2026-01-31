#!/usr/bin/env python3
"""
Setup script for Autonomous AI Agent
Installs dependencies and prepares the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python dependencies"""
    print("Installing dependencies...")
    
    requirements = [
        "aiohttp>=3.9.0",
        "colorlog>=6.7.0",
        "pyinstaller>=5.13.0"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Warning: Failed to install {package}")
            print(result.stderr)
        else:
            print(f"Successfully installed {package}")

def create_directories():
    """Create required directories"""
    directories = ["workspace", "data", "logs", "config"]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("  Autonomous AI Agent - Setup Script")
    print("=" * 60)
    print()
    
    # Install dependencies
    install_requirements()
    
    # Create directories
    create_directories()
    
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run 'build.bat' to build the Windows executable")
    print("2. Or run 'python main.py' to start immediately")
    print()
    print("Don't forget to configure your API keys in Settings!")

if __name__ == "__main__":
    main()
