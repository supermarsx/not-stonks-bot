#!/usr/bin/env python3
"""
Day Trading Orchestrator Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="day-trading-orchestrator",
    version="1.0.0",
    author="Trading Orchestrator Team",
    author_email="team@trading-orchestrator.com",
    description="AI-Powered Multi-Broker Day Trading Orchestrator with Matrix Terminal Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trading-orchestrator/day-trading-orchestrator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1", 
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.0",
            "isort>=5.13.2",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "ibkr": ["ibapi>=10.19.3", "ib-insync==0.9.86"],
        "ai": ["openai>=1.3.7", "anthropic>=0.7.7", "langchain>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "trading-orchestrator=main:main",
            "trading-orchestrator-demo=ui.demo:run_demo",
            "trading-orchestrator-config=main:create_default_config",
        ],
    },
    include_package_data=True,
    package_data={
        "ui": ["themes/*", "templates/*"],
        "ai": ["prompts/*", "schemas/*"],
        "docs": ["*"],
    },
    keywords="trading, day-trading, ai, broker-integration, matrix-terminal, risk-management",
    project_urls={
        "Bug Reports": "https://github.com/trading-orchestrator/day-trading-orchestrator/issues",
        "Source": "https://github.com/trading-orchestrator/day-trading-orchestrator",
        "Documentation": "https://trading-orchestrator.readthedocs.io/",
    },
)