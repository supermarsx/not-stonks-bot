# Local Development Setup Guide
## Day Trading Orchestrator - Complete Development Environment

<div align="center">

![Development](https://img.shields.io/badge/Development-v1.0.0-green?style=for-the-badge&logo=code)
[![VS Code](https://img.shields.io/badge/VS%20Code-Ready-blue.svg)](https://code.visualstudio.com/)
[![PyCharm](https://img.shields.io/badge/PyCharm-Ready-blue.svg)](https://www.jetbrains.com/pycharm/)
[![Debug](https://img.shields.io/badge/Debug-Setup-orange.svg)](https://docs.python.org/3/library/pdb.html)

**Professional Development Environment Setup**

[🚀 Quick Start](#-quick-start) • [📦 Requirements](#️-requirements) • [🛠️ IDE Setup](#️-ide-setup) • [🧪 Testing](#-testing) • [🐛 Debugging](#-debugging)

</div>

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [System Requirements](#-system-requirements)
3. [Development Environment Setup](#-development-environment-setup)
4. [IDE Configuration](#️-ide-configuration)
5. [Debugging Setup](#️-debugging-setup)
6. [Testing Environment](#-testing-environment)
7. [Code Quality Tools](#️-code-quality-tools)
8. [Git Workflow](#️-git-workflow)
9. [Performance Profiling](#️-performance-profiling)
10. [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Automated Development Setup

```bash
# Clone repository
git clone https://github.com/trading-orchestrator/day-trading-orchestrator.git
cd day-trading-orchestrator

# Run development setup script
python setup_dev.py

# This will:
# - Check system requirements
# - Create virtual environment
# - Install dependencies
# - Set up IDE configuration
# - Configure debugging
# - Initialize database
# - Create sample data

# Start development environment
./start.sh dev

# Open in your IDE
code .  # VS Code
# or
pycharm .  # PyCharm
```

### Manual Setup

```bash
# 1. Clone and setup virtual environment
git clone https://github.com/trading-orchestrator/day-trading-orchestrator.git
cd day-trading-orchestrator
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Setup pre-commit hooks
pre-commit install

# 4. Start development server
python main.py --dev

# 5. Run tests
python -m pytest tests/ -v
```

## 🔧 System Requirements

### Minimum Requirements

| Component | Windows | macOS | Linux |
|-----------|---------|-------|-------|
| **OS** | Windows 10+ | macOS 11.0+ | Ubuntu 20.04+ |
| **CPU** | Intel i5 / AMD Ryzen 5 | Apple M1 / Intel i5 | Intel i5 / AMD Ryzen 5 |
| **RAM** | 8 GB | 8 GB | 8 GB |
| **Storage** | 20 GB | 20 GB | 20 GB |
| **Python** | 3.11+ | 3.11+ | 3.11+ |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel i7 / AMD Ryzen 7 / Apple M1 Pro |
| **RAM** | 16 GB or more |
| **Storage** | 50 GB SSD |
| **Python** | 3.11+ (3.12 recommended) |
| **IDE** | VS Code or PyCharm Professional |
| **Git** | 2.40+ |

### Development Tools

```bash
# Essential development tools
├── Python 3.11+
├── Git 2.40+
├── Node.js 18+ (for frontend)
├── Visual Studio Code / PyCharm
├── Docker Desktop (optional)
├── Postman (for API testing)
├── MongoDB Compass (for database inspection)
├── Redis Desktop Manager (for cache inspection)
└── GitHub CLI (optional)
```

## 🛠️ Development Environment Setup

### Python Virtual Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Create requirements file for your IDE
pip freeze > requirements-dev-freeze.txt
```

#### Using Poetry (Alternative)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize project
poetry init
poetry add fastapi uvicorn sqlalchemy alembic
poetry add --group dev pytest black isort mypy

# Install and activate environment
poetry install
poetry shell
```

#### Using Conda (Data Science Focus)

```bash
# Create conda environment
conda create -n trading-orchestrator python=3.11
conda activate trading-orchestrator

# Install dependencies
conda install -c conda-forge pandas numpy scikit-learn
pip install -r requirements.txt
```

### Project Structure

```
day-trading-orchestrator/
├── .vscode/                    # VS Code configuration
│   ├── settings.json
│   ├── launch.json
│   └── extensions.json
├── .idea/                      # PyCharm configuration
├── docs/                       # Documentation
├── tests/                      # Test files
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── trading_orchestrator/       # Main application
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   ├── database/
│   ├── models/
│   └── utils/
├── scripts/                    # Development scripts
│   ├── setup_dev.py
│   ├── run_tests.py
│   └── lint_fix.py
├── frontend/                   # Frontend (if applicable)
│   ├── src/
│   ├── public/
│   └── package.json
├── config/                     # Configuration files
│   ├── development.yaml
│   └── testing.yaml
├── data/                       # Development data
│   ├── logs/
│   ├── cache/
│   └── test_data/
├── .env.example                # Environment template
├── .gitignore
├── .pre-commit-config.yaml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

### Environment Variables

```bash
# .env - Development environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=sqlite:///data/dev_database.db
# For PostgreSQL: postgresql://dev_user:dev_password@localhost:5432/trading_orchestrator_dev

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# API Keys (development/test keys)
OPENAI_API_KEY=sk-test-your-test-key
ANTHROPIC_API_KEY=sk-ant-test-your-test-key

# Broker configurations (test accounts)
ALPACA_API_KEY=your_test_api_key
ALPACA_SECRET_KEY=your_test_secret_key
ALPACA_PAPER=true

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET=your-dev-jwt-secret
ENABLE_CORS=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001

# Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## 🏗️ IDE Configuration

### Visual Studio Code

#### Extensions

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.pylint",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.mypy-type-checker",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "ms-vscode-remote.remote-containers",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-toolsai.jupyter",
    "ms-vscode.test-adapter-converter",
    "littlefoxteam.vscode-python-test-adapter",
    "formulahendry.code-runner",
    "christian-kohler.path-intellisense",
    "ms-vscode.vscode-github-actions",
    "github.vscode-pull-request-github"
  ]
}
```

#### Settings

```json
// .vscode/settings.json
{
  // Python configuration
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.envFile": "${workspaceFolder}/.env",
  "python.terminal.activateEnvironment": true,
  "python.terminal.executeInFileDir": false,

  // Code formatting
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,

  // Linting
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.lintOnSave": true,

  // Import sorting
  "python.sortImports.args": ["--profile", "black"],
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },

  // Testing
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.testing.unittestEnabled": false,

  // Auto-completion
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.autoSearchPaths": true,

  // Git configuration
  "git.ignoreLimitWarning": true,
  "git.autofetch": true,
  "git.enableSmartCommit": true,

  // File associations
  "files.associations": {
    "*.yaml": "yaml",
    "*.yml": "yaml",
    "*.env": "dotenv",
    "*.json": "json",
    "*.toml": "toml"
  },

  // Exclude files
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/venv": true,
    "**/.pytest_cache": true,
    "**/node_modules": true,
    "**/dist": true,
    "**/build": true,
    "**/.mypy_cache": true,
    "**/*.egg-info": true
  },

  // Terminal
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "${workspaceFolder}"
  },

  // Search
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/venv": true
  },

  // Debugging
  "python.analysis.indexing": true,
  "python.analysis.packageIndexDepth": 2,

  // Jupyter
  "jupyter.askForKernelRestart": false,
  "jupyter.sendSelectionToInteractiveWindow": true,

  // Code Runner
  "code-runner.executorMap": {
    "python": "cd $dir && python $fileName"
  },
  "code-runner.runInTerminal": true
}
```

#### Debug Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Trading Orchestrator",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/trading_orchestrator/main.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "ENVIRONMENT": "development",
        "DEBUG": "true"
      },
      "args": ["--dev"],
      "stopOnEntry": false,
      "justMyCode": true
    },
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": ["-v", "--cov=trading_orchestrator", "--cov-report=html"],
      "justMyCode": false
    },
    {
      "name": "Debug Specific Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": ["-v", "${file}"],
      "justMyCode": false
    },
    {
      "name": "Debug with Arguments",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/trading_orchestrator/main.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": ["--dev", "--config", "${workspaceFolder}/config/development.yaml"],
      "justMyCode": true
    },
    {
      "name": "Attach to Process",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    },
    {
      "name": "Debug Flask App",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/frontend/server.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}/frontend",
      "env": {
        "FLASK_ENV": "development",
        "FLASK_DEBUG": "1"
      }
    }
  ]
}
```

#### Tasks Configuration

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Setup Development Environment",
      "type": "shell",
      "command": "python",
      "args": ["setup_dev.py"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "python",
      "args": ["-m", "pytest", "tests/", "-v", "--cov=trading_orchestrator"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Run Linting",
      "type": "shell",
      "command": "python",
      "args": ["-m", "flake8", "trading_orchestrator/"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Format Code",
      "type": "shell",
      "command": "python",
      "args": ["-m", "black", "trading_orchestrator/", "tests/"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Start Development Server",
      "type": "shell",
      "command": "python",
      "args": ["main.py", "--dev"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "new"
      },
      "isBackground": true
    },
    {
      "label": "Run Type Checking",
      "type": "shell",
      "command": "python",
      "args": ["-m", "mypy", "trading_orchestrator/"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

### PyCharm Professional

#### Project Configuration

```python
# PyCharm Configuration
# File > Settings > Project > Python Interpreter
# Set interpreter to: ./venv/bin/python

# Enable code inspection
# File > Settings > Editor > Inspections
# Enable: Python, Spelling, SQL
```

#### Run/Debug Configurations

```xml
<!-- .idea/runConfigurations/Trading_Orchestrator.xml -->
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="Trading Orchestrator" type="PythonConfigurationType" factoryName="Python">
    <module name="trading-orchestrator" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONPATH" value="$PROJECT_DIR$" />
      <env name="ENVIRONMENT" value="development" />
      <env name="DEBUG" value="true" />
    </envs>
    <option name="SDK_HOME" value="" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="true" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME" value="trading_orchestrator/main.py" />
    <option name="PARAMETERS" value="--dev" />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="EMULATE_TERMINAL" value="true" />
    <option name="MODULE_MODE" value="false" />
    <option name="REDIRECT_INPUT" value="false" />
    <option name="INPUT_FILE" value="" />
    <method v="2" />
  </configuration>
</component>

<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="Run Tests" type="tests" factoryName="py.test">
    <module name="trading-orchestrator" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs />
    <option name="SDK_HOME" value="" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="true" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="_new_keywords" value="&quot;&quot;" />
    <option name="_new_parameters" value="&quot;&quot;" />
    <option name="_new_additionalArguments" value="&quot;-v --cov=trading_orchestrator&quot;" />
    <option name="_new_target" value="&quot;$PROJECT_DIR$/tests&quot;" />
    <option name="_new_targetType" value="&quot;PATH&quot;" />
    <method v="2" />
  </configuration>
</component>
```

#### Code Style Settings

```xml
<!-- .idea/codeStyles/Project.xml -->
<component name="ProjectCodeStyleConfiguration">
  <code_scheme name="Project" version="173">
    <Python>
      <option name="OPTIMIZE_IMPORTS_SORT_NAMES_IN_FROM_IMPORTS" value="true" />
      <option name="OPTIMIZE_IMPORTS_JOIN_FROM_IMPORTS_WITH_SAME_SOURCE" value="true" />
      <option name="OPTIMIZE_IMPORTS_ALWAYS_SPLIT_FROM_IMPORTS" value="true" />
    </Python>
    <codeStyleSettings language="Python">
      <option name="RIGHT_MARGIN" value="88" />
      <option name="ALIGN_MULTILINE_PARAMETERS_IN_CALLS" value="true" />
      <option name="ALIGN_MULTILINE_BINARY_OPERATION" value="false" />
      <option name="ALIGN_MULTILINE_ASSIGNMENT" value="true" />
      <option name="ALIGN_MULTILINE_FOR" value="true" />
      <option name="SPACE_AFTER_PY_COMMA" value="true" />
      <option name="SPACE_BEFORE_PY_COLON" value="false" />
      <option name="SPACE_AROUND_EQ_IN_NAMED_PARAMETER" value="false" />
      <option name="SPACE_AROUND_EQ_IN_KEYWORD_ARGUMENT" value="false" />
      <option name="BINARY_OPERATION_SIGN_ON_NEXT_LINE" value="true" />
      <option name="FOR_STATEMENT_LPAREN_ON_NEXT_LINE" value="true" />
      <option name="FOR_STATEMENT_RPAREN_ON_NEXT_LINE" value="true" />
      <option name="WHILE_STATEMENT_LPAREN_ON_NEXT_LINE" value="true" />
      <option name="WHILE_STATEMENT_RPAREN_ON_NEXT_LINE" value="true" />
      <option name="IF_STATEMENT_LPAREN_ON_NEXT_LINE" value="true" />
      <option name="IF_STATEMENT_RPAREN_ON_NEXT_LINE" value="true" />
    </codeStyleSettings>
  </code_scheme>
</component>
```

## 🐛 Debugging Setup

### Python Debugging

#### Using pdb (Built-in)

```python
# Trading orchestrator main.py
import pdb

def main():
    config = load_config()
    
    # Set breakpoint
    pdb.set_trace()  # Debugger will stop here
    
    # Continue execution
    # l (list) - show current code
    # n (next) - next line
    # s (step) - step into function
    # c (continue) - continue execution
    # p <var> - print variable
    # pp <var> - pretty print variable
    # q (quit) - quit debugger
    
    start_application(config)

if __name__ == "__main__":
    main()
```

#### Using ipdb (Enhanced pdb)

```python
# Install ipdb
pip install ipdb

# Usage
import ipdb

def complex_function():
    result = calculate_something()
    ipdb.set_trace()  # Enhanced debugger with syntax highlighting
    return result
```

#### Using Debugpy (VS Code Remote Debugging)

```python
# Install debugpy
pip install debugpy

# trading_orchestrator/main.py
import debugpy

# Listen on port 5678 (default)
debugpy.listen(('localhost', 5678))
print("Waiting for debugger attach")
debugpy.wait_for_client()  # Blocks execution until debugger attaches

def main():
    # Your code here
    pass
```

### VS Code Debugging

#### Remote Attach Configuration

```json
// .vscode/launch.json - Remote debugging
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

### Advanced Debugging Techniques

#### Debug Trading Logic

```python
# trading_orchestrator/debug_trading.py
import logging
from functools import wraps

def debug_trading(func):
    """Decorator to debug trading functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call
        logger = logging.getLogger(__name__)
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        
        # Set breakpoint for debugging
        import pdb
        pdb.set_trace()
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper

# Usage
@debug_trading
def execute_trade(symbol: str, quantity: int, side: str):
    # Your trading logic here
    return {"status": "executed", "price": 100.50}
```

#### Debug Database Operations

```python
# trading_orchestrator/database/debug.py
from sqlalchemy import event
from sqlalchemy.engine import Engine
import logging

# Enable SQL query logging
@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    logging.debug("SQL Query: %s", statement)
    logging.debug("Parameters: %s", parameters)

# Debug ORM queries
def debug_query(session):
    """Debug database queries in session"""
    for query in session.query_logs:
        logging.debug("Generated SQL: %s", query.statement)
        logging.debug("Parameters: %s", query.parameters)
```

#### Debug API Calls

```python
# trading_orchestrator/api/debug.py
import requests
import logging

def debug_api_call(func):
    """Decorator to debug API calls"""
    def wrapper(*args, **kwargs):
        logging.debug(f"API Call: {func.__name__}")
        logging.debug(f"Args: {args}")
        logging.debug(f"Kwargs: {kwargs}")
        
        response = func(*args, **kwargs)
        
        logging.debug(f"Response Status: {response.status_code}")
        logging.debug(f"Response Headers: {response.headers}")
        logging.debug(f"Response Body: {response.text[:500]}...")
        
        return response
    return wrapper

# Usage
@debug_api_call
def get_market_data(symbol: str):
    return requests.get(f"https://api.example.com/market/{symbol}")
```

### Jupyter Notebook Debugging

```python
# notebooks/debug_analysis.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore data
df = pd.read_csv('data/market_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Visualize data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='price')
plt.title('Price Distribution')

plt.subplot(1, 2, 2)
plt.plot(df['timestamp'], df['price'])
plt.title('Price Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Interactive debugging with ipdb
import ipdb
ipdb.set_trace()  # Set breakpoint and explore data interactively
```

## 🧪 Testing Environment

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trading_orchestrator.database import get_db, Base
from trading_orchestrator.main import app

# Test database
SQLITE_DATABASE_URL = "sqlite:///./test_database.db"
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_db):
    """Create database session"""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    """Create test client"""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def mock_brokers():
    """Mock broker clients"""
    with patch('trading_orchestrator.brokers.alpaca.AlpacaBroker') as mock_alpaca:
        with patch('trading_orchestrator.brokers.binance.BinanceBroker') as mock_binance:
            yield {
                'alpaca': mock_alpaca.return_value,
                'binance': mock_binance.return_value
            }

@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing"""
    return {
        'symbol': 'AAPL',
        'quantity': 10,
        'side': 'buy',
        'order_type': 'market',
        'price': 150.25
    }

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return [
        {'timestamp': '2024-01-01T09:30:00Z', 'symbol': 'AAPL', 'price': 150.25, 'volume': 1000},
        {'timestamp': '2024-01-01T09:31:00Z', 'symbol': 'AAPL', 'price': 150.50, 'volume': 800},
        {'timestamp': '2024-01-01T09:32:00Z', 'symbol': 'AAPL', 'price': 150.75, 'volume': 1200},
    ]

@pytest.fixture
def mock_ai_provider():
    """Mock AI provider"""
    with patch('trading_orchestrator.ai.providers.OpenAIProvider') as mock_openai:
        mock_openai.return_value.generate_response.return_value = {
            'analysis': 'Market shows positive momentum',
            'confidence': 0.85,
            'recommendation': 'BUY'
        }
        yield mock_openai.return_value
```

### Test Types

#### Unit Tests

```python
# tests/unit/test_trading_engine.py
import pytest
from unittest.mock import Mock, patch
from trading_orchestrator.engine import TradingEngine
from trading_orchestrator.risk import RiskManager

class TestTradingEngine:
    
    def test_init(self, mock_brokers):
        """Test TradingEngine initialization"""
        engine = TradingEngine(
            broker_factory=mock_brokers,
            risk_manager=Mock(spec=RiskManager)
        )
        assert len(engine.brokers) == 2
        assert 'alpaca' in engine.brokers
        assert 'binance' in engine.brokers
    
    def test_execute_trade_success(self, client, sample_trade_data, mock_brokers):
        """Test successful trade execution"""
        # Mock successful trade
        mock_brokers['alpaca'].execute_trade.return_value = {
            'status': 'filled',
            'filled_quantity': 10,
            'average_fill_price': 150.25
        }
        
        # Mock risk check passes
        with patch.object(RiskManager, 'check_trade') as mock_risk_check:
            mock_risk_check.return_value = True
            
            engine = TradingEngine(mock_brokers, Mock(spec=RiskManager))
            result = engine.execute_trade(sample_trade_data)
            
            assert result['status'] == 'filled'
            mock_brokers['alpaca'].execute_trade.assert_called_once()
    
    def test_execute_trade_risk_rejection(self, client, sample_trade_data, mock_brokers):
        """Test trade rejection due to risk"""
        # Mock risk check fails
        with patch.object(RiskManager, 'check_trade') as mock_risk_check:
            mock_risk_check.return_value = False
            
            engine = TradingEngine(mock_brokers, Mock(spec=RiskManager))
            result = engine.execute_trade(sample_trade_data)
            
            assert result['status'] == 'rejected'
            assert 'risk' in result['reason'].lower()
    
    @pytest.mark.asyncio
    async def test_async_market_data_fetch(self, sample_market_data):
        """Test asynchronous market data fetching"""
        with patch('trading_orchestrator.market_data.BrokerDataProvider') as mock_provider:
            mock_provider.return_value.get_market_data.return_value = sample_market_data
            
            from trading_orchestrator.market_data import MarketDataManager
            manager = MarketDataManager()
            data = await manager.get_realtime_data('AAPL')
            
            assert len(data) == 3
            assert all(item['symbol'] == 'AAPL' for item in data)
```

#### Integration Tests

```python
# tests/integration/test_full_trading_flow.py
import pytest
from fastapi.testclient import TestClient
from trading_orchestrator.main import app
from trading_orchestrator.database import get_db

class TestFullTradingFlow:
    
    def test_complete_trading_flow(self, client, sample_trade_data):
        """Test complete trading flow from API to execution"""
        # Step 1: Login and get token
        login_response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # Step 2: Create trade order
        headers = {"Authorization": f"Bearer {token}"}
        order_response = client.post("/api/orders", json=sample_trade_data, headers=headers)
        assert order_response.status_code == 201
        order_data = order_response.json()
        
        # Step 3: Check order status
        status_response = client.get(f"/api/orders/{order_data['id']}", headers=headers)
        assert status_response.status_code == 200
        
        # Step 4: Verify order in database
        db_response = client.get("/api/orders", headers=headers)
        assert db_response.status_code == 200
        orders = db_response.json()
        assert len(orders) > 0
    
    def test_error_handling(self, client):
        """Test error handling in trading flow"""
        # Test invalid trade data
        invalid_trade = {
            "symbol": "",
            "quantity": -10,
            "side": "invalid_side"
        }
        
        response = client.post("/api/orders", json=invalid_trade)
        assert response.status_code == 422
        
        error_detail = response.json()["detail"]
        assert any("symbol" in error for error in error_detail)
        assert any("quantity" in error for error in error_detail)
        assert any("side" in error for error in error_detail)
```

#### Performance Tests

```python
# tests/performance/test_performance.py
import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from trading_orchestrator.database import get_db

class TestPerformance:
    
    def test_trade_execution_time(self, client, sample_trade_data):
        """Test trade execution time"""
        start_time = time.time()
        
        response = client.post("/api/orders", json=sample_trade_data)
        
        execution_time = time.time() - start_time
        
        assert response.status_code in [200, 201]
        assert execution_time < 1.0  # Should execute within 1 second
    
    def test_concurrent_trades(self, client, sample_trade_data):
        """Test handling concurrent trades"""
        def execute_trade(trade_data):
            response = client.post("/api/orders", json=trade_data)
            return response.status_code
        
        # Create 10 concurrent trades
        trade_data_list = [sample_trade_data.copy() for _ in range(10)]
        for i, trade_data in enumerate(trade_data_list):
            trade_data['symbol'] = f'SYM{i}'
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(execute_trade, trade_data_list))
        
        # All should succeed
        assert all(status in [200, 201] for status in results)
    
    @pytest.mark.slow
    def test_database_performance(self, client):
        """Test database query performance"""
        start_time = time.time()
        
        # Fetch large dataset
        response = client.get("/api/orders?limit=1000")
        
        query_time = time.time() - start_time
        
        assert response.status_code == 200
        assert query_time < 2.0  # Should query within 2 seconds
```

### Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_trading_engine.py -v

# Run tests with coverage
python -m pytest tests/ -v --cov=trading_orchestrator --cov-report=html

# Run tests in parallel
python -m pytest tests/ -v -n auto

# Run specific test
python -m pytest tests/unit/test_trading_engine.py::TestTradingEngine::test_execute_trade_success -v

# Run tests with markers
python -m pytest tests/ -v -m "not slow"

# Run tests with specific fixtures
python -m pytest tests/ -v --tb=short --maxfail=3

# Generate test report
python -m pytest tests/ --html=reports/test_report.html --self-contained-html
```

## 🔧 Code Quality Tools

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
      - id: check-case-conflict
      - id: check-merge-conflict
      - id: check-executables-have-shebangs
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile, black]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --strict-optional]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, trading_orchestrator/]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: python -m pytest tests/ --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

### Black Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### Flake8 Configuration

```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
    E501,  # line too long (handled by black)
exclude = 
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist,
    *.egg-info,
    .pytest_cache,
    .mypy_cache

per-file-ignores =
    __init__.py:F401
    tests/*:F401,F811

max-complexity = 10
```

### MyPy Configuration

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False

[mypy-trading_orchestrator.external.*]
ignore_missing_imports = True

[mypy-trading_orchestrator.brokers.*]
ignore_errors = True
```

### Bandit Security Configuration

```yaml
# .bandit
# Bandit configuration for security scanning
skips:
  - B101  # Skip assert_used test
  - B601  # Skip paramiko_calls test

exclude_dirs:
  - tests/
  - venv/
  - .venv/

tests:
  - B201
  - B301
  - B302
  - B303
  - B304
  - B305
  - B306
  - B307
  - B308
  - B309
  - B310
  - B311
  - B312
  - B313
  - B314
  - B315
  - B316
  - B317
  - B318
  - B319
  - B320
  - B321
  - B322
  - B323
  - B324
  - B325
```

## 🌿 Git Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/new-trading-strategy

# Development flow
main (production)
├── develop (integration)
│   ├── feature/user-authentication
│   ├── feature/trading-engine-v2
│   └── feature/ai-integration
├── hotfix/critical-bug-fix
└── release/v1.2.0
```

### Commit Messages

```bash
# Conventional commits format
type(scope): description

# Examples:
feat(trading): add mean reversion strategy
fix(database): resolve connection pool timeout
docs(api): update authentication documentation
test(engine): add integration tests for order execution
refactor(risk): simplify position sizing logic
chore(dependencies): update pandas to version 2.1.0
```

### Git Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

set -e

echo "🧹 Running pre-commit checks..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run 'python setup_dev.py' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run code quality checks
echo "🔍 Running flake8..."
flake8 trading_orchestrator/ tests/ --max-line-length=88 --extend-ignore=E203,W503

echo "🎨 Running black..."
black --check trading_orchestrator/ tests/

echo "📦 Running isort..."
isort --check-only trading_orchestrator/ tests/

echo "🔒 Running security scan..."
bandit -r trading_orchestrator/ -f json -o bandit-report.json || true

echo "🧪 Running tests..."
python -m pytest tests/ -x --tb=short

echo "✅ All checks passed!"
```

## ⚡ Performance Profiling

### Memory Profiling

```python
# scripts/profile_memory.py
import memory_profiler
import pandas as pd
from trading_orchestrator.strategies import MeanReversionStrategy

@memory_profiler.profile
def analyze_strategy_performance():
    """Profile memory usage of strategy analysis"""
    strategy = MeanReversionStrategy()
    
    # Simulate data processing
    data = pd.read_csv('data/market_data.csv')
    result = strategy.analyze(data)
    
    return result

if __name__ == "__main__":
    analyze_strategy_performance()
```

### CPU Profiling

```python
# scripts/profile_cpu.py
import cProfile
import pstats
from trading_orchestrator.engine import TradingEngine

def main():
    engine = TradingEngine()
    
    # Profile trading operations
    for i in range(100):
        trade = {
            'symbol': f'SYM{i}',
            'quantity': 10,
            'side': 'buy'
        }
        engine.process_trade(trade)

if __name__ == "__main__":
    cProfile.run('main()', 'trading_profile.stats')
    
    # Analyze results
    stats = pstats.Stats('trading_profile.stats')
    stats.sort_stats('cumulative').print_stats(20)
```

### Line Profiling

```python
# scripts/line_profile.py
from line_profiler import LineProfiler

def profile_trading_engine():
    """Profile specific functions line by line"""
    from trading_orchestrator.engine import TradingEngine
    from trading_orchestrator.strategies import MeanReversionStrategy
    
    engine = TradingEngine()
    strategy = MeanReversionStrategy()
    
    # Profile these functions
    functions = [
        engine.execute_trade,
        strategy.analyze_market_data,
        engine.calculate_position_size
    ]
    
    profiler = LineProfiler()
    for func in functions:
        profiler.add_function(func)
    
    # Run profiling
    profiler.enable_by_count()
    
    # Your trading operations here
    engine.execute_trade({'symbol': 'AAPL', 'quantity': 10})
    
    profiler.print_stats()
```

### Benchmarking

```python
# scripts/benchmark.py
import time
import statistics
from functools import wraps
import matplotlib.pyplot as plt

def benchmark(func):
    """Decorator to benchmark function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        times = []
        
        # Run multiple times for accuracy
        for _ in range(100):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"{func.__name__}:")
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Std Dev: {std_time:.4f}s")
        print(f"  Min: {min_time:.4f}s")
        print(f"  Max: {max_time:.4f}s")
        
        return result
    
    return wrapper

# Usage
@benchmark
def complex_calculation():
    """Example function to benchmark"""
    import numpy as np
    data = np.random.randn(1000, 1000)
    result = np.linalg.svd(data)
    return result

if __name__ == "__main__":
    complex_calculation()
```

### Performance Monitoring

```python
# scripts/performance_monitor.py
import psutil
import time
import logging
from threading import Thread
from trading_orchestrator.main import app

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = []
    
    def start_monitoring(self, interval=5):
        """Start monitoring system performance"""
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                stats = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'active_connections': len(psutil.net_connections())
                }
                self.stats.append(stats)
                time.sleep(interval)
        
        thread = Thread(target=monitor, daemon=True)
        thread.start()
        
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logging.info("Performance monitoring stopped")
    
    def get_summary(self):
        """Get performance summary"""
        if not self.stats:
            return None
        
        import pandas as pd
        
        df = pd.DataFrame(self.stats)
        return {
            'avg_cpu': df['cpu_percent'].mean(),
            'avg_memory': df['memory_percent'].mean(),
            'avg_disk': df['disk_usage'].mean(),
            'max_cpu': df['cpu_percent'].max(),
            'max_memory': df['memory_percent'].max()
        }
```

## ❓ Troubleshooting

### Common Issues

#### Virtual Environment Issues

```bash
# Problem: Virtual environment not activating
# Solution:
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Problem: pip not found in virtual environment
# Solution:
which pip  # Check pip location
python -m pip install package_name

# Problem: Package not installing
# Solution:
pip install --upgrade pip setuptools wheel
pip install package_name --no-cache-dir
```

#### IDE Issues

```bash
# Problem: VS Code not recognizing virtual environment
# Solution:
# 1. Open Command Palette (Ctrl+Shift+P)
# 2. Select "Python: Select Interpreter"
# 3. Choose ./venv/bin/python

# Problem: PyCharm not recognizing project
# Solution:
# 1. File > Open > Select project directory
# 2. Mark as > Mark as > Project Root
# 3. Configure Python interpreter in Settings

# Problem: IntelliSense not working
# Solution:
# VS Code: Reload window (Ctrl+Shift+P > "Developer: Reload Window")
# PyCharm: File > Invalidate Caches and Restart
```

#### Database Issues

```python
# Problem: SQLite database locked
# Solution:
import sqlite3
conn = sqlite3.connect('database.db', timeout=30)
conn.execute('PRAGMA journal_mode=WAL')

# Problem: PostgreSQL connection issues
# Solution:
# Check connection string format
DATABASE_URL = "postgresql://username:password@host:port/database"

# Test connection
import psycopg2
try:
    conn = psycopg2.connect(DATABASE_URL)
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

#### Import Issues

```python
# Problem: Module not found
# Solution 1: Check PYTHONPATH
import sys
print(sys.path)

# Solution 2: Add project root to Python path
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Solution 3: Use relative imports
from .module import function
from ..parent_module import ClassName

# Problem: Circular imports
# Solution: Use TYPE_CHECKING for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import MyClass

class MyClass:
    pass
```

#### Performance Issues

```python
# Problem: Slow imports
# Solution: Use lazy imports
from importlib import import_module

def lazy_import(module_name):
    return import_module(module_name)

# Problem: Memory leaks
# Solution: Use context managers
with closing(requests.Session()) as session:
    response = session.get(url)

# Problem: Database N+1 queries
# Solution: Use eager loading
from sqlalchemy.orm import joinedload

results = session.query(Model).options(joinedload(Model.related)).all()
```

### Debugging Commands

```bash
# Check Python version and location
python --version
which python
python -m site

# Check installed packages
pip list
pip show package_name

# Check virtual environment
pip --version
which pip
python -c "import sys; print(sys.executable)"

# Test database connection
python -c "
from sqlalchemy import create_engine
engine = create_engine('sqlite:///test.db')
connection = engine.connect()
print('Database connection successful')
connection.close()
"

# Test API endpoints
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/api/orders \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","quantity":10,"side":"buy"}'

# Check system resources
htop  # Process monitor
iotop  # I/O monitor
nethogz  # Network monitor
```

### Environment Diagnostics

```python
# scripts/diagnostics.py
#!/usr/bin/env python3
"""Development environment diagnostics"""

import sys
import os
import platform
import subprocess

def check_python():
    """Check Python installation"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print()

def check_virtual_environment():
    """Check virtual environment status"""
    print(f"Virtual environment: {sys.prefix}")
    print(f"Real prefix: {getattr(sys, 'real_prefix', None)}")
    print(f"Base prefix: {sys.base_prefix}")
    
    in_venv = sys.prefix != sys.base_prefix
    print(f"In virtual environment: {in_venv}")
    print()

def check_dependencies():
    """Check key dependencies"""
    dependencies = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'alembic',
        'pandas', 'numpy', 'requests', 'redis'
    ]
    
    print("Checking dependencies:")
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {dep}: {version}")
        except ImportError:
            print(f"  ✗ {dep}: not installed")
    print()

def check_database():
    """Check database connectivity"""
    try:
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///test.db')
        connection = engine.connect()
        connection.close()
        print("  ✓ SQLite: connected")
    except Exception as e:
        print(f"  ✗ SQLite: {e}")
    
    # Check PostgreSQL if configured
    db_url = os.getenv('DATABASE_URL')
    if db_url and 'postgresql' in db_url:
        try:
            engine = create_engine(db_url)
            connection = engine.connect()
            connection.close()
            print("  ✓ PostgreSQL: connected")
        except Exception as e:
            print(f"  ✗ PostgreSQL: {e}")
    print()

def check_git():
    """Check Git status"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Git repository: found")
            if result.stdout.strip():
                print("Uncommitted changes:")
                print(result.stdout)
            else:
                print("Working directory: clean")
        else:
            print("Git repository: not found")
    except FileNotFoundError:
        print("Git: not found")
    print()

if __name__ == "__main__":
    print("🔍 Development Environment Diagnostics")
    print("=" * 50)
    
    check_python()
    check_virtual_environment()
    check_dependencies()
    check_database()
    check_git()
    
    print("✅ Diagnostics complete!")
```

---

**Need Help?**

- 📖 [Full Documentation](docs/)
- 🐛 [Troubleshooting Guide](docs/troubleshooting.md)
- 💬 [Developer Community](https://discord.gg/trading-orchestrator)
- 📧 [Developer Support](mailto:dev@trading-orchestrator.com)

<div align="center">

**Happy Coding! 💻**

Made with ❤️ by the Trading Orchestrator Team

</div>
