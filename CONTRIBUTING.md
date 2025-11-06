# Contributing to not-stonks-bot

Thank you for your interest in contributing to not-stonks-bot! We welcome contributions from developers of all skill levels.

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** - Required for development
- **Node.js 18+** - Required for frontend development
- **Git** - For version control
- **Docker** - For containerized development

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/not-stonks-bot.git
   cd not-stonks-bot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r trading_orchestrator/requirements.txt
   pip install -r trading_orchestrator/requirements-dev.txt  # Development dependencies
   ```

4. **Set up frontend development**
   ```bash
   cd matrix-trading-command-center
   npm install
   cd ..
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Run tests to verify setup**
   ```bash
   python -m pytest trading_orchestrator/tests/
   ```

## 🏗️ Development Workflow

### Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `hotfix/critical-fix` - Critical fixes
- `docs/documentation-update` - Documentation updates
- `refactor/code-improvement` - Code refactoring

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(broker): add Binance Futures support
fix(risk): resolve position limit calculation error
docs(readme): update installation instructions
refactor(ui): simplify matrix component structure
```

## 📝 Code Style

### Python Code Style

We use several tools to maintain code quality:

```bash
# Format code
black trading_orchestrator/
isort trading_orchestrator/

# Lint code
flake8 trading_orchestrator/

# Type checking
mypy trading_orchestrator/
```

**Configuration files:**
- `pyproject.toml` - Black, isort, and pytest configuration
- `.flake8` - Flake8 configuration
- `mypy.ini` - MyPy configuration

### JavaScript/TypeScript Code Style

```bash
# Lint frontend code
cd matrix-trading-command-center
npm run lint

# Format frontend code
npm run format
```

### Code Quality Standards

1. **Follow PEP 8** for Python code
2. **Use type hints** where possible
3. **Write docstrings** for all public functions and classes
4. **Maintain test coverage** above 80%
5. **Use meaningful variable and function names**
6. **Keep functions small and focused**
7. **Avoid duplicate code**

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=trading_orchestrator --cov-report=html

# Run specific test file
python -m pytest trading_orchestrator/tests/test_broker.py

# Run frontend tests
cd matrix-trading-command-center
npm test
```

### Writing Tests

1. **Test Structure**: Use Arrange-Act-Assert pattern
2. **Test Coverage**: Aim for 80%+ coverage
3. **Test Categories**:
   - Unit tests for individual functions/classes
   - Integration tests for component interactions
   - End-to-end tests for full workflows

**Example test:**
```python
import pytest
from trading_orchestrator.brokers.alpaca_broker import AlpacaBroker

class TestAlpacaBroker:
    def test_connect_success(self, mock_alpaca_config):
        """Test successful connection to Alpaca."""
        broker = AlpacaBroker(mock_alpaca_config)
        result = broker.connect()
        assert result is True
        assert broker.is_connected is True
    
    def test_get_account_info(self, broker):
        """Test retrieving account information."""
        account = broker.get_account()
        assert account is not None
        assert 'equity' in account
```

## 🏗️ Architecture Guidelines

### Project Structure

```
not-stonks-bot/
├── trading_orchestrator/     # Main Python package
│   ├── ai/                  # AI/ML components
│   ├── analytics/           # Data analysis
│   ├── api/                 # REST API
│   ├── brokers/             # Broker integrations
│   ├── config/              # Configuration
│   ├── risk/                # Risk management
│   ├── strategies/          # Trading strategies
│   └── ui/                  # Terminal UI
├── matrix-trading-command-center/  # Frontend React app
├── docs/                    # Documentation
└── tests/                   # Test files
```

### Design Principles

1. **Modularity**: Keep components small and focused
2. **Separation of Concerns**: Separate business logic from UI
3. **Dependency Injection**: Use interfaces for broker integrations
4. **Error Handling**: Always handle exceptions gracefully
5. **Logging**: Use structured logging throughout
6. **Configuration**: Externalize all configuration

## 🔒 Security Guidelines

### API Keys and Secrets

- **Never commit** API keys or secrets to version control
- **Use environment variables** for all sensitive data
- **Implement proper key rotation** mechanisms
- **Validate** all external API responses

### Code Security

- **Input validation**: Validate all user inputs
- **SQL injection prevention**: Use parameterized queries
- **XSS prevention**: Sanitize user-provided data
- **Access control**: Implement proper authentication/authorization

### Security Scanning

Run security scans before submitting PRs:

```bash
# Python security scanning
pip install bandit[toml]
bandit -r trading_orchestrator/

# Frontend security scanning
cd matrix-trading-command-center
npm audit
```

## 📋 Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Run full test suite** 
3. **Check code coverage**
4. **Update changelog** if applicable
5. **Ensure security scan passes**

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Code is well-commented
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one maintainer** review required
3. **Code review** for architecture and security
4. **Testing review** for test quality and coverage
5. **Documentation review** for accuracy and completeness

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Screenshots/logs** if applicable
- **Environment details** (OS, Python version, etc.)

### Feature Requests

Use the feature request template and include:

- **Clear problem statement** this feature solves
- **Proposed solution** description
- **Alternative solutions** considered
- **Additional context** or examples

## 💡 Development Tips

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Use Python debugger
python -m pdb script.py

# Profile code
python -m cProfile script.py
```

### Performance Monitoring

```bash
# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Line profiling
pip install line-profiler
kernprof -l -v script.py
```

### Database Development

```bash
# Apply migrations
python -m alembic upgrade head

# Create new migration
python -m alembic revision --autogenerate -m "description"

# Reset database
python -m alembic downgrade base
python -m alembic upgrade head
```

## 📚 Resources

### Documentation

- [Python Best Practices](https://docs.python.org/3/tutorial/)
- [React Documentation](https://reactjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Trading APIs](https://alpaca.markets/docs/)

### Tools

- **IDE**: VS Code with Python extensions
- **Linting**: flake8, pylint
- **Formatting**: black, isort
- **Testing**: pytest, unittest
- **Documentation**: Sphinx, mkdocs

### Community

- **GitHub Discussions**: For questions and ideas
- **Discord**: Real-time chat
- **Stack Overflow**: Use tag `not-stonks-bot`

## 🎯 Areas for Contribution

### High Priority

- [ ] **New broker integrations** (Robinhood, TD Ameritrade, etc.)
- [ ] **Advanced AI strategies** (LSTM, reinforcement learning)
- [ ] **Mobile app** (React Native or Flutter)
- [ ] **Cloud deployment** (AWS, GCP, Azure)
- [ ] **Backtesting engine** improvements

### Medium Priority

- [ ] **WebSocket streaming** for real-time data
- [ ] **Advanced charting** (TradingView integration)
- [ ] **Portfolio analytics** (Sharpe ratio, drawdown analysis)
- [ ] **Multi-language support**
- [ ] **Plugin system** for custom strategies

### Documentation

- [ ] **API reference** documentation
- [ ] **Video tutorials** for new users
- [ ] **Strategy examples** with explanations
- [ ] **Deployment guides** for various platforms

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 👥 Maintainers

- **Lead Developer**: [@your-username](https://github.com/your-username)
- **Core Contributors**: [See contributors page](https://github.com/your-username/not-stonks-bot/graphs/contributors)

## 🙏 Recognition

Contributors who make significant contributions will be:

- Listed in the **README.md** contributors section
- Added to the **CONTRIBUTORS.md** file
- Mentioned in **release notes** for their contributions
- Eligible for **commit access** to the repository

Thank you for contributing to not-stonks-bot! 🎉
