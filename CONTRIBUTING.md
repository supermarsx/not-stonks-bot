# Contributing to not-stonks-bot

Thank you for your interest in contributing to not-stonks-bot! We welcome contributions from the community to make this enterprise-grade trading platform even better.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Provide detailed information about the issue
- Include steps to reproduce the problem
- Specify your environment (OS, Python version, etc.)

### Submitting Changes

#### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/not-stonks-bot.git
cd not-stonks-bot
git remote add upstream https://github.com/supermarsx/not-stonks-bot.git
```

#### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Make Your Changes
- Follow the code style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed

#### 4. Test Your Changes
```bash
# Run the test suite
python -m pytest tests/

# Run specific tests
python health_check.py --full

# Test in demo mode
python main.py --demo
```

#### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

#### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Code Style Guidelines

### Python Code Standards
- **Black** for code formatting (100 character line limit)
- **isort** for import sorting
- **flake8** for linting and style checking
- **mypy** for type checking
- **pytest** for testing framework

### Code Quality Requirements
- 80%+ test coverage
- All functions and classes must have docstrings
- Type hints for all function parameters and return values
- No linting errors or warnings

### Example Code Style
```python
def calculate_position_size(
    risk_percentage: float,
    account_balance: float,
    stop_loss_distance: float
) -> float:
    """
    Calculate optimal position size based on risk parameters.
    
    Args:
        risk_percentage: Maximum risk as percentage of account (0.01 = 1%)
        account_balance: Total account balance
        stop_loss_distance: Distance to stop loss in percentage
    
    Returns:
        Optimal position size in base currency
        
    Raises:
        ValueError: If risk_percentage is invalid
    """
    if not 0 < risk_percentage <= 1:
        raise ValueError("Risk percentage must be between 0 and 1")
    
    max_risk_amount = account_balance * risk_percentage
    position_size = max_risk_amount / stop_loss_distance
    
    return position_size
```

## 🏗️ Architecture Guidelines

### Module Structure
```
trading_orchestrator/
├── brokers/           # Broker integrations
├── strategies/        # Trading strategies
├── risk/             # Risk management
├── analytics/        # Performance analytics
├── crawlers/         # Data collection
├── ai/              # AI integrations
└── api/             # REST API
```

### Design Principles
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Avoid hardcoded dependencies
- **Error Handling**: Comprehensive error handling and logging
- **Configuration**: All settings should be configurable

## 🧪 Testing Guidelines

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **System Tests**: End-to-end functionality tests
- **Performance Tests**: Load and stress testing

### Testing Best Practices
```python
import pytest
from unittest.mock import Mock, patch
from trading_orchestrator.brokers.alpaca import AlpacaBroker

class TestAlpacaBroker:
    def test_place_order_success(self):
        """Test successful order placement."""
        broker = AlpacaBroker(api_key="test", secret_key="test")
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                'id': 'order_123',
                'status': 'accepted'
            }
            
            result = broker.place_order(
                symbol='AAPL',
                side='buy',
                quantity=10
            )
            
            assert result['id'] == 'order_123'
            assert result['status'] == 'accepted'
```

## 📝 Documentation Standards

### Required Documentation
- **README updates**: Update README for new features
- **API documentation**: Document all new endpoints
- **Code comments**: Complex logic should be commented
- **CHANGELOG**: Update CHANGELOG.md for all changes

### Documentation Format
```markdown
## Feature Description
Brief description of what this feature does.

### Usage Example
```python
# Example code showing how to use the feature
```

### Configuration
```json
{
  "feature_name": {
    "enabled": true,
    "option": "value"
  }
}
```
```

## 🚀 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Setup Commands
```bash
# Clone repository
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## 🔒 Security Guidelines

### Security Best Practices
- **Never commit API keys or sensitive data**
- **Use environment variables for configuration**
- **Validate all inputs**
- **Implement proper error handling**
- **Follow OWASP security guidelines**

### Security Review Checklist
- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] Error messages don't leak sensitive data
- [ ] Authentication/authorization properly implemented
- [ ] Encryption used for sensitive data

## 📊 Performance Guidelines

### Performance Standards
- **Sub-second response times** for critical operations
- **Memory usage optimization** for long-running processes
- **Efficient database queries** with proper indexing
- **Caching strategy** for frequently accessed data

### Performance Testing
```bash
# Performance benchmark
python benchmark.py

# Memory usage analysis
python -m memory_profiler trading_orchestrator/main.py

# Load testing
python load_test.py --concurrent-users 50 --duration 300s
```

## 🤖 AI Integration Guidelines

### AI Model Integration
- Support multiple AI providers (OpenAI, Anthropic, Local)
- Implement cost management and token throttling
- Provide fallbacks for AI service failures
- Document API usage and costs

### Example AI Integration
```python
from trading_orchestrator.ai.providers import OpenAIProvider

class MarketAnalyzer:
    def __init__(self):
        self.ai_provider = OpenAIProvider(
            api_key="your-api-key",
            model="gpt-4",
            cost_per_token=0.00003
        )
    
    async def analyze_sentiment(self, news_text: str) -> float:
        """Analyze market sentiment from news text."""
        prompt = f"Analyze the market sentiment of this news: {news_text}"
        
        response = await self.ai_provider.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        return self._parse_sentiment(response.content)
```

## ⚠️ Risk Disclaimer

All contributions related to trading functionality must include appropriate risk warnings and disclaimers. Trading involves substantial risk of loss and should not be promoted without proper warnings.

## 📞 Getting Help

- **GitHub Discussions**: For general questions and feature discussions
- **GitHub Issues**: For bug reports and specific issues
- **Documentation**: Check the docs/ directory for detailed guides

## 📜 License

By contributing to not-stonks-bot, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to not-stonks-bot! Your efforts help make this trading platform better for everyone. 🚀