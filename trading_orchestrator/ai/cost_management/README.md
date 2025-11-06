# LLM Cost Management System

A comprehensive, enterprise-grade system for tracking, monitoring, optimizing, and controlling LLM costs across multiple providers. This system provides real-time cost tracking, predictive analytics, automated optimization, and intelligent budget management for AI operations.

## 🚀 Features

### Core Cost Management
- **Real-time Cost Tracking**: Monitor LLM usage and costs in real-time across all providers
- **Multi-Provider Support**: Support for OpenAI, Anthropic, and custom providers
- **Token Usage Analytics**: Detailed breakdown of token consumption by provider, model, and task
- **Cost Per Request Tracking**: Granular cost analysis for each API call

### Budget Management
- **Dynamic Budget Tiers**: Low, Medium, High, and Enterprise budget profiles
- **Real-time Budget Monitoring**: Track spending against daily, weekly, and monthly limits
- **Automated Budget Controls**: Automatic model switching and request throttling when limits are reached
- **Budget Alerts**: Configurable alerts at 50%, 80%, 90%, and 95% budget utilization

### Provider Optimization
- **Intelligent Provider Selection**: Cost-based model selection based on task requirements
- **Provider Health Monitoring**: Real-time monitoring of provider availability and performance
- **Automatic Failover**: Seamless switching between providers based on cost and availability
- **Provider Performance Analytics**: Track response times, success rates, and cost efficiency

### Analytics & Forecasting
- **Cost Trend Analysis**: Historical cost analysis with trend identification
- **Predictive Forecasting**: AI-powered cost forecasting with confidence intervals
- **Usage Pattern Analysis**: Identify peak usage times and optimization opportunities
- **Performance Benchmarking**: Compare performance against industry standards

### Anomaly Detection
- **Real-time Anomaly Detection**: Automatic detection of unusual cost patterns
- **Statistical Outlier Detection**: Z-score based anomaly detection with configurable thresholds
- **Pattern-based Detection**: Detect deviations from historical usage patterns
- **Automated Response**: Automatic cost controls when anomalies are detected

### Interactive Dashboard
- **Real-time Cost Dashboard**: Live monitoring of costs, budgets, and provider health
- **Customizable Widgets**: Configure dashboard layout with charts, metrics, and alerts
- **Mobile Responsive**: Access cost data from any device
- **Export Capabilities**: Export reports and data in multiple formats

## 📦 Installation

### Prerequisites
- Python 3.8+
- SQLite3
- Required packages (see requirements.txt)

### Setup
1. Clone or copy the cost management system to your project
2. Ensure the directory structure matches the expected layout
3. Install dependencies:
```bash
pip install loguru numpy sqlite3
```

### Database Initialization
The system automatically creates SQLite databases in the `data/` directory:
- `llm_cost_management.db` - Main cost tracking database
- `provider_stats.db` - Provider performance statistics
- `anomaly_storage.db` - Anomaly detection events

## 🔧 Quick Start

### Basic Integration

```python
from ai.cost_management import LLMIntegratedCostManager
from ai.models.ai_models_manager import AIModelsManager

# Initialize AI models manager (existing)
ai_models = AIModelsManager(
    openai_api_key="your_openai_key",
    anthropic_api_key="your_anthropic_key"
)

# Initialize integrated cost management
cost_system = LLMIntegratedCostManager(
    ai_models_manager=ai_models,
    database_path="data/my_costs.db",
    enable_real_time_monitoring=True
)

# Track AI requests automatically
result = await cost_system.track_ai_request(
    provider="openai",
    model="gpt-4-turbo",
    tokens_used=1500,
    cost=0.015,
    task_type="reasoning"
)
```

### Cost-Optimal Model Selection

```python
# Select the most cost-effective model for a task
optimal_model = await cost_system.select_cost_optimal_model(
    task_type="reasoning",
    token_count=2000,
    budget_constraint=0.1,
    max_response_time=5.0
)

print(f"Selected model: {optimal_model}")
```

### Budget Management

```python
# Create a custom budget profile
budget_result = await cost_system.create_budget_profile(
    name="Production Budget",
    monthly_limit=5000.0,
    daily_limit=200.0,
    tier="high",
    auto_optimization=True
)

# Get current budget status
budget_status = await cost_system.budget_manager.get_budget_status()
print(f"Budget utilization: {budget_status['daily_percentage']}")
```

### Cost Analytics and Forecasting

```python
from datetime import datetime, timedelta

# Generate comprehensive cost report
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

report = await cost_system.get_comprehensive_cost_report(
    start_date=start_date,
    end_date=end_date,
    report_type="comprehensive"
)

print(f"Total cost: ${report['cost_analysis']['summary']['total_cost']:.2f}")
```

### Real-time Dashboard

```python
# Get dashboard data
dashboard_data = await cost_system.dashboard.get_dashboard_data("Main Overview")

# Check for triggered alerts
alerts = await cost_system.dashboard.check_alert_thresholds()
for alert in alerts:
    print(f"Alert: {alert['severity']} - {alert['metric']} = {alert['current_value']}")
```

## 🎯 Advanced Features

### Custom Alert Handlers

```python
def custom_cost_alert_handler(alert):
    if alert.level.value == 'critical':
        # Send emergency notification
        send_email("admin@company.com", f"Critical cost alert: {alert.message}")
        
        # Execute emergency optimization
        asyncio.create_task(cost_system._switch_to_emergency_mode())

# Register custom handler
cost_system.add_alert_handler(custom_cost_alert_handler)
```

### Anomaly Detection Rules

```python
from ai.cost_management.anomaly_detector import AnomalyRule, AnomalyType

# Add custom anomaly detection rule
custom_rule = AnomalyRule(
    name="High Frequency Requests",
    anomaly_type=AnomalyType.REQUEST_SPIKE,
    threshold=100,  # requests per hour
    window_size=1,
    severity_threshold=200,
    cooldown_period=timedelta(minutes=30)
)

cost_system.anomaly_detector.add_rule(custom_rule)
```

### Provider Customization

```python
from ai.cost_management.provider_manager import ProviderConfig

# Add custom provider
custom_provider = ProviderConfig(
    name="my-custom-llm",
    base_url="https://api.my-llm.com",
    cost_per_1k_input_tokens=0.005,
    cost_per_1k_output_tokens=0.010,
    max_tokens_per_request=8192,
    rate_limit_per_minute=100,
    backup_provider="openai-gpt35"
)

cost_system.provider_manager.add_provider(custom_provider)
```

### Performance Monitoring

```python
# Get provider performance comparison
comparison = cost_system.provider_manager.get_provider_comparison()

for provider, metrics in comparison.items():
    print(f"{provider}:")
    print(f"  Cost per request: {metrics['cost_per_request']}")
    print(f"  Success rate: {metrics['success_rate']}")
    print(f"  Avg response time: {metrics['avg_response_time']}")
```

## 📊 Dashboard Widgets

The system includes pre-built dashboard widgets:

- **Current Cost Metric**: Live cost tracking
- **Daily Cost Chart**: Trend visualization
- **Provider Breakdown**: Cost distribution by provider
- **Budget Status**: Real-time budget utilization
- **Active Alerts**: Current system alerts
- **Cost Forecast**: Predictive cost visualization
- **Anomaly Detection**: Real-time anomaly monitoring
- **Performance Benchmarks**: Industry comparison metrics

### Creating Custom Widgets

```python
from ai.cost_management.dashboard import DashboardWidget

# Create custom widget
custom_widget = DashboardWidget(
    id="custom_cost_widget",
    title="Custom Cost Analysis",
    widget_type="chart",
    position={"x": 0, "y": 0, "width": 6, "height": 4},
    data_source="custom_analysis",
    refresh_interval=60,
    configuration={"chart_type": "bar", "metric": "cost_efficiency"}
)

# Create custom dashboard
await cost_system.dashboard.create_custom_dashboard(
    name="Custom Dashboard",
    widgets=[
        {
            "id": "custom_cost_widget",
            "title": "Custom Cost Analysis",
            "type": "chart",
            "position": {"x": 0, "y": 0, "width": 6, "height": 4},
            "data_source": "custom_analysis",
            "refresh_interval": 60
        }
    ]
)
```

## 🔍 Monitoring and Alerts

### Alert Types
- **Budget Warnings**: 50%, 80%, 90% budget utilization
- **Cost Spikes**: Unusual cost increases
- **Provider Issues**: Provider availability problems
- **Anomalies**: Statistical outliers in usage patterns

### Alert Severities
- **INFO**: Informational notifications
- **WARNING**: Attention required
- **CRITICAL**: Immediate action needed
- **EMERGENCY**: System-wide response required

### Email Integration
```python
import smtplib
from email.mime.text import MimeText

async def send_cost_alert_email(alert):
    """Example email notification for cost alerts"""
    msg = MimeText(f"""
    LLM Cost Alert
    
    Alert: {alert.message}
    Level: {alert.level.value}
    Time: {alert.timestamp}
    
    Please review your LLM usage and cost controls.
    """)
    
    msg['Subject'] = f"LLM Cost Alert - {alert.level.value.upper()}"
    msg['From'] = "cost-monitor@company.com"
    msg['To'] = "admin@company.com"
    
    # Send email (configure SMTP settings)
    # server = smtplib.SMTP('smtp.company.com')
    # server.send_message(msg)
    # server.quit()
```

## 📈 Analytics and Reporting

### Built-in Reports
- **Daily Cost Summary**: Day-by-day cost breakdown
- **Provider Performance Report**: Provider comparison and analysis
- **Budget Analysis**: Budget utilization and forecasting
- **Anomaly Report**: Detailed anomaly detection results
- **Optimization Report**: Cost optimization opportunities

### Custom Analytics
```python
# Get cost trends analysis
trends = await cost_system.analytics.get_cost_trends(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    group_by='daily'
)

# Detect cost anomalies
anomalies = await cost_system.analytics.detect_cost_anomalies(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

# Export analytics data
export_data = await cost_system.analytics.export_analytics_data(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    format='json'
)
```

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration
LLM_COST_DB_PATH=data/llm_costs.db
PROVIDER_DB_PATH=data/provider_stats.db

# Alert Configuration
EMAIL_SMTP_SERVER=smtp.company.com
EMAIL_SMTP_PORT=587
ALERT_EMAIL_TO=admin@company.com
ALERT_EMAIL_FROM=cost-monitor@company.com
```

### Configuration File
```json
{
  "cost_tracking": {
    "enabled": true,
    "real_time_monitoring": true,
    "auto_optimization": true
  },
  "budgets": {
    "default_tier": "medium",
    "alert_thresholds": [0.5, 0.8, 0.95],
    "emergency_mode_threshold": 0.95
  },
  "anomaly_detection": {
    "enabled": true,
    "threshold": 2.0,
    "window_size": 24,
    "cooldown_period_minutes": 60
  },
  "providers": {
    "default_timeout": 30,
    "retry_attempts": 3,
    "health_check_interval": 300
  }
}
```

## 🚨 Emergency Procedures

### Budget Emergency Mode
When budget limits are severely exceeded, the system automatically:
1. Switches to the most cost-effective models
2. Enables request throttling
3. Sends emergency alerts
4. Stops non-critical operations
5. Requires manual override to exit emergency mode

### Manual Emergency Override
```python
# Manually enable emergency mode
await cost_system._switch_to_emergency_mode()

# Reset to normal operation
cost_system.budget_manager.reset_emergency_mode()
```

## 🛠️ Troubleshooting

### Common Issues

**High API Costs**
- Enable auto-optimization
- Set stricter budget limits
- Use cost-aware model selection
- Monitor for anomalies

**Provider Availability Issues**
- Configure backup providers
- Monitor provider health
- Use automatic failover

**Dashboard Not Updating**
- Check real-time monitoring status
- Verify database connectivity
- Restart monitoring services

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ai.cost_management')

# Check system status
overview = await cost_system.get_system_overview()
print(f"System status: {overview}")
```

## 📚 API Reference

### Core Classes
- `LLMIntegratedCostManager`: Main system controller
- `LLMCostManager`: Core cost tracking functionality
- `BudgetManager`: Budget configuration and control
- `ProviderManager`: Provider optimization and health monitoring
- `CostAnalytics`: Analytics and reporting
- `CostForecaster`: Predictive analytics
- `AnomalyDetector`: Anomaly detection and response
- `CostDashboard`: Real-time monitoring dashboard

### Key Methods
- `track_ai_request()`: Track AI usage and costs
- `select_cost_optimal_model()`: Cost-aware model selection
- `get_cost_optimization_recommendations()`: Optimization suggestions
- `create_budget_profile()`: Budget configuration
- `get_comprehensive_cost_report()`: Detailed reporting
- `get_system_overview()`: System health monitoring

## 🤝 Contributing

The LLM Cost Management System is designed to be extensible. Key areas for enhancement:

1. **Additional Providers**: Support for more LLM providers
2. **Advanced Analytics**: Machine learning-based cost prediction
3. **Custom Integrations**: Webhook and API integrations
4. **Visualization Enhancements**: Advanced charting and reporting
5. **Performance Optimization**: Faster data processing and analysis

## 📄 License

This LLM Cost Management System is part of the Day Trading Orchestrator project.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section
2. Review the demo examples
3. Enable debug logging for detailed information
4. Check system status with `get_system_overview()`

---

**Note**: This system is designed for production use with real LLM APIs. Always test in a development environment first and monitor costs carefully during initial deployment.