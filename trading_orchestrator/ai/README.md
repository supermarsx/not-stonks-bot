# AI/LLM Orchestration System

## Overview

The AI/LLM Orchestration System is the intelligent core of the Day Trading Orchestrator. It uses multi-tier language models to analyze markets, generate strategies, and make trading decisions with comprehensive tool integration and risk management.

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Orchestrator                          │
│  (Strategy execution, market analysis, decision making)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                   ↓
┌─────────┐    ┌──────────────┐    ┌─────────────┐
│  Models │    │    Tools     │    │   Brokers   │
│ Manager │    │   (Trading)  │    │   Manager   │
└─────────┘    └──────────────┘    └─────────────┘
```

### 1. AI Models Manager

**Purpose**: Multi-tier LLM orchestration with provider abstraction

**Tiers**:
- **Tier 1 (Reasoning)**: Claude 3.5 Sonnet, GPT-4 Turbo
  - Use for: Strategy generation, complex analysis, risk assessment
  - Characteristics: High quality, slower, higher cost
  
- **Tier 2 (Fast)**: GPT-3.5 Turbo, Claude Haiku
  - Use for: Quick decisions, real-time analysis, order routing
  - Characteristics: Balanced quality/speed/cost
  
- **Tier 3 (Local)**: Local SLMs (future)
  - Use for: High-frequency operations, latency-critical tasks
  - Characteristics: Fastest, lowest cost, lower quality

**Providers**:
- OpenAI (GPT-4 Turbo, GPT-3.5 Turbo)
- Anthropic (Claude 3.5 Sonnet, Claude Haiku)
- Local models (placeholder for future integration)

**Features**:
- Automatic model selection based on task type
- Function calling support for tool integration
- Conversation history management
- Usage tracking and cost monitoring
- Async/await for non-blocking operations

**File**: `ai/models/ai_models_manager.py`

### 2. Trading Tools

**Purpose**: AI-callable functions for trading operations

**Available Tools**:

1. **get_market_features**
   - Calculate technical indicators (momentum, volatility, volume, trends)
   - Identify support/resistance levels
   - Classify market regime
   - Returns: Feature dictionary with metrics

2. **backtest_strategy**
   - Run strategy simulations on historical data
   - Calculate performance metrics (Sharpe, drawdown, win rate)
   - Generate equity curves and trade logs
   - Returns: Backtest results with statistics

3. **check_risk_limits**
   - Validate trades against risk parameters
   - Check position size limits, concentration, daily limits
   - Verify compliance with trading rules
   - Returns: Approval status with violations if any

4. **get_news_sentiment**
   - Retrieve recent news for symbols or market
   - Analyze sentiment (positive/negative/neutral)
   - Identify key topics and trends
   - Returns: News articles with sentiment scores

5. **query_trading_knowledge**
   - RAG (Retrieval Augmented Generation) for trading knowledge
   - Query strategies, patterns, techniques, historical events
   - Synthesize answers from knowledge base
   - Returns: Relevant documents and synthesized answer

**File**: `ai/tools/trading_tools.py`

### 3. AI Orchestrator

**Purpose**: Main coordination layer for AI-driven trading

**Core Capabilities**:

#### Market Analysis
```python
result = await orchestrator.analyze_market(
    symbols=['AAPL', 'TSLA', 'MSFT'],
    analysis_type='comprehensive',
    use_reasoning_model=True
)
```
- Analyzes multiple symbols comprehensively
- Uses tools to gather data (features, news, sentiment)
- Provides actionable insights and recommendations
- Configurable depth (quick/comprehensive/deep)

#### Strategy Generation
```python
strategy = await orchestrator.generate_trading_strategy(
    strategy_type=StrategyType.MOMENTUM,
    symbols=['AAPL'],
    parameters={'lookback': 20, 'threshold': 0.02}
)
```
- Generates complete trading strategies
- Includes entry/exit rules, position sizing, risk management
- Uses reasoning models for complex logic
- Validates with backtesting when possible

#### Opportunity Evaluation
```python
evaluation = await orchestrator.evaluate_trading_opportunity(
    symbol='AAPL',
    opportunity_type='breakout',
    context={'price': 150.50, 'volume': 'high'}
)
```
- Fast evaluation of specific opportunities
- Provides BUY/SELL/PASS recommendation
- Suggests entry/exit levels and position size
- Uses fast models for low-latency decisions

#### Trade Execution
```python
result = await orchestrator.execute_ai_trade(
    symbol='AAPL',
    side='buy',
    reasoning='Breakout confirmed with strong volume',
    risk_check=True
)
```
- Executes trades with AI reasoning
- Automatic risk checks before execution
- Supports multiple trading modes (paper/live/analysis)
- Integrates with broker manager for routing

#### Strategy Sessions
```python
session_id = await orchestrator.run_strategy_session(
    strategy_config=my_strategy,
    duration_minutes=60,
    check_interval_seconds=30
)
```
- Run automated trading sessions
- Continuous monitoring for signals
- Automatic trade execution when conditions met
- Background task management

**File**: `ai/orchestrator.py`

## Trading Modes

The system supports multiple operating modes:

1. **PAPER**: Paper trading with simulation (no real money)
2. **LIVE**: Live trading with real money and execution
3. **ANALYSIS**: Analysis only, no trading executed
4. **BACKTEST**: Historical simulation mode

## Strategy Types

Supported strategy categories:

- **MOMENTUM**: Trend following strategies
- **MEAN_REVERSION**: Buy low, sell high patterns
- **PAIRS_TRADING**: Statistical arbitrage between correlated assets
- **CROSS_VENUE**: Cross-exchange arbitrage opportunities
- **AI_DISCRETIONARY**: Full AI decision making without rules
- **HYBRID**: Combination of AI and rule-based logic

## Usage Examples

### Basic Market Analysis

```python
from ai.orchestrator import AITradingOrchestrator, TradingMode
from ai.models.ai_models_manager import AIModelsManager
from ai.tools.trading_tools import TradingTools

# Initialize components
ai_models = AIModelsManager(
    openai_api_key="your_key",
    anthropic_api_key="your_key"
)
trading_tools = TradingTools(broker_manager=broker_mgr)

# Create orchestrator
orchestrator = AITradingOrchestrator(
    ai_models_manager=ai_models,
    trading_tools=trading_tools,
    trading_mode=TradingMode.PAPER
)

# Analyze market
analysis = await orchestrator.analyze_market(
    symbols=['AAPL', 'TSLA'],
    analysis_type='comprehensive'
)

print(analysis['analysis'])
```

### Generate and Run Strategy

```python
# Generate strategy using AI
strategy = await orchestrator.generate_trading_strategy(
    strategy_type=StrategyType.MOMENTUM,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    parameters={'momentum_period': 20, 'exit_threshold': 0.03}
)

# Run strategy session
session_id = await orchestrator.run_strategy_session(
    strategy_config=strategy['strategy_config'],
    duration_minutes=120,
    check_interval_seconds=60
)

# Check session status
status = orchestrator.get_session_status(session_id)
print(f"Signals: {status['signals_generated']}, Trades: {status['trades_executed']}")

# Stop session when done
orchestrator.stop_strategy_session(session_id)
```

### Evaluate Trading Opportunity

```python
# Real-time opportunity evaluation
opportunity = await orchestrator.evaluate_trading_opportunity(
    symbol='AAPL',
    opportunity_type='momentum_breakout',
    context={
        'current_price': 175.50,
        'volume': 'above_average',
        'news': 'positive_earnings'
    }
)

# Check recommendation
if 'BUY' in opportunity['evaluation']:
    # Execute trade
    result = await orchestrator.execute_ai_trade(
        symbol='AAPL',
        side='buy',
        reasoning=opportunity['evaluation'],
        risk_check=True
    )
```

## Performance Tracking

The orchestrator tracks key metrics:

```python
stats = orchestrator.get_performance_stats()

# Returns:
{
    'decisions_made': 150,
    'trades_executed': 45,
    'analysis_count': 32,
    'active_sessions': 2,
    'ai_model_usage': {
        'gpt-3.5-turbo': {
            'total_tokens': 50000,
            'total_cost': 0.75,
            'request_count': 100
        },
        'claude-3-5-sonnet': {
            'total_tokens': 20000,
            'total_cost': 0.30,
            'request_count': 10
        }
    },
    'trading_mode': 'paper'
}
```

## Configuration

### Environment Variables

```bash
# AI/LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Default Settings
AI_DEFAULT_TIER=fast
AI_MAX_TOOL_CALLS=10
AI_DEFAULT_TEMPERATURE=0.7
```

### Model Configuration

Modify `MODEL_REGISTRY` in `ai_models_manager.py` to:
- Add new models
- Adjust cost parameters
- Configure token limits
- Change default temperatures

## Integration Points

### With Broker Manager
- Market data retrieval
- Order execution
- Position tracking
- Account information

### With Risk Manager
- Pre-trade risk checks
- Position limit enforcement
- Compliance validation
- Circuit breakers

### With Terminal UI
- Display AI analysis
- Show strategy performance
- Interactive decision making
- Real-time monitoring

## Future Enhancements

1. **Local SLM Integration**: Add support for local small language models
2. **RAG Knowledge Base**: Implement vector database for trading knowledge
3. **Advanced Backtesting**: Full-featured backtest engine with slippage/commissions
4. **News Integration**: Connect to real news APIs (Alpaca News, Benzinga)
5. **Multi-Agent System**: Specialized agents for different tasks
6. **Reinforcement Learning**: RL agents for adaptive strategies
7. **Explainable AI**: Better transparency in AI decisions

## Best Practices

1. **Start with Paper Trading**: Always test strategies in paper mode first
2. **Use Appropriate Tiers**: Reasoning for strategy design, Fast for execution
3. **Enable Risk Checks**: Always use risk checks in live trading
4. **Monitor Costs**: Track AI model usage and costs
5. **Review Decisions**: Audit AI reasoning for critical trades
6. **Set Limits**: Configure maximum position sizes and daily loss limits
7. **Continuous Learning**: Use backtest results to improve strategies

## Troubleshooting

### Common Issues

**High API Costs**:
- Use Fast tier (GPT-3.5, Claude Haiku) for frequent operations
- Limit max_tool_calls parameter
- Cache analysis results when appropriate

**Slow Response Times**:
- Switch to Fast tier models
- Reduce analysis depth
- Limit number of symbols analyzed simultaneously

**Tool Call Failures**:
- Check broker connections
- Verify market data access
- Review tool function implementations

**Strategy Not Executing**:
- Verify trading mode (not ANALYSIS)
- Check broker manager initialization
- Review risk limit configuration

## Support

For issues or questions:
- Check logs in `logs/ai_orchestrator.log`
- Review tool call traces
- Monitor model usage statistics
- Enable debug logging for detailed traces
