"""
AI Trading Orchestrator - Main coordination layer for AI-driven trading

Coordinates between:
- Multiple LLM models (Reasoning, Fast, Local)
- Trading tools (market analysis, backtesting, risk checks)
- Broker connections and order execution
- Strategy execution and monitoring

This is the central brain that processes market data, generates insights,
and makes trading decisions using AI models.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from loguru import logger

from ai.models.ai_models_manager import AIModelsManager, ModelTier
from ai.tools.trading_tools import TradingTools, TOOL_DEFINITIONS


class TradingMode(Enum):
    """Trading operation modes"""
    PAPER = "paper"          # Paper trading with simulation
    LIVE = "live"            # Live trading with real money
    ANALYSIS = "analysis"    # Analysis only, no trading
    BACKTEST = "backtest"    # Backtest mode


class StrategyType(Enum):
    """Trading strategy types"""
    MOMENTUM = "momentum"                    # Trend following
    MEAN_REVERSION = "mean_reversion"        # Buy low, sell high
    PAIRS_TRADING = "pairs_trading"          # Statistical arbitrage
    CROSS_VENUE = "cross_venue_arbitrage"    # Cross-exchange arbitrage
    AI_DISCRETIONARY = "ai_discretionary"    # Full AI decision making
    HYBRID = "hybrid"                        # AI + rule-based


class AITradingOrchestrator:
    """
    Main AI trading orchestrator
    
    Manages the entire AI-driven trading workflow from market analysis
    to order execution, with multi-tier LLM support and comprehensive
    risk management.
    """
    
    def __init__(
        self,
        ai_models_manager: AIModelsManager,
        trading_tools: TradingTools,
        broker_manager=None,
        risk_manager=None,
        trading_mode: TradingMode = TradingMode.PAPER
    ):
        """
        Initialize AI trading orchestrator
        
        Args:
            ai_models_manager: AI models manager instance
            trading_tools: Trading tools instance
            broker_manager: Broker manager for executing trades
            risk_manager: Risk manager for limits and compliance
            trading_mode: Trading mode (paper/live/analysis/backtest)
        """
        self.ai_models = ai_models_manager
        self.tools = trading_tools
        self.broker_manager = broker_manager
        self.risk_manager = risk_manager
        self.trading_mode = trading_mode
        
        # Register tools with AI models
        self._register_tools()
        
        # Active strategies and sessions
        self.active_strategies: Dict[str, Dict] = {}
        self.active_sessions: Dict[str, Dict] = {}
        
        # Performance tracking
        self.decisions_made = 0
        self.trades_executed = 0
        self.analysis_count = 0
        
        logger.info(f"AI Trading Orchestrator initialized (mode={trading_mode.value})")
        
    def _register_tools(self):
        """Register trading tools with AI models manager"""
        self.ai_models.register_tool('get_market_features', self.tools.get_market_features)
        self.ai_models.register_tool('backtest_strategy', self.tools.backtest_strategy)
        self.ai_models.register_tool('check_risk_limits', self.tools.check_risk_limits)
        self.ai_models.register_tool('get_news_sentiment', self.tools.get_news_sentiment)
        self.ai_models.register_tool('query_trading_knowledge', self.tools.query_trading_knowledge)
        
    async def analyze_market(
        self,
        symbols: List[str],
        analysis_type: str = "comprehensive",
        use_reasoning_model: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze market conditions for given symbols
        
        Args:
            symbols: List of symbols to analyze
            analysis_type: Type of analysis (quick/comprehensive/deep)
            use_reasoning_model: Use high-quality reasoning model
            
        Returns:
            Market analysis results with insights and recommendations
        """
        try:
            self.analysis_count += 1
            session_id = f"analysis_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting market analysis for {len(symbols)} symbols (type={analysis_type})")
            
            # Determine model tier based on analysis type
            if analysis_type == "deep" or use_reasoning_model:
                tier = ModelTier.REASONING
            elif analysis_type == "quick":
                tier = ModelTier.FAST
            else:
                tier = ModelTier.FAST
                
            # Construct analysis prompt
            prompt = self._build_analysis_prompt(symbols, analysis_type)
            
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'You are an expert quantitative trader and market analyst. '
                        'Analyze market data using available tools and provide actionable insights. '
                        'Consider technical indicators, market sentiment, risk factors, and trading opportunities. '
                        'Be specific and data-driven in your analysis.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            # Get model to use
            model_config = self.ai_models.get_model_for_task('market_analysis', tier)
            
            # Generate analysis with tool calling
            response = await self.ai_models.generate_completion(
                messages=messages,
                model_key=model_config.name.split('-')[0] + '-' + model_config.name.split('-')[1],
                tools=TOOL_DEFINITIONS,
                max_tool_calls=10,
                session_id=session_id
            )
            
            result = {
                'session_id': session_id,
                'symbols': symbols,
                'analysis_type': analysis_type,
                'model_used': response.get('model'),
                'tier': response.get('tier'),
                'analysis': response.get('content', ''),
                'tool_calls_made': response.get('tool_calls_made', 0),
                'timestamp': datetime.utcnow().isoformat(),
                'mode': self.trading_mode.value
            }
            
            logger.info(f"Market analysis complete (session={session_id}, tools_used={result['tool_calls_made']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {'error': str(e)}
            
    async def generate_trading_strategy(
        self,
        strategy_type: StrategyType,
        symbols: List[str],
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading strategy using AI
        
        Args:
            strategy_type: Type of strategy to generate
            symbols: Symbols to trade
            parameters: Optional strategy parameters
            
        Returns:
            Generated strategy configuration
        """
        try:
            session_id = f"strategy_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Generating {strategy_type.value} strategy for {symbols}")
            
            # Construct strategy generation prompt
            prompt = self._build_strategy_prompt(strategy_type, symbols, parameters)
            
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'You are an expert quantitative strategy developer. '
                        'Design robust trading strategies with clear entry/exit rules, '
                        'position sizing, and risk management. Use available tools to analyze '
                        'market conditions and validate strategy logic.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            # Use reasoning model for strategy generation
            response = await self.ai_models.generate_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tool_calls=15,
                session_id=session_id
            )
            
            result = {
                'session_id': session_id,
                'strategy_type': strategy_type.value,
                'symbols': symbols,
                'strategy_config': response.get('content', ''),
                'model_used': response.get('model'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Strategy generated (session={session_id})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return {'error': str(e)}
            
    async def evaluate_trading_opportunity(
        self,
        symbol: str,
        opportunity_type: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a specific trading opportunity
        
        Args:
            symbol: Symbol for the opportunity
            opportunity_type: Type of opportunity (breakout/reversal/arbitrage/etc)
            context: Additional context information
            
        Returns:
            Evaluation results with recommendation
        """
        try:
            self.decisions_made += 1
            
            logger.info(f"Evaluating {opportunity_type} opportunity for {symbol}")
            
            # Quick evaluation using fast model
            prompt = f"""
Evaluate this {opportunity_type} trading opportunity for {symbol}.

Context: {context if context else 'None provided'}

Use available tools to:
1. Get current market features and technical indicators
2. Check recent news sentiment
3. Verify risk limits for potential trade
4. Query trading knowledge for similar patterns

Provide a clear recommendation: BUY, SELL, or PASS with reasoning.
Include suggested entry price, position size, stop loss, and take profit levels if recommending action.
"""
            
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'You are a professional day trader. Evaluate trading opportunities quickly '
                        'and decisively. Consider risk/reward, market conditions, and timing. '
                        'Be specific with entry/exit levels and position sizing.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            response = await self.ai_models.generate_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tool_calls=5
            )
            
            result = {
                'symbol': symbol,
                'opportunity_type': opportunity_type,
                'evaluation': response.get('content', ''),
                'model_used': response.get('model'),
                'tier': response.get('tier'),
                'timestamp': datetime.utcnow().isoformat(),
                'mode': self.trading_mode.value
            }
            
            logger.info(f"Opportunity evaluated for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating opportunity: {e}")
            return {'error': str(e)}
            
    async def execute_ai_trade(
        self,
        symbol: str,
        side: str,
        reasoning: str,
        risk_check: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a trade with AI reasoning and risk checks
        
        Args:
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            reasoning: AI reasoning for the trade
            risk_check: Perform risk checks before execution
            
        Returns:
            Trade execution result
        """
        try:
            if self.trading_mode == TradingMode.ANALYSIS:
                logger.info(f"Analysis mode: Would execute {side} {symbol}")
                return {
                    'status': 'simulated',
                    'symbol': symbol,
                    'side': side,
                    'reasoning': reasoning,
                    'note': 'Analysis mode - no actual trade executed'
                }
                
            if not self.broker_manager:
                return {'error': 'Broker manager not initialized'}
                
            logger.info(f"Executing AI trade: {side} {symbol}")
            
            # Perform risk checks if enabled
            if risk_check:
                risk_result = await self.tools.check_risk_limits(
                    symbol=symbol,
                    side=side,
                    quantity=100  # TODO: Calculate appropriate position size
                )
                
                if not risk_result.get('approved', False):
                    logger.warning(f"Risk check failed for {symbol}: {risk_result.get('violations', [])}")
                    return {
                        'status': 'rejected',
                        'reason': 'risk_limit_violation',
                        'details': risk_result
                    }
                    
            # Execute trade via broker
            # TODO: Implement actual broker execution
            self.trades_executed += 1
            
            result = {
                'status': 'executed',
                'symbol': symbol,
                'side': side,
                'reasoning': reasoning,
                'timestamp': datetime.utcnow().isoformat(),
                'mode': self.trading_mode.value,
                'note': 'Broker execution integration pending'
            }
            
            logger.success(f"Trade executed: {side} {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'error': str(e)}
            
    async def run_strategy_session(
        self,
        strategy_config: Dict[str, Any],
        duration_minutes: int = 60,
        check_interval_seconds: int = 60
    ) -> str:
        """
        Run an automated trading strategy session
        
        Args:
            strategy_config: Strategy configuration
            duration_minutes: How long to run the strategy
            check_interval_seconds: How often to check for signals
            
        Returns:
            Session ID for tracking
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        self.active_sessions[session_id] = {
            'strategy_config': strategy_config,
            'start_time': datetime.utcnow(),
            'duration_minutes': duration_minutes,
            'status': 'running',
            'signals_generated': 0,
            'trades_executed': 0
        }
        
        logger.info(f"Starting strategy session {session_id} (duration={duration_minutes}min)")
        
        # Run session in background
        asyncio.create_task(
            self._run_strategy_loop(session_id, duration_minutes, check_interval_seconds)
        )
        
        return session_id
        
    async def _run_strategy_loop(
        self,
        session_id: str,
        duration_minutes: int,
        check_interval_seconds: int
    ):
        """Background loop for strategy execution"""
        end_time = datetime.utcnow().timestamp() + (duration_minutes * 60)
        
        while datetime.utcnow().timestamp() < end_time:
            if session_id not in self.active_sessions:
                logger.info(f"Session {session_id} stopped")
                break
                
            if self.active_sessions[session_id]['status'] != 'running':
                break
                
            # TODO: Implement actual strategy logic
            # - Check for signals
            # - Evaluate opportunities
            # - Execute trades if conditions met
            
            await asyncio.sleep(check_interval_seconds)
            
        # Mark session as completed
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'completed'
            logger.info(f"Strategy session {session_id} completed")
            
    def stop_strategy_session(self, session_id: str) -> bool:
        """Stop a running strategy session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'stopped'
            logger.info(f"Stopped strategy session {session_id}")
            return True
        return False
        
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of a strategy session"""
        return self.active_sessions.get(session_id)
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        return {
            'decisions_made': self.decisions_made,
            'trades_executed': self.trades_executed,
            'analysis_count': self.analysis_count,
            'active_sessions': len([s for s in self.active_sessions.values() if s['status'] == 'running']),
            'ai_model_usage': self.ai_models.get_usage_stats(),
            'trading_mode': self.trading_mode.value
        }
        
    # Helper methods for prompt construction
    
    def _build_analysis_prompt(self, symbols: List[str], analysis_type: str) -> str:
        """Build market analysis prompt"""
        return f"""
Perform a {analysis_type} market analysis for the following symbols: {', '.join(symbols)}

For each symbol, use the get_market_features tool to analyze:
- Price momentum and trends
- Volatility patterns
- Volume profiles
- Technical indicators
- Support/resistance levels
- Market regime

Also check recent news sentiment using get_news_sentiment.

Provide a comprehensive analysis with:
1. Current market conditions for each symbol
2. Identified trading opportunities
3. Risk factors and concerns
4. Recommended actions or watchlist additions

Be specific and data-driven in your analysis.
"""
    
    def _build_strategy_prompt(
        self,
        strategy_type: StrategyType,
        symbols: List[str],
        parameters: Optional[Dict]
    ) -> str:
        """Build strategy generation prompt"""
        params_str = f"\nParameters: {parameters}" if parameters else ""
        
        return f"""
Design a {strategy_type.value} trading strategy for: {', '.join(symbols)}
{params_str}

Create a complete strategy configuration including:

1. **Entry Rules**: Specific conditions for entering positions
2. **Exit Rules**: Take profit and stop loss logic
3. **Position Sizing**: How to calculate position size based on risk
4. **Risk Management**: Maximum drawdown, position limits, daily loss limits
5. **Timeframe**: What timeframes to analyze and trade on

Use available tools to:
- Analyze historical patterns using get_market_features
- Query similar strategies using query_trading_knowledge
- Validate with backtesting if possible

Provide the strategy in a structured format that can be executed automatically.
"""