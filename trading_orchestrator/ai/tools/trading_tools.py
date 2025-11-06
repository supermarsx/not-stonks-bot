"""
AI Trading Tools - Functions that AI models can call to execute trading operations

These tools provide the AI with capabilities to:
- Analyze market data and generate trading features
- Execute backtests on strategies
- Check risk limits and compliance
- Retrieve news sentiment
- Access trading knowledge base (RAG)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from loguru import logger


class TradingTools:
    """
    Trading tools accessible to AI models
    
    Each method is designed to be called by an LLM with natural language parameters
    converted to function arguments.
    """
    
    def __init__(self, broker_manager=None, risk_manager=None):
        """
        Initialize trading tools
        
        Args:
            broker_manager: Broker manager instance for market data and trading
            risk_manager: Risk manager instance for limit checks
        """
        self.broker_manager = broker_manager
        self.risk_manager = risk_manager
        
    async def get_market_features(
        self,
        symbol: str,
        lookback_days: int = 30,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators and market features for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USDT')
            lookback_days: Number of days of historical data to analyze
            features: List of features to calculate (None = all)
            
        Returns:
            Dictionary containing calculated features and indicators
            
        Available features:
        - price_momentum: Recent price changes and momentum
        - volatility: Historical and implied volatility metrics
        - volume_profile: Volume analysis and anomalies
        - trend_indicators: Moving averages, MACD, RSI
        - support_resistance: Key price levels
        - market_regime: Current market regime classification
        """
        try:
            if not self.broker_manager:
                return {'error': 'Broker manager not initialized'}
                
            logger.info(f"Calculating market features for {symbol}")
            
            # Fetch historical data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get data from first available broker
            market_data = await self._fetch_market_data(symbol, start_date, end_date)
            
            if not market_data:
                return {'error': f'No market data available for {symbol}'}
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame(market_data)
            
            # Calculate requested features
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'data_points': len(df),
                'features': {}
            }
            
            # Price momentum
            if not features or 'price_momentum' in features:
                result['features']['price_momentum'] = self._calculate_momentum(df)
                
            # Volatility
            if not features or 'volatility' in features:
                result['features']['volatility'] = self._calculate_volatility(df)
                
            # Volume profile
            if not features or 'volume_profile' in features:
                result['features']['volume_profile'] = self._calculate_volume_profile(df)
                
            # Trend indicators
            if not features or 'trend_indicators' in features:
                result['features']['trend_indicators'] = self._calculate_trend_indicators(df)
                
            # Support and resistance
            if not features or 'support_resistance' in features:
                result['features']['support_resistance'] = self._find_support_resistance(df)
                
            # Market regime
            if not features or 'market_regime' in features:
                result['features']['market_regime'] = self._classify_market_regime(df)
                
            return result
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return {'error': str(e)}
            
    async def backtest_strategy(
        self,
        strategy_config: Dict[str, Any],
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Run a backtest on a trading strategy
        
        Args:
            strategy_config: Strategy configuration including rules and parameters
            symbols: List of symbols to trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital for backtest
            
        Returns:
            Backtest results including performance metrics, trades, and equity curve
            
        Strategy config format:
        {
            'name': 'Strategy name',
            'type': 'momentum' | 'mean_reversion' | 'pairs' | 'custom',
            'entry_rules': {...},
            'exit_rules': {...},
            'position_sizing': {...},
            'parameters': {...}
        }
        """
        try:
            logger.info(f"Running backtest: {strategy_config.get('name', 'Unnamed')}")
            
            # Parse dates
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            # Fetch historical data for all symbols
            all_data = {}
            for symbol in symbols:
                data = await self._fetch_market_data(symbol, start, end)
                if data:
                    all_data[symbol] = pd.DataFrame(data)
                    
            if not all_data:
                return {'error': 'No historical data available for backtest'}
                
            # Initialize backtest state
            capital = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            
            # Run backtest simulation
            # This is a simplified implementation - production would use a proper backtest engine
            result = {
                'strategy_name': strategy_config.get('name', 'Unnamed'),
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'performance': {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                },
                'trades': [],
                'equity_curve': [],
                'note': 'Detailed backtest engine integration pending'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
            
    async def check_risk_limits(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if a proposed trade complies with risk limits
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price (None = market price)
            
        Returns:
            Risk check result with approval status and any violations
        """
        try:
            if not self.risk_manager:
                return {
                    'approved': True,
                    'note': 'Risk manager not initialized - trade allowed by default'
                }
                
            logger.info(f"Checking risk limits for {side} {quantity} {symbol}")
            
            # Get current price if not provided
            if not price:
                quote = await self._get_current_price(symbol)
                price = quote.get('last', 0) if quote else 0
                
            # Calculate trade value
            trade_value = quantity * price
            
            # Check various risk limits
            violations = []
            
            # Position size limit
            max_position_value = 50000  # Example: $50k max per position
            if trade_value > max_position_value:
                violations.append(f"Trade value ${trade_value:.2f} exceeds max position size ${max_position_value}")
                
            # Portfolio concentration limit
            # TODO: Check against total portfolio value
            
            # Daily trade limit
            # TODO: Check against daily trade count
            
            # Sector exposure limit
            # TODO: Check sector concentration
            
            result = {
                'approved': len(violations) == 0,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'violations': violations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if not result['approved']:
                logger.warning(f"Risk check FAILED for {symbol}: {violations}")
            else:
                logger.info(f"Risk check PASSED for {symbol}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {'approved': False, 'error': str(e)}
            
    async def get_news_sentiment(
        self,
        symbol: Optional[str] = None,
        lookback_hours: int = 24,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve and analyze news sentiment
        
        Args:
            symbol: Specific symbol (None = market-wide news)
            lookback_hours: Hours of news to analyze
            sources: List of news sources to include
            
        Returns:
            News articles with sentiment scores and analysis
        """
        try:
            logger.info(f"Fetching news sentiment for {symbol or 'market'}")
            
            # Placeholder for news API integration
            # In production, integrate with news APIs (Alpaca News, Benzinga, etc.)
            
            result = {
                'symbol': symbol,
                'lookback_hours': lookback_hours,
                'articles': [],
                'summary': {
                    'total_articles': 0,
                    'average_sentiment': 0.0,
                    'sentiment_trend': 'neutral',
                    'key_topics': []
                },
                'note': 'News API integration pending'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return {'error': str(e)}
            
    async def query_trading_knowledge(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the trading knowledge base using RAG (Retrieval Augmented Generation)
        
        Args:
            query: Natural language query
            context: Optional context to narrow search
            
        Returns:
            Relevant knowledge base entries and synthesized answer
            
        Knowledge base includes:
        - Trading strategies and patterns
        - Market analysis techniques
        - Risk management principles
        - Historical market events
        - Asset class characteristics
        """
        try:
            logger.info(f"Querying knowledge base: {query}")
            
            # Placeholder for RAG implementation
            # In production, integrate with vector database (Pinecone, Weaviate, etc.)
            # and knowledge base content
            
            result = {
                'query': query,
                'context': context,
                'relevant_documents': [],
                'synthesized_answer': 'Knowledge base RAG integration pending',
                'sources': [],
                'confidence_score': 0.0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return {'error': str(e)}
            
    # Helper methods for feature calculation
    
    async def _fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fetch market data from broker"""
        if not self.broker_manager:
            return []
            
        try:
            # Get first active broker
            brokers = self.broker_manager.get_active_brokers()
            if not brokers:
                return []
                
            broker = brokers[0]
            data = await broker.get_market_data(
                symbol=symbol,
                timeframe='1d',
                start=start_date,
                end=end_date,
                limit=365
            )
            
            return [
                {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                for bar in data
            ]
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return []
            
    async def _get_current_price(self, symbol: str) -> Dict:
        """Get current price quote"""
        if not self.broker_manager:
            return {}
            
        try:
            brokers = self.broker_manager.get_active_brokers()
            if not brokers:
                return {}
                
            broker = brokers[0]
            return await broker.get_quote(symbol)
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return {}
            
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict:
        """Calculate price momentum indicators"""
        current_price = df['close'].iloc[-1]
        
        return {
            '1d_return': ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0,
            '5d_return': ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0,
            '20d_return': ((df['close'].iloc[-1] / df['close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0,
            'current_price': float(current_price)
        }
        
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        returns = df['close'].pct_change().dropna()
        
        return {
            'daily_volatility': float(returns.std() * 100),
            'annualized_volatility': float(returns.std() * (252 ** 0.5) * 100),
            'volatility_trend': 'increasing' if len(returns) > 10 and returns.tail(10).std() > returns.head(10).std() else 'decreasing'
        }
        
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate volume analysis"""
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        
        return {
            'average_volume': float(avg_volume),
            'current_volume': float(current_volume),
            'volume_ratio': float(current_volume / avg_volume) if avg_volume > 0 else 0,
            'volume_trend': 'high' if current_volume > avg_volume * 1.5 else 'normal'
        }
        
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate trend indicators"""
        # Simple moving averages
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else df['close'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        return {
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'price_vs_sma20': float(((current_price / sma_20) - 1) * 100) if sma_20 > 0 else 0,
            'trend': 'bullish' if current_price > sma_20 and sma_20 > sma_50 else 'bearish'
        }
        
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify support and resistance levels"""
        recent_high = df['high'].tail(20).max() if len(df) >= 20 else df['high'].max()
        recent_low = df['low'].tail(20).min() if len(df) >= 20 else df['low'].min()
        
        return {
            'resistance': float(recent_high),
            'support': float(recent_low),
            'range': float(recent_high - recent_low)
        }
        
    def _classify_market_regime(self, df: pd.DataFrame) -> Dict:
        """Classify current market regime"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        trend = 'bullish' if returns.mean() > 0 else 'bearish'
        
        # Simple regime classification
        if volatility > returns.std() * 1.5:
            regime = 'high_volatility'
        elif abs(returns.mean()) < 0.001:
            regime = 'ranging'
        else:
            regime = 'trending'
            
        return {
            'regime': regime,
            'trend': trend,
            'confidence': 0.7  # Placeholder confidence score
        }


# Tool definitions for LLM function calling
TOOL_DEFINITIONS = [
    {
        'name': 'get_market_features',
        'description': 'Calculate technical indicators and market features for a symbol including momentum, volatility, volume, trends, and support/resistance levels',
        'parameters': {
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Trading symbol (e.g., AAPL, BTC/USDT)'
                },
                'lookback_days': {
                    'type': 'integer',
                    'description': 'Number of days of historical data to analyze',
                    'default': 30
                },
                'features': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of specific features to calculate (optional)'
                }
            },
            'required': ['symbol']
        }
    },
    {
        'name': 'backtest_strategy',
        'description': 'Run a backtest on a trading strategy with historical data',
        'parameters': {
            'type': 'object',
            'properties': {
                'strategy_config': {
                    'type': 'object',
                    'description': 'Strategy configuration including rules and parameters'
                },
                'symbols': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of symbols to trade'
                },
                'start_date': {
                    'type': 'string',
                    'description': 'Backtest start date (YYYY-MM-DD)'
                },
                'end_date': {
                    'type': 'string',
                    'description': 'Backtest end date (YYYY-MM-DD)'
                },
                'initial_capital': {
                    'type': 'number',
                    'description': 'Starting capital for backtest',
                    'default': 100000.0
                }
            },
            'required': ['strategy_config', 'symbols', 'start_date', 'end_date']
        }
    },
    {
        'name': 'check_risk_limits',
        'description': 'Check if a proposed trade complies with risk management limits',
        'parameters': {
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Trading symbol'
                },
                'side': {
                    'type': 'string',
                    'enum': ['buy', 'sell'],
                    'description': 'Trade direction'
                },
                'quantity': {
                    'type': 'number',
                    'description': 'Trade quantity'
                },
                'price': {
                    'type': 'number',
                    'description': 'Trade price (optional, uses market price if not provided)'
                }
            },
            'required': ['symbol', 'side', 'quantity']
        }
    },
    {
        'name': 'get_news_sentiment',
        'description': 'Retrieve and analyze news sentiment for a symbol or the overall market',
        'parameters': {
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Specific symbol (optional, market-wide if not provided)'
                },
                'lookback_hours': {
                    'type': 'integer',
                    'description': 'Hours of news to analyze',
                    'default': 24
                },
                'sources': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of news sources to include (optional)'
                }
            }
        }
    },
    {
        'name': 'query_trading_knowledge',
        'description': 'Query the trading knowledge base for strategies, techniques, and market insights',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Natural language query'
                },
                'context': {
                    'type': 'string',
                    'description': 'Optional context to narrow search'
                }
            },
            'required': ['query']
        }
    }
]
