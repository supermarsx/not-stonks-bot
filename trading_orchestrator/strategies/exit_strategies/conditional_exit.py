"""
@file conditional_exit.py
@brief Conditional Exit Strategy Implementation

@details
This module implements conditional exit strategies that evaluate complex
market conditions to determine optimal exit timing. Conditional exits use
multiple criteria and logical combinations to make exit decisions.

Key Features:
- Multi-condition evaluation
- Logical operator combinations (AND, OR, NOT)
- Technical indicator conditions
- Market state conditions
- News and event-based exits

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
Conditional exits provide flexibility to handle complex market scenarios
and can be customized for specific trading strategies.

@see base_exit_strategy.py for base framework
"""

from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import asyncio

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitSignal,
    ExitReason,
    ExitType,
    ExitCondition,
    ExitConfiguration,
    ExitMetrics,
    ExitStatus
)


@dataclass
class ConditionalRule:
    """Single conditional rule for exit evaluation"""
    condition_id: str
    name: str
    condition_func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    priority: int = 1


class ConditionalExitStrategy(BaseExitStrategy):
    """
    @class ConditionalExitStrategy
    @brief Conditional exit strategy
    
    @details
    Evaluates complex market conditions using multiple criteria and logical
    combinations to determine optimal exit timing. Supports AND, OR, and
    NOT operations between conditions.
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Conditional logic configuration
        self.logic_operator = config.parameters.get('logic_operator', 'AND')  # AND, OR
        self.conditions = config.parameters.get('conditions', [])
        self.rules: List[ConditionalRule] = []
        
        # Initialize conditional rules
        self._initialize_conditional_rules()
        
        # Evaluation tracking
        self.condition_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Conditional exit strategy initialized: {self.config.name}")
    
    def _initialize_conditional_rules(self):
        """Initialize conditional rules based on configuration"""
        try:
            # Default conditions if none provided
            if not self.conditions:
                self.conditions = [
                    {
                        'id': 'price_decline',
                        'name': 'Price decline threshold',
                        'type': 'price_change',
                        'threshold': -0.05,  # 5% decline
                        'period': 10
                    },
                    {
                        'id': 'volume_spike',
                        'name': 'Volume spike',
                        'type': 'volume_ratio',
                        'threshold': 2.0,  # 2x average volume
                        'period': 20
                    }
                ]
            
            # Convert conditions to rules
            for condition in self.conditions:
                rule = ConditionalRule(
                    condition_id=condition['id'],
                    name=condition['name'],
                    condition_func=self._get_condition_function(condition['type']),
                    parameters=condition,
                    priority=condition.get('priority', 1)
                )
                self.rules.append(rule)
            
            logger.info(f"Initialized {len(self.rules)} conditional rules")
            
        except Exception as e:
            logger.error(f"Error initializing conditional rules: {e}")
    
    def _get_condition_function(self, condition_type: str) -> Callable:
        """Get condition evaluation function"""
        condition_functions = {
            'price_change': self._evaluate_price_change,
            'volume_ratio': self._evaluate_volume_ratio,
            'technical_indicator': self._evaluate_technical_indicator,
            'time_based': self._evaluate_time_condition,
            'volatility': self._evaluate_volatility_condition,
            'market_sentiment': self._evaluate_sentiment_condition
        }
        
        return condition_functions.get(condition_type, self._evaluate_price_change)
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Evaluate conditional exit conditions"""
        try:
            if not self.rules:
                return False
            
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Evaluate all conditions
            condition_results = []
            for rule in self.rules:
                if rule.is_active:
                    result = await self._evaluate_rule(rule, position)
                    condition_results.append(result)
                    
                    # Store result for analysis
                    if position_id not in self.condition_history:
                        self.condition_history[position_id] = []
                    
                    self.condition_history[position_id].append({
                        'condition_id': rule.condition_id,
                        'result': result,
                        'timestamp': datetime.utcnow()
                    })
            
            # Apply logical operator
            final_result = self._apply_logic_operator(condition_results)
            
            if final_result:
                logger.info(f"Conditional exit triggered for position {position_id}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error evaluating conditional conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """Generate conditional exit signal"""
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            # Calculate confidence based on conditions met
            confidence = await self._calculate_conditional_confidence(position)
            urgency = await self._calculate_conditional_urgency(position)
            
            # Determine which conditions were triggered
            triggered_conditions = await self._get_triggered_conditions(position)
            
            exit_signal = ExitSignal(
                signal_id=f"conditional_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.MARKET_CONDITION,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                metadata={
                    'logic_operator': self.logic_operator,
                    'triggered_conditions': triggered_conditions,
                    'total_conditions': len(self.rules),
                    'conditions_met': len(triggered_conditions)
                }
            )
            
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating conditional exit signal: {e}")
            return None
    
    async def _evaluate_rule(self, rule: ConditionalRule, position: Dict[str, Any]) -> bool:
        """Evaluate a single conditional rule"""
        try:
            result = await rule.condition_func(position, rule.parameters)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.condition_id}: {e}")
            return False
    
    async def _evaluate_price_change(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate price change condition"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return False
            
            period = params.get('period', 10)
            threshold = params.get('threshold', -0.05)  # Default 5% decline
            
            # Get historical data
            data = await self.context.get_historical_data(symbol, '1h', period + 1)
            if len(data) < period + 1:
                return False
            
            # Calculate price change
            start_price = Decimal(str(data[0].get('close', 0)))
            end_price = Decimal(str(data[-1].get('close', 0)))
            
            if start_price <= 0:
                return False
            
            price_change = (end_price - start_price) / start_price
            
            # Check if condition is met
            if threshold < 0:
                return price_change <= threshold  # Decline condition
            else:
                return price_change >= threshold  # Rise condition
            
        except Exception as e:
            logger.error(f"Error evaluating price change: {e}")
            return False
    
    async def _evaluate_volume_ratio(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate volume ratio condition"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return False
            
            period = params.get('period', 20)
            threshold = params.get('threshold', 2.0)  # Default 2x average
            
            # Get volume data
            data = await self.context.get_historical_data(symbol, '1h', period + 1)
            if len(data) < period:
                return False
            
            # Calculate volume ratio
            current_volume = data[-1].get('volume', 0)
            historical_volumes = [item.get('volume', 0) for item in data[:-1]]
            
            if not historical_volumes or sum(historical_volumes) == 0:
                return False
            
            avg_volume = sum(historical_volumes) / len(historical_volumes)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return volume_ratio >= threshold
            
        except Exception as e:
            logger.error(f"Error evaluating volume ratio: {e}")
            return False
    
    async def _evaluate_technical_indicator(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate technical indicator condition"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return False
            
            indicator = params.get('indicator', 'rsi')
            threshold = params.get('threshold', 70)
            period = params.get('period', 14)
            
            # Get historical data
            data = await self.context.get_historical_data(symbol, '1h', period + 1)
            if len(data) < period + 1:
                return False
            
            # Calculate indicator
            prices = [Decimal(str(item.get('close', 0))) for item in data]
            
            if indicator.lower() == 'rsi':
                rsi = self._calculate_rsi(prices, period)
                return rsi >= threshold if threshold > 50 else rsi <= threshold
            elif indicator.lower() == 'macd':
                macd_line, macd_signal, _ = self._calculate_macd(prices)
                return macd_line <= macd_signal  # MACD bearish crossover
            elif indicator.lower() == 'bollinger':
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20)
                current_price = float(prices[-1])
                return current_price <= float(bb_lower)  # Price below lower band
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating technical indicator: {e}")
            return False
    
    async def _evaluate_time_condition(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate time-based condition"""
        try:
            time_type = params.get('time_type', 'hold_duration')
            threshold = params.get('threshold', 24)  # hours
            
            created_at = position.get('created_at')
            if not created_at:
                return False
            
            if time_type == 'hold_duration':
                hold_duration = (datetime.utcnow() - created_at).total_seconds() / 3600
                return hold_duration >= threshold
            elif time_type == 'time_of_day':
                current_hour = datetime.utcnow().hour
                start_hour = params.get('start_hour', 15)  # 3 PM
                end_hour = params.get('end_hour', 16)  # 4 PM
                return start_hour <= current_hour <= end_hour
            elif time_type == 'day_of_week':
                current_day = datetime.utcnow().weekday()
                target_days = params.get('days', [4])  # Friday
                return current_day in target_days
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating time condition: {e}")
            return False
    
    async def _evaluate_volatility_condition(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate volatility condition"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return False
            
            threshold = params.get('threshold', 0.03)  # 3% volatility
            period = params.get('period', 14)
            
            # Get price data
            data = await self.context.get_historical_data(symbol, '1h', period + 1)
            if len(data) < period + 1:
                return False
            
            # Calculate volatility
            returns = []
            for i in range(1, len(data)):
                prev_close = Decimal(str(data[i-1].get('close', 0)))
                current_close = Decimal(str(data[i].get('close', 0)))
                
                if prev_close > 0:
                    daily_return = (current_close - prev_close) / prev_close
                    returns.append(float(daily_return))
            
            if not returns:
                return False
            
            # Calculate standard deviation (volatility)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            return volatility >= threshold
            
        except Exception as e:
            logger.error(f"Error evaluating volatility condition: {e}")
            return False
    
    async def _evaluate_sentiment_condition(self, position: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate market sentiment condition"""
        try:
            sentiment_threshold = params.get('sentiment_threshold', 0.0)
            # Simplified sentiment evaluation
            # In real implementation, would use news/social sentiment APIs
            
            return False  # Placeholder
            
        except Exception as e:
            logger.error(f"Error evaluating sentiment condition: {e}")
            return False
    
    def _apply_logic_operator(self, condition_results: List[bool]) -> bool:
        """Apply logical operator to condition results"""
        try:
            if not condition_results:
                return False
            
            if self.logic_operator.upper() == 'AND':
                return all(condition_results)
            elif self.logic_operator.upper() == 'OR':
                return any(condition_results)
            else:
                # Default to AND
                return all(condition_results)
                
        except Exception as e:
            logger.error(f"Error applying logic operator: {e}")
            return False
    
    async def _calculate_conditional_confidence(self, position: Dict[str, Any]) -> float:
        """Calculate confidence for conditional exit"""
        try:
            base_confidence = 0.70
            
            # Adjust based on number of conditions met
            triggered_conditions = await self._get_triggered_conditions(position)
            condition_ratio = len(triggered_conditions) / len(self.rules) if self.rules else 0
            
            confidence_boost = condition_ratio * 0.25
            final_confidence = min(0.95, base_confidence + confidence_boost)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating conditional confidence: {e}")
            return 0.70
    
    async def _calculate_conditional_urgency(self, position: Dict[str, Any]) -> float:
        """Calculate urgency for conditional exit"""
        try:
            triggered_conditions = await self._get_triggered_conditions(position)
            
            # Higher urgency for more conditions met
            urgency = 0.6 + (len(triggered_conditions) / len(self.rules)) * 0.3
            
            return min(0.95, urgency)
            
        except Exception as e:
            logger.error(f"Error calculating conditional urgency: {e}")
            return 0.70
    
    async def _get_triggered_conditions(self, position: Dict[str, Any]) -> List[str]:
        """Get list of triggered conditions"""
        try:
            triggered = []
            for rule in self.rules:
                if rule.is_active:
                    result = await self._evaluate_rule(rule, position)
                    if result:
                        triggered.append(rule.condition_id)
            return triggered
            
        except Exception as e:
            logger.error(f"Error getting triggered conditions: {e}")
            return []
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = float(prices[i] - prices[i-1])
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: List[Decimal]) -> tuple:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return 0.0, 0.0, 0.0
            
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd_line = float(ema_12 - ema_26)
            macd_signal = float(self._calculate_ema([Decimal(str(macd_line))] * 26, 9))
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> Decimal:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return Decimal('0')
            
            multiplier = Decimal('2') / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = price * multiplier + ema * (Decimal('1') - multiplier)
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return Decimal('0')
    
    def _calculate_bollinger_bands(self, prices: List[Decimal], period: int) -> tuple:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else Decimal('0')
                return current_price, current_price, current_price
            
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / len(recent_prices)
            
            variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
            std_dev = variance ** Decimal('0.5')
            
            upper_band = sma + (std_dev * Decimal('2'))
            lower_band = sma - (std_dev * Decimal('2'))
            
            return upper_band, sma, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = prices[-1] if prices else Decimal('0')
            return current_price, current_price, current_price


# Factory functions

def create_conditional_exit_strategy(
    strategy_id: str,
    symbol: str,
    conditions: List[Dict[str, Any]],
    logic_operator: str = 'AND',
    **kwargs
) -> ConditionalExitStrategy:
    """
    Create a conditional exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param conditions List of conditional rules
    @param logic_operator Logical operator ('AND', 'OR')
    @param kwargs Additional configuration parameters
    
    @returns Configured conditional exit strategy instance
    """
    parameters = {
        'conditions': conditions,
        'logic_operator': logic_operator,
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.CONDITIONAL,
        name=f"Conditional Exit ({symbol})",
        description=f"Conditional exit strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return ConditionalExitStrategy(config)


def create_multi_condition_exit_strategy(
    strategy_id: str,
    symbol: str,
    price_decline_threshold: float = -0.05,
    volume_spike_threshold: float = 2.0,
    rsi_threshold: float = 70,
    **kwargs
) -> ConditionalExitStrategy:
    """Create multi-condition exit strategy with common parameters"""
    conditions = [
        {
            'id': 'price_decline',
            'name': 'Price decline threshold',
            'type': 'price_change',
            'threshold': price_decline_threshold,
            'period': 10
        },
        {
            'id': 'volume_spike',
            'name': 'Volume spike',
            'type': 'volume_ratio',
            'threshold': volume_spike_threshold,
            'period': 20
        },
        {
            'id': 'rsi_overbought',
            'name': 'RSI overbought',
            'type': 'technical_indicator',
            'indicator': 'rsi',
            'threshold': rsi_threshold,
            'period': 14
        }
    ]
    
    return create_conditional_exit_strategy(
        strategy_id=strategy_id,
        symbol=symbol,
        conditions=conditions,
        logic_operator='OR',  # Exit if any condition is met
        **kwargs
    )
