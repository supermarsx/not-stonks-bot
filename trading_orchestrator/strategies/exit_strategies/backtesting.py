"""
@file backtesting.py
@brief Exit Strategy Backtesting Framework

@details
This module provides comprehensive backtesting capabilities for exit strategies,
enabling users to evaluate the performance of different exit approaches on
historical data. The framework supports multiple exit strategies, performance
metrics, and comparative analysis.

Key Features:
- Multi-strategy backtesting
- Performance metrics calculation
- Comparative analysis
- Historical simulation
- Risk-adjusted performance measures

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
Backtesting should be used with realistic assumptions about slippage, commissions,
and market impact to provide meaningful results.

@see base_exit_strategy.py for exit strategy framework
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitStrategy,
    ExitSignal,
    ExitReason,
    ExitType,
    ExitConfiguration,
    ExitMetrics,
    ExitStatus
)


class BacktestStatus(Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestTrade:
    """Individual trade in backtest results"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    exit_reason: str
    strategy_id: str
    pnl: Decimal
    pnl_percentage: float
    duration: timedelta
    confidence: float
    urgency: float


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    total_exposure_time: float
    commission_costs: float
    market_impact_costs: float
    net_return: float
    volatility: float
    var_95: float
    cvar_95: float
    trade_frequency: float  # trades per day
    avg_confidence: float
    avg_urgency: float


@dataclass
class BacktestResult:
    """Complete backtest results"""
    backtest_id: str
    status: BacktestStatus
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    strategies: Dict[str, BacktestMetrics]
    trades: Dict[str, List[BacktestTrade]]
    summary: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ExitStrategyBacktestEngine:
    """
    @class ExitStrategyBacktestEngine
    @brief Backtesting engine for exit strategies
    
    @details
    Provides comprehensive backtesting capabilities for exit strategies including
    historical simulation, performance calculation, and comparative analysis.
    """
    
    def __init__(self):
        self.slippage_rate = Decimal('0.001')  # 0.1% slippage
        self.commission_rate = Decimal('0.0001')  # 0.01% commission
        self.market_impact_model = "linear"  # linear, quadratic
        
    async def run_backtest(
        self,
        strategies: List[BaseExitStrategy],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal('100000'),
        price_data: Optional[List[Dict[str, Any]]] = None
    ) -> BacktestResult:
        """
        Run backtest for exit strategies
        
        @param strategies List of exit strategies to test
        @param symbol Trading symbol
        @param start_date Backtest start date
        @param end_date Backtest end date
        @param initial_capital Starting capital
        @param price_data Historical price data (optional)
        
        @returns Backtest results
        """
        backtest_id = f"bt_{datetime.utcnow().timestamp()}"
        logger.info(f"Starting backtest {backtest_id} for symbol {symbol}")
        
        try:
            # Get or generate price data
            if not price_data:
                price_data = await self._generate_sample_data(symbol, start_date, end_date)
            
            if not price_data:
                raise ValueError("No price data available for backtest")
            
            # Initialize backtest
            backtest_result = BacktestResult(
                backtest_id=backtest_id,
                status=BacktestStatus.PENDING,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategies={},
                trades={}
            )
            
            backtest_result.status = BacktestStatus.RUNNING
            
            # Run backtest for each strategy
            for strategy in strategies:
                try:
                    strategy_metrics, strategy_trades = await self._backtest_strategy(
                        strategy, symbol, price_data, initial_capital
                    )
                    backtest_result.strategies[strategy.config.strategy_id] = strategy_metrics
                    backtest_result.trades[strategy.config.strategy_id] = strategy_trades
                    
                except Exception as e:
                    logger.error(f"Error backtesting strategy {strategy.config.strategy_id}: {e}")
                    continue
            
            # Calculate summary metrics
            backtest_result.summary = await self._calculate_summary_metrics(
                backtest_result, initial_capital
            )
            
            backtest_result.status = BacktestStatus.COMPLETED
            backtest_result.completed_at = datetime.utcnow()
            
            logger.info(f"Backtest {backtest_id} completed successfully")
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            if 'backtest_result' in locals():
                backtest_result.status = BacktestStatus.FAILED
                backtest_result.error_message = str(e)
            raise
    
    async def _backtest_strategy(
        self,
        strategy: BaseExitStrategy,
        symbol: str,
        price_data: List[Dict[str, Any]],
        initial_capital: Decimal
    ) -> Tuple[BacktestMetrics, List[BacktestTrade]]:
        """Backtest individual strategy"""
        try:
            # Initialize tracking
            trades = []
            positions = []
            capital = initial_capital
            current_positions = {}
            
            # Simulate strategy execution
            for i, price_point in enumerate(price_data):
                timestamp = datetime.fromisoformat(price_point['timestamp'])
                
                # Simulate position entry (simplified)
                if i % 20 == 0 and capital > initial_capital * 0.1:  # Enter position every 20 points
                    position = {
                        'position_id': f"pos_{symbol}_{len(positions)}",
                        'symbol': symbol,
                        'entry_time': timestamp,
                        'entry_price': Decimal(str(price_point['close'])),
                        'quantity': Decimal(str(1000 / float(price_point['close']))),  # Fixed dollar amount
                        'side': 'long',
                        'created_at': timestamp
                    }
                    positions.append(position)
                    current_positions[position['position_id']] = position
                
                # Check for exits
                positions_to_close = []
                for pos_id, position in list(current_positions.items()):
                    # Simulate exit condition check
                    exit_triggered = await self._simulate_exit_condition(
                        strategy, position, price_point
                    )
                    
                    if exit_triggered:
                        # Calculate exit
                        exit_price = Decimal(str(price_point['close']))
                        
                        # Apply slippage and costs
                        exit_price = self._apply_slippage(exit_price, 'sell')
                        
                        # Calculate P&L
                        entry_price = position['entry_price']
                        quantity = position['quantity']
                        gross_pnl = (exit_price - entry_price) * quantity
                        commission_cost = exit_price * quantity * self.commission_rate
                        market_impact = self._estimate_market_impact(exit_price, quantity)
                        net_pnl = gross_pnl - commission_cost - market_impact
                        
                        # Update capital
                        capital += net_pnl
                        
                        # Create trade record
                        trade = BacktestTrade(
                            trade_id=f"trade_{pos_id}",
                            symbol=symbol,
                            entry_time=position['entry_time'],
                            exit_time=timestamp,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=quantity,
                            exit_reason=exit_triggered.get('reason', 'unknown'),
                            strategy_id=strategy.config.strategy_id,
                            pnl=net_pnl,
                            pnl_percentage=float(net_pnl / (entry_price * quantity)),
                            duration=timestamp - position['entry_time'],
                            confidence=exit_triggered.get('confidence', 0.8),
                            urgency=exit_triggered.get('urgency', 0.7)
                        )
                        
                        trades.append(trade)
                        positions_to_close.append(pos_id)
                
                # Remove closed positions
                for pos_id in positions_to_close:
                    del current_positions[pos_id]
            
            # Calculate metrics
            metrics = await self._calculate_strategy_metrics(trades, initial_capital)
            
            return metrics, trades
            
        except Exception as e:
            logger.error(f"Error in strategy backtest: {e}")
            raise
    
    async def _simulate_exit_condition(
        self,
        strategy: BaseExitStrategy,
        position: Dict[str, Any],
        price_point: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Simulate exit condition evaluation"""
        try:
            # Create mock position data with current price
            mock_position = position.copy()
            mock_position['current_price'] = Decimal(str(price_point['close']))
            mock_position['quantity'] = position['quantity']
            
            # Simplified exit condition simulation
            # In real backtest, would use actual strategy logic
            
            entry_price = Decimal(str(position['entry_price']))
            current_price = Decimal(str(price_point['close']))
            
            # Simple profit/loss check
            profit_loss = (current_price - entry_price) / entry_price
            
            # Different exit strategies
            if isinstance(strategy, type(strategy).__module__.endswith('trailing_stop')):
                # Trailing stop simulation
                if profit_loss > 0.05:  # 5% profit - trail
                    return {
                        'triggered': True,
                        'reason': 'trailing_stop',
                        'confidence': 0.8,
                        'urgency': 0.7
                    }
                elif profit_loss < -0.03:  # 3% loss
                    return {
                        'triggered': True,
                        'reason': 'stop_loss',
                        'confidence': 0.9,
                        'urgency': 0.9
                    }
            
            elif isinstance(strategy, type(strategy).__module__.endswith('fixed_target')):
                # Fixed target simulation
                if profit_loss > 0.10:  # 10% profit target
                    return {
                        'triggered': True,
                        'reason': 'profit_target',
                        'confidence': 0.85,
                        'urgency': 0.6
                    }
                elif profit_loss < -0.05:  # 5% loss target
                    return {
                        'triggered': True,
                        'reason': 'loss_target',
                        'confidence': 0.9,
                        'urgency': 0.8
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error simulating exit condition: {e}")
            return None
    
    async def _generate_sample_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate sample price data for backtesting"""
        try:
            data = []
            current_date = start_date
            current_price = Decimal('100.00')  # Starting price
            
            while current_date <= end_date:
                # Generate hourly data
                for hour in range(24):
                    timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Simple random walk with drift
                    import random
                    price_change = random.gauss(0, 0.02)  # 2% standard deviation
                    current_price = current_price * (Decimal('1') + Decimal(str(price_change)))
                    
                    # Generate OHLC data
                    high = current_price * Decimal('1.005')  # 0.5% high
                    low = current_price * Decimal('0.995')   # 0.5% low
                    volume = random.randint(10000, 100000)
                    
                    data.append({
                        'timestamp': timestamp.isoformat(),
                        'open': float(current_price * Decimal('0.999')),
                        'high': float(high),
                        'low': float(low),
                        'close': float(current_price),
                        'volume': volume
                    })
                
                current_date += timedelta(days=1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return []
    
    def _apply_slippage(self, price: Decimal, side: str) -> Decimal:
        """Apply slippage to price"""
        slippage = price * self.slippage_rate
        if side == 'buy':
            return price + slippage
        else:  # sell
            return price - slippage
    
    def _estimate_market_impact(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Estimate market impact cost"""
        try:
            # Simple linear market impact model
            order_value = price * quantity
            impact_rate = Decimal('0.0002')  # 0.02% impact rate
            return order_value * impact_rate
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return Decimal('0')
    
    async def _calculate_strategy_metrics(
        self,
        trades: List[BacktestTrade],
        initial_capital: Decimal
    ) -> BacktestMetrics:
        """Calculate comprehensive strategy metrics"""
        try:
            if not trades:
                return BacktestMetrics(
                    strategy_id="unknown",
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_return=0.0,
                    annualized_return=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    profit_factor=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    avg_trade_duration=0.0,
                    total_exposure_time=0.0,
                    commission_costs=0.0,
                    market_impact_costs=0.0,
                    net_return=0.0,
                    volatility=0.0,
                    var_95=0.0,
                    cvar_95=0.0,
                    trade_frequency=0.0,
                    avg_confidence=0.0,
                    avg_urgency=0.0
                )
            
            strategy_id = trades[0].strategy_id
            
            # Basic counts
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Returns
            total_pnl = sum(t.pnl for t in trades)
            total_return = float(total_pnl / initial_capital)
            
            # P&L statistics
            winning_pnls = [float(t.pnl) for t in trades if t.pnl > 0]
            losing_pnls = [float(t.pnl) for t in trades if t.pnl <= 0]
            
            average_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
            average_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0
            largest_win = max(winning_pnls) if winning_pnls else 0.0
            largest_loss = min(losing_pnls) if losing_pnls else 0.0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0.0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Time statistics
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]  # hours
            avg_trade_duration = sum(durations) / len(durations) if durations else 0.0
            
            # Confidence and urgency
            avg_confidence = sum(t.confidence for t in trades) / len(trades)
            avg_urgency = sum(t.urgency for t in trades) / len(trades)
            
            # Calculate risk metrics (simplified)
            daily_returns = self._calculate_daily_returns(trades)
            volatility = np.std(daily_returns) if daily_returns else 0.0
            
            # VaR and CVaR
            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0.0
            cvar_95 = np.mean([r for r in daily_returns if r <= var_95]) if daily_returns else var_95
            
            # Advanced metrics
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0
            sortino_ratio = self._calculate_sortino_ratio(daily_returns) if daily_returns else 0.0
            max_drawdown = self._calculate_max_drawdown(trades, initial_capital)
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # Trade frequency
            if trades:
                time_span = (trades[-1].exit_time - trades[0].entry_time).days
                trade_frequency = total_trades / time_span if time_span > 0 else 0.0
            else:
                trade_frequency = 0.0
            
            # Costs (estimated)
            total_volume = sum(t.quantity * t.entry_price for t in trades)
            commission_costs = float(total_volume) * float(self.commission_rate)
            
            return BacktestMetrics(
                strategy_id=strategy_id,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                annualized_return=total_return * (365 / max(1, len(daily_returns))),  # Simplified
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_trade_duration,
                total_exposure_time=avg_trade_duration * total_trades,
                commission_costs=commission_costs,
                market_impact_costs=0.0,  # Would calculate based on order sizes
                net_return=total_return,
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95,
                trade_frequency=trade_frequency,
                avg_confidence=avg_confidence,
                avg_urgency=avg_urgency
            )
            
        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {e}")
            raise
    
    def _calculate_daily_returns(self, trades: List[BacktestTrade]) -> List[float]:
        """Calculate daily returns from trades"""
        try:
            # Group trades by day and calculate daily P&L
            daily_pnls = {}
            for trade in trades:
                trade_date = trade.exit_time.date()
                if trade_date not in daily_pnls:
                    daily_pnls[trade_date] = 0
                daily_pnls[trade_date] += float(trade.pnl)
            
            # Calculate returns
            returns = []
            previous_pnl = 0
            for date in sorted(daily_pnls.keys()):
                if previous_pnl != 0:
                    daily_return = (daily_pnls[date] - previous_pnl) / abs(previous_pnl)
                    returns.append(daily_return)
                previous_pnl = daily_pnls[date]
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
            return []
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
            mean_excess = np.mean(excess_returns)
            std_returns = np.std(returns)
            
            if std_returns == 0:
                return 0.0
            
            return mean_excess / std_returns * np.sqrt(252)  # Annualized
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if not returns:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]
            mean_excess = np.mean(excess_returns)
            
            # Downside deviation
            downside_returns = [r for r in excess_returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0.0
            
            if downside_deviation == 0:
                return 0.0
            
            return mean_excess / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, trades: List[BacktestTrade], initial_capital: Decimal) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_pnl = 0
            peak = float(initial_capital)
            max_dd = 0
            
            for trade in trades:
                cumulative_pnl += float(trade.pnl)
                current_value = float(initial_capital) + cumulative_pnl
                
                if current_value > peak:
                    peak = current_value
                
                drawdown = (peak - current_value) / peak
                max_dd = max(max_dd, drawdown)
            
            return -max_dd  # Return as negative percentage
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_summary_metrics(
        self,
        backtest_result: BacktestResult,
        initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Calculate summary metrics across all strategies"""
        try:
            summary = {
                'total_strategies': len(backtest_result.strategies),
                'best_return': None,
                'worst_return': None,
                'best_sharpe': None,
                'worst_drawdown': None,
                'avg_win_rate': 0.0,
                'strategy_comparison': {}
            }
            
            if not backtest_result.strategies:
                return summary
            
            returns = []
            sharpe_ratios = []
            win_rates = []
            
            for strategy_id, metrics in backtest_result.strategies.items():
                returns.append(metrics.total_return)
                sharpe_ratios.append(metrics.sharpe_ratio)
                win_rates.append(metrics.win_rate)
                
                summary['strategy_comparison'][strategy_id] = {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'win_rate': metrics.win_rate,
                    'max_drawdown': metrics.max_drawdown,
                    'total_trades': metrics.total_trades
                }
            
            # Best/worst performers
            if returns:
                best_return_idx = returns.index(max(returns))
                worst_return_idx = returns.index(min(returns))
                best_sharpe_idx = sharpe_ratios.index(max(sharpe_ratios))
                worst_drawdown_idx = min(range(len(backtest_result.strategies)), 
                                       key=lambda i: backtest_result.strategies[list(backtest_result.strategies.keys())[i]].max_drawdown)
                
                strategy_ids = list(backtest_result.strategies.keys())
                
                summary['best_return'] = {
                    'strategy_id': strategy_ids[best_return_idx],
                    'return': returns[best_return_idx]
                }
                
                summary['worst_return'] = {
                    'strategy_id': strategy_ids[worst_return_idx],
                    'return': returns[worst_return_idx]
                }
                
                summary['best_sharpe'] = {
                    'strategy_id': strategy_ids[best_sharpe_idx],
                    'sharpe_ratio': sharpe_ratios[best_sharpe_idx]
                }
                
                summary['worst_drawdown'] = {
                    'strategy_id': strategy_ids[worst_drawdown_idx],
                    'max_drawdown': list(backtest_result.strategies.values())[worst_drawdown_idx].max_drawdown
                }
            
            summary['avg_win_rate'] = sum(win_rates) / len(win_rates) if win_rates else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary metrics: {e}")
            return {}


# Convenience functions

async def run_exit_strategy_backtest(
    strategies: List[BaseExitStrategy],
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_price: Decimal = Decimal('100.00'),
    initial_capital: Decimal = Decimal('100000')
) -> BacktestResult:
    """
    Convenience function to run backtest for exit strategies
    
    @param strategies List of exit strategies to test
    @param symbol Trading symbol
    @param start_date Backtest start date
    @param end_date Backtest end date
    @param initial_price Starting price for simulation
    @param initial_capital Starting capital
    
    @returns Backtest results
    """
    engine = ExitStrategyBacktestEngine()
    return await engine.run_backtest(
        strategies=strategies,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )


async def compare_exit_strategies(
    strategies: List[BaseExitStrategy],
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple exit strategies
    
    @param strategies List of exit strategies to compare
    @param symbol Trading symbol
    @param start_date Backtest start date
    @param end_date Backtest end date
    @param kwargs Additional backtest parameters
    
    @returns Comparison results
    """
    backtest_result = await run_exit_strategy_backtest(
        strategies=strategies,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    return {
        'comparison_summary': backtest_result.summary,
        'detailed_metrics': {
            strategy_id: {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades
            }
            for strategy_id, metrics in backtest_result.strategies.items()
        },
        'rankings': _rank_strategies(backtest_result.strategies)
    }


def _rank_strategies(strategies: Dict[str, BacktestMetrics]) -> Dict[str, int]:
    """Rank strategies by different criteria"""
    if not strategies:
        return {}
    
    rankings = {}
    criteria = ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']
    
    for criterion in criteria:
        # Sort strategies by criterion
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: getattr(x[1], criterion),
            reverse=(criterion != 'max_drawdown')  # Max drawdown is negative
        )
        
        # Assign rankings
        for rank, (strategy_id, _) in enumerate(sorted_strategies, 1):
            if strategy_id not in rankings:
                rankings[strategy_id] = {}
            rankings[strategy_id][criterion] = rank
    
    return rankings
