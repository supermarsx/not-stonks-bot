"""
Demo Strategy Backtesting - Historical simulation and strategy performance testing
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json
from pathlib import Path

from loguru import logger

from .demo_mode_manager import DemoModeManager
from .virtual_broker import VirtualBroker
from .virtual_portfolio import VirtualPortfolio
from .demo_logging import DemoLogger
# Import from base brokers (will be available when used in the full system)
try:
    from ..brokers.base import MarketDataPoint, OrderInfo
except ImportError:
    from typing import Any
    MarketDataPoint = Any
    OrderInfo = Any


class BacktestMode(Enum):
    """Backtest execution modes"""
    FAST = "fast"  # Basic execution
    REALISTIC = "realistic"  # With slippage and market impact
    DETAILED = "detailed"  # Full simulation with all features
    MONTE_CARLO = "monte_carlo"  # Multiple scenario simulation


class StrategyPerformance(Enum):
    """Strategy performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTED_SHORTFALL = "expected_shortfall"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    # Basic settings
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Strategy settings
    strategy_config: Dict[str, Any] = None
    position_sizing: str = "fixed"  # fixed, kelly, optimal_f
    
    # Risk management
    max_position_size: float = 0.1  # 10% max per position
    max_sector_exposure: float = 0.2  # 20% max per sector
    stop_loss: Optional[float] = None  # e.g., 0.05 for 5% stop loss
    take_profit: Optional[float] = None  # e.g., 0.10 for 10% take profit
    
    # Execution settings
    mode: BacktestMode = BacktestMode.REALISTIC
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    
    # Data settings
    data_frequency: str = "1d"  # 1m, 5m, 1h, 1d
    lookback_period: int = 252  # days for calculations
    
    # Output settings
    save_trades: bool = True
    save_positions: bool = True
    generate_reports: bool = True
    export_results: bool = True


@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # long or short
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    duration: Optional[float]  # days
    strategy_signal: str


@dataclass
class BacktestPosition:
    """Position at a point in time"""
    timestamp: datetime
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float


@dataclass
class BacktestSnapshot:
    """Portfolio snapshot during backtest"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    invested_value: float
    total_pnl: float
    daily_return: float
    cumulative_return: float
    positions: List[BacktestPosition]
    trades_count: int
    win_rate: float


@dataclass
class BacktestResults:
    """Complete backtest results"""
    config: BacktestConfig
    trades: List[BacktestTrade]
    snapshots: List[BacktestSnapshot]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    trade_statistics: Dict[str, Any]
    benchmark_comparison: Dict[str, float]
    execution_summary: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    duration: float


class DemoStrategy:
    """Base class for demo strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.positions = {}
        self.cash = 0
        self.trades = []
    
    async def initialize(self, initial_capital: float):
        """Initialize strategy with capital"""
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
    
    async def generate_signals(self, market_data: Dict[str, List[MarketDataPoint]]) -> Dict[str, str]:
        """Generate trading signals for all symbols
        
        Returns:
            Dict mapping symbol to signal ('long', 'short', 'close', 'hold')
        """
        signals = {}
        for symbol, data in market_data.items():
            signal = await self._generate_signal(symbol, data)
            if signal:
                signals[symbol] = signal
        return signals
    
    async def calculate_position_size(self, symbol: str, signal: str, portfolio_value: float) -> float:
        """Calculate position size based on signal and risk parameters"""
        if signal == 'hold':
            return 0
        
        # Base position size (5% of portfolio)
        base_size = portfolio_value * 0.05
        
        # Adjust based on volatility if configured
        if 'use_volatility_sizing' in self.config and self.config['use_volatility_sizing']:
            # Simplified volatility adjustment
            vol_adjustment = 0.5  # Reduced size for high vol
            base_size *= vol_adjustment
        
        # Convert to shares (simplified)
        position_value = base_size
        
        # Return absolute quantity (notional value / price)
        # Price will be determined during execution
        return position_value
    
    async def update_positions(self, signals: Dict[str, str], current_prices: Dict[str, float]):
        """Update strategy positions based on signals"""
        for symbol, signal in signals.items():
            if signal == 'close' and symbol in self.positions:
                # Close existing position
                self.positions[symbol] = 0
            elif signal in ['long', 'short'] and symbol not in self.positions:
                # Open new position
                self.positions[symbol] = signal
    
    def get_current_positions(self) -> Dict[str, str]:
        """Get current strategy positions"""
        return {symbol: side for symbol, side in self.positions.items() if side}
    
    async def _generate_signal(self, symbol: str, data: List[MarketDataPoint]) -> str:
        """Override in subclasses to implement strategy logic"""
        if len(data) < 20:
            return 'hold'
        
        # Simple momentum strategy as default
        prices = [d.close for d in data[-20:]]
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        
        if short_ma > long_ma * 1.02:  # 2% threshold
            return 'long'
        elif short_ma < long_ma * 0.98:
            return 'short'
        else:
            return 'hold'


class DemoBacktester:
    """
    Demo strategy backtesting engine
    
    Provides comprehensive backtesting capabilities including:
    - Historical data simulation
    - Strategy execution
    - Performance calculation
    - Risk analysis
    - Monte Carlo simulation
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.logger = None  # Will be initialized when needed
    
    async def run_backtest(
        self,
        strategy: DemoStrategy,
        symbols: List[str],
        config: BacktestConfig
    ) -> BacktestResults:
        """Run complete backtest"""
        try:
            self.logger = await get_demo_logger()
            
            await self.logger.log_system_event(
                "backtest_started", "demo_backtester", "info", 0.0,
                {
                    "strategy": strategy.name,
                    "symbols": symbols,
                    "start_date": config.start_date.isoformat(),
                    "end_date": config.end_date.isoformat(),
                    "mode": config.mode.value
                }
            )
            
            # Initialize backtest
            portfolio = VirtualPortfolio(self.demo_manager)
            portfolio.initial_value = config.initial_capital
            portfolio.cash_balance = config.initial_capital
            
            # Load or generate historical data
            market_data = await self._load_historical_data(symbols, config)
            
            # Execute backtest
            results = await self._execute_backtest(strategy, market_data, config, portfolio)
            
            # Calculate performance metrics
            results.performance_metrics = await self._calculate_performance_metrics(results)
            results.risk_metrics = await self._calculate_risk_metrics(results)
            results.trade_statistics = await self._calculate_trade_statistics(results)
            results.benchmark_comparison = await self._calculate_benchmark_comparison(results, config)
            results.execution_summary = await self._generate_execution_summary(results)
            
            await self.logger.log_system_event(
                "backtest_completed", "demo_backtester", "info", 0.0,
                {
                    "total_return": results.performance_metrics.get('total_return', 0),
                    "trades_count": len(results.trades),
                    "duration": results.duration
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            if self.logger:
                await self.logger.log_system_event(
                    "backtest_failed", "demo_backtester", "error", 0.0,
                    {"error": str(e)}
                )
            raise
    
    async def run_multiple_scenarios(
        self,
        strategy: DemoStrategy,
        symbols: List[str],
        base_config: BacktestConfig,
        scenarios: int = 100
    ) -> List[BacktestResults]:
        """Run Monte Carlo simulation with multiple scenarios"""
        try:
            await self.logger.log_system_event(
                "monte_carlo_started", "demo_backtester", "info", 0.0,
                {"scenarios": scenarios, "strategy": strategy.name}
            )
            
            results_list = []
            
            for i in range(scenarios):
                # Vary configuration for each scenario
                scenario_config = await self._create_scenario_config(base_config, i)
                
                try:
                    result = await self.run_backtest(strategy, symbols, scenario_config)
                    results_list.append(result)
                    
                    if (i + 1) % 10 == 0:
                        await self.logger.log_system_event(
                            "monte_carlo_progress", "demo_backtester", "info", 0.0,
                            {"completed": i + 1, "total": scenarios}
                        )
                
                except Exception as e:
                    logger.warning(f"Scenario {i} failed: {e}")
                    continue
            
            # Calculate Monte Carlo statistics
            await self._calculate_monte_carlo_statistics(results_list)
            
            await self.logger.log_system_event(
                "monte_carlo_completed", "demo_backtester", "info", 0.0,
                {"successful_scenarios": len(results_list), "total_scenarios": scenarios}
            )
            
            return results_list
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            raise
    
    async def optimize_strategy(
        self,
        strategy_class,
        symbols: List[str],
        config_template: BacktestConfig,
        parameter_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            await self.logger.log_system_event(
                "optimization_started", "demo_backtester", "info", 0.0,
                {"strategy": strategy_class.__name__, "parameters": parameter_ranges}
            )
            
            best_result = None
            best_score = float('-inf')
            optimization_results = []
            
            # Generate all parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            for i, params in enumerate(param_combinations):
                try:
                    # Create config with current parameters
                    config = await self._create_optimization_config(config_template, params)
                    
                    # Create strategy instance
                    strategy = strategy_class(config.strategy_config)
                    
                    # Run backtest
                    result = await self.run_backtest(strategy, symbols, config)
                    
                    # Calculate optimization score (Sharpe ratio by default)
                    score = result.performance_metrics.get('sharpe_ratio', 0)
                    
                    optimization_results.append({
                        "parameters": params,
                        "score": score,
                        "metrics": result.performance_metrics,
                        "trades_count": len(result.trades)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                    
                    if (i + 1) % 10 == 0:
                        await self.logger.log_system_event(
                            "optimization_progress", "demo_backtester", "info", 0.0,
                            {"tested": i + 1, "total": len(param_combinations)}
                        )
                
                except Exception as e:
                    logger.warning(f"Optimization run {i} failed: {e}")
                    continue
            
            # Sort results by score
            optimization_results.sort(key=lambda x: x['score'], reverse=True)
            
            await self.logger.log_system_event(
                "optimization_completed", "demo_backtester", "info", 0.0,
                {
                    "best_score": best_score,
                    "best_parameters": optimization_results[0]['parameters'] if optimization_results else None,
                    "total_runs": len(optimization_results)
                }
            )
            
            return {
                "best_result": best_result,
                "best_score": best_score,
                "best_parameters": optimization_results[0]['parameters'] if optimization_results else None,
                "all_results": optimization_results[:20],  # Top 20 results
                "optimization_summary": {
                    "total_runs": len(optimization_results),
                    "average_score": np.mean([r['score'] for r in optimization_results]) if optimization_results else 0,
                    "best_sharpe": max([r['score'] for r in optimization_results]) if optimization_results else 0,
                    "worst_sharpe": min([r['score'] for r in optimization_results]) if optimization_results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            raise
    
    # Private methods
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        config: BacktestConfig
    ) -> Dict[str, List[MarketDataPoint]]:
        """Load or generate historical market data"""
        try:
            market_data = {}
            
            for symbol in symbols:
                # Generate synthetic historical data
                data = await self._generate_synthetic_historical_data(symbol, config)
                market_data[symbol] = data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    async def _generate_synthetic_historical_data(
        self,
        symbol: str,
        config: BacktestConfig
    ) -> List[MarketDataPoint]:
        """Generate synthetic historical data for backtesting"""
        try:
            data = []
            current_date = config.start_date
            current_price = 100.0  # Starting price
            
            # Calculate number of periods
            if config.data_frequency == "1d":
                total_periods = (config.end_date - config.start_date).days
                time_delta = timedelta(days=1)
            elif config.data_frequency == "1h":
                total_periods = int((config.end_date - config.start_date).total_seconds() / 3600)
                time_delta = timedelta(hours=1)
            else:  # Default to daily
                total_periods = (config.end_date - config.start_date).days
                time_delta = timedelta(days=1)
            
            # Generate OHLCV data
            for i in range(total_periods):
                # Random walk with trend and volatility
                daily_return = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% volatility
                current_price *= (1 + daily_return)
                
                # Generate OHLC from close price
                volatility = current_price * 0.01  # 1% intraday volatility
                open_price = current_price * (1 + np.random.normal(0, 0.002))
                high_price = max(open_price, current_price) + abs(np.random.normal(0, volatility))
                low_price = min(open_price, current_price) - abs(np.random.normal(0, volatility))
                close_price = current_price
                volume = np.random.lognormal(15, 1)  # Realistic volume distribution
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    broker_name="demo_backtester",
                    timestamp=current_date,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe=config.data_frequency,
                    metadata={"synthetic": True, "backtest": True}
                )
                
                data.append(data_point)
                current_date += time_delta
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return []
    
    async def _execute_backtest(
        self,
        strategy: DemoStrategy,
        market_data: Dict[str, List[MarketDataPoint]],
        config: BacktestConfig,
        portfolio: VirtualPortfolio
    ) -> BacktestResults:
        """Execute the backtest simulation"""
        try:
            start_time = datetime.now()
            trades = []
            snapshots = []
            
            # Initialize strategy
            await strategy.initialize(config.initial_capital)
            
            # Get all unique timestamps
            all_timestamps = set()
            for data in market_data.values():
                all_timestamps.update([d.timestamp for d in data])
            
            timestamps = sorted(list(all_timestamps))
            
            # Iterate through time periods
            for i, timestamp in enumerate(timestamps):
                # Get current market data for all symbols
                current_data = {}
                current_prices = {}
                
                for symbol, data in market_data.items():
                    # Get data up to current timestamp
                    symbol_data = [d for d in data if d.timestamp <= timestamp]
                    if symbol_data:
                        current_data[symbol] = symbol_data
                        current_prices[symbol] = symbol_data[-1].close
                
                if not current_prices:
                    continue
                
                # Generate trading signals
                signals = await strategy.generate_signals(current_data)
                
                # Update portfolio
                await portfolio.update_market_prices(current_prices)
                
                # Execute trades based on signals
                await self._execute_signals_as_trades(
                    strategy, signals, current_prices, config, portfolio, timestamp, trades
                )
                
                # Take snapshot
                if i % max(1, len(timestamps) // 252) == 0:  # Daily snapshots
                    snapshot = await self._create_backtest_snapshot(
                        portfolio, timestamp, current_prices
                    )
                    snapshots.append(snapshot)
            
            # Create final snapshot
            final_snapshot = await self._create_backtest_snapshot(
                portfolio, timestamps[-1], current_prices
            )
            snapshots.append(final_snapshot)
            
            end_time = datetime.now()
            
            return BacktestResults(
                config=config,
                trades=trades,
                snapshots=snapshots,
                performance_metrics={},
                risk_metrics={},
                trade_statistics={},
                benchmark_comparison={},
                execution_summary={},
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error executing backtest: {e}")
            raise
    
    async def _execute_signals_as_trades(
        self,
        strategy: DemoStrategy,
        signals: Dict[str, str],
        current_prices: Dict[str, float],
        config: BacktestConfig,
        portfolio: VirtualPortfolio,
        timestamp: datetime,
        trades: List[BacktestTrade]
    ):
        """Execute trading signals as actual trades"""
        try:
            portfolio_value = await portfolio.get_portfolio_value()
            
            for symbol, signal in signals.items():
                if signal in ['long', 'short'] and symbol in current_prices:
                    # Calculate position size
                    position_value = await strategy.calculate_position_size(symbol, signal, portfolio_value)
                    
                    if position_value > 1000:  # Minimum trade size
                        price = current_prices[symbol]
                        quantity = position_value / price
                        
                        # Execute trade
                        commission = position_value * config.commission_rate
                        slippage = position_value * config.slippage_rate
                        
                        # Simulate trade execution
                        side = signal
                        trade_id = str(uuid.uuid4())
                        
                        # Create trade record
                        trade = BacktestTrade(
                            trade_id=trade_id,
                            entry_time=timestamp,
                            exit_time=None,
                            symbol=symbol,
                            side=side,
                            entry_price=price,
                            exit_price=None,
                            quantity=quantity,
                            commission=commission,
                            slippage=slippage,
                            pnl=0,  # Will be calculated on exit
                            pnl_pct=0,
                            duration=None,
                            strategy_signal=signal
                        )
                        
                        trades.append(trade)
                        
                        # Update portfolio
                        quantity_change = quantity if side == 'long' else -quantity
                        await portfolio.update_position(
                            symbol, quantity_change, price, commission, slippage
                        )
                
                elif signal == 'close' and symbol in current_prices:
                    # Close existing position
                    position = await portfolio.get_position(symbol)
                    if position and position.quantity != 0:
                        price = current_prices[symbol]
                        
                        # Find corresponding entry trade
                        entry_trade = None
                        for trade in reversed(trades):
                            if trade.symbol == symbol and trade.exit_time is None:
                                entry_trade = trade
                                break
                        
                        if entry_trade:
                            # Calculate P&L
                            if entry_trade.side == 'long':
                                pnl = (price - entry_trade.entry_price) * entry_trade.quantity
                            else:  # short
                                pnl = (entry_trade.entry_price - price) * entry_trade.quantity
                            
                            pnl_pct = (pnl / (entry_trade.entry_price * entry_trade.quantity)) * 100
                            duration = (timestamp - entry_trade.entry_time).days
                            
                            # Update trade record
                            entry_trade.exit_time = timestamp
                            entry_trade.exit_price = price
                            entry_trade.pnl = pnl - entry_trade.commission
                            entry_trade.pnl_pct = pnl_pct
                            entry_trade.duration = duration
                        
                        # Update portfolio
                        await portfolio.update_position(symbol, -position.quantity, price)
        
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
    
    async def _create_backtest_snapshot(
        self,
        portfolio: VirtualPortfolio,
        timestamp: datetime,
        current_prices: Dict[str, float]
    ) -> BacktestSnapshot:
        """Create a backtest portfolio snapshot"""
        try:
            portfolio_value = await portfolio.get_portfolio_value()
            positions_data = await portfolio.get_positions_summary()
            
            # Convert to backtest positions
            positions = []
            for pos_data in positions_data:
                if pos_data.quantity != 0:
                    position = BacktestPosition(
                        timestamp=timestamp,
                        symbol=pos_data.symbol,
                        quantity=pos_data.quantity,
                        avg_price=pos_data.avg_entry_price,
                        current_price=pos_data.current_price,
                        market_value=pos_data.market_value,
                        unrealized_pnl=pos_data.unrealized_pnl,
                        weight=pos_data.weight
                    )
                    positions.append(position)
            
            return BacktestSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash=portfolio.cash_balance,
                invested_value=portfolio_value - portfolio.cash_balance,
                total_pnl=portfolio_value - portfolio.initial_value,
                daily_return=0,  # Would need previous snapshot to calculate
                cumulative_return=((portfolio_value - portfolio.initial_value) / portfolio.initial_value) * 100,
                positions=positions,
                trades_count=len(portfolio.trades),
                win_rate=0  # Would calculate from trade history
            )
            
        except Exception as e:
            logger.error(f"Error creating backtest snapshot: {e}")
            raise
    
    async def _calculate_performance_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if not results.snapshots:
                return {}
            
            # Extract portfolio values
            values = [snap.portfolio_value for snap in results.snapshots]
            returns = [snap.cumulative_return for snap in results.snapshots]
            
            # Basic metrics
            initial_value = results.config.initial_capital
            final_value = values[-1]
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Time period
            total_days = (results.snapshots[-1].timestamp - results.snapshots[0].timestamp).days
            annualized_return = ((final_value / initial_value) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0
            
            # Risk metrics
            daily_returns = self._calculate_daily_returns(values)
            volatility = np.std(daily_returns) * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
            
            # Risk-adjusted metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            peak_value = max(values)
            max_drawdown = ((peak_value - min(values)) / peak_value) * 100 if peak_value > 0 else 0
            
            # Other metrics
            win_rate = await self._calculate_win_rate(results.trades)
            profit_factor = await self._calculate_profit_factor(results.trades)
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "final_value": final_value,
                "total_days": total_days
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _calculate_risk_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            if not results.snapshots:
                return {}
            
            values = [snap.portfolio_value for snap in results.snapshots]
            daily_returns = self._calculate_daily_returns(values)
            
            if len(daily_returns) < 30:
                return {}
            
            returns_array = np.array(daily_returns)
            
            # Value at Risk
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            # Expected Shortfall
            expected_shortfall = np.mean(returns_array[returns_array <= var_95])
            
            # Volatility measures
            volatility = np.std(returns_array) * np.sqrt(252) * 100
            downside_deviation = np.std(returns_array[returns_array < 0]) * np.sqrt(252) * 100
            
            # Sortino ratio
            mean_return = np.mean(returns_array) * 252 * 100
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            
            return {
                "var_95": var_95 * 100,
                "var_99": var_99 * 100,
                "expected_shortfall": expected_shortfall * 100,
                "volatility": volatility,
                "downside_deviation": downside_deviation,
                "sortino_ratio": sortino_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _calculate_trade_statistics(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate trade statistics"""
        try:
            if not results.trades:
                return {}
            
            completed_trades = [t for t in results.trades if t.exit_time is not None]
            
            if not completed_trades:
                return {"total_trades": len(results.trades)}
            
            # Basic trade stats
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = (winning_trades / total_trades) * 100
            
            # P&L statistics
            pnls = [t.pnl for t in completed_trades]
            avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([p for p in pnls if p < 0]) if losing_trades > 0 else 0
            
            # Trade duration
            durations = [t.duration for t in completed_trades if t.duration is not None]
            avg_duration = np.mean(durations) if durations else 0
            
            # Largest trades
            largest_win = max(pnls) if pnls else 0
            largest_loss = min(pnls) if pnls else 0
            
            return {
                "total_trades": len(results.trades),
                "completed_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_duration": avg_duration,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "avg_trade_size": np.mean([abs(t.entry_price * t.quantity) for t in completed_trades])
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    async def _calculate_benchmark_comparison(
        self,
        results: BacktestResults,
        config: BacktestConfig
    ) -> Dict[str, float]:
        """Compare strategy performance to benchmark"""
        try:
            # Generate benchmark data (simplified)
            benchmark_return = await self._generate_benchmark_return(results, config)
            
            strategy_return = results.performance_metrics.get('total_return', 0)
            
            # Calculate excess return
            excess_return = strategy_return - benchmark_return
            
            return {
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "information_ratio": excess_return / results.risk_metrics.get('volatility', 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}
    
    async def _generate_benchmark_return(
        self,
        results: BacktestResults,
        config: BacktestConfig
    ) -> float:
        """Generate benchmark return for comparison"""
        try:
            # Simplified benchmark calculation
            total_days = (results.snapshots[-1].timestamp - results.snapshots[0].timestamp).days
            benchmark_annual_return = 0.08  # 8% assumed market return
            benchmark_return = (1 + benchmark_annual_return) ** (total_days / 365) - 1
            return benchmark_return * 100
        except:
            return 0
    
    async def _generate_execution_summary(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate execution summary"""
        try:
            return {
                "strategy_name": "DemoStrategy",
                "symbols": list(set([t.symbol for t in results.trades])),
                "total_trades": len(results.trades),
                "completed_trades": len([t for t in results.trades if t.exit_time is not None]),
                "total_commission": sum(t.commission for t in results.trades),
                "total_slippage": sum(t.slippage for t in results.trades),
                "execution_time": results.duration,
                "data_points_processed": sum(len(snap.positions) for snap in results.snapshots),
                "configuration": asdict(results.config)
            }
        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return {}
    
    def _calculate_daily_returns(self, values: List[float]) -> List[float]:
        """Calculate daily returns from portfolio values"""
        returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            returns.append(daily_return)
        return returns
    
    async def _calculate_win_rate(self, trades: List[BacktestTrade]) -> float:
        """Calculate win rate from trades"""
        completed_trades = [t for t in trades if t.exit_time is not None]
        if not completed_trades:
            return 0
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        return (winning_trades / len(completed_trades)) * 100
    
    async def _calculate_profit_factor(self, trades: List[BacktestTrade]) -> float:
        """Calculate profit factor from trades"""
        completed_trades = [t for t in trades if t.exit_time is not None]
        if not completed_trades:
            return 0
        
        gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    async def _create_scenario_config(self, base_config: BacktestConfig, scenario_id: int) -> BacktestConfig:
        """Create configuration for Monte Carlo scenario"""
        # Vary some parameters for each scenario
        config = asdict(base_config)
        
        # Add random variations
        config['commission_rate'] *= (1 + np.random.normal(0, 0.1))
        config['slippage_rate'] *= (1 + np.random.normal(0, 0.2))
        config['max_position_size'] *= (1 + np.random.normal(0, 0.1))
        
        return BacktestConfig(**config)
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for optimization"""
        import itertools
        
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    async def _create_optimization_config(self, base_config: BacktestConfig, parameters: Dict[str, Any]) -> BacktestConfig:
        """Create configuration with optimization parameters"""
        config_dict = asdict(base_config)
        
        # Update strategy config with parameters
        if 'strategy_config' not in config_dict:
            config_dict['strategy_config'] = {}
        config_dict['strategy_config'].update(parameters)
        
        return BacktestConfig(**config_dict)
    
    async def _calculate_monte_carlo_statistics(self, results_list: List[BacktestResults]):
        """Calculate statistics across Monte Carlo scenarios"""
        try:
            returns = [r.performance_metrics.get('total_return', 0) for r in results_list]
            sharpe_ratios = [r.performance_metrics.get('sharpe_ratio', 0) for r in results_list]
            
            if not returns:
                return
            
            await self.logger.log_system_event(
                "monte_carlo_statistics", "demo_backtester", "info", 0.0,
                {
                    "mean_return": np.mean(returns),
                    "std_return": np.std(returns),
                    "mean_sharpe": np.mean(sharpe_ratios),
                    "best_return": max(returns),
                    "worst_return": min(returns),
                    "success_rate": len([r for r in results_list if r.performance_metrics.get('total_return', 0) > 0]) / len(results_list) * 100
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo statistics: {e}")


# Global backtester instance
demo_backtester = None


async def get_demo_backtester() -> DemoBacktester:
    """Get global demo backtester instance"""
    global demo_backtester
    if demo_backtester is None:
        manager = await get_demo_manager()
        demo_backtester = DemoBacktester(manager)
    return demo_backtester


# Utility functions
async def run_simple_backtest(
    strategy: DemoStrategy,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000
) -> BacktestResults:
    """Run a simple backtest with default settings"""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    backtester = await get_demo_backtester()
    return await backtester.run_backtest(strategy, symbols, config)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager and enable demo mode
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Create simple demo strategy
        class SimpleMomentumStrategy(DemoStrategy):
            async def _generate_signal(self, symbol: str, data: List[MarketDataPoint]) -> str:
                if len(data) < 20:
                    return 'hold'
                
                prices = [d.close for d in data[-10:]]
                current_price = prices[-1]
                ma_10 = np.mean(prices)
                
                if current_price > ma_10 * 1.01:
                    return 'long'
                elif current_price < ma_10 * 0.99:
                    return 'short'
                else:
                    return 'hold'
        
        # Run backtest
        strategy = SimpleMomentumStrategy({})
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        results = await run_simple_backtest(strategy, symbols, start_date, end_date)
        
        print(f"Backtest completed:")
        print(f"Total return: {results.performance_metrics.get('total_return', 0):.2f}%")
        print(f"Sharpe ratio: {results.performance_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Total trades: {len(results.trades)}")
    
    asyncio.run(main())
