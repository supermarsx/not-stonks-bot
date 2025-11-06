"""
Backtesting Framework
Comprehensive backtesting system for strategy evaluation
"""

from typing import Dict, Any, List, Optional, Union, Protocol
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from collections import defaultdict

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel
)


class BacktestStatus(Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: Decimal
    exit_price: Optional[Decimal]
    quantity: Decimal
    pnl: Decimal = Decimal('0')
    pnl_pct: float = 0.0
    commission: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')
    hold_time: Optional[timedelta] = None
    strategy_id: str = ""
    signal_id: Optional[str] = None


@dataclass
class Position:
    """Position tracking during backtest"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    value_at_risk: float = 0.0
    conditional_var: float = 0.0
    
    # Trade analysis
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Performance ratios
    recovery_factor: float = 0.0
    return_risk_ratio: float = 0.0
    
    # Additional metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    average_hold_time: timedelta = field(default_factory=lambda: timedelta(0))
    commission_paid: Decimal = Decimal('0')
    total_slippage: Decimal = Decimal('0')


@dataclass
class BacktestResult:
    """Complete backtest result"""
    backtest_id: str
    strategy_config: StrategyConfig
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    status: BacktestStatus
    metrics: BacktestMetrics
    trades: List[Trade]
    positions: List[Position]
    equity_curve: List[Dict[str, Any]]
    daily_returns: List[float]
    trade_history: List[Dict[str, Any]]
    execution_time: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class BacktestContext:
    """Mock context for backtesting strategies"""
    
    def __init__(self, market_data: Dict[str, List[Dict[str, Any]]]):
        self.market_data = market_data
        self.current_time = None
        self.current_prices = {}
        self.submitted_orders = []
        self.positions = {}
    
    async def get_market_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get historical market data"""
        return self.market_data.get(symbol, [])
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol"""
        return self.current_prices.get(symbol)
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Mock order submission"""
        order_id = f"BACKTEST_{len(self.submitted_orders)}"
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price or self.current_prices.get(symbol),
            'timestamp': self.current_time,
            'status': 'filled'
        }
        
        self.submitted_orders.append(order)
        
        return {
            'success': True,
            'order_id': order_id,
            'filled_quantity': quantity,
            'filled_price': order['price']
        }
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions"""
        if symbol:
            if symbol in self.positions:
                return [self.positions[symbol]]
            return []
        
        return list(self.positions.values())
    
    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value"""
        total_value = Decimal('0')
        for position in self.positions.values():
            total_value += position.market_value
        return total_value
    
    def log_message(self, level: str, message: str, **kwargs):
        """Log backtest messages"""
        logger.debug(f"[BACKTEST {level.upper()}] {message}")


class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self):
        self.slippage_model = 'fixed'  # 'fixed', 'percentage', 'volume_based'
        self.commission_model = 'fixed'  # 'fixed', 'percentage'
        self.default_slippage = Decimal('0.0001')  # 1 basis point
        self.default_commission = Decimal('1.0')  # $1 per trade
        self.commission_pct = Decimal('0.001')  # 0.1%
        
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal('100000'),
        timeframe: str = "1h",
        **kwargs
    ) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            strategy: Strategy to backtest
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            timeframe: Data timeframe
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        start_time = datetime.utcnow()
        backtest_id = f"BT_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting backtest: {backtest_id}")
        
        try:
            # Generate or load market data
            market_data = await self._load_market_data(symbols, start_date, end_date, timeframe)
            
            # Create backtest context
            context = BacktestContext(market_data)
            strategy.set_context(context)
            
            # Initialize backtest state
            current_capital = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            daily_returns = []
            
            # Main backtest loop
            current_time = start_date
            day_count = 0
            
            while current_time <= end_date:
                context.current_time = current_time
                
                # Update current prices and positions
                await self._update_market_state(context, symbols, current_time)
                
                # Update portfolio value
                portfolio_value = await self._calculate_portfolio_value(context, current_capital)
                
                # Generate and process signals
                signals = await strategy.generate_signals()
                
                for signal in signals:
                    if await strategy.validate_signal(signal):
                        trade = await self._execute_signal(signal, context, current_capital)
                        if trade:
                            trades.append(trade)
                            current_capital += trade.pnl - trade.commission
                            
                            # Update positions
                            await self._update_positions(positions, trade)
                
                # Record equity curve
                equity_curve.append({
                    'timestamp': current_time,
                    'portfolio_value': float(portfolio_value),
                    'cash': float(current_capital),
                    'positions': len(positions)
                })
                
                # Calculate daily returns
                if day_count > 0:
                    prev_value = equity_curve[day_count-1]['portfolio_value']
                    current_value = portfolio_value
                    daily_return = (float(current_value) - float(prev_value)) / float(prev_value)
                    daily_returns.append(daily_return)
                
                # Advance time (simplified - would use real market hours)
                current_time += timedelta(hours=1)
                day_count += 1
                
                # Progress logging
                if day_count % 24 == 0:
                    progress = (current_time - start_date) / (end_date - start_date)
                    logger.info(f"Backtest progress: {progress:.1%}")
            
            # Calculate final metrics
            final_capital = current_capital + await self._calculate_unrealized_pnl(positions)
            
            metrics = await self._calculate_metrics(
                initial_capital=initial_capital,
                final_capital=final_capital,
                trades=trades,
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                start_date=start_date,
                end_date=end_date
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create result
            result = BacktestResult(
                backtest_id=backtest_id,
                strategy_config=strategy.config,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                status=BacktestStatus.COMPLETED,
                metrics=metrics,
                trades=trades,
                positions=list(positions.values()),
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                trade_history=[self._trade_to_dict(trade) for trade in trades],
                execution_time=execution_time
            )
            
            logger.info(f"Backtest completed: {backtest_id}")
            logger.info(f"Total return: {metrics.total_return:.2%}")
            logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"Max drawdown: {metrics.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return BacktestResult(
                backtest_id=backtest_id,
                strategy_config=strategy.config,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=initial_capital,
                status=BacktestStatus.FAILED,
                metrics=BacktestMetrics(),
                trades=[],
                positions=[],
                equity_curve=[],
                daily_returns=[],
                trade_history=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _load_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load or generate market data for backtesting"""
        market_data = {}
        
        for symbol in symbols:
            # In a real implementation, this would load from database or API
            # For now, generate synthetic data
            data = await self._generate_synthetic_data(symbol, start_date, end_date)
            market_data[symbol] = data
        
        return market_data
    
    async def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate synthetic market data for testing"""
        import random
        
        data = []
        current_price = 100.0
        current_time = start_date
        
        # Generate daily data
        while current_time <= end_date:
            # Random walk with trend
            trend_factor = 0.0001  # 0.01% daily trend
            volatility = 0.02  # 2% daily volatility
            
            price_change = random.gauss(trend_factor, volatility)
            current_price *= (1 + price_change)
            
            # Generate OHLC from close price
            open_price = current_price * random.uniform(0.99, 1.01)
            high_price = max(current_price, open_price) * random.uniform(1.0, 1.02)
            low_price = min(current_price, open_price) * random.uniform(0.98, 1.0)
            volume = random.randint(100000, 1000000)
            
            data.append({
                'timestamp': current_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume
            })
            
            # Next day
            current_time += timedelta(days=1)
        
        return data
    
    async def _update_market_state(
        self,
        context: BacktestContext,
        symbols: List[str],
        current_time: datetime
    ):
        """Update current market state for backtest"""
        for symbol in symbols:
            # Find closest price data point
            market_data = context.market_data.get(symbol, [])
            if market_data:
                # Simplified: use last available price
                # In real implementation, would find exact timestamp
                latest_data = market_data[-1]
                context.current_prices[symbol] = Decimal(str(latest_data['close']))
    
    async def _calculate_portfolio_value(
        self,
        context: BacktestContext,
        cash: Decimal
    ) -> Decimal:
        """Calculate total portfolio value including positions"""
        total_value = cash
        
        for position in context.positions.values():
            total_value += position.market_value
        
        return total_value
    
    async def _execute_signal(
        self,
        signal: TradingSignal,
        context: BacktestContext,
        current_capital: Decimal
    ) -> Optional[Trade]:
        """Execute trading signal and create trade record"""
        try:
            symbol = signal.symbol
            price = signal.price or context.current_prices.get(symbol)
            
            if not price:
                return None
            
            # Apply slippage
            execution_price = self._apply_slippage(price, signal.signal_type)
            
            # Check if we have enough capital
            trade_value = signal.quantity * execution_price
            
            if signal.signal_type == SignalType.BUY and trade_value > current_capital:
                return None
            
            # Calculate commission
            commission = self._calculate_commission(trade_value)
            
            # Create trade record
            trade = Trade(
                trade_id=signal.signal_id,
                entry_time=context.current_time,
                exit_time=None,
                symbol=symbol,
                side=signal.signal_type.value,
                entry_price=execution_price,
                exit_price=None,
                quantity=signal.quantity,
                pnl=Decimal('0'),
                pnl_pct=0.0,
                commission=commission,
                slippage=execution_price - price,
                strategy_id=signal.strategy_id,
                signal_id=signal.signal_id
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None
    
    def _apply_slippage(self, price: Decimal, signal_type: SignalType) -> Decimal:
        """Apply slippage to execution price"""
        if self.slippage_model == 'fixed':
            if signal_type == SignalType.BUY:
                return price * (1 + self.default_slippage)
            else:  # SELL
                return price * (1 - self.default_slippage)
        else:
            return price  # No slippage
    
    def _calculate_commission(self, trade_value: Decimal) -> Decimal:
        """Calculate trading commission"""
        if self.commission_model == 'fixed':
            return self.default_commission
        elif self.commission_model == 'percentage':
            return trade_value * self.commission_pct
        else:
            return Decimal('0')
    
    async def _update_positions(self, positions: Dict[str, Position], trade: Trade):
        """Update position records after trade"""
        symbol = trade.symbol
        
        if symbol not in positions:
            positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal('0'),
                avg_price=Decimal('0'),
                market_value=Decimal('0'),
                unrealized_pnl=Decimal('0')
            )
        
        position = positions[symbol]
        
        if trade.side == 'buy':
            # Calculate new average price
            old_value = position.quantity * position.avg_price
            new_value = trade.quantity * trade.entry_price
            total_quantity = position.quantity + trade.quantity
            total_value = old_value + new_value
            
            if total_quantity > 0:
                position.avg_price = total_value / total_quantity
            position.quantity += trade.quantity
        else:  # sell
            position.quantity -= trade.quantity
            
            # Calculate realized P&L
            if position.quantity >= 0:  # Long position
                pnl = trade.quantity * (trade.entry_price - position.avg_price)
                position.realized_pnl += pnl
            
            if position.quantity == 0:
                position.avg_price = Decimal('0')
        
        # Update market value and unrealized P&L
        current_price = trade.entry_price  # Simplified
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
        position.last_update = trade.entry_time
    
    async def _calculate_unrealized_pnl(self, positions: Dict[str, Position]) -> Decimal:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in positions.values())
    
    async def _calculate_metrics(
        self,
        initial_capital: Decimal,
        final_capital: Decimal,
        trades: List[Trade],
        equity_curve: List[Dict[str, Any]],
        daily_returns: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        
        # Basic metrics
        total_return = (float(final_capital) - float(initial_capital)) / float(initial_capital)
        
        # Time period
        trading_days = (end_date - start_date).days
        years = trading_days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Trade analysis
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        # P&L analysis
        all_pnls = [float(t.pnl) for t in trades]
        winning_pnls = [pnl for pnl in all_pnls if pnl > 0]
        losing_pnls = [pnl for pnl in all_pnls if pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0
        largest_win = max(winning_pnls) if winning_pnls else 0
        largest_loss = abs(min(losing_pnls)) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        if daily_returns and len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (using downside deviation)
            negative_returns = [r for r in daily_returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        else:
            volatility = sharpe_ratio = sortino_ratio = 0
        
        # Maximum drawdown
        if equity_curve:
            values = [point['portfolio_value'] for point in equity_curve]
            peak = values[0]
            max_dd = 0
            current_dd = 0
            
            for value in values:
                if value > peak:
                    peak = value
                    current_dd = 0
                else:
                    current_dd = (peak - value) / peak
                    max_dd = max(max_dd, current_dd)
        else:
            max_dd = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0
        
        # Create metrics object
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            volatility=volatility,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            average_win=avg_win,
            average_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            recovery_factor=total_return / max_dd if max_dd > 0 else 0
        )
        
        return metrics
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'trade_id': trade.trade_id,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'quantity': float(trade.quantity),
            'pnl': float(trade.pnl),
            'pnl_pct': trade.pnl_pct,
            'commission': float(trade.commission),
            'slippage': float(trade.slippage),
            'hold_time': trade.hold_time.total_seconds() if trade.hold_time else None,
            'strategy_id': trade.strategy_id,
            'signal_id': trade.signal_id
        }


# Utility functions for backtesting
def compare_backtests(results: List[BacktestResult]) -> Dict[str, Any]:
    """Compare multiple backtest results"""
    if not results:
        return {}
    
    comparison = {
        'count': len(results),
        'best_return': max(r.metrics.total_return for r in results),
        'worst_return': min(r.metrics.total_return for r in results),
        'average_return': np.mean([r.metrics.total_return for r in results]),
        'best_sharpe': max(r.metrics.sharpe_ratio for r in results),
        'worst_sharpe': min(r.metrics.sharpe_ratio for r in results),
        'best_drawdown': min(r.metrics.max_drawdown for r in results),
        'worst_drawdown': max(r.metrics.max_drawdown for r in results),
        'results': []
    }
    
    # Add individual results
    for result in results:
        comparison['results'].append({
            'backtest_id': result.backtest_id,
            'strategy_name': result.strategy_config.name,
            'total_return': result.metrics.total_return,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown': result.metrics.max_drawdown,
            'win_rate': result.metrics.win_rate,
            'total_trades': result.metrics.total_trades
        })
    
    # Sort by total return
    comparison['results'].sort(key=lambda x: x['total_return'], reverse=True)
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    async def test_backtest_engine():
        # Create a simple test strategy
        from .trend_following import TrendFollowingStrategy, create_trend_following_strategy
        
        strategy = create_trend_following_strategy(
            strategy_id="test_tf",
            symbols=['AAPL'],
            fast_period=10,
            slow_period=30
        )
        
        # Create backtest engine
        engine = BacktestEngine()
        
        # Run backtest
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        result = await engine.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal('100000')
        )
        
        print("Backtest Results:")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  Annualized Return: {result.metrics.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        print(f"  Win Rate: {result.metrics.win_rate:.2%}")
        print(f"  Total Trades: {result.metrics.total_trades}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
    
    import asyncio
    asyncio.run(test_backtest_engine())