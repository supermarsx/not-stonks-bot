"""
Virtual Portfolio Management - Track virtual positions, P&L, and portfolio metrics
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import math

from loguru import logger

from .demo_mode_manager import DemoModeManager
from .virtual_broker import VirtualBroker


class PortfolioMetric(Enum):
    """Portfolio performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VAR_95 = "var_95"  # Value at Risk 95%
    BETA = "beta"  # Market beta
    ALPHA = "alpha"  # Portfolio alpha


@dataclass
class PortfolioPosition:
    """Virtual portfolio position"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_percentage: float
    weight: float  # Portfolio weight
    commission_paid: float
    first_trade_date: datetime
    last_trade_date: datetime
    trade_count: int


@dataclass
class PortfolioTrade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    side: str  # buy or sell
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    unrealized_before: float
    unrealized_after: float
    realized_pnl: float
    portfolio_value_before: float
    portfolio_value_after: float


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    return_pct: float
    positions_count: int
    largest_position: str
    largest_position_weight: float


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


class VirtualPortfolio:
    """
    Virtual portfolio management system
    
    Tracks virtual positions, calculates P&L, and provides portfolio analytics.
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trades: List[PortfolioTrade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.cash_balance = demo_manager.config.demo_account_balance
        self.initial_value = demo_manager.config.demo_account_balance
        self.peak_value = demo_manager.config.demo_account_balance
        self.drawdown_start: Optional[datetime] = None
        self.last_update = datetime.now()
        
        # Risk tracking
        self.daily_returns: List[float] = []
        self.max_drawdown = 0.0
        self.max_drawdown_date: Optional[datetime] = None
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage = 0.0
        
        # Benchmark data (simplified)
        self.benchmark_symbol = "SPY"
        self.benchmark_prices: List[float] = []
        
    async def update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> PortfolioPosition:
        """Update portfolio position after a trade"""
        try:
            current_time = datetime.now()
            
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                old_quantity = position.quantity
                old_avg_price = position.avg_entry_price
                
                # Calculate new quantity and average price
                if old_quantity * quantity_change > 0:
                    # Same direction (add to position)
                    total_cost = old_quantity * old_avg_price + quantity_change * price
                    new_quantity = old_quantity + quantity_change
                    new_avg_price = total_cost / new_quantity if new_quantity != 0 else 0
                    
                elif old_quantity + quantity_change == 0:
                    # Position closed
                    new_quantity = 0
                    new_avg_price = 0
                    realized_pnl = (price - old_avg_price) * old_quantity
                    position.realized_pnl += realized_pnl
                    
                else:
                    # Reducing or reversing position
                    if abs(quantity_change) < abs(old_quantity):
                        # Partial close
                        realized_pnl = (price - old_avg_price) * quantity_change
                        position.realized_pnl += realized_pnl
                        new_quantity = old_quantity + quantity_change
                        new_avg_price = old_avg_price
                    else:
                        # Full close with possible reverse
                        realized_pnl = (price - old_avg_price) * old_quantity
                        position.realized_pnl += realized_pnl
                        new_quantity = old_quantity + quantity_change
                        new_avg_price = price if new_quantity != 0 else 0
                
                # Update position
                position.quantity = new_quantity
                position.avg_entry_price = new_avg_price
                position.commission_paid += commission
                position.last_trade_date = current_time
                
                # If position is closed, remove it
                if new_quantity == 0:
                    position.current_price = price
                    position.market_value = 0
                    position.unrealized_pnl = 0
                else:
                    # Update with new market price (will be updated separately)
                    pass
                
            else:
                # New position
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity_change,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=quantity_change * price,
                    cost_basis=quantity_change * price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0,
                    pnl_percentage=0.0,
                    weight=0.0,
                    commission_paid=commission,
                    first_trade_date=current_time,
                    last_trade_date=current_time,
                    trade_count=1
                )
                self.positions[symbol] = position
            
            # Update portfolio cash
            trade_value = abs(quantity_change) * price
            if quantity_change > 0:  # Buy order
                self.cash_balance -= trade_value + commission
            else:  # Sell order
                self.cash_balance += trade_value - commission
            
            # Update tracking metrics
            self.total_commission_paid += commission
            self.total_slippage += slippage
            
            # Record trade
            await self._record_trade(
                symbol=symbol,
                side="buy" if quantity_change > 0 else "sell",
                quantity=abs(quantity_change),
                price=price,
                commission=commission,
                timestamp=current_time
            )
            
            # Take snapshot
            await self._take_snapshot()
            
            logger.debug(f"Updated position {symbol}: qty={quantity_change}, price={price}")
            return self.positions.get(symbol, position)
            
        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            raise
    
    async def update_market_prices(self, prices: Dict[str, float]):
        """Update market prices for all positions"""
        try:
            current_time = datetime.now()
            total_value = self.cash_balance
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    current_price = prices[symbol]
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    
                    # Calculate unrealized P&L
                    if position.quantity != 0:
                        cost_basis = abs(position.quantity) * position.avg_entry_price
                        position.unrealized_pnl = position.market_value - cost_basis
                        position.pnl_percentage = (position.unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                    else:
                        position.unrealized_pnl = 0
                        position.pnl_percentage = 0
                    
                    # Calculate total P&L
                    position.total_pnl = position.unrealized_pnl + position.realized_pnl
                    
                    total_value += position.market_value
            
            # Update drawdown tracking
            await self._update_drawdown_tracking(total_value)
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
    
    async def get_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        try:
            total_value = self.cash_balance
            
            for position in self.positions.values():
                total_value += position.market_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.cash_balance
    
    async def get_positions_summary(self) -> List[PortfolioPosition]:
        """Get summary of all positions"""
        try:
            # Update weights
            total_value = await self.get_portfolio_value()
            
            for position in self.positions.values():
                position.weight = (position.market_value / total_value) * 100 if total_value > 0 else 0
            
            return list(self.positions.values())
            
        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    async def get_portfolio_metrics(self, benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        try:
            current_value = await self.get_portfolio_value()
            total_return = ((current_value - self.initial_value) / self.initial_value) * 100
            
            # Basic metrics
            metrics = {
                "total_value": current_value,
                "cash_balance": self.cash_balance,
                "invested_value": current_value - self.cash_balance,
                "total_return_pct": total_return,
                "absolute_return": current_value - self.initial_value,
                "positions_count": len([p for p in self.positions.values() if p.quantity != 0]),
                "total_commission": self.total_commission_paid,
                "total_slippage": self.total_slippage,
                "trades_count": len(self.trades),
                "max_drawdown_pct": (self.max_drawdown / self.peak_value) * 100 if self.peak_value > 0 else 0
            }
            
            # Advanced metrics if we have enough data
            if len(self.daily_returns) > 30:
                metrics.update(await self._calculate_advanced_metrics(benchmark_returns))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Calculate portfolio risk metrics"""
        try:
            if len(self.daily_returns) < 30:
                # Return default metrics if insufficient data
                return RiskMetrics(
                    var_95=0.0, var_99=0.0, expected_shortfall=0.0,
                    max_drawdown=self.max_drawdown, max_drawdown_duration=0,
                    volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                    calmar_ratio=0.0
                )
            
            returns_array = np.array(self.daily_returns)
            
            # Value at Risk
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            # Expected Shortfall (Conditional VaR)
            expected_shortfall = np.mean(returns_array[returns_array <= var_95])
            
            # Volatility (annualized)
            volatility = np.std(returns_array) * np.sqrt(252)  # Assuming daily returns
            
            # Sharpe ratio (assuming 0% risk-free rate)
            mean_return = np.mean(returns_array) * 252  # Annualized
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            annualized_return = mean_return
            max_drawdown_pct = (self.max_drawdown / self.peak_value) if self.peak_value > 0 else 0
            calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=self.max_drawdown,
                max_drawdown_duration=await self._calculate_drawdown_duration(),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            recent_trades = self.trades[-limit:] if limit > 0 else self.trades
            return [asdict(trade) for trade in recent_trades]
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    async def get_performance_history(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get portfolio performance history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_snapshots = [
                snapshot for snapshot in self.snapshots 
                if snapshot.timestamp >= cutoff_date
            ]
            return recent_snapshots
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    async def calculate_attribution(self, benchmark_symbol: str = "SPY") -> Dict[str, Any]:
        """Calculate portfolio attribution analysis"""
        try:
            # Simplified attribution - would need more sophisticated analysis
            positions = await self.get_positions_summary()
            
            if not positions:
                return {}
            
            total_value = await self.get_portfolio_value()
            
            attribution = {
                "total_return": ((total_value - self.initial_value) / self.initial_value) * 100,
                "asset_allocation": {},
                "top_contributors": [],
                "top_detractors": []
            }
            
            # Asset allocation
            for position in positions:
                if position.quantity > 0:
                    symbol_allocation = {
                        "weight": position.weight,
                        "return": position.pnl_percentage,
                        "contribution": (position.weight * position.pnl_percentage) / 100
                    }
                    attribution["asset_allocation"][position.symbol] = symbol_allocation
            
            # Top contributors/detractors
            sorted_positions = sorted(positions, key=lambda x: x.pnl_percentage, reverse=True)
            attribution["top_contributors"] = [
                {"symbol": p.symbol, "return": p.pnl_percentage, "weight": p.weight}
                for p in sorted_positions[:5] if p.pnl_percentage > 0
            ]
            attribution["top_detractors"] = [
                {"symbol": p.symbol, "return": p.pnl_percentage, "weight": p.weight}
                for p in sorted_positions[-5:] if p.pnl_percentage < 0
            ]
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            return {}
    
    async def reset_portfolio(self):
        """Reset portfolio to initial state"""
        try:
            self.positions.clear()
            self.trades.clear()
            self.snapshots.clear()
            self.cash_balance = self.initial_value
            self.daily_returns.clear()
            self.max_drawdown = 0
            self.peak_value = self.initial_value
            self.total_commission_paid = 0
            self.total_slippage = 0
            self.drawdown_start = None
            self.last_update = datetime.now()
            
            logger.info("Portfolio reset to initial state")
            
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")
    
    async def export_portfolio_data(self, filepath: str):
        """Export portfolio data to JSON file"""
        try:
            export_data = {
                "positions": {symbol: asdict(position) for symbol, position in self.positions.items()},
                "trades": [asdict(trade) for trade in self.trades],
                "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
                "metrics": await self.get_portfolio_metrics(),
                "risk_metrics": asdict(await self.get_risk_metrics()),
                "daily_returns": self.daily_returns,
                "summary": {
                    "initial_value": self.initial_value,
                    "current_value": await self.get_portfolio_value(),
                    "total_return_pct": ((await self.get_portfolio_value() - self.initial_value) / self.initial_value) * 100,
                    "total_trades": len(self.trades),
                    "total_commission": self.total_commission_paid,
                    "max_drawdown": self.max_drawdown
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Portfolio data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
    
    # Private methods
    
    async def _record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime
    ):
        """Record trade in history"""
        try:
            portfolio_value_before = await self.get_portfolio_value()
            
            # Simplified unrealized P&L calculation
            unrealized_before = self.positions.get(symbol, PortfolioPosition(
                symbol=symbol, quantity=0, avg_entry_price=0, current_price=0,
                market_value=0, cost_basis=0, unrealized_pnl=0, realized_pnl=0,
                total_pnl=0, pnl_percentage=0, weight=0, commission_paid=0,
                first_trade_date=timestamp, last_trade_date=timestamp, trade_count=0
            )).unrealized_pnl
            
            # Simulate portfolio value after trade (simplified)
            portfolio_value_after = portfolio_value_before + (price * quantity if side == "sell" else -price * quantity)
            
            trade = PortfolioTrade(
                trade_id=f"trade_{len(self.trades) + 1}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                commission=commission,
                timestamp=timestamp,
                unrealized_before=unrealized_before,
                unrealized_after=unrealized_before,  # Simplified
                realized_pnl=0,  # Calculated during position updates
                portfolio_value_before=portfolio_value_before,
                portfolio_value_after=portfolio_value_after
            )
            
            self.trades.append(trade)
            
            # Keep only recent trades to manage memory
            if len(self.trades) > 10000:
                self.trades = self.trades[-5000:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    async def _take_snapshot(self):
        """Take portfolio snapshot"""
        try:
            total_value = await self.get_portfolio_value()
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                cash=self.cash_balance,
                invested_value=total_value - self.cash_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=unrealized_pnl + realized_pnl,
                return_pct=((total_value - self.initial_value) / self.initial_value) * 100,
                positions_count=len([p for p in self.positions.values() if p.quantity != 0]),
                largest_position=max(self.positions.keys(), key=lambda x: self.positions[x].market_value) if self.positions else "None",
                largest_position_weight=max((p.weight for p in self.positions.values()), default=0)
            )
            
            self.snapshots.append(snapshot)
            
            # Update daily returns if more than a day has passed
            if len(self.snapshots) > 1:
                last_snapshot = self.snapshots[-2]
                time_diff = (snapshot.timestamp - last_snapshot.timestamp).total_seconds()
                if time_diff >= 86400:  # 24 hours
                    daily_return = (snapshot.total_value - last_snapshot.total_value) / last_snapshot.total_value
                    self.daily_returns.append(daily_return)
            
            # Keep only recent snapshots
            if len(self.snapshots) > 1000:
                self.snapshots = self.snapshots[-500:]
                
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
    
    async def _update_drawdown_tracking(self, current_value: float):
        """Update drawdown tracking"""
        try:
            # Update peak value
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.drawdown_start = None
            
            # Calculate current drawdown
            current_drawdown = self.peak_value - current_value
            
            # Update maximum drawdown
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                self.max_drawdown_date = datetime.now()
                if self.drawdown_start is None:
                    self.drawdown_start = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating drawdown tracking: {e}")
    
    async def _calculate_advanced_metrics(self, benchmark_returns: Optional[List[float]]) -> Dict[str, float]:
        """Calculate advanced portfolio metrics"""
        try:
            returns_array = np.array(self.daily_returns)
            
            # Annualized metrics
            mean_return = np.mean(returns_array) * 252
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns_array)
            calmar_ratio = self._calculate_calmar_ratio()
            
            # Beta and Alpha (if benchmark data available)
            beta = None
            alpha = None
            if benchmark_returns and len(benchmark_returns) == len(self.daily_returns):
                beta = self._calculate_beta(returns_array, np.array(benchmark_returns))
                alpha = self._calculate_alpha(returns_array, np.array(benchmark_returns), beta)
            
            return {
                "annualized_return": mean_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "beta": beta,
                "alpha": alpha
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        try:
            mean_return = np.mean(returns) * 252
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else np.std(returns) * np.sqrt(252)
            return mean_return / downside_deviation if downside_deviation > 0 else 0
        except:
            return 0
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            current_value = asyncio.create_task(self.get_portfolio_value())
            annualized_return = np.mean(self.daily_returns) * 252
            max_drawdown_pct = (self.max_drawdown / self.peak_value) if self.peak_value > 0 else 0
            return annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        except:
            return 0
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_alpha(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, beta: float) -> float:
        """Calculate portfolio alpha"""
        try:
            portfolio_return = np.mean(portfolio_returns) * 252
            benchmark_return = np.mean(benchmark_returns) * 252
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            
            alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            return alpha
        except:
            return 0
    
    async def _calculate_drawdown_duration(self) -> int:
        """Calculate drawdown duration in days"""
        try:
            if self.drawdown_start is None:
                return 0
            return (datetime.now() - self.drawdown_start).days
        except:
            return 0


# Global virtual portfolio instance
virtual_portfolio = None


async def get_virtual_portfolio() -> VirtualPortfolio:
    """Get global virtual portfolio instance"""
    global virtual_portfolio
    if virtual_portfolio is None:
        manager = await get_demo_manager()
        virtual_portfolio = VirtualPortfolio(manager)
    return virtual_portfolio


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager and enable demo mode
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Get virtual portfolio
        portfolio = await get_virtual_portfolio()
        
        # Simulate some trades
        await portfolio.update_position("AAPL", 10, 150.0, 1.50)
        await portfolio.update_position("MSFT", 5, 300.0, 1.50)
        
        # Update market prices
        await portfolio.update_market_prices({"AAPL": 155.0, "MSFT": 310.0})
        
        # Get portfolio metrics
        metrics = await portfolio.get_portfolio_metrics()
        print(f"Portfolio metrics: {metrics}")
        
        # Get risk metrics
        risk_metrics = await portfolio.get_risk_metrics()
        print(f"Risk metrics: {risk_metrics}")
        
        # Export data
        await portfolio.export_portfolio_data("portfolio_export.json")
    
    asyncio.run(main())
