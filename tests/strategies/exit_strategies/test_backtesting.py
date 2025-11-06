"""
@file test_backtesting.py
@brief Comprehensive unit tests for exit strategy backtesting module

@details
This module provides comprehensive unit tests for the ExitStrategyBacktestEngine
and related backtesting functionality. Tests cover initialization, backtest execution,
performance metrics calculation, and error handling.

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@see backtesting.py for implementation details
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import pandas as pd

from trading_orchestrator.strategies.exit_strategies.backtesting import (
    ExitStrategyBacktestEngine,
    BacktestStatus,
    BacktestTrade,
    BacktestMetrics,
    BacktestResult,
    run_exit_strategy_backtest,
    compare_exit_strategies,
    _rank_strategies
)
from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    BaseExitStrategy,
    ExitConfiguration,
    ExitType,
    ExitStatus
)


class TestBacktestDataClasses:
    """Test backtesting data classes"""
    
    def test_backtest_trade_creation(self):
        """Test BacktestTrade dataclass creation"""
        trade = BacktestTrade(
            trade_id="test_trade_001",
            symbol="AAPL",
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow() + timedelta(hours=1),
            entry_price=Decimal("100.00"),
            exit_price=Decimal("105.00"),
            quantity=Decimal("10"),
            exit_reason="profit_target",
            strategy_id="trailing_stop_001",
            pnl=Decimal("50.00"),
            pnl_percentage=0.05,
            duration=timedelta(hours=1),
            confidence=0.85,
            urgency=0.7
        )
        
        assert trade.trade_id == "test_trade_001"
        assert trade.symbol == "AAPL"
        assert trade.pnl == Decimal("50.00")
        assert trade.confidence == 0.85
    
    def test_backtest_metrics_creation(self):
        """Test BacktestMetrics dataclass creation"""
        metrics = BacktestMetrics(
            strategy_id="test_strategy",
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            total_return=0.15,
            annualized_return=0.18,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.08,
            calmar_ratio=2.25,
            profit_factor=1.8,
            average_win=100.0,
            average_loss=-60.0,
            largest_win=500.0,
            largest_loss=-200.0,
            avg_trade_duration=2.0,
            total_exposure_time=200.0,
            commission_costs=50.0,
            market_impact_costs=25.0,
            net_return=0.125,
            volatility=0.20,
            var_95=-0.05,
            cvar_95=-0.07,
            trade_frequency=5.0,
            avg_confidence=0.8,
            avg_urgency=0.6
        )
        
        assert metrics.total_trades == 100
        assert metrics.win_rate == 0.60
        assert metrics.sharpe_ratio == 1.2
        assert metrics.strategy_id == "test_strategy"
    
    def test_backtest_result_creation(self):
        """Test BacktestResult dataclass creation"""
        result = BacktestResult(
            backtest_id="bt_001",
            status=BacktestStatus.PENDING,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            initial_capital=Decimal("100000"),
            strategies={},
            trades={},
            summary={}
        )
        
        assert result.backtest_id == "bt_001"
        assert result.status == BacktestStatus.PENDING
        assert result.initial_capital == Decimal("100000")
        assert isinstance(result.created_at, datetime)
    
    def test_backtest_status_enum(self):
        """Test BacktestStatus enum values"""
        assert BacktestStatus.PENDING.value == "pending"
        assert BacktestStatus.RUNNING.value == "running"
        assert BacktestStatus.COMPLETED.value == "completed"
        assert BacktestStatus.FAILED.value == "failed"


class TestExitStrategyBacktestEngine:
    """Test ExitStrategyBacktestEngine class"""
    
    @pytest.fixture
    def engine(self):
        """Create backtest engine instance"""
        return ExitStrategyBacktestEngine()
    
    @pytest.fixture
    def mock_strategy(self):
        """Create mock exit strategy"""
        strategy = Mock(spec=BaseExitStrategy)
        strategy.config = ExitConfiguration(
            strategy_id="test_strategy",
            strategy_type=ExitType.TRAILING_STOP,
            symbol="AAPL",
            parameters={}
        )
        return strategy
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data"""
        data = []
        base_price = 100.00
        current_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(168):  # 7 days of hourly data
            price = base_price + (i * 0.1)  # Trending up
            data.append({
                'timestamp': (current_time + timedelta(hours=i)).isoformat(),
                'open': price - 0.5,
                'high': price + 1.0,
                'low': price - 1.0,
                'close': price,
                'volume': 10000
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test backtest engine initialization"""
        assert engine.slippage_rate == Decimal('0.001')
        assert engine.commission_rate == Decimal('0.0001')
        assert engine.market_impact_model == "linear"
    
    @pytest.mark.asyncio
    async def test_run_backtest_with_strategies(self, engine, mock_strategy, sample_price_data):
        """Test running backtest with strategies"""
        # Mock the strategy backtest method
        with patch.object(engine, '_backtest_strategy', new_callable=AsyncMock) as mock_backtest:
            mock_backtest.return_value = (
                BacktestMetrics(
                    strategy_id="test_strategy",
                    total_trades=10,
                    winning_trades=6,
                    losing_trades=4,
                    win_rate=0.6,
                    total_return=0.15,
                    annualized_return=0.18,
                    sharpe_ratio=1.2,
                    sortino_ratio=1.5,
                    max_drawdown=-0.08,
                    calmar_ratio=2.25,
                    profit_factor=1.8,
                    average_win=100.0,
                    average_loss=-60.0,
                    largest_win=500.0,
                    largest_loss=-200.0,
                    avg_trade_duration=2.0,
                    total_exposure_time=20.0,
                    commission_costs=50.0,
                    market_impact_costs=25.0,
                    net_return=0.125,
                    volatility=0.20,
                    var_95=-0.05,
                    cvar_95=-0.07,
                    trade_frequency=5.0,
                    avg_confidence=0.8,
                    avg_urgency=0.6
                ),
                []
            )
            
            result = await engine.run_backtest(
                strategies=[mock_strategy],
                symbol="AAPL",
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow(),
                price_data=sample_price_data
            )
            
            assert result.status == BacktestStatus.COMPLETED
            assert "test_strategy" in result.strategies
            assert result.strategies["test_strategy"].total_trades == 10
    
    @pytest.mark.asyncio
    async def test_run_backtest_no_price_data(self, engine, mock_strategy):
        """Test backtest without price data"""
        result = await engine.run_backtest(
            strategies=[mock_strategy],
            symbol="AAPL",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        assert result.status == BacktestStatus.COMPLETED
        assert len(result.strategies) == 0  # No strategies executed
    
    @pytest.mark.asyncio
    async def test_run_backtest_exception_handling(self, engine, mock_strategy):
        """Test backtest exception handling"""
        with patch.object(engine, '_generate_sample_data', side_effect=Exception("Data error")):
            with pytest.raises(Exception) as exc_info:
                await engine.run_backtest(
                    strategies=[mock_strategy],
                    symbol="AAPL",
                    start_date=datetime.utcnow() - timedelta(days=7),
                    end_date=datetime.utcnow()
                )
            
            assert "Data error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_backtest_strategy_individual(self, engine, mock_strategy, sample_price_data):
        """Test individual strategy backtesting"""
        initial_capital = Decimal("100000")
        
        # Mock strategy simulation
        with patch.object(engine, '_simulate_exit_condition') as mock_sim:
            mock_sim.return_value = {
                'triggered': True,
                'reason': 'profit_target',
                'confidence': 0.85,
                'urgency': 0.7
            }
            
            metrics, trades = await engine._backtest_strategy(
                mock_strategy,
                "AAPL",
                sample_price_data,
                initial_capital
            )
            
            assert isinstance(metrics, BacktestMetrics)
            assert isinstance(trades, list)
            assert len(trades) > 0
    
    @pytest.mark.asyncio
    async def test_simulate_exit_condition_trailing_stop(self, engine, mock_strategy):
        """Test exit condition simulation for trailing stop"""
        position = {
            'position_id': 'pos_1',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(hours=2),
            'entry_price': Decimal("100.00"),
            'quantity': Decimal("10"),
            'side': 'long',
            'created_at': datetime.utcnow() - timedelta(hours=2)
        }
        
        # Price moved up 6%
        price_point = {'close': '106.00'}
        
        # Mock strategy type as trailing stop
        mock_strategy.__class__.__module__ = 'trailing_stop'
        
        result = await engine._simulate_exit_condition(mock_strategy, position, price_point)
        
        assert result is not None
        assert result['triggered'] == True
        assert result['reason'] == 'trailing_stop'
        assert 'confidence' in result
        assert 'urgency' in result
    
    @pytest.mark.asyncio
    async def test_simulate_exit_condition_fixed_target(self, engine, mock_strategy):
        """Test exit condition simulation for fixed target"""
        position = {
            'position_id': 'pos_1',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(hours=2),
            'entry_price': Decimal("100.00"),
            'quantity': Decimal("10"),
            'side': 'long',
            'created_at': datetime.utcnow() - timedelta(hours=2)
        }
        
        # Price moved up 11%
        price_point = {'close': '111.00'}
        
        # Mock strategy type as fixed target
        mock_strategy.__class__.__module__ = 'fixed_target'
        
        result = await engine._simulate_exit_condition(mock_strategy, position, price_point)
        
        assert result is not None
        assert result['triggered'] == True
        assert result['reason'] == 'profit_target'
    
    @pytest.mark.asyncio
    async def test_generate_sample_data(self, engine):
        """Test sample data generation"""
        start_date = datetime.utcnow() - timedelta(days=2)
        end_date = datetime.utcnow()
        
        data = await engine._generate_sample_data("AAPL", start_date, end_date)
        
        assert len(data) > 0
        assert all('timestamp' in item for item in data)
        assert all('close' in item for item in data)
        assert all('volume' in item for item in data)
        
        # Verify daily progression
        timestamps = [datetime.fromisoformat(item['timestamp']) for item in data]
        assert timestamps[0].date() == start_date.date()
    
    def test_apply_slippage(self, engine):
        """Test slippage application"""
        price = Decimal("100.00")
        
        # Buy with slippage
        buy_price = engine._apply_slippage(price, 'buy')
        assert buy_price > price
        
        # Sell with slippage
        sell_price = engine._apply_slippage(price, 'sell')
        assert sell_price < price
    
    def test_estimate_market_impact(self, engine):
        """Test market impact estimation"""
        price = Decimal("100.00")
        quantity = Decimal("1000")
        
        impact = engine._estimate_market_impact(price, quantity)
        
        assert isinstance(impact, Decimal)
        assert impact >= Decimal('0')
        assert impact > 0  # Should have some impact
    
    @pytest.mark.asyncio
    async def test_calculate_strategy_metrics_empty_trades(self, engine):
        """Test metrics calculation with no trades"""
        metrics = await engine._calculate_strategy_metrics([], Decimal("100000"))
        
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_strategy_metrics_with_trades(self, engine):
        """Test metrics calculation with trades"""
        trades = [
            BacktestTrade(
                trade_id="t1", symbol="AAPL",
                entry_time=datetime.utcnow() - timedelta(hours=2),
                exit_time=datetime.utcnow(),
                entry_price=Decimal("100"), exit_price=Decimal("110"),
                quantity=Decimal("10"), exit_reason="profit",
                strategy_id="test", pnl=Decimal("100"),
                pnl_percentage=0.1, duration=timedelta(hours=2),
                confidence=0.8, urgency=0.7
            ),
            BacktestTrade(
                trade_id="t2", symbol="AAPL",
                entry_time=datetime.utcnow() - timedelta(hours=3),
                exit_time=datetime.utcnow(),
                entry_price=Decimal("100"), exit_price=Decimal("90"),
                quantity=Decimal("10"), exit_reason="stop_loss",
                strategy_id="test", pnl=Decimal("-100"),
                pnl_percentage=-0.1, duration=timedelta(hours=3),
                confidence=0.9, urgency=0.9
            )
        ]
        
        metrics = await engine._calculate_strategy_metrics(trades, Decimal("100000"))
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.profit_factor > 0
    
    def test_calculate_daily_returns(self, engine):
        """Test daily returns calculation"""
        trades = [
            BacktestTrade(
                trade_id="t1", symbol="AAPL",
                entry_time=datetime.utcnow().replace(day=1) + timedelta(hours=1),
                exit_time=datetime.utcnow().replace(day=1) + timedelta(hours=2),
                entry_price=Decimal("100"), exit_price=Decimal("110"),
                quantity=Decimal("10"), exit_reason="profit",
                strategy_id="test", pnl=Decimal("100"),
                pnl_percentage=0.1, duration=timedelta(hours=1),
                confidence=0.8, urgency=0.7
            ),
            BacktestTrade(
                trade_id="t2", symbol="AAPL",
                entry_time=datetime.utcnow().replace(day=2) + timedelta(hours=1),
                exit_time=datetime.utcnow().replace(day=2) + timedelta(hours=2),
                entry_price=Decimal("100"), exit_price=Decimal("90"),
                quantity=Decimal("10"), exit_reason="stop_loss",
                strategy_id="test", pnl=Decimal("-50"),
                pnl_percentage=-0.05, duration=timedelta(hours=1),
                confidence=0.9, urgency=0.9
            )
        ]
        
        returns = engine._calculate_daily_returns(trades)
        assert isinstance(returns, list)
    
    def test_calculate_sharpe_ratio(self, engine):
        """Test Sharpe ratio calculation"""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]  # Sample returns
        
        sharpe = engine._calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert sharpe != 0  # Should have some value
    
    def test_calculate_sortino_ratio(self, engine):
        """Test Sortino ratio calculation"""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        
        sortino = engine._calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_calculate_max_drawdown(self, engine):
        """Test maximum drawdown calculation"""
        trades = [
            BacktestTrade(
                trade_id="t1", symbol="AAPL",
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow() + timedelta(hours=1),
                entry_price=Decimal("100"), exit_price=Decimal("110"),
                quantity=Decimal("10"), exit_reason="profit",
                strategy_id="test", pnl=Decimal("100"),
                pnl_percentage=0.1, duration=timedelta(hours=1),
                confidence=0.8, urgency=0.7
            ),
            BacktestTrade(
                trade_id="t2", symbol="AAPL",
                entry_time=datetime.utcnow() + timedelta(hours=2),
                exit_time=datetime.utcnow() + timedelta(hours=3),
                entry_price=Decimal("100"), exit_price=Decimal("80"),
                quantity=Decimal("10"), exit_reason="stop_loss",
                strategy_id="test", pnl=Decimal("-200"),
                pnl_percentage=-0.2, duration=timedelta(hours=1),
                confidence=0.9, urgency=0.9
            )
        ]
        
        drawdown = engine._calculate_max_drawdown(trades, Decimal("100000"))
        assert isinstance(drawdown, float)
        assert drawdown <= 0  # Drawdown should be negative or zero
    
    @pytest.mark.asyncio
    async def test_calculate_summary_metrics(self, engine):
        """Test summary metrics calculation"""
        backtest_result = BacktestResult(
            backtest_id="test_bt",
            status=BacktestStatus.COMPLETED,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            initial_capital=Decimal("100000"),
            strategies={
                "strategy1": BacktestMetrics(
                    strategy_id="strategy1", total_trades=50, winning_trades=30,
                    losing_trades=20, win_rate=0.6, total_return=0.15,
                    annualized_return=0.18, sharpe_ratio=1.2, sortino_ratio=1.5,
                    max_drawdown=-0.08, calmar_ratio=2.25, profit_factor=1.8,
                    average_win=100.0, average_loss=-60.0, largest_win=500.0,
                    largest_loss=-200.0, avg_trade_duration=2.0,
                    total_exposure_time=100.0, commission_costs=50.0,
                    market_impact_costs=25.0, net_return=0.125,
                    volatility=0.20, var_95=-0.05, cvar_95=-0.07,
                    trade_frequency=5.0, avg_confidence=0.8, avg_urgency=0.6
                ),
                "strategy2": BacktestMetrics(
                    strategy_id="strategy2", total_trades=40, winning_trades=25,
                    losing_trades=15, win_rate=0.625, total_return=0.12,
                    annualized_return=0.15, sharpe_ratio=1.0, sortino_ratio=1.3,
                    max_drawdown=-0.06, calmar_ratio=2.5, profit_factor=2.0,
                    average_win=80.0, average_loss=-50.0, largest_win=300.0,
                    largest_loss=-150.0, avg_trade_duration=1.5,
                    total_exposure_time=60.0, commission_costs=40.0,
                    market_impact_costs=20.0, net_return=0.10,
                    volatility=0.18, var_95=-0.04, cvar_95=-0.06,
                    trade_frequency=4.0, avg_confidence=0.85, avg_urgency=0.65
                )
            },
            trades={},
            summary={}
        )
        
        summary = await engine._calculate_summary_metrics(backtest_result, Decimal("100000"))
        
        assert 'total_strategies' in summary
        assert summary['total_strategies'] == 2
        assert 'best_return' in summary
        assert 'worst_return' in summary
        assert 'strategy_comparison' in summary
        assert 'avg_win_rate' in summary


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def mock_strategies(self):
        """Create mock strategies for testing"""
        strategies = []
        for i in range(2):
            strategy = Mock(spec=BaseExitStrategy)
            strategy.config = ExitConfiguration(
                strategy_id=f"strategy_{i+1}",
                strategy_type=ExitType.TRAILING_STOP,
                symbol="AAPL",
                parameters={}
            )
            strategies.append(strategy)
        return strategies
    
    @pytest.mark.asyncio
    async def test_run_exit_strategy_backtest(self, mock_strategies):
        """Test run_exit_strategy_backtest convenience function"""
        with patch('trading_orchestrator.strategies.exit_strategies.backtesting.ExitStrategyBacktestEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.run_backtest = AsyncMock()
            
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow()
            
            result = await run_exit_strategy_backtest(
                strategies=mock_strategies,
                symbol="AAPL",
                start_date=start_date,
                end_date=end_date
            )
            
            mock_engine.run_backtest.assert_called_once()
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_compare_exit_strategies(self, mock_strategies):
        """Test compare_exit_strategies convenience function"""
        with patch('trading_orchestrator.strategies.exit_strategies.backtesting.run_exit_strategy_backtest') as mock_run:
            # Mock backtest result
            mock_result = Mock()
            mock_result.summary = {'best_return': {'strategy_id': 'strategy_1', 'return': 0.15}}
            mock_result.strategies = {
                'strategy_1': BacktestMetrics(
                    strategy_id='strategy_1', total_trades=50, winning_trades=30,
                    losing_trades=20, win_rate=0.6, total_return=0.15,
                    annualized_return=0.18, sharpe_ratio=1.2, sortino_ratio=1.5,
                    max_drawdown=-0.08, calmar_ratio=2.25, profit_factor=1.8,
                    average_win=100.0, average_loss=-60.0, largest_win=500.0,
                    largest_loss=-200.0, avg_trade_duration=2.0,
                    total_exposure_time=100.0, commission_costs=50.0,
                    market_impact_costs=25.0, net_return=0.125,
                    volatility=0.20, var_95=-0.05, cvar_95=-0.07,
                    trade_frequency=5.0, avg_confidence=0.8, avg_urgency=0.6
                )
            }
            mock_run.return_value = mock_result
            
            comparison = await compare_exit_strategies(
                strategies=mock_strategies,
                symbol="AAPL",
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow()
            )
            
            assert 'comparison_summary' in comparison
            assert 'detailed_metrics' in comparison
            assert 'rankings' in comparison
            assert 'strategy_1' in comparison['detailed_metrics']
    
    def test_rank_strategies(self):
        """Test strategy ranking function"""
        strategies = {
            'strategy_1': BacktestMetrics(
                strategy_id='strategy_1', total_trades=50, winning_trades=30,
                losing_trades=20, win_rate=0.6, total_return=0.15,
                annualized_return=0.18, sharpe_ratio=1.2, sortino_ratio=1.5,
                max_drawdown=-0.08, calmar_ratio=2.25, profit_factor=1.8,
                average_win=100.0, average_loss=-60.0, largest_win=500.0,
                largest_loss=-200.0, avg_trade_duration=2.0,
                total_exposure_time=100.0, commission_costs=50.0,
                market_impact_costs=25.0, net_return=0.125,
                volatility=0.20, var_95=-0.05, cvar_95=-0.07,
                trade_frequency=5.0, avg_confidence=0.8, avg_urgency=0.6
            ),
            'strategy_2': BacktestMetrics(
                strategy_id='strategy_2', total_trades=40, winning_trades=25,
                losing_trades=15, win_rate=0.625, total_return=0.12,
                annualized_return=0.15, sharpe_ratio=1.0, sortino_ratio=1.3,
                max_drawdown=-0.06, calmar_ratio=2.5, profit_factor=2.0,
                average_win=80.0, average_loss=-50.0, largest_win=300.0,
                largest_loss=-150.0, avg_trade_duration=1.5,
                total_exposure_time=60.0, commission_costs=40.0,
                market_impact_costs=20.0, net_return=0.10,
                volatility=0.18, var_95=-0.04, cvar_95=-0.06,
                trade_frequency=4.0, avg_confidence=0.85, avg_urgency=0.65
            )
        }
        
        rankings = _rank_strategies(strategies)
        
        assert 'strategy_1' in rankings
        assert 'strategy_2' in rankings
        assert 'total_return' in rankings['strategy_1']
        assert 'sharpe_ratio' in rankings['strategy_1']
        assert 'win_rate' in rankings['strategy_1']
        assert 'max_drawdown' in rankings['strategy_1']
    
    def test_rank_strategies_empty(self):
        """Test strategy ranking with empty strategies"""
        rankings = _rank_strategies({})
        assert rankings == {}


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_backtest_with_zero_capital(self):
        """Test backtest with zero initial capital"""
        engine = ExitStrategyBacktestEngine()
        
        result = await engine.run_backtest(
            strategies=[],
            symbol="AAPL",
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow(),
            initial_capital=Decimal("0")
        )
        
        assert result.status == BacktestStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_backtest_with_invalid_dates(self):
        """Test backtest with invalid date range"""
        engine = ExitStrategyBacktestEngine()
        
        with pytest.raises(Exception):
            await engine.run_backtest(
                strategies=[],
                symbol="AAPL",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() - timedelta(days=1)  # End before start
            )
    
    @pytest.mark.asyncio
    async def test_metrics_with_infinite_values(self):
        """Test metrics calculation with edge case values"""
        engine = ExitStrategyBacktestEngine()
        
        # Trade with zero entry price
        trade = BacktestTrade(
            trade_id="t1", symbol="AAPL",
            entry_time=datetime.utcnow() - timedelta(hours=1),
            exit_time=datetime.utcnow(),
            entry_price=Decimal("0.01"), exit_price=Decimal("0.02"),
            quantity=Decimal("1000"), exit_reason="profit",
            strategy_id="test", pnl=Decimal("10"),
            pnl_percentage=0.01, duration=timedelta(hours=1),
            confidence=0.8, urgency=0.7
        )
        
        metrics = await engine._calculate_strategy_metrics([trade], Decimal("100000"))
        
        assert isinstance(metrics, BacktestMetrics)
        assert not np.isnan(metrics.total_return)
    
    @pytest.mark.asyncio
    async def test_backtest_error_recovery(self):
        """Test backtest error recovery"""
        engine = ExitStrategyBacktestEngine()
        mock_strategy = Mock(spec=BaseExitStrategy)
        mock_strategy.config = ExitConfiguration(
            strategy_id="failing_strategy",
            strategy_type=ExitType.TRAILING_STOP,
            symbol="AAPL",
            parameters={}
        )
        
        # Make strategy fail during backtest
        with patch.object(engine, '_backtest_strategy', side_effect=Exception("Strategy error")):
            result = await engine.run_backtest(
                strategies=[mock_strategy],
                symbol="AAPL",
                start_date=datetime.utcnow() - timedelta(days=1),
                end_date=datetime.utcnow()
            )
            
            # Should still complete but without this strategy
            assert result.status == BacktestStatus.COMPLETED


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])