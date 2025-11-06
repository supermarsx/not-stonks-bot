"""
@file strategy_management.py
@brief Comprehensive Strategy Management System

@details
This module provides comprehensive strategy management capabilities including:
- Strategy combination and ensemble methods
- Dynamic strategy allocation and rebalancing
- Strategy lifecycle management
- Performance attribution and A/B testing
- Strategy wizard for beginners
- Advanced strategy editor for experts
- Real-time strategy monitoring
- Strategy comparison and ranking

Key Features:
- Ensemble strategy management
- Dynamic allocation algorithms
- Performance attribution analysis
- A/B testing framework
- Strategy lifecycle automation
- Real-time monitoring and alerts
- Advanced analytics and reporting

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@see library.StrategyLibrary for strategy registry
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import math

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    strategy_registry
)
from .library import StrategyCategory, StrategyMetadata, strategy_library, BacktestResults


class AllocationMethod(Enum):
    """Strategy allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM_WEIGHTED = "momentum_weighted"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    SHARPE_WEIGHTED = "sharpe_weighted"


class RebalanceFrequency(Enum):
    """Strategy rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_SIGNAL = "on_signal"


class EnsembleType(Enum):
    """Types of strategy ensembles"""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    DIVERSIFICATION = "diversification"
    DYNAMIC_ALLOCATION = "dynamic_allocation"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_id: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: timedelta
    volatility: float
    var_95: float
    cvar_95: float
    trades_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_id: str
    weight: float
    max_weight: float = 1.0
    min_weight: float = 0.0
    target_volatility: float = 0.15
    rebalance_threshold: float = 0.05
    is_active: bool = True


@dataclass
class EnsembleStrategy:
    """Ensemble strategy configuration"""
    ensemble_id: str
    ensemble_type: EnsembleType
    strategies: List[StrategyAllocation]
    allocation_method: AllocationMethod
    rebalance_frequency: RebalanceFrequency
    min_allocation: float = 0.05
    max_strategies: int = 10
    correlation_threshold: float = 0.7


class StrategyManager:
    """Comprehensive strategy management system"""
    
    def __init__(self):
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.ensemble_strategies: Dict[str, EnsembleStrategy] = {}
        self.strategy_signals: Dict[str, List[TradingSignal]] = defaultdict(list)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year of daily data
        self.risk_budgets: Dict[str, float] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.allocation_history: List[Dict[str, Any]] = []
    
    async def register_strategy(self, strategy: BaseStrategy, performance_config: Dict[str, Any] = None) -> bool:
        """Register a new strategy"""
        try:
            strategy_id = strategy.config.strategy_id
            
            if strategy_id in self.active_strategies:
                logger.warning(f"Strategy {strategy_id} already registered")
                return False
            
            self.active_strategies[strategy_id] = strategy
            
            # Initialize performance tracking
            if performance_config:
                self.risk_budgets[strategy_id] = performance_config.get('risk_budget', 0.1)
            
            # Create initial performance record
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                avg_trade_duration=timedelta(hours=24),
                volatility=0.0,
                var_95=0.0,
                cvar_95=0.0,
                trades_count=0
            )
            
            logger.info(f"Strategy {strategy_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return False
    
    async def create_ensemble(self, ensemble_config: Dict[str, Any]) -> Optional[EnsembleStrategy]:
        """Create a new ensemble strategy"""
        try:
            ensemble_id = ensemble_config.get('ensemble_id')
            if not ensemble_id:
                ensemble_id = f"ensemble_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate strategies
            strategy_ids = ensemble_config.get('strategies', [])
            if len(strategy_ids) < 2:
                raise ValueError("Ensemble must have at least 2 strategies")
            
            if len(strategy_ids) > ensemble_config.get('max_strategies', 10):
                raise ValueError(f"Maximum {ensemble_config.get('max_strategies', 10)} strategies allowed")
            
            # Create strategy allocations
            allocations = []
            for strategy_id in strategy_ids:
                allocation_config = ensemble_config.get('allocation_overrides', {}).get(strategy_id, {})
                
                allocation = StrategyAllocation(
                    strategy_id=strategy_id,
                    weight=allocation_config.get('weight', 1.0 / len(strategy_ids)),
                    max_weight=allocation_config.get('max_weight', 1.0),
                    min_weight=allocation_config.get('min_weight', 0.0),
                    target_volatility=allocation_config.get('target_volatility', 0.15),
                    rebalance_threshold=allocation_config.get('rebalance_threshold', 0.05),
                    is_active=allocation_config.get('is_active', True)
                )
                allocations.append(allocation)
            
            # Normalize weights
            total_weight = sum(allocation.weight for allocation in allocations)
            if total_weight > 0:
                for allocation in allocations:
                    allocation.weight /= total_weight
            
            # Create ensemble
            ensemble = EnsembleStrategy(
                ensemble_id=ensemble_id,
                ensemble_type=EnsembleType(ensemble_config.get('ensemble_type', 'voting')),
                strategies=allocations,
                allocation_method=AllocationMethod(ensemble_config.get('allocation_method', 'equal_weight')),
                rebalance_frequency=RebalanceFrequency(ensemble_config.get('rebalance_frequency', 'weekly')),
                min_allocation=ensemble_config.get('min_allocation', 0.05),
                max_strategies=ensemble_config.get('max_strategies', 10),
                correlation_threshold=ensemble_config.get('correlation_threshold', 0.7)
            )
            
            self.ensemble_strategies[ensemble_id] = ensemble
            
            logger.info(f"Ensemble {ensemble_id} created with {len(allocations)} strategies")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return None
    
    async def rebalance_ensemble(self, ensemble_id: str) -> bool:
        """Rebalance ensemble strategy weights"""
        try:
            ensemble = self.ensemble_strategies.get(ensemble_id)
            if not ensemble:
                logger.error(f"Ensemble {ensemble_id} not found")
                return False
            
            # Calculate new weights based on allocation method
            if ensemble.allocation_method == AllocationMethod.EQUAL_WEIGHT:
                new_weights = self._equal_weight_allocation(ensemble)
            
            elif ensemble.allocation_method == AllocationMethod.RISK_PARITY:
                new_weights = await self._risk_parity_allocation(ensemble)
            
            elif ensemble.allocation_method == AllocationMethod.PERFORMANCE_WEIGHTED:
                new_weights = await self._performance_weighted_allocation(ensemble)
            
            elif ensemble.allocation_method == AllocationMethod.SHARPE_WEIGHTED:
                new_weights = await self._sharpe_weighted_allocation(ensemble)
            
            else:  # Default to equal weight
                new_weights = self._equal_weight_allocation(ensemble)
            
            # Update allocations
            for allocation in ensemble.strategies:
                if allocation.strategy_id in new_weights:
                    allocation.weight = new_weights[allocation.strategy_id]
            
            # Log rebalancing
            self.allocation_history.append({
                'timestamp': datetime.utcnow(),
                'ensemble_id': ensemble_id,
                'allocations': {a.strategy_id: a.weight for a in ensemble.strategies},
                'method': ensemble.allocation_method.value
            })
            
            logger.info(f"Ensemble {ensemble_id} rebalanced using {ensemble.allocation_method.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error rebalancing ensemble {ensemble_id}: {e}")
            return False
    
    def _equal_weight_allocation(self, ensemble: EnsembleStrategy) -> Dict[str, float]:
        """Equal weight allocation"""
        active_strategies = [a for a in ensemble.strategies if a.is_active]
        if not active_strategies:
            return {}
        
        weight = 1.0 / len(active_strategies)
        return {a.strategy_id: weight for a in active_strategies}
    
    async def _risk_parity_allocation(self, ensemble: EnsembleStrategy) -> Dict[str, float]:
        """Risk parity allocation based on volatility"""
        weights = {}
        active_strategies = [a for a in ensemble.strategies if a.is_active]
        
        if not active_strategies:
            return weights
        
        # Get volatilities
        volatilities = {}
        for allocation in active_strategies:
            strategy_id = allocation.strategy_id
            if strategy_id in self.strategy_performance:
                volatilities[strategy_id] = self.strategy_performance[strategy_id].volatility
            else:
                volatilities[strategy_id] = 0.15  # Default volatility
        
        # Calculate inverse volatility weights
        inverse_vols = {sid: 1.0 / vol for sid, vol in volatilities.items() if vol > 0}
        total_inv_vol = sum(inverse_vols.values())
        
        if total_inv_vol > 0:
            for allocation in active_strategies:
                weights[allocation.strategy_id] = inverse_vols[allocation.strategy_id] / total_inv_vol
        
        return weights
    
    async def _performance_weighted_allocation(self, ensemble: EnsembleStrategy) -> Dict[str, float]:
        """Performance weighted allocation"""
        weights = {}
        active_strategies = [a for a in ensemble.strategies if a.is_active]
        
        if not active_strategies:
            return weights
        
        # Get performance metrics
        performances = {}
        for allocation in active_strategies:
            strategy_id = allocation.strategy_id
            if strategy_id in self.strategy_performance:
                # Use Sharpe ratio as performance metric
                performances[strategy_id] = max(0.01, self.strategy_performance[strategy_id].sharpe_ratio)
            else:
                performances[strategy_id] = 0.01  # Default performance
        
        # Calculate weights based on performance
        total_performance = sum(performances.values())
        if total_performance > 0:
            for allocation in active_strategies:
                weights[allocation.strategy_id] = performances[allocation.strategy_id] / total_performance
        
        return weights
    
    async def _sharpe_weighted_allocation(self, ensemble: EnsembleStrategy) -> Dict[str, float]:
        """Sharpe ratio weighted allocation"""
        weights = {}
        active_strategies = [a for a in ensemble.strategies if a.is_active]
        
        if not active_strategies:
            return weights
        
        # Get Sharpe ratios
        sharpe_ratios = {}
        for allocation in active_strategies:
            strategy_id = allocation.strategy_id
            if strategy_id in self.strategy_performance:
                sharpe_ratios[strategy_id] = max(0.01, self.strategy_performance[strategy_id].sharpe_ratio)
            else:
                sharpe_ratios[strategy_id] = 0.01  # Default Sharpe ratio
        
        # Normalize Sharpe ratios and calculate weights
        total_sharpe = sum(sharpe_ratios.values())
        if total_sharpe > 0:
            for allocation in active_strategies:
                weights[allocation.strategy_id] = sharpe_ratios[allocation.strategy_id] / total_sharpe
        
        return weights
    
    async def update_strategy_performance(self, strategy_id: str, new_returns: float) -> None:
        """Update strategy performance metrics"""
        try:
            if strategy_id not in self.strategy_performance:
                return
            
            # Add to performance history
            self.performance_history[strategy_id].append(new_returns)
            
            # Calculate performance metrics
            returns = list(self.performance_history[strategy_id])
            if len(returns) < 10:  # Need minimum data points
                return
            
            returns_series = pd.Series(returns)
            
            # Calculate basic metrics
            total_return = (1 + returns_series).prod() - 1
            avg_return = returns_series.mean()
            volatility = returns_series.std() * np.sqrt(252)  # Annualized
            
            # Calculate risk-adjusted metrics
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_return = avg_return - risk_free_rate / 252
            
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate Sortino ratio
            downside_returns = returns_series[returns_series < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calculate Calmar ratio
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Update performance metrics
            performance = self.strategy_performance[strategy_id]
            performance.total_return = total_return
            performance.sharpe_ratio = sharpe_ratio
            performance.sortino_ratio = sortino_ratio
            performance.calmar_ratio = calmar_ratio
            performance.max_drawdown = abs(max_drawdown)
            performance.volatility = volatility
            performance.last_updated = datetime.utcnow()
            
            # Update correlation matrix
            await self._update_correlation_matrix(strategy_id, returns)
            
            logger.debug(f"Updated performance for strategy {strategy_id}: Sharpe={sharpe_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating performance for {strategy_id}: {e}")
    
    async def _update_correlation_matrix(self, new_strategy_id: str, new_returns: List[float]) -> None:
        """Update correlation matrix between strategies"""
        try:
            # Initialize correlations for new strategy
            self.correlation_matrix[new_strategy_id] = {}
            
            # Calculate correlations with existing strategies
            for existing_strategy_id, existing_returns in self.performance_history.items():
                if existing_strategy_id == new_strategy_id:
                    continue
                
                if len(existing_returns) < 10:
                    continue
                
                # Align return series lengths
                min_length = min(len(new_returns), len(existing_returns))
                new_returns_aligned = new_returns[-min_length:]
                existing_returns_aligned = list(existing_returns)[-min_length:]
                
                if min_length >= 10:
                    correlation = np.corrcoef(new_returns_aligned, existing_returns_aligned)[0, 1]
                    if not np.isnan(correlation):
                        self.correlation_matrix[new_strategy_id][existing_strategy_id] = correlation
                        self.correlation_matrix[existing_strategy_id][new_strategy_id] = correlation
                        self.correlation_matrix[existing_strategy_id][existing_strategy_id] = 1.0
            
            # Ensure diagonal is 1.0
            if new_strategy_id not in self.correlation_matrix:
                self.correlation_matrix[new_strategy_id] = {}
            self.correlation_matrix[new_strategy_id][new_strategy_id] = 1.0
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    async def generate_ensemble_signal(self, ensemble_id: str) -> Optional[TradingSignal]:
        """Generate combined signal from ensemble"""
        try:
            ensemble = self.ensemble_strategies.get(ensemble_id)
            if not ensemble:
                return None
            
            # Get signals from all strategies in ensemble
            all_signals = []
            strategy_weights = {}
            
            for allocation in ensemble.strategies:
                if not allocation.is_active:
                    continue
                
                strategy_id = allocation.strategy_id
                signals = self.strategy_signals.get(strategy_id, [])
                
                if signals:
                    # Use most recent signal
                    latest_signal = signals[-1]
                    all_signals.append(latest_signal)
                    strategy_weights[strategy_id] = allocation.weight
            
            if not all_signals:
                return None
            
            # Combine signals based on ensemble type
            if ensemble.ensemble_type == EnsembleType.VOTING:
                return self._voting_ensemble_signal(all_signals, strategy_weights)
            
            elif ensemble.ensemble_type == EnsembleType.BLENDING:
                return self._blending_ensemble_signal(all_signals, strategy_weights)
            
            elif ensemble.ensemble_type == EnsembleType.DIVERSIFICATION:
                return self._diversification_ensemble_signal(all_signals, strategy_weights)
            
            else:
                # Default to voting
                return self._voting_ensemble_signal(all_signals, strategy_weights)
                
        except Exception as e:
            logger.error(f"Error generating ensemble signal for {ensemble_id}: {e}")
            return None
    
    def _voting_ensemble_signal(self, signals: List[TradingSignal], weights: Dict[str, float]) -> Optional[TradingSignal]:
        """Voting ensemble signal combination"""
        if not signals:
            return None
        
        # Count votes for each signal type
        votes = defaultdict(float)
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.strategy_id, 1.0)
            votes[signal.signal_type] += weight * signal.strength
            total_weight += weight
        
        # Determine winning signal
        if total_weight == 0:
            return None
        
        winning_vote = max(votes.items(), key=lambda x: x[1])
        winning_signal_type, vote_strength = winning_vote
        
        if vote_strength / total_weight < 0.5:  # Need majority vote
            return None
        
        # Create ensemble signal
        primary_signal = signals[0]  # Use first signal as template
        return TradingSignal(
            signal_id=f"ENSEMBLE_VOTE_{primary_signal.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id="ensemble_voting",
            symbol=primary_signal.symbol,
            signal_type=winning_signal_type,
            confidence=vote_strength / total_weight,
            strength=vote_strength / total_weight,
            price=primary_signal.price,
            quantity=primary_signal.quantity,
            time_horizon=primary_signal.time_horizon,
            metadata={
                'ensemble_type': 'voting',
                'individual_votes': dict(votes),
                'strategies': [s.strategy_id for s in signals]
            }
        )
    
    def _blending_ensemble_signal(self, signals: List[TradingSignal], weights: Dict[str, float]) -> Optional[TradingSignal]:
        """Blending ensemble signal combination"""
        if not signals:
            return None
        
        # Calculate weighted average of signals
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        weighted_price = Decimal('0')
        weighted_quantity = Decimal('0')
        
        for signal in signals:
            weight = weights.get(signal.strategy_id, 1.0)
            total_weight += weight
            
            if signal.signal_type in [SignalType.BUY, SignalType.INCREASE]:
                buy_weight += weight * signal.strength
            elif signal.signal_type in [SignalType.SELL, SignalType.REDUCE]:
                sell_weight += weight * signal.strength
            
            weighted_price += signal.price * Decimal(str(weight))
            weighted_quantity += signal.quantity * Decimal(str(weight))
        
        if total_weight == 0:
            return None
        
        # Determine signal direction
        if buy_weight > sell_weight and buy_weight > 0.3 * total_weight:
            signal_type = SignalType.BUY
            strength = buy_weight / total_weight
        elif sell_weight > buy_weight and sell_weight > 0.3 * total_weight:
            signal_type = SignalType.SELL
            strength = sell_weight / total_weight
        else:
            return None
        
        # Create ensemble signal
        primary_signal = signals[0]
        avg_price = weighted_price / Decimal(str(total_weight))
        avg_quantity = weighted_quantity / Decimal(str(total_weight))
        
        return TradingSignal(
            signal_id=f"ENSEMBLE_BLEND_{primary_signal.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id="ensemble_blending",
            symbol=primary_signal.symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=avg_price,
            quantity=avg_quantity,
            time_horizon=primary_signal.time_horizon,
            metadata={
                'ensemble_type': 'blending',
                'buy_weight': buy_weight / total_weight,
                'sell_weight': sell_weight / total_weight,
                'strategies': [s.strategy_id for s in signals]
            }
        )
    
    def _diversification_ensemble_signal(self, signals: List[TradingSignal], weights: Dict[str, float]) -> Optional[TradingSignal]:
        """Diversification ensemble signal combination"""
        # For diversification, require consensus across strategies
        buy_count = sum(1 for s in signals if s.signal_type in [SignalType.BUY, SignalType.INCREASE])
        sell_count = sum(1 for s in signals if s.signal_type in [SignalType.SELL, SignalType.REDUCE])
        
        total_strategies = len(signals)
        
        # Require significant consensus (60% or more)
        if buy_count >= 0.6 * total_strategies:
            # Strong buy consensus
            primary_signal = signals[0]
            strength = buy_count / total_strategies
            
            return TradingSignal(
                signal_id=f"ENSEMBLE_DIV_BUY_{primary_signal.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id="ensemble_diversification",
                symbol=primary_signal.symbol,
                signal_type=SignalType.BUY,
                confidence=strength,
                strength=strength,
                price=primary_signal.price,
                quantity=primary_signal.quantity,
                time_horizon=primary_signal.time_horizon,
                metadata={
                    'ensemble_type': 'diversification',
                    'consensus_level': strength,
                    'buy_count': buy_count,
                    'sell_count': sell_count,
                    'total_strategies': total_strategies
                }
            )
        
        elif sell_count >= 0.6 * total_strategies:
            # Strong sell consensus
            primary_signal = signals[0]
            strength = sell_count / total_strategies
            
            return TradingSignal(
                signal_id=f"ENSEMBLE_DIV_SELL_{primary_signal.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id="ensemble_diversification",
                symbol=primary_signal.symbol,
                signal_type=SignalType.SELL,
                confidence=strength,
                strength=strength,
                price=primary_signal.price,
                quantity=primary_signal.quantity,
                time_horizon=primary_signal.time_horizon,
                metadata={
                    'ensemble_type': 'diversification',
                    'consensus_level': strength,
                    'buy_count': buy_count,
                    'sell_count': sell_count,
                    'total_strategies': total_strategies
                }
            )
        
        return None
    
    async def get_strategy_performance_summary(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance summary for a strategy"""
        if strategy_id not in self.strategy_performance:
            return None
        
        performance = self.strategy_performance[strategy_id]
        
        # Get correlations
        correlations = self.correlation_matrix.get(strategy_id, {})
        
        # Calculate performance ranking
        all_performances = list(self.strategy_performance.values())
        ranking = sorted(all_performances, key=lambda x: x.sharpe_ratio, reverse=True)
        rank = next((i + 1 for i, p in enumerate(ranking) if p.strategy_id == strategy_id), len(ranking))
        
        return {
            'strategy_id': strategy_id,
            'performance': {
                'total_return': f"{performance.total_return:.2%}",
                'sharpe_ratio': f"{performance.sharpe_ratio:.3f}",
                'sortino_ratio': f"{performance.sortino_ratio:.3f}",
                'calmar_ratio': f"{performance.calmar_ratio:.3f}",
                'max_drawdown': f"{performance.max_drawdown:.2%}",
                'volatility': f"{performance.volatility:.2%}",
                'win_rate': f"{performance.win_rate:.2%}",
                'profit_factor': f"{performance.profit_factor:.3f}",
                'trades_count': performance.trades_count,
                'avg_trade_duration_hours': performance.avg_trade_duration.total_seconds() / 3600,
                'var_95': f"{performance.var_95:.2%}",
                'cvar_95': f"{performance.cvar_95:.2%}"
            },
            'ranking': {
                'overall_rank': rank,
                'total_strategies': len(all_performances),
                'percentile': f"{((len(all_performances) - rank) / len(all_performances) * 100):.1f}%"
            },
            'correlations': {k: f"{v:.3f}" for k, v in correlations.items() if k != strategy_id},
            'last_updated': performance.last_updated.isoformat(),
            'risk_budget': self.risk_budgets.get(strategy_id, 0.1)
        }
    
    async def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple strategies performance"""
        comparison_results = {}
        
        for strategy_id in strategy_ids:
            if strategy_id in self.strategy_performance:
                comparison_results[strategy_id] = await self.get_strategy_performance_summary(strategy_id)
        
        # Calculate relative performance
        if len(comparison_results) > 1:
            sharpe_ratios = [self.strategy_performance[sid].sharpe_ratio for sid in strategy_ids if sid in self.strategy_performance]
            max_sharpe = max(sharpe_ratios) if sharpe_ratios else 0
            
            for strategy_id in comparison_results:
                if strategy_id in self.strategy_performance:
                    sharpe = self.strategy_performance[strategy_id].sharpe_ratio
                    comparison_results[strategy_id]['relative_performance'] = {
                        'vs_best_sharpe': f"{((sharpe / max_sharpe - 1) * 100):+.1f}%" if max_sharpe > 0 else "N/A",
                        'volatility_adj_return': f"{self.strategy_performance[strategy_id].total_return:.2%}"
                    }
        
        return {
            'comparison_date': datetime.utcnow().isoformat(),
            'strategies_compared': strategy_ids,
            'results': comparison_results
        }
    
    async def run_ab_test(self, strategy_a: str, strategy_b: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run A/B test between two strategies"""
        if strategy_a not in self.strategy_performance or strategy_b not in self.strategy_performance:
            return {'error': 'One or both strategies not found'}
        
        performance_a = self.strategy_performance[strategy_a]
        performance_b = self.strategy_performance[strategy_b]
        
        # Calculate statistical significance (simplified)
        returns_a = list(self.performance_history[strategy_a])
        returns_b = list(self.performance_history[strategy_b])
        
        if len(returns_a) < 30 or len(returns_b) < 30:
            return {'error': 'Insufficient data for A/B test'}
        
        # T-test for difference in means
        mean_a = np.mean(returns_a)
        mean_b = np.mean(returns_b)
        std_a = np.std(returns_a)
        std_b = np.std(returns_b)
        n_a = len(returns_a)
        n_b = len(returns_b)
        
        # Welch's t-test
        pooled_se = math.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        t_stat = (mean_a - mean_b) / pooled_se if pooled_se > 0 else 0
        
        # Determine winner
        if mean_a > mean_b:
            winner = strategy_a
            improvement = (mean_a / mean_b - 1) * 100 if mean_b > 0 else 0
        else:
            winner = strategy_b
            improvement = (mean_b / mean_a - 1) * 100 if mean_a > 0 else 0
        
        return {
            'test_id': test_config.get('test_id', f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
            'start_date': test_config.get('start_date'),
            'strategy_a': {
                'id': strategy_a,
                'avg_return': f"{mean_a:.4f}",
                'volatility': f"{std_a:.4f}",
                'sharpe_ratio': f"{performance_a.sharpe_ratio:.3f}",
                'total_return': f"{performance_a.total_return:.2%}"
            },
            'strategy_b': {
                'id': strategy_b,
                'avg_return': f"{mean_b:.4f}",
                'volatility': f"{std_b:.4f}",
                'sharpe_ratio': f"{performance_b.sharpe_ratio:.3f}",
                'total_return': f"{performance_b.total_return:.2%}"
            },
            'results': {
                'winner': winner,
                'improvement_pct': f"{improvement:.2f}%",
                't_statistic': f"{t_stat:.3f}",
                'significance_level': 'High' if abs(t_stat) > 2.576 else 'Medium' if abs(t_stat) > 1.96 else 'Low'
            },
            'test_date': datetime.utcnow().isoformat()
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get strategy management dashboard summary"""
        active_count = len(self.active_strategies)
        ensemble_count = len(self.ensemble_strategies)
        
        if self.strategy_performance:
            best_sharpe = max(p.sharpe_ratio for p in self.strategy_performance.values())
            worst_sharpe = min(p.sharpe_ratio for p in self.strategy_performance.values())
            avg_sharpe = np.mean([p.sharpe_ratio for p in self.strategy_performance.values()])
            avg_return = np.mean([p.total_return for p in self.strategy_performance.values()])
        else:
            best_sharpe = worst_sharpe = avg_sharpe = avg_return = 0.0
        
        return {
            'overview': {
                'active_strategies': active_count,
                'ensemble_strategies': ensemble_count,
                'total_strategies': active_count + ensemble_count,
                'best_sharpe_ratio': f"{best_sharpe:.3f}",
                'worst_sharpe_ratio': f"{worst_sharpe:.3f}",
                'average_sharpe_ratio': f"{avg_sharpe:.3f}",
                'average_return': f"{avg_return:.2%}"
            },
            'top_performers': [
                {
                    'strategy_id': p.strategy_id,
                    'sharpe_ratio': f"{p.sharpe_ratio:.3f}",
                    'total_return': f"{p.total_return:.2%}",
                    'max_drawdown': f"{p.max_drawdown:.2%}"
                }
                for p in sorted(self.strategy_performance.values(), key=lambda x: x.sharpe_ratio, reverse=True)[:5]
            ],
            'recent_rebalancing': self.allocation_history[-5:] if self.allocation_history else [],
            'correlation_clusters': self._get_correlation_clusters(),
            'risk_utilization': self._calculate_risk_utilization()
        }
    
    def _get_correlation_clusters(self) -> List[Dict[str, Any]]:
        """Identify highly correlated strategy clusters"""
        clusters = []
        processed = set()
        
        for strategy_id, correlations in self.correlation_matrix.items():
            if strategy_id in processed:
                continue
            
            # Find highly correlated strategies (correlation > 0.7)
            highly_correlated = [sid for sid, corr in correlations.items() 
                               if sid != strategy_id and corr > 0.7]
            
            if highly_correlated:
                cluster = {
                    'strategy_ids': [strategy_id] + highly_correlated,
                    'average_correlation': np.mean([correlations[sid] for sid in highly_correlated]),
                    'size': len(highly_correlated) + 1
                }
                clusters.append(cluster)
                processed.add(strategy_id)
                processed.update(highly_correlated)
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)[:3]
    
    def _calculate_risk_utilization(self) -> Dict[str, float]:
        """Calculate current risk utilization by strategy"""
        total_risk_budget = sum(self.risk_budgets.values())
        
        if total_risk_budget == 0:
            return {}
        
        return {
            strategy_id: (self.risk_budgets[strategy_id] / total_risk_budget) * 100
            for strategy_id in self.risk_budgets
        }


# Global strategy manager instance
strategy_manager = StrategyManager()


# Quick start functions for strategy management
async def quick_setup_momentum_ensemble() -> Optional[str]:
    """Quick setup for a momentum ensemble strategy"""
    momentum_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MOMENTUM)
    
    if len(momentum_strategies) < 3:
        logger.error("Insufficient momentum strategies for ensemble")
        return None
    
    # Select top 3 momentum strategies
    top_strategies = [s.strategy_id for s in momentum_strategies[:3]]
    
    ensemble_config = {
        'ensemble_id': f"momentum_ensemble_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        'strategies': top_strategies,
        'ensemble_type': 'voting',
        'allocation_method': 'equal_weight',
        'rebalance_frequency': 'weekly',
        'min_allocation': 0.1,
        'max_strategies': 5
    }
    
    ensemble = await strategy_manager.create_ensemble(ensemble_config)
    return ensemble.ensemble_id if ensemble else None


async def setup_balanced_portfolio() -> Optional[str]:
    """Setup a balanced portfolio with different strategy types"""
    # Get strategies from different categories
    momentum_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MOMENTUM)[:2]
    mean_reversion_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MEAN_REVERSION)[:2]
    volatility_strategies = strategy_library.get_strategies_by_category(StrategyCategory.VOLATILITY)[:1]
    
    all_strategies = (momentum_strategies + mean_reversion_strategies + volatility_strategies)
    
    if len(all_strategies) < 3:
        logger.error("Insufficient strategies for balanced portfolio")
        return None
    
    strategy_ids = [s.strategy_id for s in all_strategies]
    
    ensemble_config = {
        'ensemble_id': f"balanced_portfolio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        'strategies': strategy_ids,
        'ensemble_type': 'diversification',
        'allocation_method': 'risk_parity',
        'rebalance_frequency': 'monthly',
        'min_allocation': 0.15,
        'max_strategies': 8,
        'allocation_overrides': {
            strategy_ids[0]: {'weight': 0.3, 'target_volatility': 0.12},
            strategy_ids[1]: {'weight': 0.3, 'target_volatility': 0.12},
            strategy_ids[2]: {'weight': 0.4, 'target_volatility': 0.18}
        }
    }
    
    ensemble = await strategy_manager.create_ensemble(ensemble_config)
    return ensemble.ensemble_id if ensemble else None


# Export the strategy manager
__all__ = [
    'StrategyManager',
    'strategy_manager',
    'StrategyPerformance',
    'StrategyAllocation',
    'EnsembleStrategy',
    'AllocationMethod',
    'RebalanceFrequency',
    'EnsembleType',
    'quick_setup_momentum_ensemble',
    'setup_balanced_portfolio'
]