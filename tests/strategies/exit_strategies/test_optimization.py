"""
@file test_optimization.py
@brief Comprehensive unit tests for exit strategy optimization module

@details
This module provides comprehensive unit tests for the ExitStrategyOptimizer
and related optimization functionality. Tests cover optimization algorithms,
parameter ranges, objective functions, and optimization results.

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@see optimization.py for implementation details
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np

from trading_orchestrator.strategies.exit_strategies.optimization import (
    ExitStrategyOptimizer,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationParameter,
    OptimizationResult,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GeneticAlgorithmOptimizer,
    create_parameter_ranges_for_trailing_stop,
    create_parameter_ranges_for_fixed_target,
    run_exit_strategy_optimization
)
from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitType,
    ExitConfiguration
)
from trading_orchestrator.strategies.exit_strategies.backtesting import (
    BacktestMetrics
)


class TestOptimizationDataClasses:
    """Test optimization data classes"""
    
    def test_optimization_parameter_creation(self):
        """Test OptimizationParameter dataclass"""
        param = OptimizationParameter(
            name="trailing_distance",
            param_type="float",
            min_value=0.01,
            max_value=0.10,
            step=0.005,
            description="Trailing stop distance as percentage"
        )
        
        assert param.name == "trailing_distance"
        assert param.param_type == "float"
        assert param.min_value == 0.01
        assert param.max_value == 0.10
        assert param.description == "Trailing stop distance as percentage"
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult dataclass"""
        result = OptimizationResult(
            optimization_id="opt_001",
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            best_parameters={"trailing_distance": 0.05},
            best_score=1.25,
            optimization_history=[],
            total_evaluations=100,
            optimization_time=5.2,
            convergence_achieved=False
        )
        
        assert result.optimization_id == "opt_001"
        assert result.method == OptimizationMethod.GRID_SEARCH
        assert result.best_score == 1.25
        assert isinstance(result.created_at, datetime)
    
    def test_optimization_enums(self):
        """Test optimization enums"""
        assert OptimizationMethod.GRID_SEARCH.value == "grid_search"
        assert OptimizationMethod.RANDOM_SEARCH.value == "random_search"
        assert OptimizationMethod.GENETIC_ALGORITHM.value == "genetic_algorithm"
        
        assert OptimizationObjective.MAXIMIZE_SHARPE_RATIO.value == "maximize_sharpe_ratio"
        assert OptimizationObjective.MINIMIZE_MAX_DRAWDOWN.value == "minimize_max_drawdown"
        assert OptimizationObjective.MAXIMIZE_TOTAL_RETURN.value == "maximize_total_return"


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer class"""
    
    @pytest.fixture
    def optimizer(self):
        """Create grid search optimizer instance"""
        return GridSearchOptimizer()
    
    @pytest.fixture
    def mock_objective_function(self):
        """Create mock objective function"""
        async def objective(params):
            # Simple scoring function for testing
            return params.get("param1", 0) + params.get("param2", 0)
        return objective
    
    @pytest.fixture
    def sample_parameter_space(self):
        """Create sample parameter space"""
        return [
            OptimizationParameter(
                name="param1",
                param_type="float",
                min_value=0.1,
                max_value=0.3,
                step=0.1,
                description="Test parameter 1"
            ),
            OptimizationParameter(
                name="param2",
                param_type="choice",
                choices=[True, False],
                description="Test parameter 2"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_grid_search_optimize_basic(self, optimizer, mock_objective_function, sample_parameter_space):
        """Test basic grid search optimization"""
        result = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=10
        )
        
        assert result.method == OptimizationMethod.GRID_SEARCH
        assert result.total_evaluations > 0
        assert len(result.optimization_history) > 0
        assert result.best_score is not None
        assert isinstance(result.best_parameters, dict)
    
    @pytest.mark.asyncio
    async def test_grid_search_parameter_grid_generation(self, optimizer, sample_parameter_space):
        """Test parameter grid generation"""
        grid = optimizer._generate_parameter_grid(sample_parameter_space)
        
        assert len(grid) > 0
        
        # Should have combinations of param1 values and param2 choices
        expected_combinations = 3 * 2  # 3 param1 values × 2 param2 choices
        assert len(grid) >= expected_combinations
        
        # Verify all combinations have required parameters
        for combination in grid:
            assert "param1" in combination
            assert "param2" in combination
            assert isinstance(combination["param1"], float)
            assert isinstance(combination["param2"], bool)
    
    @pytest.mark.asyncio
    async def test_grid_search_evaluation_limit(self, optimizer, mock_objective_function, sample_parameter_space):
        """Test grid search with evaluation limit"""
        result = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=3  # Very low limit
        )
        
        assert result.total_evaluations <= 3
        assert len(result.optimization_history) <= 3
    
    @pytest.mark.asyncio
    async def test_grid_search_error_handling(self, optimizer, sample_parameter_space):
        """Test grid search error handling"""
        async def failing_objective(params):
            raise Exception("Objective function error")
        
        with pytest.raises(Exception) as exc_info:
            await optimizer.optimize(
                objective_function=failing_objective,
                parameter_space=sample_parameter_space
            )
        
        assert "Objective function error" in str(exc_info.value)


class TestRandomSearchOptimizer:
    """Test RandomSearchOptimizer class"""
    
    @pytest.fixture
    def optimizer(self):
        """Create random search optimizer instance"""
        return RandomSearchOptimizer()
    
    @pytest.fixture
    def mock_objective_function(self):
        """Create mock objective function"""
        async def objective(params):
            # Simple scoring function for testing
            return params.get("param1", 0) * 2 + params.get("param2", 0)
        return objective
    
    @pytest.fixture
    def sample_parameter_space(self):
        """Create sample parameter space"""
        return [
            OptimizationParameter(
                name="param1",
                param_type="float",
                min_value=0.0,
                max_value=1.0,
                description="Test float parameter"
            ),
            OptimizationParameter(
                name="param2",
                param_type="int",
                min_value=1,
                max_value=10,
                description="Test integer parameter"
            ),
            OptimizationParameter(
                name="param3",
                param_type="choice",
                choices=["option1", "option2", "option3"],
                description="Test choice parameter"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_random_search_optimize_basic(self, optimizer, mock_objective_function, sample_parameter_space):
        """Test basic random search optimization"""
        result = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=20,
            random_seed=42
        )
        
        assert result.method == OptimizationMethod.RANDOM_SEARCH
        assert result.total_evaluations == 20
        assert len(result.optimization_history) == 20
        assert result.best_score is not None
        assert isinstance(result.best_parameters, dict)
    
    @pytest.mark.asyncio
    async def test_random_search_reproducibility(self, optimizer, mock_objective_function, sample_parameter_space):
        """Test random search reproducibility with seed"""
        result1 = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=10,
            random_seed=123
        )
        
        result2 = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=10,
            random_seed=123
        )
        
        # With same seed, should get same best parameters (approximately)
        assert result1.best_parameters == result2.best_parameters
        assert result1.best_score == result2.best_score
    
    @pytest.mark.asyncio
    async def test_random_search_parameter_generation(self, optimizer, sample_parameter_space):
        """Test random parameter generation"""
        # Generate many random parameter sets to test distribution
        for _ in range(100):
            params = optimizer._generate_random_parameters(sample_parameter_space)
            
            assert "param1" in params
            assert "param2" in params
            assert "param3" in params
            
            assert 0.0 <= params["param1"] <= 1.0
            assert 1 <= params["param2"] <= 10
            assert params["param3"] in ["option1", "option2", "option3"]


class TestGeneticAlgorithmOptimizer:
    """Test GeneticAlgorithmOptimizer class"""
    
    @pytest.fixture
    def optimizer(self):
        """Create genetic algorithm optimizer instance"""
        return GeneticAlgorithmOptimizer()
    
    @pytest.fixture
    def mock_objective_function(self):
        """Create mock objective function for GA"""
        async def objective(params):
            # Simple function that rewards certain parameter combinations
            score = 0
            if params.get("param1", 0) > 0.5:
                score += 10
            if params.get("param2", 0) > 5:
                score += 5
            return score
        return objective
    
    @pytest.fixture
    def sample_parameter_space(self):
        """Create parameter space for GA testing"""
        return [
            OptimizationParameter(
                name="param1",
                param_type="float",
                min_value=0.0,
                max_value=1.0,
                description="Float parameter"
            ),
            OptimizationParameter(
                name="param2",
                param_type="int",
                min_value=1,
                max_value=10,
                description="Integer parameter"
            ),
            OptimizationParameter(
                name="param3",
                param_type="choice",
                choices=[True, False],
                description="Choice parameter"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_basic(self, optimizer, mock_objective_function, sample_parameter_space):
        """Test basic genetic algorithm optimization"""
        result = await optimizer.optimize(
            objective_function=mock_objective_function,
            parameter_space=sample_parameter_space,
            max_evaluations=100
        )
        
        assert result.method == OptimizationMethod.GENETIC_ALGORITHM
        assert result.total_evaluations > 0
        assert len(result.optimization_history) > 0
        assert result.convergence_achieved == True
        assert isinstance(result.best_parameters, dict)
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_population_initialization(self, optimizer, sample_parameter_space):
        """Test GA population initialization"""
        population_size = 20
        population = optimizer._initialize_population(sample_parameter_space, population_size)
        
        assert len(population) == population_size
        
        # Verify each individual has all required parameters
        for individual in population:
            assert "param1" in individual
            assert "param2" in individual
            assert "param3" in individual
            
            assert isinstance(individual["param1"], float)
            assert isinstance(individual["param2"], int)
            assert isinstance(individual["param3"], bool)
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_evolution(self, optimizer, sample_parameter_space):
        """Test GA population evolution"""
        # Create initial population
        population = optimizer._initialize_population(sample_parameter_space, 10)
        fitness_scores = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2, 2.8, 1.8]
        
        # Evolve population
        new_population = optimizer._evolve_population(population, fitness_scores, sample_parameter_space)
        
        assert len(new_population) == len(population)
        assert len(new_population) == 10
        
        # Verify all individuals still have correct parameter structure
        for individual in new_population:
            assert "param1" in individual
            assert "param2" in individual
            assert "param3" in individual
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_crossover(self, optimizer, sample_parameter_space):
        """Test GA crossover operation"""
        parent1 = {"param1": 0.2, "param2": 3, "param3": True}
        parent2 = {"param1": 0.8, "param2": 7, "param3": False}
        
        offspring1, offspring2 = optimizer._crossover(parent1, parent2, sample_parameter_space)
        
        # Verify offspring have correct structure
        assert "param1" in offspring1 and "param1" in offspring2
        assert "param2" in offspring1 and "param2" in offspring2
        assert "param3" in offspring1 and "param3" in offspring2
        
        # Verify parameters are from parents
        assert offspring1["param1"] in [parent1["param1"], parent2["param1"]]
        assert offspring2["param2"] in [parent1["param2"], parent2["param2"]]
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_mutation(self, optimizer, sample_parameter_space):
        """Test GA mutation operation"""
        individual = {"param1": 0.5, "param2": 5, "param3": True}
        original = individual.copy()
        
        mutated = optimizer._mutate(individual, sample_parameter_space)
        
        # Verify structure is maintained
        assert "param1" in mutated and "param2" in mutated and "param3" in mutated
        assert isinstance(mutated["param1"], float)
        assert isinstance(mutated["param2"], int)
        assert isinstance(mutated["param3"], bool)
        
        # Verify bounds are respected
        assert 0.0 <= mutated["param1"] <= 1.0
        assert 1 <= mutated["param2"] <= 10


class TestExitStrategyOptimizer:
    """Test ExitStrategyOptimizer class"""
    
    @pytest.fixture
    def optimizer(self):
        """Create exit strategy optimizer instance"""
        return ExitStrategyOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.backtest_engine is not None
        assert OptimizationMethod.GRID_SEARCH in optimizer.algorithms
        assert OptimizationMethod.RANDOM_SEARCH in optimizer.algorithms
        assert OptimizationMethod.GENETIC_ALGORITHM in optimizer.algorithms
    
    @pytest.mark.asyncio
    async def test_optimize_exit_strategy_parameters(self, optimizer):
        """Test exit strategy parameter optimization"""
        parameter_ranges = [
            OptimizationParameter(
                name="trailing_distance",
                param_type="float",
                min_value=0.01,
                max_value=0.05,
                step=0.01
            )
        ]
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        with patch.object(optimizer, '_create_objective_function') as mock_obj_func:
            mock_obj_func.return_value = lambda params: 1.5  # Mock objective score
            
            with patch.object(optimizer, 'algorithms', {OptimizationMethod.RANDOM_SEARCH: Mock()}):
                mock_algorithm = Mock()
                mock_algorithm.optimize = AsyncMock()
                mock_algorithm.optimize.return_value = OptimizationResult(
                    optimization_id="test_opt",
                    method=OptimizationMethod.RANDOM_SEARCH,
                    objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                    best_parameters={"trailing_distance": 0.03},
                    best_score=1.5,
                    optimization_history=[],
                    total_evaluations=10,
                    optimization_time=2.5,
                    convergence_achieved=False
                )
                optimizer.algorithms[OptimizationMethod.RANDOM_SEARCH] = mock_algorithm
                
                result = await optimizer.optimize_exit_strategy_parameters(
                    strategy_type=ExitType.TRAILING_STOP,
                    symbol="AAPL",
                    parameter_ranges=parameter_ranges,
                    objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                    method=OptimizationMethod.RANDOM_SEARCH,
                    start_date=start_date,
                    end_date=end_date,
                    max_evaluations=10
                )
                
                assert result is not None
                assert result.optimization_id == "test_opt"
    
    @pytest.mark.asyncio
    async def test_create_objective_function(self, optimizer):
        """Test objective function creation"""
        parameter_ranges = [
            OptimizationParameter(
                name="trailing_distance",
                param_type="float",
                min_value=0.01,
                max_value=0.05
            )
        ]
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        # Mock backtest engine
        mock_backtest_engine = Mock()
        mock_backtest_engine.run_backtest = AsyncMock()
        
        # Mock backtest result
        mock_result = Mock()
        mock_result.strategies = {
            "test_strategy": BacktestMetrics(
                strategy_id="test_strategy",
                total_trades=50, winning_trades=30, losing_trades=20,
                win_rate=0.6, total_return=0.15, annualized_return=0.18,
                sharpe_ratio=1.2, sortino_ratio=1.5, max_drawdown=-0.08,
                calmar_ratio=2.25, profit_factor=1.8, average_win=100.0,
                average_loss=-60.0, largest_win=500.0, largest_loss=-200.0,
                avg_trade_duration=2.0, total_exposure_time=100.0,
                commission_costs=50.0, market_impact_costs=25.0,
                net_return=0.125, volatility=0.20, var_95=-0.05,
                cvar_95=-0.07, trade_frequency=5.0, avg_confidence=0.8,
                avg_urgency=0.6
            )
        }
        mock_backtest_engine.run_backtest.return_value = mock_result
        
        optimizer.backtest_engine = mock_backtest_engine
        
        # Mock strategy creation
        with patch('trading_orchestrator.strategies.exit_strategies.optimization.create_trailing_stop_strategy') as mock_create:
            mock_strategy = Mock()
            mock_create.return_value = mock_strategy
            
            objective_function = await optimizer._create_objective_function(
                strategy_type=ExitType.TRAILING_STOP,
                symbol="AAPL",
                parameter_ranges=parameter_ranges,
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal("100000")
            )
            
            # Test objective function
            test_params = {"trailing_distance": 0.03}
            score = await objective_function(test_params)
            
            assert isinstance(score, float)
            assert score == 1.2  # Sharpe ratio from mock metrics
    
    def test_calculate_objective_score(self, optimizer):
        """Test objective score calculation"""
        # Create sample metrics
        metrics = BacktestMetrics(
            strategy_id="test_strategy",
            total_trades=50, winning_trades=30, losing_trades=20,
            win_rate=0.6, total_return=0.15, annualized_return=0.18,
            sharpe_ratio=1.2, sortino_ratio=1.5, max_drawdown=-0.08,
            calmar_ratio=2.25, profit_factor=1.8, average_win=100.0,
            average_loss=-60.0, largest_win=500.0, largest_loss=-200.0,
            avg_trade_duration=2.0, total_exposure_time=100.0,
            commission_costs=50.0, market_impact_costs=25.0,
            net_return=0.125, volatility=0.20, var_95=-0.05,
            cvar_95=-0.07, trade_frequency=5.0, avg_confidence=0.8,
            avg_urgency=0.6
        )
        
        # Test different objectives
        sharpe_score = optimizer._calculate_objective_score(
            metrics, OptimizationObjective.MAXIMIZE_SHARPE_RATIO
        )
        assert sharpe_score == 1.2
        
        drawdown_score = optimizer._calculate_objective_score(
            metrics, OptimizationObjective.MINIMIZE_MAX_DRAWDOWN
        )
        assert drawdown_score == -0.08  # Negative for minimization
        
        return_score = optimizer._calculate_objective_score(
            metrics, OptimizationObjective.MAXIMIZE_TOTAL_RETURN
        )
        assert return_score == 0.15
        
        volatility_score = optimizer._calculate_objective_score(
            metrics, OptimizationObjective.MINIMIZE_VOLATILITY
        )
        assert volatility_score == -0.20  # Negative for minimization


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_parameter_ranges_for_trailing_stop(self):
        """Test trailing stop parameter ranges"""
        param_ranges = create_parameter_ranges_for_trailing_stop()
        
        assert len(param_ranges) == 3
        
        param_names = [p.name for p in param_ranges]
        assert "trailing_distance" in param_names
        assert "initial_stop" in param_names
        assert "update_frequency" in param_names
        
        # Verify parameter types and bounds
        trailing_param = next(p for p in param_ranges if p.name == "trailing_distance")
        assert trailing_param.param_type == "float"
        assert trailing_param.min_value == 0.01
        assert trailing_param.max_value == 0.10
        assert trailing_param.step == 0.005
    
    def test_create_parameter_ranges_for_fixed_target(self):
        """Test fixed target parameter ranges"""
        param_ranges = create_parameter_ranges_for_fixed_target()
        
        assert len(param_ranges) == 3
        
        param_names = [p.name for p in param_ranges]
        assert "profit_target" in param_names
        assert "loss_target" in param_names
        assert "partial_exits" in param_names
        
        # Verify choice parameter
        partial_exits_param = next(p for p in param_ranges if p.name == "partial_exits")
        assert partial_exits_param.param_type == "choice"
        assert partial_exits_param.choices == [True, False]
    
    @pytest.mark.asyncio
    async def test_run_exit_strategy_optimization(self):
        """Test run_exit_strategy_optimization convenience function"""
        with patch('trading_orchestrator.strategies.exit_strategies.optimization.ExitStrategyOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_exit_strategy_parameters = AsyncMock()
            
            result = await run_exit_strategy_optimization(
                strategy_type=ExitType.TRAILING_STOP,
                symbol="AAPL",
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                method=OptimizationMethod.RANDOM_SEARCH,
                max_evaluations=50
            )
            
            mock_optimizer.optimize_exit_strategy_parameters.assert_called_once()
            assert result is not None


class TestOptimizationEdgeCases:
    """Test optimization edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_optimization_with_empty_parameter_space(self):
        """Test optimization with empty parameter space"""
        optimizer = ExitStrategyOptimizer()
        
        result = await optimizer.optimize_exit_strategy_parameters(
            strategy_type=ExitType.TRAILING_STOP,
            symbol="AAPL",
            parameter_ranges=[],  # Empty parameter space
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            method=OptimizationMethod.RANDOM_SEARCH,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            max_evaluations=10
        )
        
        # Should handle gracefully
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_optimization_unsupported_method(self):
        """Test optimization with unsupported method"""
        optimizer = ExitStrategyOptimizer()
        
        with pytest.raises(ValueError) as exc_info:
            await optimizer.optimize_exit_strategy_parameters(
                strategy_type=ExitType.TRAILING_STOP,
                symbol="AAPL",
                parameter_ranges=[],
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                method="unsupported_method",  # Invalid method
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow()
            )
        
        assert "Unsupported optimization method" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_optimization_backtest_failure(self):
        """Test optimization when backtest fails"""
        optimizer = ExitStrategyOptimizer()
        
        # Mock backtest engine to fail
        mock_backtest_engine = Mock()
        mock_backtest_engine.run_backtest.side_effect = Exception("Backtest failed")
        optimizer.backtest_engine = mock_backtest_engine
        
        parameter_ranges = [
            OptimizationParameter(
                name="trailing_distance",
                param_type="float",
                min_value=0.01,
                max_value=0.05
            )
        ]
        
        # Mock objective function creation
        with patch.object(optimizer, '_create_objective_function') as mock_create:
            async def failing_objective(params):
                return float('-inf')  # Return worst score when backtest fails
            
            mock_create.return_value = failing_objective
            
            objective_function = await optimizer._create_objective_function(
                strategy_type=ExitType.TRAILING_STOP,
                symbol="AAPL",
                parameter_ranges=parameter_ranges,
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow(),
                initial_capital=Decimal("100000")
            )
            
            # Test with failing backtest
            score = await objective_function({"trailing_distance": 0.03})
            assert score == float('-inf')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
