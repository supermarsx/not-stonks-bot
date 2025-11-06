"""
@file optimization.py
@brief Exit Strategy Optimization Framework

@details
This module provides optimization capabilities for exit strategies, enabling
users to find optimal parameters for different exit approaches. The framework
supports multiple optimization algorithms and objective functions.

Key Features:
- Parameter optimization for exit strategies
- Multiple optimization algorithms
- Objective function customization
- Multi-objective optimization
- Optimization result analysis

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
Optimization should be performed on out-of-sample data to avoid overfitting
and ensure robustness in live trading.

@see backtesting.py for backtesting framework
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import random
from abc import ABC, abstractmethod

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitType,
    ExitConfiguration
)
from .backtesting import (
    ExitStrategyBacktestEngine,
    BacktestResult,
    BacktestMetrics
)


class OptimizationMethod(Enum):
    """Optimization algorithm methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian"


class OptimizationObjective(Enum):
    """Optimization objective functions"""
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    MINIMIZE_MAX_DRAWDOWN = "minimize_max_drawdown"
    MAXIMIZE_TOTAL_RETURN = "maximize_total_return"
    MAXIMIZE_CALMAR_RATIO = "maximize_calmar_ratio"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"
    CUSTOM_OBJECTIVE = "custom_objective"


@dataclass
class OptimizationParameter:
    """Single optimization parameter definition"""
    name: str
    param_type: str  # "float", "int", "choice"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""


@dataclass
class OptimizationResult:
    """Optimization result"""
    optimization_id: str
    method: OptimizationMethod
    objective: OptimizationObjective
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_achieved: bool
    created_at: datetime = field(default_factory=datetime.utcnow)


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms"""
    
    @abstractmethod
    async def optimize(
        self,
        objective_function: Callable,
        parameter_space: List[OptimizationParameter],
        max_evaluations: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Run optimization"""
        pass


class GridSearchOptimizer(OptimizationAlgorithm):
    """Grid search optimization algorithm"""
    
    async def optimize(
        self,
        objective_function: Callable,
        parameter_space: List[OptimizationParameter],
        max_evaluations: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Run grid search optimization"""
        optimization_id = f"grid_{datetime.utcnow().timestamp()}"
        logger.info(f"Starting grid search optimization {optimization_id}")
        
        try:
            # Generate parameter grid
            parameter_grid = self._generate_parameter_grid(parameter_space)
            
            # Limit evaluations if grid is too large
            if len(parameter_grid) > max_evaluations:
                logger.warning(f"Grid size ({len(parameter_grid)}) exceeds max evaluations ({max_evaluations})")
                parameter_grid = parameter_grid[:max_evaluations]
            
            # Evaluate all parameter combinations
            results = []
            best_score = float('-inf')
            best_params = {}
            
            for i, params in enumerate(parameter_grid):
                try:
                    score = await objective_function(params)
                    result = {
                        'parameters': params,
                        'score': score,
                        'evaluation': i + 1
                    }
                    results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Grid search progress: {i + 1}/{len(parameter_grid)}")
                
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    continue
            
            # Create optimization result
            optimization_result = OptimizationResult(
                optimization_id=optimization_id,
                method=OptimizationMethod.GRID_SEARCH,
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,  # Default
                best_parameters=best_params,
                best_score=best_score,
                optimization_history=results,
                total_evaluations=len(results),
                optimization_time=0.0,  # Would calculate actual time
                convergence_achieved=False  # Grid search doesn't converge
            )
            
            logger.info(f"Grid search optimization completed with {len(results)} evaluations")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            raise
    
    def _generate_parameter_grid(
        self, 
        parameter_space: List[OptimizationParameter]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        try:
            # Start with empty parameter combination
            grid = [{}]
            
            for param in parameter_space:
                new_grid = []
                
                for base_combination in grid:
                    if param.param_type == "choice":
                        # Add all choices
                        for choice in param.choices:
                            new_combination = base_combination.copy()
                            new_combination[param.name] = choice
                            new_grid.append(new_combination)
                    
                    elif param.param_type in ["float", "int"]:
                        # Generate value range
                        if param.min_value is None or param.max_value is None:
                            continue
                        
                        step = param.step or (param.max_value - param.min_value) / 10
                        
                        values = []
                        current = param.min_value
                        while current <= param.max_value:
                            values.append(current)
                            current += step
                        
                        # Add all values
                        for value in values:
                            new_combination = base_combination.copy()
                            new_combination[param.name] = value
                            new_grid.append(new_combination)
                
                grid = new_grid
            
            return grid
            
        except Exception as e:
            logger.error(f"Error generating parameter grid: {e}")
            return []


class RandomSearchOptimizer(OptimizationAlgorithm):
    """Random search optimization algorithm"""
    
    async def optimize(
        self,
        objective_function: Callable,
        parameter_space: List[OptimizationParameter],
        max_evaluations: int = 1000,
        random_seed: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run random search optimization"""
        optimization_id = f"random_{datetime.utcnow().timestamp()}"
        logger.info(f"Starting random search optimization {optimization_id}")
        
        try:
            # Set random seed
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)
            
            results = []
            best_score = float('-inf')
            best_params = {}
            
            for i in range(max_evaluations):
                # Generate random parameters
                params = self._generate_random_parameters(parameter_space)
                
                try:
                    score = await objective_function(params)
                    result = {
                        'parameters': params,
                        'score': score,
                        'evaluation': i + 1
                    }
                    results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Random search progress: {i + 1}/{max_evaluations}")
                
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    continue
            
            # Create optimization result
            optimization_result = OptimizationResult(
                optimization_id=optimization_id,
                method=OptimizationMethod.RANDOM_SEARCH,
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                best_parameters=best_params,
                best_score=best_score,
                optimization_history=results,
                total_evaluations=len(results),
                optimization_time=0.0,
                convergence_achieved=False
            )
            
            logger.info(f"Random search optimization completed with {len(results)} evaluations")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            raise
    
    def _generate_random_parameters(
        self,
        parameter_space: List[OptimizationParameter]
    ) -> Dict[str, Any]:
        """Generate random parameter values"""
        params = {}
        
        for param in parameter_space:
            if param.param_type == "choice":
                params[param.name] = random.choice(param.choices)
            
            elif param.param_type == "float":
                params[param.name] = random.uniform(param.min_value, param.max_value)
            
            elif param.param_type == "int":
                params[param.name] = random.randint(int(param.min_value), int(param.max_value))
        
        return params


class GeneticAlgorithmOptimizer(OptimizationAlgorithm):
    """Genetic algorithm optimization"""
    
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.generations = 100
    
    async def optimize(
        self,
        objective_function: Callable,
        parameter_space: List[OptimizationParameter],
        max_evaluations: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        optimization_id = f"ga_{datetime.utcnow().timestamp()}"
        logger.info(f"Starting genetic algorithm optimization {optimization_id}")
        
        try:
            # Calculate generations needed
            max_generations = max_evaluations // self.population_size
            generations = min(max_generations, self.generations)
            
            # Initialize population
            population = self._initialize_population(parameter_space, self.population_size)
            
            best_fitness = float('-inf')
            best_individual = None
            optimization_history = []
            
            for generation in range(generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    try:
                        score = await objective_function(individual)
                        fitness_scores.append(score)
                        
                        if score > best_fitness:
                            best_fitness = score
                            best_individual = individual.copy()
                    
                    except Exception as e:
                        logger.error(f"Error evaluating individual {individual}: {e}")
                        fitness_scores.append(float('-inf'))
                
                # Store generation results
                generation_result = {
                    'generation': generation + 1,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'population_diversity': np.std(fitness_scores)
                }
                optimization_history.append(generation_result)
                
                # Selection and reproduction
                population = self._evolve_population(
                    population, fitness_scores, parameter_space
                )
                
                logger.info(f"GA Generation {generation + 1}/{generations}, Best Fitness: {best_fitness:.4f}")
            
            # Create optimization result
            optimization_result = OptimizationResult(
                optimization_id=optimization_id,
                method=OptimizationMethod.GENETIC_ALGORITHM,
                objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
                best_parameters=best_individual or {},
                best_score=best_fitness,
                optimization_history=optimization_history,
                total_evaluations=generations * self.population_size,
                optimization_time=0.0,
                convergence_achieved=True
            )
            
            logger.info(f"Genetic algorithm optimization completed")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Genetic algorithm optimization failed: {e}")
            raise
    
    def _initialize_population(
        self,
        parameter_space: List[OptimizationParameter],
        population_size: int
    ) -> List[Dict[str, Any]]:
        """Initialize random population"""
        population = []
        for _ in range(population_size):
            individual = {}
            for param in parameter_space:
                if param.param_type == "choice":
                    individual[param.name] = random.choice(param.choices)
                elif param.param_type == "float":
                    individual[param.name] = random.uniform(param.min_value, param.max_value)
                elif param.param_type == "int":
                    individual[param.name] = random.randint(int(param.min_value), int(param.max_value))
            population.append(individual)
        return population
    
    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        parameter_space: List[OptimizationParameter]
    ) -> List[Dict[str, Any]]:
        """Evolve population using genetic operations"""
        try:
            # Sort by fitness
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            
            # Keep top performers (elitism)
            elite_count = max(1, len(population) // 10)
            new_population = [population[i].copy() for i in sorted_indices[:elite_count]]
            
            # Tournament selection and reproduction
            while len(new_population) < len(population):
                # Select parents
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2, parameter_space)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self._mutate(offspring1, parameter_space)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutate(offspring2, parameter_space)
                
                new_population.extend([offspring1, offspring2])
            
            return new_population[:len(population)]
            
        except Exception as e:
            logger.error(f"Error evolving population: {e}")
            return population
    
    def _tournament_selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> Dict[str, Any]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        parameter_space: List[OptimizationParameter]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover"""
        offspring1 = {}
        offspring2 = {}
        
        for param in parameter_space:
            if random.random() < 0.5:
                offspring1[param.name] = parent1[param.name]
                offspring2[param.name] = parent2[param.name]
            else:
                offspring1[param.name] = parent2[param.name]
                offspring2[param.name] = parent1[param.name]
        
        return offspring1, offspring2
    
    def _mutate(
        self,
        individual: Dict[str, Any],
        parameter_space: List[OptimizationParameter]
    ) -> Dict[str, Any]:
        """Gaussian mutation"""
        mutated = individual.copy()
        
        for param in parameter_space:
            if param.param_type == "choice":
                if random.random() < 0.1:  # 10% chance to change choice
                    other_choices = [c for c in param.choices if c != individual[param.name]]
                    if other_choices:
                        mutated[param.name] = random.choice(other_choices)
            
            elif param.param_type == "float":
                if random.random() < 0.1:  # 10% chance to mutate
                    mutation_strength = (param.max_value - param.min_value) * 0.1
                    mutated[param.name] = np.clip(
                        individual[param.name] + random.gauss(0, mutation_strength),
                        param.min_value, param.max_value
                    )
            
            elif param.param_type == "int":
                if random.random() < 0.1:  # 10% chance to mutate
                    mutation_range = int((param.max_value - param.min_value) * 0.1)
                    if mutation_range > 0:
                        mutation = random.randint(-mutation_range, mutation_range)
                        mutated[param.name] = int(np.clip(
                            individual[param.name] + mutation,
                            param.min_value, param.max_value
                        ))
        
        return mutated


class ExitStrategyOptimizer:
    """
    @class ExitStrategyOptimizer
    @brief Exit strategy parameter optimization
    
    @details
    Provides comprehensive parameter optimization for exit strategies using
    various optimization algorithms and objective functions.
    """
    
    def __init__(self):
        self.backtest_engine = ExitStrategyBacktestEngine()
        self.algorithms = {
            OptimizationMethod.GRID_SEARCH: GridSearchOptimizer(),
            OptimizationMethod.RANDOM_SEARCH: RandomSearchOptimizer(),
            OptimizationMethod.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer()
        }
    
    async def optimize_exit_strategy_parameters(
        self,
        strategy_type: ExitType,
        symbol: str,
        parameter_ranges: List[OptimizationParameter],
        objective: OptimizationObjective,
        method: OptimizationMethod,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal('100000'),
        max_evaluations: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize exit strategy parameters
        
        @param strategy_type Type of exit strategy
        @param symbol Trading symbol
        @parameter_ranges Parameter ranges to optimize
        @param objective Optimization objective
        @param method Optimization algorithm
        @param start_date Backtest start date
        @param end_date Backtest end date
        @param initial_capital Starting capital
        @param max_evaluations Maximum function evaluations
        @param kwargs Additional parameters
        
        @returns Optimization results
        """
        logger.info(f"Starting parameter optimization for {strategy_type.value} strategy")
        
        try:
            # Create objective function
            objective_function = await self._create_objective_function(
                strategy_type, symbol, parameter_ranges, objective,
                start_date, end_date, initial_capital, **kwargs
            )
            
            # Run optimization
            algorithm = self.algorithms.get(method)
            if not algorithm:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            result = await algorithm.optimize(
                objective_function=objective_function,
                parameter_space=parameter_ranges,
                max_evaluations=max_evaluations,
                **kwargs
            )
            
            # Set the actual objective
            result.objective = objective
            
            logger.info(f"Parameter optimization completed with best score: {result.best_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise
    
    async def _create_objective_function(
        self,
        strategy_type: ExitType,
        symbol: str,
        parameter_ranges: List[OptimizationParameter],
        objective: OptimizationObjective,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal,
        **kwargs
    ) -> Callable:
        """Create objective function for optimization"""
        
        async def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Create strategy with current parameters
                strategy = await self._create_strategy_with_parameters(
                    strategy_type, symbol, parameters
                )
                
                # Run backtest
                backtest_result = await self.backtest_engine.run_backtest(
                    strategies=[strategy],
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                
                if not backtest_result.strategies:
                    return float('-inf')
                
                metrics = list(backtest_result.strategies.values())[0]
                
                # Calculate objective score
                score = self._calculate_objective_score(metrics, objective)
                
                logger.debug(f"Parameters {parameters} achieved score {score:.4f}")
                return score
                
            except Exception as e:
                logger.error(f"Error in objective function for parameters {parameters}: {e}")
                return float('-inf')
        
        return objective_function
    
    async def _create_strategy_with_parameters(
        self,
        strategy_type: ExitType,
        symbol: str,
        parameters: Dict[str, Any]
    ) -> BaseExitStrategy:
        """Create strategy with specific parameters"""
        # This would import and create the appropriate strategy
        # For now, return a placeholder
        from .trailing_stop import create_trailing_stop_strategy
        from .fixed_target import create_fixed_target_strategy
        from .stop_loss import create_stop_loss_strategy
        from .volatility_stop import create_volatility_stop_strategy
        from .ai_exit_strategy import create_ai_exit_strategy
        from .time_based_exit import create_time_based_exit_strategy
        from .conditional_exit import create_conditional_exit_strategy
        
        if strategy_type == ExitType.TRAILING_STOP:
            return create_trailing_stop_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.FIXED_TARGET:
            return create_fixed_target_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.STOP_LOSS:
            return create_stop_loss_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.VOLATILITY_STOP:
            return create_volatility_stop_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.AI_DRIVEN:
            return create_ai_exit_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.TIME_BASED:
            return create_time_based_exit_strategy(
                strategy_id="optimized",
                symbol=symbol,
                **parameters
            )
        elif strategy_type == ExitType.CONDITIONAL:
            return create_conditional_exit_strategy(
                strategy_id="optimized",
                symbol=symbol,
                conditions=parameters.get('conditions', []),
                **parameters
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _calculate_objective_score(
        self,
        metrics: BacktestMetrics,
        objective: OptimizationObjective
    ) -> float:
        """Calculate objective score based on metrics"""
        try:
            if objective == OptimizationObjective.MAXIMIZE_SHARPE_RATIO:
                return metrics.sharpe_ratio
            
            elif objective == OptimizationObjective.MINIMIZE_MAX_DRAWDOWN:
                return -abs(metrics.max_drawdown)  # Negative because we want to minimize
            
            elif objective == OptimizationObjective.MAXIMIZE_TOTAL_RETURN:
                return metrics.total_return
            
            elif objective == OptimizationObjective.MAXIMIZE_CALMAR_RATIO:
                return metrics.calmar_ratio
            
            elif objective == OptimizationObjective.MINIMIZE_VOLATILITY:
                return -metrics.volatility  # Negative because we want to minimize
            
            elif objective == OptimizationObjective.MAXIMIZE_WIN_RATE:
                return metrics.win_rate
            
            elif objective == OptimizationObjective.MAXIMIZE_PROFIT_FACTOR:
                return metrics.profit_factor
            
            else:
                # Default to Sharpe ratio
                return metrics.sharpe_ratio
                
        except Exception as e:
            logger.error(f"Error calculating objective score: {e}")
            return float('-inf')


# Convenience functions

def create_parameter_ranges_for_trailing_stop() -> List[OptimizationParameter]:
    """Create parameter ranges for trailing stop optimization"""
    return [
        OptimizationParameter(
            name="trailing_distance",
            param_type="float",
            min_value=0.01,
            max_value=0.10,
            step=0.005,
            description="Trailing stop distance as percentage"
        ),
        OptimizationParameter(
            name="initial_stop",
            param_type="float",
            min_value=0.02,
            max_value=0.08,
            step=0.005,
            description="Initial stop loss percentage"
        ),
        OptimizationParameter(
            name="update_frequency",
            param_type="int",
            min_value=30,
            max_value=300,
            step=30,
            description="Update frequency in seconds"
        )
    ]


def create_parameter_ranges_for_fixed_target() -> List[OptimizationParameter]:
    """Create parameter ranges for fixed target optimization"""
    return [
        OptimizationParameter(
            name="profit_target",
            param_type="float",
            min_value=0.05,
            max_value=0.25,
            step=0.01,
            description="Profit target percentage"
        ),
        OptimizationParameter(
            name="loss_target",
            param_type="float",
            min_value=0.02,
            max_value=0.15,
            step=0.01,
            description="Loss target percentage"
        ),
        OptimizationParameter(
            name="partial_exits",
            param_type="choice",
            choices=[True, False],
            description="Whether to use partial exits"
        )
    ]


async def run_exit_strategy_optimization(
    strategy_type: ExitType,
    symbol: str,
    objective: OptimizationObjective,
    method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_evaluations: int = 500,
    **kwargs
) -> OptimizationResult:
    """
    Convenience function to run exit strategy optimization
    
    @param strategy_type Type of exit strategy
    @param symbol Trading symbol
    @param objective Optimization objective
    @param method Optimization algorithm
    @param start_date Backtest start date
    @param end_date Backtest end date
    @param max_evaluations Maximum evaluations
    @param kwargs Additional parameters
    
    @returns Optimization results
    """
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=365)
    if not end_date:
        end_date = datetime.utcnow()
    
    # Get parameter ranges based on strategy type
    if strategy_type == ExitType.TRAILING_STOP:
        parameter_ranges = create_parameter_ranges_for_trailing_stop()
    elif strategy_type == ExitType.FIXED_TARGET:
        parameter_ranges = create_parameter_ranges_for_fixed_target()
    else:
        raise ValueError(f"Parameter ranges not defined for strategy type: {strategy_type}")
    
    optimizer = ExitStrategyOptimizer()
    
    return await optimizer.optimize_exit_strategy_parameters(
        strategy_type=strategy_type,
        symbol=symbol,
        parameter_ranges=parameter_ranges,
        objective=objective,
        method=method,
        start_date=start_date,
        end_date=end_date,
        max_evaluations=max_evaluations,
        **kwargs
    )
