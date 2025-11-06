"""
Strategy Parameter Optimization System
Implements various optimization algorithms for strategy parameter tuning
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import itertools
import random
import numpy as np
from abc import ABC, abstractmethod

from loguru import logger

from .base import StrategyConfig, StrategyType
from .backtesting import BacktestEngine, BacktestResult


class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PARTICLE_SWARM = "particle_swarm"


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAXIMIZE_RETURNS = "maximize_returns"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    MAXIMIZE_SORTINO_RATIO = "maximize_sortino_ratio"
    MINIMIZE_VAR = "minimize_var"
    MAXIMIZE_CALMAR_RATIO = "maximize_calmar_ratio"
    CUSTOM = "custom"


@dataclass
class ParameterRange:
    """Parameter range definition"""
    name: str
    param_type: str  # 'int', 'float', 'choice'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass
class OptimizationResult:
    """Parameter optimization result"""
    optimization_id: str
    method: OptimizationMethod
    objective: OptimizationObjective
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    iterations: int
    convergence_achieved: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies"""
    
    def __init__(self, objective: OptimizationObjective):
        self.objective = objective
        self.best_score = float('-inf') if self._is_maximize() else float('inf')
        self.best_params = {}
        self.convergence_threshold = 1e-6
        self.max_iterations = 1000
    
    @abstractmethod
    async def optimize(
        self,
        parameter_ranges: List[ParameterRange],
        objective_function: Callable,
        **kwargs
    ) -> OptimizationResult:
        """Run optimization"""
        pass
    
    def _is_maximize(self) -> bool:
        """Check if objective should be maximized"""
        return self.objective in [
            OptimizationObjective.MAXIMIZE_RETURNS,
            OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            OptimizationObjective.MAXIMIZE_SORTINO_RATIO,
            OptimizationObjective.MAXIMIZE_CALMAR_RATIO
        ]
    
    def _evaluate_objective(self, score: float) -> float:
        """Convert score based on objective (minimize vs maximize)"""
        return score if self._is_maximize() else -score


class GridSearchOptimizer(OptimizationStrategy):
    """Grid Search Optimization"""
    
    def __init__(self, objective: OptimizationObjective):
        super().__init__(objective)
        self.max_iterations = 10000  # Limit for grid search
    
    async def optimize(
        self,
        parameter_ranges: List[ParameterRange],
        objective_function: Callable,
        **kwargs
    ) -> OptimizationResult:
        """Run grid search optimization"""
        start_time = datetime.utcnow()
        optimization_id = f"grid_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting grid search optimization: {optimization_id}")
        
        # Generate parameter grid
        param_combinations = self._generate_grid(parameter_ranges)
        
        if len(param_combinations) > self.max_iterations:
            logger.warning(f"Grid too large ({len(param_combinations)}), sampling randomly")
            param_combinations = random.sample(
                param_combinations, 
                min(self.max_iterations, len(param_combinations))
            )
        
        results = []
        iterations = 0
        
        for params in param_combinations:
            try:
                iterations += 1
                
                # Evaluate parameters
                score = await objective_function(params)
                
                # Store result
                result = {
                    'parameters': params.copy(),
                    'score': score,
                    'iteration': iterations
                }
                results.append(result)
                
                # Check if this is the best score
                evaluated_score = self._evaluate_objective(score)
                if evaluated_score > self.best_score:
                    self.best_score = evaluated_score
                    self.best_params = params.copy()
                
                # Progress logging
                if iterations % 100 == 0:
                    logger.info(f"Grid search progress: {iterations}/{len(param_combinations)}")
                
            except Exception as e:
                logger.error(f"Error evaluating parameters: {e}")
                continue
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create result
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=OptimizationMethod.GRID_SEARCH,
            objective=self.objective,
            best_parameters=self.best_params,
            best_score=score,
            all_results=results,
            optimization_time=optimization_time,
            iterations=iterations,
            convergence_achieved=iterations < self.max_iterations
        )
        
        logger.info(f"Grid search completed: {iterations} iterations, best score: {score:.4f}")
        
        return result
    
    def _generate_grid(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        grids = []
        
        for param_range in parameter_ranges:
            param_grid = []
            
            if param_range.param_type == 'choice':
                param_grid = [(param_range.name, choice) for choice in param_range.choices]
            elif param_range.param_type == 'int':
                values = list(range(
                    int(param_range.min_value),
                    int(param_range.max_value) + 1,
                    int(param_range.step or 1)
                ))
                param_grid = [(param_range.name, value) for value in values]
            elif param_range.param_type == 'float':
                values = np.arange(
                    param_range.min_value,
                    param_range.max_value + param_range.step,
                    param_range.step
                ).tolist()
                param_grid = [(param_range.name, value) for value in values]
            
            grids.append(param_grid)
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*grids):
            param_dict = dict(combo)
            combinations.append(param_dict)
        
        return combinations


class RandomSearchOptimizer(OptimizationStrategy):
    """Random Search Optimization"""
    
    def __init__(self, objective: OptimizationObjective):
        super().__init__(objective)
        self.max_iterations = 1000
    
    async def optimize(
        self,
        parameter_ranges: List[ParameterRange],
        objective_function: Callable,
        **kwargs
    ) -> OptimizationResult:
        """Run random search optimization"""
        start_time = datetime.utcnow()
        optimization_id = f"random_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        n_iterations = kwargs.get('n_iterations', self.max_iterations)
        
        logger.info(f"Starting random search optimization: {optimization_id} ({n_iterations} iterations)")
        
        results = []
        iterations = 0
        
        while iterations < n_iterations:
            try:
                iterations += 1
                
                # Generate random parameters
                params = self._generate_random_parameters(parameter_ranges)
                
                # Evaluate parameters
                score = await objective_function(params)
                
                # Store result
                result = {
                    'parameters': params.copy(),
                    'score': score,
                    'iteration': iterations
                }
                results.append(result)
                
                # Check if this is the best score
                evaluated_score = self._evaluate_objective(score)
                if evaluated_score > self.best_score:
                    self.best_score = evaluated_score
                    self.best_params = params.copy()
                
                # Progress logging
                if iterations % 100 == 0:
                    logger.info(f"Random search progress: {iterations}/{n_iterations}")
                
            except Exception as e:
                logger.error(f"Error evaluating parameters: {e}")
                continue
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create result
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=OptimizationMethod.RANDOM_SEARCH,
            objective=self.objective,
            best_parameters=self.best_params,
            best_score=score,
            all_results=results,
            optimization_time=optimization_time,
            iterations=iterations,
            convergence_achieved=False  # Random search doesn't guarantee convergence
        )
        
        logger.info(f"Random search completed: {iterations} iterations, best score: {score:.4f}")
        
        return result
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameters within ranges"""
        params = {}
        
        for param_range in parameter_ranges:
            if param_range.param_type == 'choice':
                params[param_range.name] = random.choice(param_range.choices)
            elif param_range.param_type == 'int':
                min_val = int(param_range.min_value)
                max_val = int(param_range.max_value)
                step = int(param_range.step or 1)
                value = random.randint(min_val, max_val)
                # Snap to step if specified
                if step > 1:
                    value = min_val + ((value - min_val) // step) * step
                params[param_range.name] = value
            elif param_range.param_type == 'float':
                min_val = param_range.min_value
                max_val = param_range.max_value
                step = param_range.step or (max_val - min_val) / 100
                value = random.uniform(min_val, max_val)
                # Snap to step if specified
                if step > 0:
                    steps_from_min = int((value - min_val) / step)
                    value = min_val + steps_from_min * step
                params[param_range.name] = value
        
        return params


class GeneticAlgorithmOptimizer(OptimizationStrategy):
    """Genetic Algorithm Optimization"""
    
    def __init__(self, objective: OptimizationObjective):
        super().__init__(objective)
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
    
    async def optimize(
        self,
        parameter_ranges: List[ParameterRange],
        objective_function: Callable,
        **kwargs
    ) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        start_time = datetime.utcnow()
        optimization_id = f"ga_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Override defaults with kwargs
        self.population_size = kwargs.get('population_size', self.population_size)
        self.generations = kwargs.get('generations', self.generations)
        self.mutation_rate = kwargs.get('mutation_rate', self.mutation_rate)
        self.crossover_rate = kwargs.get('crossover_rate', self.crossover_rate)
        self.elite_size = kwargs.get('elite_size', self.elite_size)
        
        logger.info(f"Starting GA optimization: {optimization_id}")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self._generate_random_parameters(parameter_ranges)
            population.append(individual)
        
        best_individual = None
        best_fitness = float('-inf') if self._is_maximize() else float('inf')
        
        results = []
        generation = 0
        
        for generation in range(self.generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                try:
                    score = await objective_function(individual)
                    fitness = self._evaluate_objective(score)
                    fitness_scores.append((individual, score, fitness))
                except Exception as e:
                    logger.error(f"Error evaluating individual: {e}")
                    fitness_scores.append((individual, float('-inf'), float('-inf')))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Track best
            current_best_individual, current_best_score, current_best_fitness = fitness_scores[0]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
                self.best_params = current_best_individual.copy()
                self.best_score = best_fitness
            
            # Store generation result
            generation_result = {
                'generation': generation + 1,
                'best_score': current_best_score,
                'avg_score': np.mean([fs[1] for fs in fitness_scores]),
                'best_fitness': current_best_fitness
            }
            results.append(generation_result)
            
            # Create next generation
            next_population = []
            
            # Elitism: keep top individuals
            for i in range(self.elite_size):
                next_population.append(fitness_scores[i][0].copy())
            
            # Generate rest through crossover and mutation
            while len(next_population) < self.population_size:
                # Selection (tournament selection)
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, parameter_ranges)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, parameter_ranges)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, parameter_ranges)
                
                next_population.extend([child1, child2])
            
            # Trim population to exact size
            population = next_population[:self.population_size]
            
            logger.info(f"GA generation {generation + 1}/{self.generations}, best fitness: {best_fitness:.4f}")
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get final score for best individual
        final_score = await objective_function(best_individual) if best_individual else 0
        
        # Create result
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=OptimizationMethod.GENETIC_ALGORITHM,
            objective=self.objective,
            best_parameters=self.best_params,
            best_score=final_score,
            all_results=results,
            optimization_time=optimization_time,
            iterations=self.generations,
            convergence_achieved=True
        )
        
        logger.info(f"GA optimization completed: {self.generations} generations, best score: {final_score:.4f}")
        
        return result
    
    def _tournament_selection(self, fitness_scores: List[Tuple], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        tournament.sort(key=lambda x: x[2], reverse=True)
        return tournament[0][0].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], parameter_ranges: List[ParameterRange]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover for categorical parameters, blend for continuous"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for param_range in parameter_ranges:
            param_name = param_range.name
            
            if param_range.param_type == 'choice':
                # Random choice for categorical
                if random.random() < 0.5:
                    child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
            elif param_range.param_type in ['int', 'float']:
                # Blend crossover for continuous
                if random.random() < 0.5:
                    alpha = random.random()
                    val1 = child1[param_name]
                    val2 = child2[param_name]
                    
                    if param_range.param_type == 'int':
                        child1[param_name] = int(alpha * val1 + (1 - alpha) * val2)
                        child2[param_name] = int(alpha * val2 + (1 - alpha) * val1)
                    else:
                        child1[param_name] = alpha * val1 + (1 - alpha) * val2
                        child2[param_name] = alpha * val2 + (1 - alpha) * val1
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Mutate individual parameters"""
        mutated = individual.copy()
        
        for param_range in parameter_ranges:
            param_name = param_range.name
            
            if param_range.param_type == 'choice':
                # Random choice mutation
                mutated[param_name] = random.choice(param_range.choices)
            elif param_range.param_type == 'int':
                # Gaussian mutation
                current_val = mutated[param_name]
                step = param_range.step or 1
                noise = int(random.gauss(0, step * 2))
                new_val = max(param_range.min_value, min(param_range.max_value, current_val + noise))
                mutated[param_name] = int(new_val)
            elif param_range.param_type == 'float':
                # Gaussian mutation
                current_val = mutated[param_name]
                step = param_range.step or (param_range.max_value - param_range.min_value) / 100
                noise = random.gauss(0, step * 2)
                new_val = max(param_range.min_value, min(param_range.max_value, current_val + noise))
                mutated[param_name] = new_val
        
        return mutated
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameters (reuse from RandomSearchOptimizer)"""
        params = {}
        
        for param_range in parameter_ranges:
            if param_range.param_type == 'choice':
                params[param_range.name] = random.choice(param_range.choices)
            elif param_range.param_type == 'int':
                min_val = int(param_range.min_value)
                max_val = int(param_range.max_value)
                step = int(param_range.step or 1)
                value = random.randint(min_val, max_val)
                if step > 1:
                    value = min_val + ((value - min_val) // step) * step
                params[param_range.name] = value
            elif param_range.param_type == 'float':
                min_val = param_range.min_value
                max_val = param_range.max_value
                step = param_range.step or (max_val - min_val) / 100
                value = random.uniform(min_val, max_val)
                if step > 0:
                    steps_from_min = int((value - min_val) / step)
                    value = min_val + steps_from_min * step
                params[param_range.name] = value
        
        return params


class ParameterOptimizer:
    """Main parameter optimization orchestrator"""
    
    def __init__(self):
        self.optimizers = {
            OptimizationMethod.GRID_SEARCH: GridSearchOptimizer,
            OptimizationMethod.RANDOM_SEARCH: RandomSearchOptimizer,
            OptimizationMethod.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer
        }
    
    def add_optimizer(self, method: OptimizationMethod, optimizer_class: type):
        """Add custom optimizer"""
        self.optimizers[method] = optimizer_class
    
    async def optimize_parameters(
        self,
        method: OptimizationMethod,
        objective: OptimizationObjective,
        parameter_ranges: List[ParameterRange],
        objective_function: Callable,
        **kwargs
    ) -> OptimizationResult:
        """Run parameter optimization using specified method"""
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimizer_class = self.optimizers[method]
        optimizer = optimizer_class(objective)
        
        # Override optimizer settings if provided
        for key, value in kwargs.items():
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
        
        # Run optimization
        result = await optimizer.optimize(parameter_ranges, objective_function, **kwargs)
        
        return result
    
    def create_strategy_objective_function(
        self,
        strategy_class,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective
    ) -> Callable:
        """Create objective function for strategy optimization"""
        
        async def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Create strategy with parameters
                config = StrategyConfig(
                    strategy_id="opt_test",
                    strategy_type=StrategyType.TREND_FOLLOWING,  # Default
                    name="Optimization Test Strategy",
                    description="Strategy for parameter optimization",
                    parameters=parameters,
                    risk_level=kwargs.get('risk_level', 'medium'),  # Default
                    symbols=[symbol],
                    max_position_size=Decimal('10000'),
                    max_daily_loss=Decimal('1000')
                )
                
                strategy = strategy_class(config)
                
                # Run backtest
                backtest_engine = BacktestEngine()
                backtest_result = await backtest_engine.run_backtest(
                    strategy=strategy,
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=Decimal('100000')
                )
                
                # Extract objective value
                return self._extract_objective_value(backtest_result, objective)
                
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return float('-inf') if self._is_maximize_objective(objective) else float('inf')
        
        return objective_function
    
    def _extract_objective_value(self, backtest_result: BacktestResult, objective: OptimizationObjective) -> float:
        """Extract objective value from backtest result"""
        try:
            if objective == OptimizationObjective.MAXIMIZE_RETURNS:
                return backtest_result.total_return
            elif objective == OptimizationObjective.MINIMIZE_DRAWDOWN:
                return -backtest_result.max_drawdown  # Negative for minimization
            elif objective == OptimizationObjective.MAXIMIZE_SHARPE_RATIO:
                return backtest_result.sharpe_ratio
            elif objective == OptimizationObjective.MAXIMIZE_SORTINO_RATIO:
                return getattr(backtest_result, 'sortino_ratio', 0.0)
            elif objective == OptimizationObjective.MINIMIZE_VAR:
                return getattr(backtest_result, 'value_at_risk', 0.0)
            elif objective == OptimizationObjective.MAXIMIZE_CALMAR_RATIO:
                return getattr(backtest_result, 'calmar_ratio', 0.0)
            else:
                # Default to total return
                return backtest_result.total_return
                
        except Exception as e:
            logger.error(f"Error extracting objective value: {e}")
            return 0.0
    
    def _is_maximize_objective(self, objective: OptimizationObjective) -> bool:
        """Check if objective should be maximized"""
        return objective in [
            OptimizationObjective.MAXIMIZE_RETURNS,
            OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            OptimizationObjective.MAXIMIZE_SORTINO_RATIO,
            OptimizationObjective.MAXIMIZE_CALMAR_RATIO
        ]


# Utility functions for common parameter ranges
def create_parameter_ranges(strategy_type: StrategyType) -> List[ParameterRange]:
    """Create standard parameter ranges for a strategy type"""
    if strategy_type == StrategyType.TREND_FOLLOWING:
        return [
            ParameterRange("fast_period", "int", 5, 50, 1),
            ParameterRange("slow_period", "int", 20, 200, 5),
            ParameterRange("signal_threshold", "float", 0.1, 0.9, 0.1),
            ParameterRange("stop_loss", "float", 0.01, 0.05, 0.005)
        ]
    elif strategy_type == StrategyType.MEAN_REVERSION:
        return [
            ParameterRange("rsi_period", "int", 10, 30, 2),
            ParameterRange("rsi_oversold", "int", 20, 40, 5),
            ParameterRange("rsi_overbought", "int", 60, 80, 5),
            ParameterRange("bb_std", "float", 1.5, 3.0, 0.1)
        ]
    else:
        # Generic ranges
        return [
            ParameterRange("param1", "float", 0.1, 2.0, 0.1),
            ParameterRange("param2", "int", 1, 100, 5)
        ]


# Example usage and testing
if __name__ == "__main__":
    async def test_parameter_optimization():
        # Create optimizer
        optimizer = ParameterOptimizer()
        
        # Define parameter ranges for trend following
        parameter_ranges = [
            ParameterRange("fast_period", "int", 5, 20, 1),
            ParameterRange("slow_period", "int", 20, 50, 5),
            ParameterRange("signal_threshold", "float", 0.3, 0.8, 0.1)
        ]
        
        # Simple objective function for testing
        async def dummy_objective(params):
            # Simulate strategy performance
            fast = params["fast_period"]
            slow = params["slow_period"]
            threshold = params["signal_threshold"]
            
            # Dummy calculation
            score = (slow - fast) * threshold * random.uniform(0.5, 1.5)
            return score
        
        # Run optimization
        result = await optimizer.optimize_parameters(
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.MAXIMIZE_RETURNS,
            parameter_ranges=parameter_ranges,
            objective_function=dummy_objective
        )
        
        print("Optimization Result:")
        print(f"  Method: {result.method.value}")
        print(f"  Best Parameters: {result.best_parameters}")
        print(f"  Best Score: {result.best_score:.4f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {result.optimization_time:.2f}s")
    
    import asyncio
    asyncio.run(test_parameter_optimization())