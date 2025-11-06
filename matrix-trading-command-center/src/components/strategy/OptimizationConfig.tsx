import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Settings, 
  Zap, 
  Target, 
  TrendingUp,
  BarChart3,
  Play,
  Pause,
  Save,
  RotateCcw,
  Download,
  Upload,
  Brain,
  GitBranch,
  Search,
  Timer,
  Cpu,
  MemoryStick,
  Clock,
  CheckCircle,
  AlertTriangle,
  Info,
  Plus,
  Trash2,
  Copy,
  Sliders
} from 'lucide-react';

interface OptimizationConfig {
  genetic: {
    populationSize: number;
    generations: number;
    mutationRate: number;
    crossoverRate: number;
    elitismRate: number;
    tournamentSize: number;
    maxFitness: number;
    convergenceThreshold: number;
    parallelFitness: boolean;
  };
  gridSearch: {
    enabled: boolean;
    parameters: Record<string, any[]>;
    metric: string;
    direction: 'maximize' | 'minimize';
    parallel: boolean;
    maxWorkers: number;
  };
  bayesian: {
    enabled: boolean;
    acquisitionFunction: 'ucb' | 'ei' | 'pi' | 'lcb';
    nInitialPoints: number;
    nWarmupSteps: number;
    explorationExploitation: number;
    randomState: number;
  };
  constraints: {
    maxDrawdown: number;
    maxExposure: number;
    maxPositions: number;
    minSharpeRatio: number;
    maxVolatility: number;
    allowedAssets: string[];
    forbiddenCombinations: string[];
  };
  objectiveFunction: {
    primary: 'sharpe_ratio' | 'total_return' | 'calmar_ratio' | 'sortino_ratio' | 'information_ratio';
    secondary?: string;
    weights: Record<string, number>;
    penalties: {
      turnover: number;
      transactionCosts: number;
      concentration: number;
    };
  };
  runConfig: {
    timeLimit: number;
    memoryLimit: number;
    checkpointFrequency: number;
    resumeFrom: string | null;
    saveIntermediate: boolean;
    parallelRuns: number;
  };
}

const OptimizationConfig: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'genetic' | 'grid' | 'bayesian' | 'constraints' | 'objectives' | 'runtime'>('genetic');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [config, setConfig] = useState<OptimizationConfig>({
    genetic: {
      populationSize: 100,
      generations: 200,
      mutationRate: 0.1,
      crossoverRate: 0.8,
      elitismRate: 0.05,
      tournamentSize: 3,
      maxFitness: 10.0,
      convergenceThreshold: 0.001,
      parallelFitness: true
    },
    gridSearch: {
      enabled: false,
      parameters: {
        'ma_period': [10, 20, 30, 50],
        'rsi_period': [14, 21, 30],
        'stop_loss': [0.02, 0.03, 0.05],
        'take_profit': [0.04, 0.06, 0.08]
      },
      metric: 'sharpe_ratio',
      direction: 'maximize',
      parallel: true,
      maxWorkers: 4
    },
    bayesian: {
      enabled: false,
      acquisitionFunction: 'ucb',
      nInitialPoints: 20,
      nWarmupSteps: 100,
      explorationExploitation: 0.2,
      randomState: 42
    },
    constraints: {
      maxDrawdown: 0.15,
      maxExposure: 0.8,
      maxPositions: 10,
      minSharpeRatio: 0.5,
      maxVolatility: 0.25,
      allowedAssets: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
      forbiddenCombinations: []
    },
    objectiveFunction: {
      primary: 'sharpe_ratio',
      weights: {
        'sharpe_ratio': 0.6,
        'total_return': 0.3,
        'max_drawdown': 0.1
      },
      penalties: {
        turnover: 0.01,
        transactionCosts: 0.005,
        concentration: 0.02
      }
    },
    runConfig: {
      timeLimit: 3600,
      memoryLimit: 4096,
      checkpointFrequency: 10,
      resumeFrom: null,
      saveIntermediate: true,
      parallelRuns: 1
    }
  });

  const tabs = [
    { id: 'genetic', label: 'Genetic Algorithm', icon: GitBranch },
    { id: 'grid', label: 'Grid Search', icon: Search },
    { id: 'bayesian', label: 'Bayesian', icon: Brain },
    { id: 'constraints', label: 'Constraints', icon: Shield },
    { id: 'objectives', label: 'Objectives', icon: Target },
    { id: 'runtime', label: 'Runtime', icon: Cpu }
  ];

  const updateConfig = (section: keyof OptimizationConfig, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const handleStartOptimization = async () => {
    setIsRunning(true);
    setProgress(0);
    
    // Simulate optimization progress
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          return 100;
        }
        return newProgress;
      });
    }, 500);
  };

  const handleStopOptimization = () => {
    setIsRunning(false);
    setProgress(0);
  };

  const resetConfig = () => {
    setConfig({
      genetic: {
        populationSize: 100,
        generations: 200,
        mutationRate: 0.1,
        crossoverRate: 0.8,
        elitismRate: 0.05,
        tournamentSize: 3,
        maxFitness: 10.0,
        convergenceThreshold: 0.001,
        parallelFitness: true
      },
      gridSearch: {
        enabled: false,
        parameters: {
          'ma_period': [10, 20, 30, 50],
          'rsi_period': [14, 21, 30],
          'stop_loss': [0.02, 0.03, 0.05],
          'take_profit': [0.04, 0.06, 0.08]
        },
        metric: 'sharpe_ratio',
        direction: 'maximize',
        parallel: true,
        maxWorkers: 4
      },
      bayesian: {
        enabled: false,
        acquisitionFunction: 'ucb',
        nInitialPoints: 20,
        nWarmupSteps: 100,
        explorationExploitation: 0.2,
        randomState: 42
      },
      constraints: {
        maxDrawdown: 0.15,
        maxExposure: 0.8,
        maxPositions: 10,
        minSharpeRatio: 0.5,
        maxVolatility: 0.25,
        allowedAssets: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        forbiddenCombinations: []
      },
      objectiveFunction: {
        primary: 'sharpe_ratio',
        weights: {
          'sharpe_ratio': 0.6,
          'total_return': 0.3,
          'max_drawdown': 0.1
        },
        penalties: {
          turnover: 0.01,
          transactionCosts: 0.005,
          concentration: 0.02
        }
      },
      runConfig: {
        timeLimit: 3600,
        memoryLimit: 4096,
        checkpointFrequency: 10,
        resumeFrom: null,
        saveIntermediate: true,
        parallelRuns: 1
      }
    });
  };

  const renderGeneticConfig = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Population Size</label>
          <MatrixInput
            type="number"
            value={config.genetic.populationSize}
            onChange={(e) => updateConfig('genetic', 'populationSize', parseInt(e.target.value))}
            className="w-full"
            placeholder="100"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Generations</label>
          <MatrixInput
            type="number"
            value={config.genetic.generations}
            onChange={(e) => updateConfig('genetic', 'generations', parseInt(e.target.value))}
            className="w-full"
            placeholder="200"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Mutation Rate</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.genetic.mutationRate}
            onChange={(e) => updateConfig('genetic', 'mutationRate', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Crossover Rate</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.genetic.crossoverRate}
            onChange={(e) => updateConfig('genetic', 'crossoverRate', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.8"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Elitism Rate</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="0.5"
            value={config.genetic.elitismRate}
            onChange={(e) => updateConfig('genetic', 'elitismRate', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.05"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Tournament Size</label>
          <MatrixInput
            type="number"
            min="2"
            max="10"
            value={config.genetic.tournamentSize}
            onChange={(e) => updateConfig('genetic', 'tournamentSize', parseInt(e.target.value))}
            className="w-full"
            placeholder="3"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Max Fitness</label>
          <MatrixInput
            type="number"
            step="0.1"
            value={config.genetic.maxFitness}
            onChange={(e) => updateConfig('genetic', 'maxFitness', parseFloat(e.target.value))}
            className="w-full"
            placeholder="10.0"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Convergence Threshold</label>
          <MatrixInput
            type="number"
            step="0.0001"
            min="0"
            value={config.genetic.convergenceThreshold}
            onChange={(e) => updateConfig('genetic', 'convergenceThreshold', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.001"
          />
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <input
          type="checkbox"
          id="parallelFitness"
          checked={config.genetic.parallelFitness}
          onChange={(e) => updateConfig('genetic', 'parallelFitness', e.target.checked)}
          className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400 focus:ring-2"
        />
        <label htmlFor="parallelFitness" className="text-sm text-gray-300">
          Enable Parallel Fitness Evaluation
        </label>
      </div>
    </div>
  );

  const renderGridSearchConfig = () => (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <input
          type="checkbox"
          id="gridEnabled"
          checked={config.gridSearch.enabled}
          onChange={(e) => updateConfig('gridSearch', 'enabled', e.target.checked)}
          className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400 focus:ring-2"
        />
        <label htmlFor="gridEnabled" className="text-sm font-medium text-cyan-400">
          Enable Grid Search
        </label>
      </div>

      {config.gridSearch.enabled && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Optimization Metric</label>
              <select
                value={config.gridSearch.metric}
                onChange={(e) => updateConfig('gridSearch', 'metric', e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                <option value="sharpe_ratio">Sharpe Ratio</option>
                <option value="total_return">Total Return</option>
                <option value="calmar_ratio">Calmar Ratio</option>
                <option value="sortino_ratio">Sortino Ratio</option>
                <option value="information_ratio">Information Ratio</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Direction</label>
              <select
                value={config.gridSearch.direction}
                onChange={(e) => updateConfig('gridSearch', 'direction', e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                <option value="maximize">Maximize</option>
                <option value="minimize">Minimize</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="gridParallel"
                checked={config.gridSearch.parallel}
                onChange={(e) => updateConfig('gridSearch', 'parallel', e.target.checked)}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400 focus:ring-2"
              />
              <label htmlFor="gridParallel" className="text-sm text-gray-300">
                Parallel Processing
              </label>
            </div>
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Max Workers</label>
              <MatrixInput
                type="number"
                min="1"
                max="16"
                value={config.gridSearch.maxWorkers}
                onChange={(e) => updateConfig('gridSearch', 'maxWorkers', parseInt(e.target.value))}
                className="w-full"
                placeholder="4"
              />
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-cyan-400 mb-3">Parameter Grid</h4>
            <div className="space-y-3">
              {Object.entries(config.gridSearch.parameters).map(([param, values]) => (
                <div key={param} className="flex items-center space-x-3">
                  <span className="text-sm text-gray-300 w-32">{param}:</span>
                  <MatrixInput
                    type="text"
                    value={values.join(', ')}
                    onChange={(e) => {
                      const newValues = e.target.value.split(',').map(v => v.trim()).filter(v => v !== '');
                      updateConfig('gridSearch', 'parameters', {
                        ...config.gridSearch.parameters,
                        [param]: newValues
                      });
                    }}
                    className="flex-1"
                    placeholder="comma-separated values"
                  />
                  <MatrixButton
                    variant="destructive"
                    size="sm"
                    onClick={() => {
                      const newParams = { ...config.gridSearch.parameters };
                      delete newParams[param];
                      updateConfig('gridSearch', 'parameters', newParams);
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                  </MatrixButton>
                </div>
              ))}
              <MatrixButton
                variant="secondary"
                size="sm"
                onClick={() => {
                  const paramName = prompt('Parameter name:');
                  if (paramName) {
                    updateConfig('gridSearch', 'parameters', {
                      ...config.gridSearch.parameters,
                      [paramName]: ['1', '2', '3']
                    });
                  }
                }}
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Parameter
              </MatrixButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderBayesianConfig = () => (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <input
          type="checkbox"
          id="bayesianEnabled"
          checked={config.bayesian.enabled}
          onChange={(e) => updateConfig('bayesian', 'enabled', e.target.checked)}
          className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400 focus:ring-2"
        />
        <label htmlFor="bayesianEnabled" className="text-sm font-medium text-cyan-400">
          Enable Bayesian Optimization
        </label>
      </div>

      {config.bayesian.enabled && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Acquisition Function</label>
              <select
                value={config.bayesian.acquisitionFunction}
                onChange={(e) => updateConfig('bayesian', 'acquisitionFunction', e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                <option value="ucb">Upper Confidence Bound (UCB)</option>
                <option value="ei">Expected Improvement (EI)</option>
                <option value="pi">Probability of Improvement (PI)</option>
                <option value="lcb">Lower Confidence Bound (LCB)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Random State</label>
              <MatrixInput
                type="number"
                value={config.bayesian.randomState}
                onChange={(e) => updateConfig('bayesian', 'randomState', parseInt(e.target.value))}
                className="w-full"
                placeholder="42"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Initial Points</label>
              <MatrixInput
                type="number"
                min="1"
                value={config.bayesian.nInitialPoints}
                onChange={(e) => updateConfig('bayesian', 'nInitialPoints', parseInt(e.target.value))}
                className="w-full"
                placeholder="20"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-cyan-400 mb-2">Warmup Steps</label>
              <MatrixInput
                type="number"
                min="0"
                value={config.bayesian.nWarmupSteps}
                onChange={(e) => updateConfig('bayesian', 'nWarmupSteps', parseInt(e.target.value))}
                className="w-full"
                placeholder="100"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-cyan-400 mb-2">
              Exploration-Exploitation Balance
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.bayesian.explorationExploitation}
              onChange={(e) => updateConfig('bayesian', 'explorationExploitation', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Exploitation</span>
              <span className="text-cyan-400">{config.bayesian.explorationExploitation}</span>
              <span>Exploration</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderConstraintsConfig = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Max Drawdown</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.constraints.maxDrawdown}
            onChange={(e) => updateConfig('constraints', 'maxDrawdown', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.15"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Max Exposure</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.constraints.maxExposure}
            onChange={(e) => updateConfig('constraints', 'maxExposure', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.8"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Max Positions</label>
          <MatrixInput
            type="number"
            min="1"
            value={config.constraints.maxPositions}
            onChange={(e) => updateConfig('constraints', 'maxPositions', parseInt(e.target.value))}
            className="w-full"
            placeholder="10"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Min Sharpe Ratio</label>
          <MatrixInput
            type="number"
            step="0.1"
            min="0"
            value={config.constraints.minSharpeRatio}
            onChange={(e) => updateConfig('constraints', 'minSharpeRatio', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.5"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Max Volatility</label>
          <MatrixInput
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.constraints.maxVolatility}
            onChange={(e) => updateConfig('constraints', 'maxVolatility', parseFloat(e.target.value))}
            className="w-full"
            placeholder="0.25"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-cyan-400 mb-2">Allowed Assets</label>
        <MatrixInput
          type="text"
          value={config.constraints.allowedAssets.join(', ')}
          onChange={(e) => {
            const assets = e.target.value.split(',').map(a => a.trim()).filter(a => a !== '');
            updateConfig('constraints', 'allowedAssets', assets);
          }}
          className="w-full"
          placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-cyan-400 mb-2">Forbidden Combinations</label>
        <div className="space-y-2">
          {config.constraints.forbiddenCombinations.map((combo, index) => (
            <div key={index} className="flex items-center space-x-2">
              <span className="text-sm text-gray-300">{combo}</span>
              <MatrixButton
                variant="destructive"
                size="sm"
                onClick={() => {
                  const newCombos = config.constraints.forbiddenCombinations.filter((_, i) => i !== index);
                  updateConfig('constraints', 'forbiddenCombinations', newCombos);
                }}
              >
                <Trash2 className="w-4 h-4" />
              </MatrixButton>
            </div>
          ))}
          <MatrixButton
            variant="secondary"
            size="sm"
            onClick={() => {
              const combo = prompt('Enter forbidden combination:');
              if (combo) {
                updateConfig('constraints', 'forbiddenCombinations', [
                  ...config.constraints.forbiddenCombinations,
                  combo
                ]);
              }
            }}
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Combination
          </MatrixButton>
        </div>
      </div>
    </div>
  );

  const renderObjectivesConfig = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-cyan-400 mb-2">Primary Objective</label>
        <select
          value={config.objectiveFunction.primary}
          onChange={(e) => updateConfig('objectiveFunction', 'primary', e.target.value)}
          className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
        >
          <option value="sharpe_ratio">Sharpe Ratio</option>
          <option value="total_return">Total Return</option>
          <option value="calmar_ratio">Calmar Ratio</option>
          <option value="sortino_ratio">Sortino Ratio</option>
          <option value="information_ratio">Information Ratio</option>
        </select>
      </div>

      <div>
        <h4 className="text-sm font-medium text-cyan-400 mb-3">Objective Weights</h4>
        <div className="space-y-3">
          {Object.entries(config.objectiveFunction.weights).map(([objective, weight]) => (
            <div key={objective} className="flex items-center space-x-3">
              <span className="text-sm text-gray-300 w-32 capitalize">
                {objective.replace('_', ' ')}
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={weight}
                onChange={(e) => {
                  updateConfig('objectiveFunction', 'weights', {
                    ...config.objectiveFunction.weights,
                    [objective]: parseFloat(e.target.value)
                  });
                }}
                className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm text-cyan-400 w-12">{weight}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-cyan-400 mb-3">Penalty Factors</h4>
        <div className="space-y-3">
          {Object.entries(config.objectiveFunction.penalties).map(([penalty, value]) => (
            <div key={penalty} className="flex items-center space-x-3">
              <span className="text-sm text-gray-300 w-32 capitalize">
                {penalty.replace(/([A-Z])/g, ' $1').trim()}
              </span>
              <MatrixInput
                type="number"
                step="0.001"
                min="0"
                value={value}
                onChange={(e) => {
                  updateConfig('objectiveFunction', 'penalties', {
                    ...config.objectiveFunction.penalties,
                    [penalty]: parseFloat(e.target.value)
                  });
                }}
                className="flex-1"
                placeholder="0.01"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderRuntimeConfig = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Time Limit (seconds)</label>
          <MatrixInput
            type="number"
            min="60"
            value={config.runConfig.timeLimit}
            onChange={(e) => updateConfig('runConfig', 'timeLimit', parseInt(e.target.value))}
            className="w-full"
            placeholder="3600"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Memory Limit (MB)</label>
          <MatrixInput
            type="number"
            min="512"
            value={config.runConfig.memoryLimit}
            onChange={(e) => updateConfig('runConfig', 'memoryLimit', parseInt(e.target.value))}
            className="w-full"
            placeholder="4096"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Checkpoint Frequency</label>
          <MatrixInput
            type="number"
            min="1"
            value={config.runConfig.checkpointFrequency}
            onChange={(e) => updateConfig('runConfig', 'checkpointFrequency', parseInt(e.target.value))}
            className="w-full"
            placeholder="10"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-cyan-400 mb-2">Parallel Runs</label>
          <MatrixInput
            type="number"
            min="1"
            max="8"
            value={config.runConfig.parallelRuns}
            onChange={(e) => updateConfig('runConfig', 'parallelRuns', parseInt(e.target.value))}
            className="w-full"
            placeholder="1"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-cyan-400 mb-2">Resume From Checkpoint</label>
        <MatrixInput
          type="text"
          value={config.runConfig.resumeFrom || ''}
          onChange={(e) => updateConfig('runConfig', 'resumeFrom', e.target.value || null)}
          className="w-full"
          placeholder="checkpoint_id or leave empty"
        />
      </div>

      <div className="space-y-3">
        <div className="flex items-center space-x-3">
          <input
            type="checkbox"
            id="saveIntermediate"
            checked={config.runConfig.saveIntermediate}
            onChange={(e) => updateConfig('runConfig', 'saveIntermediate', e.target.checked)}
            className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400 focus:ring-2"
          />
          <label htmlFor="saveIntermediate" className="text-sm text-gray-300">
            Save Intermediate Results
          </label>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'genetic':
        return renderGeneticConfig();
      case 'grid':
        return renderGridSearchConfig();
      case 'bayesian':
        return renderBayesianConfig();
      case 'constraints':
        return renderConstraintsConfig();
      case 'objectives':
        return renderObjectivesConfig();
      case 'runtime':
        return renderRuntimeConfig();
      default:
        return null;
    }
  };

  return (
    <div className="p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <MatrixCard className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Zap className="w-6 h-6 text-cyan-400" />
              <h2 className="text-xl font-bold text-white">Strategy Optimization</h2>
            </div>
            <div className="flex items-center space-x-2">
              {!isRunning ? (
                <MatrixButton
                  onClick={handleStartOptimization}
                  disabled={isRunning}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Optimization
                </MatrixButton>
              ) : (
                <MatrixButton
                  onClick={handleStopOptimization}
                  variant="destructive"
                >
                  <Pause className="w-4 h-4 mr-2" />
                  Stop
                </MatrixButton>
              )}
              <MatrixButton
                onClick={resetConfig}
                variant="secondary"
                size="sm"
              >
                <RotateCcw className="w-4 h-4" />
              </MatrixButton>
              <MatrixButton variant="secondary" size="sm">
                <Download className="w-4 h-4" />
              </MatrixButton>
              <MatrixButton variant="secondary" size="sm">
                <Upload className="w-4 h-4" />
              </MatrixButton>
            </div>
          </div>

          {isRunning && (
            <div className="mb-6">
              <div className="flex items-center justify-between text-sm text-gray-300 mb-2">
                <span>Optimization Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-cyan-400 to-green-400 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          )}

          <div className="grid grid-cols-6 gap-2 mb-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    flex flex-col items-center p-3 rounded-lg transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400/30'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                    }
                  `}
                >
                  <Icon className="w-5 h-5 mb-1" />
                  <span className="text-xs text-center leading-tight">{tab.label}</span>
                </button>
              );
            })}
          </div>

          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <MatrixCard className="p-4 bg-gray-800/50">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                {React.createElement(tabs.find(t => t.id === activeTab)?.icon || Settings, {
                  className: "w-5 h-5 mr-2 text-cyan-400"
                })}
                {tabs.find(t => t.id === activeTab)?.label}
              </h3>
              {renderTabContent()}
            </MatrixCard>
          </motion.div>

          <div className="mt-6">
            <MatrixButton
              onClick={() => console.log('Optimization config:', config)}
              className="bg-cyan-600 hover:bg-cyan-700"
            >
              <Save className="w-4 h-4 mr-2" />
              Save Configuration
            </MatrixButton>
          </div>
        </MatrixCard>
      </motion.div>
    </div>
  );
};

export default OptimizationConfig;