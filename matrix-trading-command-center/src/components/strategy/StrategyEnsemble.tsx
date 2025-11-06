import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Network, 
  Target, 
  BarChart3, 
  Settings,
  Save,
  Plus,
  Trash2,
  Copy,
  Play,
  Pause,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  CheckCircle,
  AlertTriangle,
  Info,
  RotateCcw
} from 'lucide-react';

interface EnsembleStrategy {
  id: string;
  name: string;
  description: string;
  type: 'mean_reversion' | 'momentum' | 'pairs_trading' | 'scalping' | 'ai_ml' | 'arbitrage';
  weight: number; // 0-1
  enabled: boolean;
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    trades: number;
  };
  config: Record<string, any>;
  status: 'active' | 'inactive' | 'error';
  lastUpdate?: Date;
}

interface EnsembleMethod {
  id: string;
  name: string;
  description: string;
  type: 'weighted_average' | 'majority_vote' | 'rank_weighted' | 'confidence_weighted' | 'dynamic_allocation' | 'genetic_optimization';
  parameters: Record<string, any>;
}

interface PerformanceMetrics {
  totalReturn: number;
  annualReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  winRate: number;
  profitFactor: number;
  averageTrade: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

interface StrategyEnsemble {
  id: string;
  name: string;
  description: string;
  version: string;
  strategies: EnsembleStrategy[];
  ensembleMethods: EnsembleMethod[];
  activeMethod: string;
  performance: PerformanceMetrics;
  settings: {
    rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'adaptive';
    minWeight: number;
    maxWeight: number;
    correlationThreshold: number;
    performanceWindow: number; // days
    riskAdjustment: boolean;
    stopLossEnabled: boolean;
    maxDrawdownLimit: number;
  };
  optimization: {
    enabled: boolean;
    method: 'genetic' | 'particle_swarm' | 'simulated_annealing' | 'bayesian';
    parameters: {
      populationSize: number;
      generations: number;
      mutationRate: number;
      crossoverRate: number;
    };
    constraints: {
      maxWeight: number;
      minCorrelation: number;
      targetReturn: number;
      maxRisk: number;
    };
  };
  status: 'running' | 'stopped' | 'optimizing' | 'error';
  lastOptimization?: Date;
}

interface StrategyEnsembleProps {
  ensemble: StrategyEnsemble;
  onChange: (ensemble: StrategyEnsemble) => void;
  onSave?: () => void;
  onBacktest?: () => void;
}

export const StrategyEnsembleBuilder: React.FC<StrategyEnsembleProps> = ({
  ensemble,
  onChange,
  onSave,
  onBacktest
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'strategies' | 'methods' | 'performance' | 'optimization'>('overview');

  const strategyTypes = {
    mean_reversion: {
      name: 'Mean Reversion',
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'border-blue-500',
      description: 'Strategies that exploit price reversions to mean'
    },
    momentum: {
      name: 'Momentum',
      icon: <Zap className="w-4 h-4" />,
      color: 'border-green-500',
      description: 'Strategies that follow price momentum trends'
    },
    pairs_trading: {
      name: 'Pairs Trading',
      icon: <Target className="w-4 h-4" />,
      color: 'border-purple-500',
      description: 'Statistical arbitrage between correlated assets'
    },
    scalping: {
      name: 'Scalping',
      icon: <BarChart3 className="w-4 h-4" />,
      color: 'border-yellow-500',
      description: 'High-frequency short-term strategies'
    },
    ai_ml: {
      name: 'AI/ML',
      icon: <Network className="w-4 h-4" />,
      color: 'border-cyan-500',
      description: 'Machine learning based strategies'
    },
    arbitrage: {
      name: 'Arbitrage',
      icon: <RotateCcw className="w-4 h-4" />,
      color: 'border-orange-500',
      description: 'Risk-free profit from price differences'
    }
  };

  const ensembleMethodTypes = {
    weighted_average: {
      name: 'Weighted Average',
      description: 'Combine signals using fixed or dynamic weights',
      parameters: ['weights', 'normalization', 'confluence_threshold']
    },
    majority_vote: {
      name: 'Majority Vote',
      description: 'Decision based on majority of strategies',
      parameters: ['min_votes', 'tie_breaking', 'confidence_threshold']
    },
    rank_weighted: {
      name: 'Rank Weighted',
      description: 'Weight by performance rank',
      parameters: ['ranking_period', 'decay_factor', 'min_trades']
    },
    confidence_weighted: {
      name: 'Confidence Weighted',
      description: 'Weight by strategy confidence scores',
      parameters: ['confidence_method', 'smoothing', 'max_confidence']
    },
    dynamic_allocation: {
      name: 'Dynamic Allocation',
      description: 'Adjust weights based on performance',
      parameters: ['lookback_period', 'adjustment_speed', 'rebalance_threshold']
    },
    genetic_optimization: {
      name: 'Genetic Optimization',
      description: 'Evolve optimal weights using genetic algorithms',
      parameters: ['population_size', 'generations', 'mutation_rate', 'crossover_rate']
    }
  };

  const createEnsembleStrategy = (type: keyof typeof strategyTypes): EnsembleStrategy => {
    const strategyType = strategyTypes[type];
    return {
      id: `strategy_${Date.now()}`,
      name: `${strategyType.name} Strategy`,
      description: strategyType.description,
      type,
      weight: 0.2, // Equal weight initially
      enabled: true,
      performance: {
        totalReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        winRate: 0,
        trades: 0
      },
      config: getDefaultConfig(type),
      status: 'active'
    };
  };

  const createEnsembleMethod = (type: keyof typeof ensembleMethodTypes): EnsembleMethod => {
    const methodType = ensembleMethodTypes[type];
    return {
      id: `method_${Date.now()}`,
      name: methodType.name,
      description: methodType.description,
      type,
      parameters: getDefaultMethodParameters(type)
    };
  };

  const getDefaultConfig = (type: keyof typeof strategyTypes): Record<string, any> => {
    const configs = {
      mean_reversion: { lookback_period: 20, entry_threshold: 2.0, exit_threshold: 0.5 },
      momentum: { momentum_period: 14, entry_threshold: 0.02, stop_loss: 0.05 },
      pairs_trading: { correlation_threshold: 0.8, z_score_entry: 2.0, z_score_exit: 0.5 },
      scalping: { timeframe: 1, profit_target: 0.001, stop_loss: 0.0005 },
      ai_ml: { model_type: 'ensemble', confidence_threshold: 0.7, features: [] },
      arbitrage: { spread_threshold: 0.02, max_hold_time: 24, capital_allocation: 0.1 }
    };
    return configs[type] || {};
  };

  const getDefaultMethodParameters = (type: keyof typeof ensembleMethodTypes) => {
    const parameters = {
      weighted_average: { weights: 'equal', normalization: 'l1', confluence_threshold: 0.6 },
      majority_vote: { min_votes: 2, tie_breaking: 'highest_weight', confidence_threshold: 0.7 },
      rank_weighted: { ranking_period: 30, decay_factor: 0.95, min_trades: 10 },
      confidence_weighted: { confidence_method: 'recent', smoothing: 0.8, max_confidence: 1.0 },
      dynamic_allocation: { lookback_period: 20, adjustment_speed: 0.1, rebalance_threshold: 0.05 },
      genetic_optimization: { population_size: 100, generations: 50, mutation_rate: 0.1, crossover_rate: 0.8 }
    };
    return parameters[type] || {};
  };

  const addEnsembleStrategy = (type: keyof typeof strategyTypes) => {
    const newStrategy = createEnsembleStrategy(type);
    onChange({
      ...ensemble,
      strategies: [...ensemble.strategies, newStrategy]
    });
    setSelectedStrategy(newStrategy.id);
  };

  const addEnsembleMethod = (type: keyof typeof ensembleMethodTypes) => {
    const newMethod = createEnsembleMethod(type);
    onChange({
      ...ensemble,
      ensembleMethods: [...ensemble.ensembleMethods, newMethod]
    });
    setSelectedMethod(newMethod.id);
  };

  const updateStrategy = (strategyId: string, updates: Partial<EnsembleStrategy>) => {
    onChange({
      ...ensemble,
      strategies: ensemble.strategies.map(strategy =>
        strategy.id === strategyId ? { ...strategy, ...updates } : strategy
      )
    });
  };

  const updateMethod = (methodId: string, updates: Partial<EnsembleMethod>) => {
    onChange({
      ...ensemble,
      ensembleMethods: ensemble.ensembleMethods.map(method =>
        method.id === methodId ? { ...method, ...updates } : method
      )
    });
  };

  const removeStrategy = (strategyId: string) => {
    onChange({
      ...ensemble,
      strategies: ensemble.strategies.filter(strategy => strategy.id !== strategyId)
    });
    if (selectedStrategy === strategyId) {
      setSelectedStrategy(null);
    }
  };

  const removeMethod = (methodId: string) => {
    onChange({
      ...ensemble,
      ensembleMethods: ensemble.ensembleMethods.filter(method => method.id !== methodId)
    });
    if (selectedMethod === methodId) {
      setSelectedMethod(null);
    }
  };

  const normalizeWeights = () => {
    const totalWeight = ensemble.strategies.reduce((sum, strategy) => sum + strategy.weight, 0);
    if (totalWeight === 0) return;

    const normalizedStrategies = ensemble.strategies.map(strategy => ({
      ...strategy,
      weight: strategy.weight / totalWeight
    }));

    onChange({
      ...ensemble,
      strategies: normalizedStrategies
    });
  };

  const getStrategyTypeInfo = (type: EnsembleStrategy['type']) => {
    return strategyTypes[type];
  };

  const calculateTotalWeight = () => {
    return ensemble.strategies.filter(s => s.enabled).reduce((sum, strategy) => sum + strategy.weight, 0);
  };

  const renderStrategyConfig = (strategy: EnsembleStrategy) => {
    const typeInfo = getStrategyTypeInfo(strategy.type);
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Strategy Name</label>
            <MatrixInput
              value={strategy.name}
              onChange={(e) => updateStrategy(strategy.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Type</label>
            <select
              value={strategy.type}
              onChange={(e) => updateStrategy(strategy.id, { type: e.target.value as EnsembleStrategy['type'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {Object.entries(strategyTypes).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={strategy.description}
            onChange={(e) => updateStrategy(strategy.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this strategy..."
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Weight ({(strategy.weight * 100).toFixed(1)}%)
            </label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={strategy.weight}
              onChange={(e) => updateStrategy(strategy.id, { weight: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={strategy.enabled}
                onChange={(e) => updateStrategy(strategy.id, { enabled: e.target.checked })}
                className="w-4 h-4 accent-green-500"
              />
              <span className="text-sm text-green-400">Enabled</span>
            </label>
          </div>
        </div>

        {/* Strategy-specific Configuration */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Configuration</h4>
          
          {strategy.type === 'mean_reversion' && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Lookback Period</label>
                <MatrixInput
                  type="number"
                  min="5"
                  value={strategy.config.lookback_period || 20}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, lookback_period: parseInt(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Entry Threshold (σ)</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  value={strategy.config.entry_threshold || 2.0}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, entry_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Exit Threshold (σ)</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  value={strategy.config.exit_threshold || 0.5}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, exit_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {strategy.type === 'momentum' && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Momentum Period</label>
                <MatrixInput
                  type="number"
                  min="5"
                  value={strategy.config.momentum_period || 14}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, momentum_period: parseInt(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Entry Threshold (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={(strategy.config.entry_threshold || 0.02) * 100}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, entry_threshold: parseFloat(e.target.value) / 100 }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Stop Loss (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={(strategy.config.stop_loss || 0.05) * 100}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, stop_loss: parseFloat(e.target.value) / 100 }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {strategy.type === 'pairs_trading' && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Correlation Threshold</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="1"
                  value={strategy.config.correlation_threshold || 0.8}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, correlation_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Z-Score Entry</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  value={strategy.config.z_score_entry || 2.0}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, z_score_entry: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Z-Score Exit</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  value={strategy.config.z_score_exit || 0.5}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, z_score_exit: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {strategy.type === 'ai_ml' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Model Type</label>
                <select
                  value={strategy.config.model_type || 'ensemble'}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, model_type: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2 text-sm"
                >
                  <option value="ensemble">Ensemble</option>
                  <option value="random_forest">Random Forest</option>
                  <option value="neural_network">Neural Network</option>
                  <option value="svm">SVM</option>
                  <option value="xgboost">XGBoost</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Confidence Threshold</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="1"
                  value={strategy.config.confidence_threshold || 0.7}
                  onChange={(e) => updateStrategy(strategy.id, {
                    config: { ...strategy.config, confidence_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        <div className="matrix-card p-4 bg-black/30">
          <h4 className="text-sm font-bold text-green-400 mb-3">Performance Metrics</h4>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-green-600">Total Return</div>
              <div className={`text-sm font-bold ${
                strategy.performance.totalReturn > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {(strategy.performance.totalReturn * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-green-600">Sharpe Ratio</div>
              <div className="text-sm font-bold text-green-400">
                {strategy.performance.sharpeRatio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-green-600">Win Rate</div>
              <div className="text-sm font-bold text-green-400">
                {(strategy.performance.winRate * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between">
            <span className="text-xs text-green-400">Status</span>
            <div className="flex items-center gap-2">
              {strategy.status === 'active' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : strategy.status === 'error' ? (
                <AlertTriangle className="w-4 h-4 text-red-400" />
              ) : (
                <Info className="w-4 h-4 text-gray-400" />
              )}
              <span className={`text-xs font-bold ${
                strategy.status === 'active' ? 'text-green-400' :
                strategy.status === 'error' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {strategy.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderMethodConfig = (method: EnsembleMethod) => {
    const methodType = ensembleMethodTypes[method.type];
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Method Name</label>
            <MatrixInput
              value={method.name}
              onChange={(e) => updateMethod(method.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Type</label>
            <select
              value={method.type}
              onChange={(e) => updateMethod(method.id, { type: e.target.value as EnsembleMethod['type'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {Object.entries(ensembleMethodTypes).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={method.description}
            onChange={(e) => updateMethod(method.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this ensemble method..."
          />
        </div>

        {/* Method-specific Parameters */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Parameters</h4>
          
          {method.type === 'weighted_average' && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Weights</label>
                <select
                  value={method.parameters.weights || 'equal'}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, weights: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2 text-sm"
                >
                  <option value="equal">Equal</option>
                  <option value="performance">Performance-based</option>
                  <option value="inverse_vol">Inverse Volatility</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Normalization</label>
                <select
                  value={method.parameters.normalization || 'l1'}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, normalization: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2 text-sm"
                >
                  <option value="l1">L1 (Sum=1)</option>
                  <option value="l2">L2 (Euclidean)</option>
                  <option value="max">Max Normalization</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Confluence Threshold</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={method.parameters.confluence_threshold || 0.6}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, confluence_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {method.type === 'majority_vote' && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Minimum Votes</label>
                <MatrixInput
                  type="number"
                  min="1"
                  value={method.parameters.min_votes || 2}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, min_votes: parseInt(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Tie Breaking</label>
                <select
                  value={method.parameters.tie_breaking || 'highest_weight'}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, tie_breaking: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2 text-sm"
                >
                  <option value="highest_weight">Highest Weight</option>
                  <option value="highest_performance">Highest Performance</option>
                  <option value="random">Random</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Confidence Threshold</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={method.parameters.confidence_threshold || 0.7}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, confidence_threshold: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {method.type === 'genetic_optimization' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Population Size</label>
                <MatrixInput
                  type="number"
                  min="10"
                  max="1000"
                  value={method.parameters.population_size || 100}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, population_size: parseInt(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Generations</label>
                <MatrixInput
                  type="number"
                  min="10"
                  max="500"
                  value={method.parameters.generations || 50}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, generations: parseInt(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Mutation Rate</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1"
                  value={method.parameters.mutation_rate || 0.1}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, mutation_rate: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Crossover Rate</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="1"
                  value={method.parameters.crossover_rate || 0.8}
                  onChange={(e) => updateMethod(method.id, {
                    parameters: { ...method.parameters, crossover_rate: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const totalWeight = calculateTotalWeight();
  const activeStrategies = ensemble.strategies.filter(s => s.enabled);
  const avgPerformance = activeStrategies.length > 0 
    ? activeStrategies.reduce((sum, s) => sum + s.performance.totalReturn, 0) / activeStrategies.length
    : 0;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            STRATEGY ENSEMBLE BUILDER
          </h1>
          <p className="text-green-600 text-sm">Combine multiple strategies for enhanced performance</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Total Weight</div>
            <div className={`text-lg font-bold ${
              Math.abs(totalWeight - 1) < 0.01 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {(totalWeight * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Active Strategies</div>
            <div className="text-lg font-bold text-green-400">
              {activeStrategies.length}
            </div>
          </div>
          
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Avg Return</div>
            <div className={`text-lg font-bold ${
              avgPerformance > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {(avgPerformance * 100).toFixed(1)}%
            </div>
          </div>
          
          <MatrixButton onClick={normalizeWeights}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Normalize Weights
          </MatrixButton>
          
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Ensemble
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'strategies', label: 'Strategies', icon: <Network className="w-4 h-4" /> },
          { id: 'methods', label: 'Ensemble Methods', icon: <Target className="w-4 h-4" /> },
          { id: 'performance', label: 'Performance', icon: <TrendingUp className="w-4 h-4" /> },
          { id: 'optimization', label: 'Optimization', icon: <Settings className="w-4 h-4" /> }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id 
                ? 'text-green-400 border-b-2 border-green-500' 
                : 'text-green-600 hover:text-green-400'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Sidebar */}
        <div className="w-80 border-r border-green-800/30 flex flex-col">
          {/* Quick Add Strategies */}
          <div className="p-4 border-b border-green-800/30">
            <h3 className="text-sm font-bold text-green-400 mb-3">ADD STRATEGIES</h3>
            
            <div className="space-y-2">
              {Object.entries(strategyTypes).map(([key, typeInfo]) => (
                <MatrixButton
                  key={key}
                  size="sm"
                  variant="secondary"
                  onClick={() => addEnsembleStrategy(key as keyof typeof strategyTypes)}
                  className="w-full justify-start"
                >
                  <div className={`${typeInfo.color} mr-2`}>
                    {typeInfo.icon}
                  </div>
                  {typeInfo.name}
                </MatrixButton>
              ))}
            </div>
          </div>

          {/* Strategy List */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">STRATEGIES ({ensemble.strategies.length})</h3>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {ensemble.strategies.map((strategy) => {
                const typeInfo = getStrategyTypeInfo(strategy.type);
                return (
                  <MatrixCard
                    key={strategy.id}
                    className={`p-3 cursor-pointer transition-all ${
                      selectedStrategy === strategy.id 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-green-800/30 hover:border-green-700'
                    } ${!strategy.enabled ? 'opacity-50' : ''}`}
                    onClick={() => setSelectedStrategy(strategy.id)}
                  >
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={typeInfo.color}>
                            {typeInfo.icon}
                          </div>
                          <div>
                            <div className="text-sm font-bold text-green-400">
                              {strategy.name}
                            </div>
                            <div className="text-xs text-green-600">
                              Weight: {(strategy.weight * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className={`text-xs font-bold ${
                            strategy.performance.totalReturn > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {(strategy.performance.totalReturn * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="text-xs text-green-600">
                          {typeInfo.name}
                        </div>
                        
                        <div className="flex items-center gap-1">
                          {strategy.status === 'active' ? (
                            <CheckCircle className="w-3 h-3 text-green-400" />
                          ) : strategy.status === 'error' ? (
                            <AlertTriangle className="w-3 h-3 text-red-400" />
                          ) : (
                            <Info className="w-3 h-3 text-gray-400" />
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              removeStrategy(strategy.id);
                            }}
                            className="text-red-400 hover:text-red-300"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </MatrixCard>
                );
              })}
            </div>
          </div>

          {/* Ensemble Methods */}
          <div className="border-t border-green-800/30 p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">ENSEMBLE METHODS</h3>
            
            <div className="space-y-2">
              {ensemble.ensembleMethods.map((method) => (
                <MatrixCard
                  key={method.id}
                  className={`p-2 cursor-pointer transition-all ${
                    selectedMethod === method.id 
                      ? 'border-green-500 bg-green-900/20' 
                      : 'border-green-800/30 hover:border-green-700'
                  }`}
                  onClick={() => setSelectedMethod(method.id)}
                >
                  <div className="flex items-center gap-2">
                    <div className="text-green-400">
                      <Target className="w-4 h-4" />
                    </div>
                    <div className="flex-1">
                      <div className="text-xs font-bold text-green-400">
                        {method.name}
                      </div>
                      <div className="text-xs text-green-600 truncate">
                        {method.description}
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeMethod(method.id);
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </MatrixCard>
              ))}
              
              <MatrixButton
                size="sm"
                onClick={() => addEnsembleMethod('weighted_average')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Method
              </MatrixButton>
            </div>
          </div>
        </div>

        {/* Right Panel - Main Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {selectedStrategy && (
              <motion.div
                key="strategy-config"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                {(() => {
                  const strategy = ensemble.strategies.find(s => s.id === selectedStrategy);
                  if (!strategy) return null;
                  
                  const typeInfo = getStrategyTypeInfo(strategy.type);
                  return (
                    <MatrixCard className="p-6">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-6">
                          <div className={`${typeInfo.color} p-2 rounded`}>
                            {typeInfo.icon}
                          </div>
                          <div>
                            <h2 className="text-lg font-bold text-green-400">
                              {strategy.name}
                            </h2>
                            <p className="text-sm text-green-600">
                              {typeInfo.name} Strategy Configuration
                            </p>
                          </div>
                        </div>

                        {renderStrategyConfig(strategy)}
                      </div>
                    </MatrixCard>
                  );
                })()}
              </motion.div>
            )}

            {selectedMethod && !selectedStrategy && (
              <motion.div
                key="method-config"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                {(() => {
                  const method = ensemble.ensembleMethods.find(m => m.id === selectedMethod);
                  if (!method) return null;
                  
                  return (
                    <MatrixCard className="p-6">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-6">
                          <div className="border-cyan-500 border p-2 rounded">
                            <Target className="w-5 h-5 text-cyan-400" />
                          </div>
                          <div>
                            <h2 className="text-lg font-bold text-green-400">
                              {method.name}
                            </h2>
                            <p className="text-sm text-green-600">
                              Ensemble Method Configuration
                            </p>
                          </div>
                        </div>

                        {renderMethodConfig(method)}
                      </div>
                    </MatrixCard>
                  );
                })()}
              </motion.div>
            )}

            {activeTab === 'overview' && !selectedStrategy && !selectedMethod && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  {/* Ensemble Overview */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Ensemble Overview
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-xs text-green-600">Total Strategies</div>
                        <div className="text-2xl font-bold text-green-400">
                          {ensemble.strategies.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Active Strategies</div>
                        <div className="text-2xl font-bold text-green-400">
                          {activeStrategies.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Total Weight</div>
                        <div className={`text-2xl font-bold ${
                          Math.abs(totalWeight - 1) < 0.01 ? 'text-green-400' : 'text-yellow-400'
                        }`}>
                          {(totalWeight * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Avg Performance</div>
                        <div className={`text-2xl font-bold ${
                          avgPerformance > 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(avgPerformance * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </MatrixCard>

                  {/* Strategy Breakdown */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Strategy Breakdown
                    </h3>
                    
                    <div className="space-y-4">
                      {ensemble.strategies.map((strategy) => {
                        const typeInfo = getStrategyTypeInfo(strategy.type);
                        return (
                          <div key={strategy.id} className="matrix-card p-4">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center gap-2">
                                <div className={typeInfo.color}>
                                  {typeInfo.icon}
                                </div>
                                <div>
                                  <div className="text-sm font-bold text-green-400">
                                    {strategy.name}
                                  </div>
                                  <div className="text-xs text-green-600">
                                    {typeInfo.name}
                                  </div>
                                </div>
                              </div>
                              
                              <div className="text-right">
                                <div className="text-sm font-bold text-green-400">
                                  {(strategy.weight * 100).toFixed(1)}%
                                </div>
                                <div className={`text-xs ${
                                  strategy.performance.totalReturn > 0 ? 'text-green-400' : 'text-red-400'
                                }`}>
                                  {(strategy.performance.totalReturn * 100).toFixed(1)}% return
                                </div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-4 gap-4 text-xs">
                              <div>
                                <div className="text-green-600">Sharpe</div>
                                <div className="text-green-400 font-bold">
                                  {strategy.performance.sharpeRatio.toFixed(2)}
                                </div>
                              </div>
                              <div>
                                <div className="text-green-600">Max DD</div>
                                <div className="text-green-400 font-bold">
                                  {(strategy.performance.maxDrawdown * 100).toFixed(1)}%
                                </div>
                              </div>
                              <div>
                                <div className="text-green-600">Win Rate</div>
                                <div className="text-green-400 font-bold">
                                  {(strategy.performance.winRate * 100).toFixed(1)}%
                                </div>
                              </div>
                              <div>
                                <div className="text-green-600">Trades</div>
                                <div className="text-green-400 font-bold">
                                  {strategy.performance.trades}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </MatrixCard>

                  {/* Ensemble Performance */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Ensemble Performance
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(ensemble.performance).map(([key, value]) => (
                        <div key={key} className="matrix-card p-4">
                          <div className="text-xs text-green-600 mb-1">
                            {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                          </div>
                          <div className={`text-lg font-bold ${
                            typeof value === 'number' && value > 0 ? 'text-green-400' : 
                            typeof value === 'number' && value < 0 ? 'text-red-400' : 'text-green-400'
                          }`}>
                            {typeof value === 'number' ? value.toFixed(2) : value}
                            {key.includes('Return') || key.includes('Drawdown') ? '%' : ''}
                          </div>
                        </div>
                      ))}
                    </div>
                  </MatrixCard>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};