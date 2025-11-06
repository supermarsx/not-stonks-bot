import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore, Strategy } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  TrendingUp, 
  TrendingDown,
  Zap,
  BarChart3,
  Clock,
  Target,
  Brain,
  Plus,
  Trash2
} from 'lucide-react';

export const StrategyManager: React.FC = () => {
  const { strategies, activeStrategyId, addStrategy, updateStrategy, setActiveStrategy } = useTradingStore();
  const [showStrategyForm, setShowStrategyForm] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<Strategy | null>(null);

  const [strategyForm, setStrategyForm] = useState({
    name: '',
    description: '',
    type: 'mean_reversion' as 'mean_reversion' | 'momentum' | 'pairs_trading' | 'scalping',
    riskLevel: 'medium' as 'low' | 'medium' | 'high',
    config: {} as Record<string, any>
  });

  const handleCreateStrategy = () => {
    if (!strategyForm.name) return;

    const defaultConfigs = {
      mean_reversion: {
        lookback_period: 20,
        entry_threshold: 2.0,
        exit_threshold: 0.5,
        stop_loss: 0.03
      },
      momentum: {
        momentum_period: 14,
        entry_threshold: 0.02,
        stop_loss: 0.05,
        take_profit: 0.1
      },
      pairs_trading: {
        correlation_threshold: 0.8,
        z_score_entry: 2.0,
        z_score_exit: 0.5,
        max_holding_time: 48
      },
      scalping: {
        timeframe: 1,
        profit_target: 0.001,
        stop_loss: 0.0005,
        min_volume: 1000000
      }
    };

    const newStrategy: Strategy = {
      id: Date.now().toString(),
      name: strategyForm.name,
      description: strategyForm.description,
      status: 'stopped',
      performance: {
        totalPnL: 0,
        winRate: 0,
        totalTrades: 0,
        avgTradeDuration: 0
      },
      config: defaultConfigs[strategyForm.type]
    };

    addStrategy(newStrategy);
    
    setStrategyForm({
      name: '',
      description: '',
      type: 'mean_reversion',
      riskLevel: 'medium',
      config: {}
    });
    setShowStrategyForm(false);
  };

  const handleToggleStrategy = (strategyId: string) => {
    const strategy = strategies.find(s => s.id === strategyId);
    if (!strategy) return;

    const newStatus = strategy.status === 'running' ? 'stopped' : 'running';
    updateStrategy(strategyId, { status: newStatus });
    
    if (newStatus === 'running') {
      setActiveStrategy(strategyId);
    }
  };

  const getStrategyIcon = (type: string) => {
    switch (type) {
      case 'mean_reversion':
        return <TrendingUp className="w-5 h-5" />;
      case 'momentum':
        return <Zap className="w-5 h-5" />;
      case 'pairs_trading':
        return <Target className="w-5 h-5" />;
      case 'scalping':
        return <BarChart3 className="w-5 h-5" />;
      default:
        return <Brain className="w-5 h-5" />;
    }
  };

  const getPerformanceColor = (value: number, type: 'pnl' | 'winRate') => {
    if (type === 'pnl') {
      return value > 0 ? 'text-green-400' : value < 0 ? 'text-red-400' : 'text-yellow-400';
    } else {
      return value > 70 ? 'text-green-400' : value > 50 ? 'text-yellow-400' : 'text-red-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400">
            STRATEGY MANAGER
          </h1>
          <p className="text-green-600 mt-1">Configure and monitor trading strategies</p>
        </div>
        <MatrixButton
          onClick={() => setShowStrategyForm(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Strategy
        </MatrixButton>
      </div>

      {/* Strategy Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MatrixCard title="Active Strategies" glow>
          <div className="text-2xl font-bold matrix-text-glow">
            {strategies.filter(s => s.status === 'running').length}
          </div>
          <div className="text-sm text-green-600">Currently running</div>
        </MatrixCard>
        
        <MatrixCard title="Total P&L" glow>
          <div className={`text-2xl font-bold ${
            strategies.reduce((sum, s) => sum + s.performance.totalPnL, 0) > 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            ${strategies.reduce((sum, s) => sum + s.performance.totalPnL, 0).toFixed(2)}
          </div>
          <div className="text-sm text-green-600">All strategies</div>
        </MatrixCard>
        
        <MatrixCard title="Total Trades" glow>
          <div className="text-2xl font-bold matrix-text-glow">
            {strategies.reduce((sum, s) => sum + s.performance.totalTrades, 0)}
          </div>
          <div className="text-sm text-green-600">Executed trades</div>
        </MatrixCard>
        
        <MatrixCard title="Avg Win Rate" glow>
          <div className={`text-2xl font-bold ${
            strategies.length > 0 ? 
            getPerformanceColor(strategies.reduce((sum, s) => sum + s.performance.winRate, 0) / strategies.length, 'winRate') 
            : 'text-green-400'
          }`}>
            {strategies.length > 0 ? 
              (strategies.reduce((sum, s) => sum + s.performance.winRate, 0) / strategies.length).toFixed(1)
              : '0.0'
            }%
          </div>
          <div className="text-sm text-green-600">Success rate</div>
        </MatrixCard>
      </div>

      {/* Strategy List */}
      <div className="space-y-4">
        <AnimatePresence>
          {strategies.map((strategy) => (
            <motion.div
              key={strategy.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <MatrixCard 
                title={strategy.name}
                subtitle={strategy.description}
                glow={strategy.status === 'running'}
                interactive
                className={`relative ${strategy.status === 'running' ? 'border-green-500' : 'border-gray-700'}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-4">
                      {getStrategyIcon(strategy.name.toLowerCase().includes('mean') ? 'mean_reversion' : 
                                     strategy.name.toLowerCase().includes('momentum') ? 'momentum' :
                                     strategy.name.toLowerCase().includes('pairs') ? 'pairs_trading' : 'scalping')}
                      <div>
                        <div className="font-bold text-green-400">{strategy.name}</div>
                        <div className="text-sm text-green-600">{strategy.description}</div>
                      </div>
                      <StatusIndicator 
                        status={strategy.status === 'running' ? 'online' : 'offline'}
                      >
                        {strategy.status.toUpperCase()}
                      </StatusIndicator>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <div className="text-xs text-green-600">P&L</div>
                        <div className={`font-bold ${getPerformanceColor(strategy.performance.totalPnL, 'pnl')}`}>
                          ${strategy.performance.totalPnL.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Win Rate</div>
                        <div className={`font-bold ${getPerformanceColor(strategy.performance.winRate, 'winRate')}`}>
                          {strategy.performance.winRate.toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Trades</div>
                        <div className="font-bold matrix-text-glow">
                          {strategy.performance.totalTrades}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Avg Duration</div>
                        <div className="font-bold matrix-text-glow">
                          {strategy.performance.avgTradeDuration}min
                        </div>
                      </div>
                    </div>

                    {/* Performance Bar */}
                    <div className="mb-4">
                      <div className="flex items-center justify-between text-xs text-green-600 mb-1">
                        <span>PERFORMANCE</span>
                        <span>{strategy.performance.winRate.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-300 ${
                            strategy.performance.winRate > 70 ? 'bg-green-500' :
                            strategy.performance.winRate > 50 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${strategy.performance.winRate}%` }}
                        />
                      </div>
                    </div>

                    {/* Configuration Parameters */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                      {Object.entries(strategy.config).map(([key, value]) => (
                        <div key={key} className="bg-black/30 p-2 rounded">
                          <span className="text-green-600">{key.replace(/_/g, ' ')}:</span>
                          <span className="font-mono ml-1">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-col gap-2 ml-4">
                    <MatrixButton
                      size="sm"
                      onClick={() => handleToggleStrategy(strategy.id)}
                      className={strategy.status === 'running' ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                    >
                      {strategy.status === 'running' ? (
                        <>
                          <Pause className="w-3 h-3 mr-1" />
                          STOP
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 mr-1" />
                          START
                        </>
                      )}
                    </MatrixButton>
                    
                    <MatrixButton variant="secondary" size="sm">
                      <Settings className="w-3 h-3" />
                    </MatrixButton>
                    
                    <MatrixButton variant="secondary" size="sm">
                      <Trash2 className="w-3 h-3" />
                    </MatrixButton>
                  </div>
                </div>

                {/* Strategy Status Indicator */}
                {activeStrategyId === strategy.id && strategy.status === 'running' && (
                  <div className="absolute top-2 right-2">
                    <div className="flex items-center gap-1 bg-green-600 text-black px-2 py-1 rounded text-xs font-bold">
                      <div className="w-2 h-2 bg-black rounded-full animate-pulse"></div>
                      ACTIVE
                    </div>
                  </div>
                )}
              </MatrixCard>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {strategies.length === 0 && (
        <MatrixCard>
          <div className="text-center py-12">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50 text-green-400" />
            <h3 className="text-lg font-bold text-green-400 mb-2">No Strategies Configured</h3>
            <p className="text-green-600 mb-4">Create your first trading strategy to get started</p>
            <MatrixButton onClick={() => setShowStrategyForm(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Create Strategy
            </MatrixButton>
          </div>
        </MatrixCard>
      )}

      {/* Strategy Form Modal */}
      <AnimatePresence>
        {showStrategyForm && (
          <motion.div
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="matrix-card p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
            >
              <h2 className="text-xl font-bold matrix-text-glow mb-6">
                Create New Strategy
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-green-400 mb-2">Strategy Name</label>
                  <MatrixInput
                    placeholder="Mean Reversion Strategy"
                    value={strategyForm.name}
                    onChange={(e) => setStrategyForm({ ...strategyForm, name: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm text-green-400 mb-2">Description</label>
                  <textarea
                    className="matrix-input w-full px-3 py-2 rounded h-20 resize-none"
                    placeholder="Describe your strategy..."
                    value={strategyForm.description}
                    onChange={(e) => setStrategyForm({ ...strategyForm, description: e.target.value })}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-green-400 mb-2">Strategy Type</label>
                    <select
                      value={strategyForm.type}
                      onChange={(e) => setStrategyForm({ ...strategyForm, type: e.target.value as any })}
                      className="matrix-input w-full px-3 py-2 rounded"
                    >
                      <option value="mean_reversion">Mean Reversion</option>
                      <option value="momentum">Momentum</option>
                      <option value="pairs_trading">Pairs Trading</option>
                      <option value="scalping">Scalping</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-green-400 mb-2">Risk Level</label>
                    <select
                      value={strategyForm.riskLevel}
                      onChange={(e) => setStrategyForm({ ...strategyForm, riskLevel: e.target.value as any })}
                      className="matrix-input w-full px-3 py-2 rounded"
                    >
                      <option value="low">LOW</option>
                      <option value="medium">MEDIUM</option>
                      <option value="high">HIGH</option>
                    </select>
                  </div>
                </div>

                {/* Default Configuration Display */}
                <div className="bg-black/30 p-4 rounded border border-green-800/30">
                  <h3 className="text-sm text-green-400 mb-3">Default Configuration</h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {strategyForm.type === 'mean_reversion' && (
                      <>
                        <div>Lookback Period: 20</div>
                        <div>Entry Threshold: 2.0</div>
                        <div>Exit Threshold: 0.5</div>
                        <div>Stop Loss: 3%</div>
                      </>
                    )}
                    {strategyForm.type === 'momentum' && (
                      <>
                        <div>Momentum Period: 14</div>
                        <div>Entry Threshold: 2%</div>
                        <div>Stop Loss: 5%</div>
                        <div>Take Profit: 10%</div>
                      </>
                    )}
                    {strategyForm.type === 'pairs_trading' && (
                      <>
                        <div>Correlation: 0.8</div>
                        <div>Z-Score Entry: 2.0</div>
                        <div>Z-Score Exit: 0.5</div>
                        <div>Max Hold Time: 48h</div>
                      </>
                    )}
                    {strategyForm.type === 'scalping' && (
                      <>
                        <div>Timeframe: 1min</div>
                        <div>Profit Target: 0.1%</div>
                        <div>Stop Loss: 0.05%</div>
                        <div>Min Volume: 1M</div>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <MatrixButton
                  onClick={handleCreateStrategy}
                  className="flex-1"
                  disabled={!strategyForm.name}
                >
                  Create Strategy
                </MatrixButton>
                <MatrixButton
                  variant="secondary"
                  onClick={() => setShowStrategyForm(false)}
                >
                  Cancel
                </MatrixButton>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};