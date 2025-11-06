import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Target, 
  TrendingDown, 
  Clock, 
  Percent,
  DollarSign,
  AlertTriangle,
  Plus,
  Trash2,
  Settings,
  Save,
  Copy,
  Play
} from 'lucide-react';

interface ExitRule {
  id: string;
  name: string;
  type: 'profit_target' | 'stop_loss' | 'trailing_stop' | 'time_based' | 'technical' | 'volatility';
  enabled: boolean;
  priority: number;
  config: {
    threshold?: number;
    percentage?: number;
    timeFrame?: string;
    indicator?: string;
    indicatorValue?: number;
  };
  description: string;
}

interface ExitStrategy {
  id: string;
  name: string;
  description: string;
  rules: ExitRule[];
  combinationLogic: 'and' | 'or' | 'sequential';
  maxExits: number;
}

interface ExitStrategyConfigProps {
  strategy: ExitStrategy;
  onChange: (strategy: ExitStrategy) => void;
  onSave?: () => void;
}

export const ExitStrategyConfig: React.FC<ExitStrategyConfigProps> = ({
  strategy,
  onChange,
  onSave
}) => {
  const [selectedRule, setSelectedRule] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'rules' | 'logic' | 'backtest'>('rules');

  const exitRuleTypes = {
    profit_target: {
      name: 'Profit Target',
      icon: <Target className="w-4 h-4" />,
      color: 'border-green-500',
      config: ['threshold', 'percentage']
    },
    stop_loss: {
      name: 'Stop Loss',
      icon: <TrendingDown className="w-4 h-4" />,
      color: 'border-red-500',
      config: ['threshold', 'percentage']
    },
    trailing_stop: {
      name: 'Trailing Stop',
      icon: <TrendingDown className="w-4 h-4" />,
      color: 'border-orange-500',
      config: ['threshold', 'activation']
    },
    time_based: {
      name: 'Time-Based',
      icon: <Clock className="w-4 h-4" />,
      color: 'border-blue-500',
      config: ['timeFrame', 'duration']
    },
    technical: {
      name: 'Technical Signal',
      icon: <Percent className="w-4 h-4" />,
      color: 'border-purple-500',
      config: ['indicator', 'indicatorValue']
    },
    volatility: {
      name: 'Volatility Break',
      icon: <AlertTriangle className="w-4 h-4" />,
      color: 'border-yellow-500',
      config: ['threshold', 'period']
    }
  };

  const createExitRule = (type: keyof typeof exitRuleTypes): ExitRule => {
    const ruleType = exitRuleTypes[type];
    return {
      id: `${type}_${Date.now()}`,
      name: `${ruleType.name} Rule`,
      type,
      enabled: true,
      priority: strategy.rules.length + 1,
      config: getDefaultConfig(type),
      description: `Automated ${ruleType.name.toLowerCase()} exit rule`
    };
  };

  const getDefaultConfig = (type: keyof typeof exitRuleTypes) => {
    switch (type) {
      case 'profit_target':
        return { threshold: 0.06, percentage: 100 };
      case 'stop_loss':
        return { threshold: 0.03, percentage: 100 };
      case 'trailing_stop':
        return { threshold: 0.05, activation: 0.02 };
      case 'time_based':
        return { timeFrame: '4h', duration: 24 };
      case 'technical':
        return { indicator: 'RSI', indicatorValue: 70 };
      case 'volatility':
        return { threshold: 0.03, period: 14 };
      default:
        return {};
    }
  };

  const addExitRule = (type: keyof typeof exitRuleTypes) => {
    const newRule = createExitRule(type);
    const updatedStrategy = {
      ...strategy,
      rules: [...strategy.rules, newRule]
    };
    onChange(updatedStrategy);
  };

  const updateRule = (ruleId: string, updates: Partial<ExitRule>) => {
    const updatedStrategy = {
      ...strategy,
      rules: strategy.rules.map(rule => 
        rule.id === ruleId ? { ...rule, ...updates } : rule
      )
    };
    onChange(updatedStrategy);
  };

  const removeRule = (ruleId: string) => {
    const updatedStrategy = {
      ...strategy,
      rules: strategy.rules.filter(rule => rule.id !== ruleId)
    };
    onChange(updatedStrategy);
    if (selectedRule === ruleId) {
      setSelectedRule(null);
    }
  };

  const moveRule = (ruleId: string, direction: 'up' | 'down') => {
    const ruleIndex = strategy.rules.findIndex(rule => rule.id === ruleId);
    if (
      (direction === 'up' && ruleIndex === 0) ||
      (direction === 'down' && ruleIndex === strategy.rules.length - 1)
    ) {
      return;
    }

    const newRules = [...strategy.rules];
    const swapIndex = direction === 'up' ? ruleIndex - 1 : ruleIndex + 1;
    [newRules[ruleIndex], newRules[swapIndex]] = [newRules[swapIndex], newRules[ruleIndex]];
    
    // Update priorities
    newRules.forEach((rule, index) => {
      rule.priority = index + 1;
    });

    const updatedStrategy = {
      ...strategy,
      rules: newRules
    };
    onChange(updatedStrategy);
  };

  const duplicateRule = (ruleId: string) => {
    const rule = strategy.rules.find(r => r.id === ruleId);
    if (!rule) return;

    const duplicate = {
      ...rule,
      id: `${rule.type}_${Date.now()}`,
      name: `${rule.name} Copy`,
      priority: strategy.rules.length + 1
    };

    const updatedStrategy = {
      ...strategy,
      rules: [...strategy.rules, duplicate]
    };
    onChange(updatedStrategy);
  };

  const renderRuleConfig = (rule: ExitRule) => {
    const ruleType = exitRuleTypes[rule.type];
    
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Rule Name</label>
            <MatrixInput
              value={rule.name}
              onChange={(e) => updateRule(rule.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          
          <div>
            <label className="block text-xs text-green-400 mb-1">Priority</label>
            <select
              value={rule.priority}
              onChange={(e) => updateRule(rule.id, { priority: parseInt(e.target.value) })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {strategy.rules.map((_, index) => (
                <option key={index + 1} value={index + 1}>
                  {index + 1}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={rule.description}
            onChange={(e) => updateRule(rule.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this exit rule..."
          />
        </div>

        {/* Dynamic Configuration Based on Rule Type */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold text-green-400">Configuration</h4>
          
          {rule.type === 'profit_target' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Profit Target (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={rule.config.threshold || 0}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, threshold: parseFloat(e.target.value) }
                  })}
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Exit Percentage (%)</label>
                <MatrixInput
                  type="number"
                  step="1"
                  min="1"
                  max="100"
                  value={rule.config.percentage || 100}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, percentage: parseFloat(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}

          {rule.type === 'stop_loss' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Stop Loss (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={rule.config.threshold || 0}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, threshold: parseFloat(e.target.value) }
                  })}
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Exit Percentage (%)</label>
                <MatrixInput
                  type="number"
                  step="1"
                  min="1"
                  max="100"
                  value={rule.config.percentage || 100}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, percentage: parseFloat(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}

          {rule.type === 'trailing_stop' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Trail Distance (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={rule.config.threshold || 0}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, threshold: parseFloat(e.target.value) }
                  })}
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Activation (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={rule.config.activation || 0}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, activation: parseFloat(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}

          {rule.type === 'time_based' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Time Frame</label>
                <select
                  value={rule.config.timeFrame || '4h'}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, timeFrame: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2"
                >
                  <option value="1h">1 Hour</option>
                  <option value="4h">4 Hours</option>
                  <option value="1d">1 Day</option>
                  <option value="1w">1 Week</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Duration (Periods)</label>
                <MatrixInput
                  type="number"
                  min="1"
                  value={rule.config.duration || 24}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, duration: parseInt(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}

          {rule.type === 'technical' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Indicator</label>
                <select
                  value={rule.config.indicator || 'RSI'}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, indicator: e.target.value }
                  })}
                  className="matrix-input w-full px-3 py-2"
                >
                  <option value="RSI">RSI</option>
                  <option value="MACD">MACD</option>
                  <option value="Bollinger">Bollinger Bands</option>
                  <option value="Stochastic">Stochastic</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Trigger Value</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  value={rule.config.indicatorValue || 70}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, indicatorValue: parseFloat(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}

          {rule.type === 'volatility' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Volatility Threshold (%)</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  value={rule.config.threshold || 0}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, threshold: parseFloat(e.target.value) }
                  })}
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Lookback Period</label>
                <MatrixInput
                  type="number"
                  min="1"
                  value={rule.config.period || 14}
                  onChange={(e) => updateRule(rule.id, {
                    config: { ...rule.config, period: parseInt(e.target.value) }
                  })}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            EXIT STRATEGY CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure exit rules and logic</p>
        </div>
        
        <div className="flex items-center gap-2">
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Exit Strategy
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'rules', label: 'Exit Rules', icon: <Target className="w-4 h-4" /> },
          { id: 'logic', label: 'Logic & Priority', icon: <Settings className="w-4 h-4" /> },
          { id: 'backtest', label: 'Backtest', icon: <Play className="w-4 h-4" /> }
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
        {/* Left Panel - Rule Types & Rules List */}
        <div className="w-80 border-r border-green-800/30 flex flex-col">
          <div className="p-4 border-b border-green-800/30">
            <h3 className="text-sm font-bold text-green-400 mb-3">EXIT RULE TYPES</h3>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(exitRuleTypes).map(([type, ruleType]) => (
                <MatrixButton
                  key={type}
                  size="sm"
                  variant="secondary"
                  onClick={() => addExitRule(type as keyof typeof exitRuleTypes)}
                  className="flex items-center gap-2 justify-start"
                >
                  {ruleType.icon}
                  {ruleType.name}
                </MatrixButton>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">CONFIGURED RULES</h3>
            <div className="space-y-2">
              {strategy.rules.map((rule, index) => {
                const ruleType = exitRuleTypes[rule.type];
                return (
                  <MatrixCard
                    key={rule.id}
                    className={`p-3 cursor-pointer transition-all ${
                      selectedRule === rule.id 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-green-800/30 hover:border-green-700'
                    } ${!rule.enabled ? 'opacity-50' : ''}`}
                    onClick={() => setSelectedRule(rule.id)}
                  >
                    <div className="space-y-2">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <div className={ruleType.color}>
                            {ruleType.icon}
                          </div>
                          <div>
                            <div className="text-sm font-medium text-green-400">
                              {rule.name}
                            </div>
                            <div className="text-xs text-green-600">
                              Priority: {rule.priority}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              updateRule(rule.id, { enabled: !rule.enabled });
                            }}
                            className={`w-4 h-4 rounded border-2 ${
                              rule.enabled 
                                ? 'bg-green-500 border-green-500' 
                                : 'border-gray-500'
                            }`}
                          >
                            {rule.enabled && <span className="text-white text-xs">✓</span>}
                          </button>
                        </div>
                      </div>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            moveRule(rule.id, 'up');
                          }}
                          disabled={index === 0}
                          className="text-green-600 hover:text-green-400 disabled:opacity-50"
                        >
                          ↑
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            moveRule(rule.id, 'down');
                          }}
                          disabled={index === strategy.rules.length - 1}
                          className="text-green-600 hover:text-green-400 disabled:opacity-50"
                        >
                          ↓
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            duplicateRule(rule.id);
                          }}
                          className="text-green-600 hover:text-green-400"
                        >
                          <Copy className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeRule(rule.id);
                          }}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </MatrixCard>
                );
              })}

              {strategy.rules.length === 0 && (
                <div className="text-center py-8">
                  <Target className="w-12 h-12 mx-auto mb-3 opacity-50 text-green-400" />
                  <p className="text-green-600 text-sm mb-2">No exit rules configured</p>
                  <p className="text-green-600 text-xs">
                    Add rules from the left panel to get started
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Configuration */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {selectedRule ? (
              <motion.div
                key="config"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                {(() => {
                  const rule = strategy.rules.find(r => r.id === selectedRule);
                  if (!rule) return null;
                  
                  const ruleType = exitRuleTypes[rule.type];
                  return (
                    <MatrixCard className="p-6">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-6">
                          <div className={`${ruleType.color} p-2 rounded`}>
                            {ruleType.icon}
                          </div>
                          <div>
                            <h2 className="text-lg font-bold text-green-400">
                              {rule.name}
                            </h2>
                            <p className="text-sm text-green-600">
                              {ruleType.name} Exit Rule
                            </p>
                          </div>
                        </div>

                        {renderRuleConfig(rule)}
                      </div>
                    </MatrixCard>
                  );
                })()}
              </motion.div>
            ) : activeTab === 'logic' ? (
              <motion.div
                key="logic"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Exit Logic Configuration
                    </h3>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-green-400 mb-2">
                          Combination Logic
                        </label>
                        <div className="grid grid-cols-3 gap-3">
                          {[
                            { value: 'and', label: 'AND', description: 'All rules must trigger' },
                            { value: 'or', label: 'OR', description: 'Any rule can trigger' },
                            { value: 'sequential', label: 'Sequential', description: 'Rules trigger in order' }
                          ].map(option => (
                            <label
                              key={option.value}
                              className={`matrix-card p-3 cursor-pointer transition-all ${
                                strategy.combinationLogic === option.value 
                                  ? 'border-green-500 bg-green-900/20' 
                                  : 'hover:border-green-700'
                              }`}
                            >
                              <input
                                type="radio"
                                name="logic"
                                value={option.value}
                                checked={strategy.combinationLogic === option.value}
                                onChange={(e) => onChange({
                                  ...strategy,
                                  combinationLogic: e.target.value as any
                                })}
                                className="sr-only"
                              />
                              <div className="text-center">
                                <div className="font-bold text-green-400">{option.label}</div>
                                <div className="text-xs text-green-600 mt-1">
                                  {option.description}
                                </div>
                              </div>
                            </label>
                          ))}
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-2">
                          Maximum Exits Per Position
                        </label>
                        <MatrixInput
                          type="number"
                          min="1"
                          max="10"
                          value={strategy.maxExits}
                          onChange={(e) => onChange({
                            ...strategy,
                            maxExits: parseInt(e.target.value)
                          })}
                        />
                        <p className="text-xs text-green-600 mt-1">
                          Maximum number of exit orders to place per position
                        </p>
                      </div>
                    </div>
                  </MatrixCard>

                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Rule Priority Overview
                    </h3>
                    
                    <div className="space-y-3">
                      {strategy.rules
                        .sort((a, b) => a.priority - b.priority)
                        .map((rule, index) => {
                          const ruleType = exitRuleTypes[rule.type];
                          return (
                            <div
                              key={rule.id}
                              className={`matrix-card p-4 ${
                                rule.enabled ? 'border-green-800/30' : 'border-gray-600 opacity-50'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="text-green-400 font-bold">
                                    #{rule.priority}
                                  </div>
                                  <div className={`${ruleType.color}`}>
                                    {ruleType.icon}
                                  </div>
                                  <div>
                                    <div className="text-sm font-medium text-green-400">
                                      {rule.name}
                                    </div>
                                    <div className="text-xs text-green-600">
                                      {rule.description}
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="flex items-center gap-2">
                                  <div className={`w-3 h-3 rounded-full ${
                                    rule.enabled ? 'bg-green-500' : 'bg-gray-500'
                                  }`} />
                                  <span className="text-xs text-green-600">
                                    {rule.enabled ? 'Active' : 'Disabled'}
                                  </span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </MatrixCard>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="backtest"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="text-center py-12">
                  <Play className="w-16 h-16 mx-auto mb-4 opacity-50 text-green-400" />
                  <h3 className="text-lg font-bold text-green-400 mb-2">
                    Backtest Exit Strategy
                  </h3>
                  <p className="text-green-600 mb-4">
                    Test your exit rules against historical data
                  </p>
                  <MatrixButton>
                    <Play className="w-4 h-4 mr-2" />
                    Start Backtest
                  </MatrixButton>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};