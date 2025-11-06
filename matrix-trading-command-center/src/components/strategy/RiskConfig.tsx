import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  Target,
  Settings,
  Save,
  RotateCcw,
  Zap,
  DollarSign,
  BarChart3,
  Clock,
  CheckCircle,
  Info,
  Minus,
  Plus
} from 'lucide-react';

interface RiskLimit {
  id: string;
  name: string;
  type: 'position_size' | 'portfolio_value' | 'daily_loss' | 'drawdown' | 'correlation' | 'concentration';
  enabled: boolean;
  threshold: number;
  currentValue: number;
  action: 'stop_trading' | 'reduce_size' | 'alert_only' | 'pause_strategy';
  description: string;
  priority: number; // 1-10
}

interface StopLossConfig {
  id: string;
  name: string;
  type: 'fixed_percentage' | 'atr_based' | 'trailing' | 'dynamic';
  enabled: boolean;
  primary: boolean;
  config: {
    percentage?: number;
    atrMultiplier?: number;
    trailingDistance?: number;
    activationPrice?: number;
    trailAdjustment?: number;
  };
  strategies: string[]; // Which strategies use this stop loss
}

interface PositionSizing {
  id: string;
  name: string;
  method: 'fixed' | 'percentage' | 'kelly' | 'optimal_f' | 'volatility_adjusted';
  enabled: boolean;
  config: {
    fixedAmount?: number;
    percentage?: number;
    kellyFraction?: number;
    maxRisk?: number;
    volatilityAdjustment?: boolean;
  };
  constraints: {
    minSize: number;
    maxSize: number;
    maxPositions: number;
  };
}

interface RiskMetrics {
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  winRate: number;
  profitFactor: number;
  recoveryFactor: number;
}

interface RiskConfig {
  id: string;
  name: string;
  description: string;
  riskProfile: 'conservative' | 'moderate' | 'aggressive' | 'custom';
  riskMetrics: RiskMetrics;
  positionSizing: PositionSizing;
  stopLoss: StopLossConfig[];
  riskLimits: RiskLimit[];
  portfolioLimits: {
    maxPortfolioDrawdown: number;
    maxPositionRisk: number;
    maxCorrelationExposure: number;
    maxLeverage: number;
    stopTradingThreshold: number;
  };
  monitoring: {
    realTimeMonitoring: boolean;
    alertFrequency: 'immediate' | 'hourly' | 'daily';
    emailNotifications: boolean;
    smsNotifications: boolean;
    dashboardAlerts: boolean;
  };
  emergency: {
    emergencyStopEnabled: boolean;
    emergencyStopLoss: number;
    circuitBreakers: boolean;
    portfolioHalt: boolean;
    manualIntervention: boolean;
  };
}

interface RiskConfigProps {
  config: RiskConfig;
  onChange: (config: RiskConfig) => void;
  onSave?: () => void;
}

export const RiskConfig: React.FC<RiskConfigProps> = ({
  config,
  onChange,
  onSave
}) => {
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'position-sizing' | 'stop-loss' | 'limits' | 'monitoring' | 'emergency'>('overview');

  const riskProfiles = {
    conservative: {
      name: 'Conservative',
      description: 'Low risk, steady returns',
      color: 'border-green-500',
      maxDrawdown: 5,
      positionRisk: 1,
      maxLeverage: 1.5
    },
    moderate: {
      name: 'Moderate',
      description: 'Balanced risk and reward',
      color: 'border-yellow-500',
      maxDrawdown: 10,
      positionRisk: 2,
      maxLeverage: 2.5
    },
    aggressive: {
      name: 'Aggressive',
      description: 'High risk, high reward',
      color: 'border-red-500',
      maxDrawdown: 20,
      positionRisk: 5,
      maxLeverage: 5
    },
    custom: {
      name: 'Custom',
      description: 'User-defined risk parameters',
      color: 'border-blue-500',
      maxDrawdown: 15,
      positionRisk: 3,
      maxLeverage: 3
    }
  };

  const riskLimitTypes = {
    position_size: {
      name: 'Position Size',
      icon: <Target className="w-4 h-4" />,
      description: 'Maximum position size in portfolio'
    },
    portfolio_value: {
      name: 'Portfolio Value',
      icon: <DollarSign className="w-4 h-4" />,
      description: 'Minimum portfolio value to continue trading'
    },
    daily_loss: {
      name: 'Daily Loss',
      icon: <TrendingDown className="w-4 h-4" />,
      description: 'Maximum daily loss limit'
    },
    drawdown: {
      name: 'Drawdown',
      icon: <BarChart3 className="w-4 h-4" />,
      description: 'Maximum portfolio drawdown'
    },
    correlation: {
      name: 'Correlation',
      icon: <Zap className="w-4 h-4" />,
      description: 'Maximum correlation between positions'
    },
    concentration: {
      name: 'Concentration',
      icon: <AlertTriangle className="w-4 h-4" />,
      description: 'Maximum concentration in single asset'
    }
  };

  const stopLossTypes = {
    fixed_percentage: {
      name: 'Fixed Percentage',
      icon: <Target className="w-4 h-4" />,
      description: 'Stop loss at fixed percentage below entry'
    },
    atr_based: {
      name: 'ATR Based',
      icon: <BarChart3 className="w-4 h-4" />,
      description: 'Stop loss based on Average True Range'
    },
    trailing: {
      name: 'Trailing Stop',
      icon: <TrendingDown className="w-4 h-4" />,
      description: 'Trailing stop loss that follows price'
    },
    dynamic: {
      name: 'Dynamic',
      icon: <Zap className="w-4 h-4" />,
      description: 'Dynamic stop loss based on volatility'
    }
  };

  const positionSizingMethods = {
    fixed: {
      name: 'Fixed Amount',
      description: 'Trade fixed dollar amount per position',
      icon: <DollarSign className="w-4 h-4" />
    },
    percentage: {
      name: 'Percentage of Portfolio',
      description: 'Trade fixed percentage of portfolio',
      icon: <BarChart3 className="w-4 h-4" />
    },
    kelly: {
      name: 'Kelly Criterion',
      description: 'Optimal position sizing using Kelly formula',
      icon: <Target className="w-4 h-4" />
    },
    optimal_f: {
      name: 'Optimal F',
      description: 'Position sizing based on optimal F',
      icon: <Zap className="w-4 h-4" />
    },
    volatility_adjusted: {
      name: 'Volatility Adjusted',
      description: 'Size positions based on volatility',
      icon: <TrendingDown className="w-4 h-4" />
    }
  };

  const createRiskLimit = (type: keyof typeof riskLimitTypes): RiskLimit => {
    const limitType = riskLimitTypes[type];
    return {
      id: `${type}_${Date.now()}`,
      name: `${limitType.name} Limit`,
      type,
      enabled: true,
      threshold: getDefaultThreshold(type),
      currentValue: 0,
      action: 'alert_only',
      description: `${limitType.description} limit`,
      priority: 1
    };
  };

  const createStopLoss = (type: keyof typeof stopLossTypes): StopLossConfig => {
    const stopType = stopLossTypes[type];
    return {
      id: `${type}_${Date.now()}`,
      name: `${stopType.name} Stop Loss`,
      type,
      enabled: true,
      primary: false,
      config: getDefaultStopLossConfig(type),
      strategies: []
    };
  };

  const getDefaultThreshold = (type: keyof typeof riskLimitTypes): number => {
    switch (type) {
      case 'position_size': return 10;
      case 'portfolio_value': return 10000;
      case 'daily_loss': return 1000;
      case 'drawdown': return 5;
      case 'correlation': return 0.8;
      case 'concentration': return 20;
      default: return 0;
    }
  };

  const getDefaultStopLossConfig = (type: keyof typeof stopLossTypes) => {
    switch (type) {
      case 'fixed_percentage':
        return { percentage: 2.0 };
      case 'atr_based':
        return { atrMultiplier: 2.0 };
      case 'trailing':
        return { trailingDistance: 3.0, trailAdjustment: 0.5 };
      case 'dynamic':
        return { activationPrice: 1.0 };
      default:
        return {};
    }
  };

  const addRiskLimit = (type: keyof typeof riskLimitTypes) => {
    const newLimit = createRiskLimit(type);
    onChange({
      ...config,
      riskLimits: [...config.riskLimits, newLimit]
    });
  };

  const addStopLoss = (type: keyof typeof stopLossTypes) => {
    const newStopLoss = createStopLoss(type);
    onChange({
      ...config,
      stopLoss: [...config.stopLoss, newStopLoss]
    });
  };

  const updateRiskLimit = (limitId: string, updates: Partial<RiskLimit>) => {
    onChange({
      ...config,
      riskLimits: config.riskLimits.map(limit =>
        limit.id === limitId ? { ...limit, ...updates } : limit
      )
    });
  };

  const updateStopLoss = (stopLossId: string, updates: Partial<StopLossConfig>) => {
    onChange({
      ...config,
      stopLoss: config.stopLoss.map(stopLoss =>
        stopLoss.id === stopLossId ? { ...stopLoss, ...updates } : stopLoss
      )
    });
  };

  const removeRiskLimit = (limitId: string) => {
    onChange({
      ...config,
      riskLimits: config.riskLimits.filter(limit => limit.id !== limitId)
    });
  };

  const removeStopLoss = (stopLossId: string) => {
    onChange({
      ...config,
      stopLoss: config.stopLoss.filter(stopLoss => stopLoss.id !== stopLossId)
    });
  };

  const applyRiskProfile = (profile: keyof typeof riskProfiles) => {
    const profileConfig = riskProfiles[profile];
    onChange({
      ...config,
      riskProfile: profile,
      portfolioLimits: {
        ...config.portfolioLimits,
        maxPortfolioDrawdown: profileConfig.maxDrawdown,
        maxPositionRisk: profileConfig.positionRisk,
        maxLeverage: profileConfig.maxLeverage
      }
    });
  };

  const renderRiskLimitConfig = (limit: RiskLimit) => {
    const limitType = riskLimitTypes[limit.type];
    
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Limit Name</label>
            <MatrixInput
              value={limit.name}
              onChange={(e) => updateRiskLimit(limit.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Action</label>
            <select
              value={limit.action}
              onChange={(e) => updateRiskLimit(limit.id, { action: e.target.value as RiskLimit['action'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              <option value="alert_only">Alert Only</option>
              <option value="reduce_size">Reduce Position Size</option>
              <option value="pause_strategy">Pause Strategy</option>
              <option value="stop_trading">Stop Trading</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Threshold {limit.type === 'position_size' || limit.type === 'concentration' ? '(%)' : 
                         limit.type === 'correlation' ? ' ' : '($)'}
            </label>
            <MatrixInput
              type="number"
              step="0.01"
              value={limit.threshold}
              onChange={(e) => updateRiskLimit(limit.id, { threshold: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Priority (1-10)</label>
            <MatrixInput
              type="number"
              min="1"
              max="10"
              value={limit.priority}
              onChange={(e) => updateRiskLimit(limit.id, { priority: parseInt(e.target.value) })}
              className="text-sm"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Description</label>
          <textarea
            value={limit.description}
            onChange={(e) => updateRiskLimit(limit.id, { description: e.target.value })}
            className="matrix-input w-full px-3 py-2 text-sm h-20 resize-none"
            placeholder="Describe this risk limit..."
          />
        </div>

        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-green-400">Current Value</span>
            <span className="text-xs font-bold text-green-400">
              {limit.currentValue.toFixed(2)}
              {limit.type === 'position_size' || limit.type === 'concentration' ? '%' : 
               limit.type === 'correlation' ? '' : '$'}
            </span>
          </div>
          
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all ${
                limit.currentValue > limit.threshold ? 'bg-red-500' : 
                limit.currentValue > limit.threshold * 0.8 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ 
                width: `${Math.min(100, (limit.currentValue / limit.threshold) * 100)}%` 
              }}
            />
          </div>
          
          {limit.currentValue > limit.threshold && (
            <div className="text-xs text-red-400 mt-1">
              Limit exceeded!
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderStopLossConfig = (stopLoss: StopLossConfig) => {
    const stopType = stopLossTypes[stopLoss.type];
    
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Stop Loss Name</label>
            <MatrixInput
              value={stopLoss.name}
              onChange={(e) => updateStopLoss(stopLoss.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={stopLoss.primary}
                onChange={(e) => updateStopLoss(stopLoss.id, { primary: e.target.checked })}
                className="w-4 h-4 accent-green-500"
              />
              <span className="text-sm text-green-400">Primary Stop Loss</span>
            </label>
          </div>
        </div>

        {/* Dynamic Configuration Based on Type */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold text-green-400">Configuration</h4>
          
          {stopLoss.type === 'fixed_percentage' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Stop Loss Percentage (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="20"
                    value={stopLoss.config.percentage || 0}
                    onChange={(e) => updateStopLoss(stopLoss.id, {
                      config: { ...stopLoss.config, percentage: parseFloat(e.target.value) }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
            </div>
          )}

          {stopLoss.type === 'atr_based' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">ATR Multiplier</label>
                <MatrixInput
                  type="number"
                  step="0.1"
                  min="0.5"
                  max="5"
                  value={stopLoss.config.atrMultiplier || 0}
                  onChange={(e) => updateStopLoss(stopLoss.id, {
                    config: { ...stopLoss.config, atrMultiplier: parseFloat(e.target.value) }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {stopLoss.type === 'trailing' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Trailing Distance (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="10"
                    value={stopLoss.config.trailingDistance || 0}
                    onChange={(e) => updateStopLoss(stopLoss.id, {
                      config: { ...stopLoss.config, trailingDistance: parseFloat(e.target.value) }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Trail Adjustment (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={stopLoss.config.trailAdjustment || 0}
                    onChange={(e) => updateStopLoss(stopLoss.id, {
                      config: { ...stopLoss.config, trailAdjustment: parseFloat(e.target.value) }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
            </div>
          )}

          {stopLoss.type === 'dynamic' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Activation Price Change (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="5"
                    value={stopLoss.config.activationPrice || 0}
                    onChange={(e) => updateStopLoss(stopLoss.id, {
                      config: { ...stopLoss.config, activationPrice: parseFloat(e.target.value) }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Status Indicator */}
        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between">
            <span className="text-xs text-green-400">Status</span>
            <div className="flex items-center gap-2">
              {stopLoss.enabled ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-gray-400" />
              )}
              <span className={`text-xs font-bold ${
                stopLoss.enabled ? 'text-green-400' : 'text-gray-400'
              }`}>
                {stopLoss.enabled ? 'ACTIVE' : 'DISABLED'}
              </span>
            </div>
          </div>
          
          {stopLoss.primary && (
            <div className="text-xs text-green-400 mt-1">
              Primary stop loss for all strategies
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderPositionSizingConfig = () => {
    const method = positionSizingMethods[config.positionSizing.method];
    
    return (
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-bold text-green-400 mb-4">Position Sizing Method</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(positionSizingMethods).map(([key, methodInfo]) => (
              <label
                key={key}
                className={`matrix-card p-4 cursor-pointer transition-all ${
                  config.positionSizing.method === key 
                    ? 'border-green-500 bg-green-900/20' 
                    : 'border-green-800/30 hover:border-green-700'
                }`}
              >
                <input
                  type="radio"
                  name="positionSizing"
                  value={key}
                  checked={config.positionSizing.method === key}
                  onChange={(e) => onChange({
                    ...config,
                    positionSizing: { ...config.positionSizing, method: e.target.value as PositionSizing['method'] }
                  })}
                  className="sr-only"
                />
                <div className="flex items-center gap-3 mb-2">
                  {methodInfo.icon}
                  <div>
                    <div className="font-bold text-green-400">{methodInfo.name}</div>
                    <div className="text-xs text-green-600">{methodInfo.description}</div>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Configuration</h4>
          
          {config.positionSizing.method === 'fixed' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Fixed Amount ($)</label>
                <MatrixInput
                  type="number"
                  min="1"
                  value={config.positionSizing.config.fixedAmount || 1000}
                  onChange={(e) => onChange({
                    ...config,
                    positionSizing: {
                      ...config.positionSizing,
                      config: { ...config.positionSizing.config, fixedAmount: parseFloat(e.target.value) }
                    }
                  })}
                  className="text-sm"
                />
              </div>
            </div>
          )}

          {config.positionSizing.method === 'percentage' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Portfolio Percentage (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="50"
                    value={config.positionSizing.config.percentage || 10}
                    onChange={(e) => onChange({
                      ...config,
                      positionSizing: {
                        ...config.positionSizing,
                        config: { ...config.positionSizing.config, percentage: parseFloat(e.target.value) }
                      }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
            </div>
          )}

          {config.positionSizing.method === 'kelly' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Kelly Fraction</label>
                <MatrixInput
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1"
                  value={config.positionSizing.config.kellyFraction || 0.25}
                  onChange={(e) => onChange({
                    ...config,
                    positionSizing: {
                      ...config.positionSizing,
                      config: { ...config.positionSizing.config, kellyFraction: parseFloat(e.target.value) }
                    }
                  })}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Max Risk (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={config.positionSizing.config.maxRisk || 2}
                    onChange={(e) => onChange({
                      ...config,
                      positionSizing: {
                        ...config.positionSizing,
                        config: { ...config.positionSizing.config, maxRisk: parseFloat(e.target.value) }
                      }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
            </div>
          )}

          {config.positionSizing.method === 'volatility_adjusted' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Base Position Size (%)</label>
                <div className="relative">
                  <MatrixInput
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="20"
                    value={config.positionSizing.config.percentage || 5}
                    onChange={(e) => onChange({
                      ...config,
                      positionSizing: {
                        ...config.positionSizing,
                        config: { ...config.positionSizing.config, percentage: parseFloat(e.target.value) }
                      }
                    })}
                    className="text-sm"
                  />
                  <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
                </div>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.positionSizing.config.volatilityAdjustment || false}
                    onChange={(e) => onChange({
                      ...config,
                      positionSizing: {
                        ...config.positionSizing,
                        config: { 
                          ...config.positionSizing.config, 
                          volatilityAdjustment: e.target.checked 
                        }
                      }
                    })}
                    className="w-4 h-4 accent-green-500"
                  />
                  <span className="text-sm text-green-400">Enable Volatility Adjustment</span>
                </label>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Constraints</h4>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Minimum Size ($)</label>
              <MatrixInput
                type="number"
                min="1"
                value={config.positionSizing.constraints.minSize}
                onChange={(e) => onChange({
                  ...config,
                  positionSizing: {
                    ...config.positionSizing,
                    constraints: {
                      ...config.positionSizing.constraints,
                      minSize: parseFloat(e.target.value)
                    }
                  }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Maximum Size ($)</label>
              <MatrixInput
                type="number"
                min="1"
                value={config.positionSizing.constraints.maxSize}
                onChange={(e) => onChange({
                  ...config,
                  positionSizing: {
                    ...config.positionSizing,
                    constraints: {
                      ...config.positionSizing.constraints,
                      maxSize: parseFloat(e.target.value)
                    }
                  }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Max Positions</label>
              <MatrixInput
                type="number"
                min="1"
                max="100"
                value={config.positionSizing.constraints.maxPositions}
                onChange={(e) => onChange({
                  ...config,
                  positionSizing: {
                    ...config.positionSizing,
                    constraints: {
                      ...config.positionSizing.constraints,
                      maxPositions: parseInt(e.target.value)
                    }
                  }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>
      </div>
    );
  };

  const currentProfile = riskProfiles[config.riskProfile];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            RISK MANAGEMENT CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure risk parameters and controls</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Risk Profile</div>
            <div className={`text-lg font-bold ${currentProfile.color.replace('border-', 'text-')}`}>
              {currentProfile.name}
            </div>
          </div>
          
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Configuration
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'position-sizing', label: 'Position Sizing', icon: <Target className="w-4 h-4" /> },
          { id: 'stop-loss', label: 'Stop Loss', icon: <TrendingDown className="w-4 h-4" /> },
          { id: 'limits', label: 'Risk Limits', icon: <Shield className="w-4 h-4" /> },
          { id: 'monitoring', label: 'Monitoring', icon: <Clock className="w-4 h-4" /> },
          { id: 'emergency', label: 'Emergency', icon: <AlertTriangle className="w-4 h-4" /> }
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
          {/* Risk Profile Selection */}
          <div className="p-4 border-b border-green-800/30">
            <h3 className="text-sm font-bold text-green-400 mb-3">RISK PROFILE</h3>
            
            <div className="space-y-2">
              {Object.entries(riskProfiles).map(([key, profile]) => (
                <MatrixCard
                  key={key}
                  className={`p-3 cursor-pointer transition-all ${
                    config.riskProfile === key 
                      ? `${profile.color} bg-green-900/20` 
                      : 'border-green-800/30 hover:border-green-700'
                  }`}
                  onClick={() => applyRiskProfile(key as keyof typeof riskProfiles)}
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-bold text-green-400">
                        {profile.name}
                      </div>
                      {config.riskProfile === key && (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      )}
                    </div>
                    <div className="text-xs text-green-600">
                      {profile.description}
                    </div>
                    <div className="text-xs text-green-600">
                      Max DD: {profile.maxDrawdown}%, Max Risk: {profile.positionRisk}%
                    </div>
                  </div>
                </MatrixCard>
              ))}
            </div>
          </div>

          {/* Quick Add Sections */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">QUICK ACTIONS</h3>
            
            <div className="space-y-2">
              <MatrixButton 
                size="sm" 
                onClick={() => addRiskLimit('daily_loss')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Daily Loss Limit
              </MatrixButton>
              
              <MatrixButton 
                size="sm" 
                onClick={() => addRiskLimit('drawdown')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Drawdown Limit
              </MatrixButton>
              
              <MatrixButton 
                size="sm" 
                onClick={() => addStopLoss('fixed_percentage')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Fixed Stop Loss
              </MatrixButton>
              
              <MatrixButton 
                size="sm" 
                onClick={() => addStopLoss('trailing')}
                className="w-full justify-start"
              >
                <Plus className="w-3 h-3 mr-2" />
                Add Trailing Stop
              </MatrixButton>
            </div>
          </div>
        </div>

        {/* Right Panel - Main Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  {/* Risk Metrics */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Current Risk Metrics
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(config.riskMetrics).map(([key, value]) => (
                        <div key={key} className="matrix-card p-4">
                          <div className="text-xs text-green-600 mb-1">
                            {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                          </div>
                          <div className="text-xl font-bold text-green-400">
                            {typeof value === 'number' ? value.toFixed(2) : value}
                            {key.includes('Ratio') || key.includes('drawdown') ? '%' : 
                             key.includes('Var') || key.includes('drawdown') ? '$' : ''}
                          </div>
                        </div>
                      ))}
                    </div>
                  </MatrixCard>

                  {/* Portfolio Limits */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Portfolio Limits
                    </h3>
                    
                    <div className="space-y-4">
                      {Object.entries(config.portfolioLimits).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between">
                          <div>
                            <div className="text-sm text-green-400">
                              {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                            </div>
                            <div className="text-xs text-green-600">
                              {key === 'maxPortfolioDrawdown' || key === 'maxPositionRisk' ? 
                                `${value}%` : key === 'maxLeverage' ? `${value}x` : 
                                key === 'stopTradingThreshold' ? `${value}%` : `$${value.toLocaleString()}`}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-green-400">
                              {typeof value === 'number' ? value.toFixed(2) : value}
                              {key === 'maxPortfolioDrawdown' || key === 'maxPositionRisk' ? '%' : 
                               key === 'maxLeverage' ? 'x' : 
                               key === 'stopTradingThreshold' ? '%' : ''}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </MatrixCard>

                  {/* Risk Controls Status */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Risk Controls Status
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-bold text-green-400 mb-3">
                          Stop Loss Rules ({config.stopLoss.filter(s => s.enabled).length} active)
                        </h4>
                        <div className="space-y-2">
                          {config.stopLoss.map((stopLoss) => (
                            <div key={stopLoss.id} className="flex items-center justify-between">
                              <span className="text-sm text-green-600">{stopLoss.name}</span>
                              <div className="flex items-center gap-2">
                                {stopLoss.primary && (
                                  <span className="text-xs bg-green-600 text-black px-2 py-1 rounded">
                                    PRIMARY
                                  </span>
                                )}
                                {stopLoss.enabled ? (
                                  <CheckCircle className="w-4 h-4 text-green-400" />
                                ) : (
                                  <AlertTriangle className="w-4 h-4 text-gray-400" />
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-bold text-green-400 mb-3">
                          Risk Limits ({config.riskLimits.filter(l => l.enabled).length} active)
                        </h4>
                        <div className="space-y-2">
                          {config.riskLimits.slice(0, 5).map((limit) => (
                            <div key={limit.id} className="flex items-center justify-between">
                              <span className="text-sm text-green-600">{limit.name}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-green-400">
                                  {limit.currentValue > limit.threshold ? 'EXCEEDED' : 'OK'}
                                </span>
                                {limit.enabled ? (
                                  <CheckCircle className="w-4 h-4 text-green-400" />
                                ) : (
                                  <AlertTriangle className="w-4 h-4 text-gray-400" />
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </MatrixCard>
                </div>
              </motion.div>
            )}

            {activeTab === 'position-sizing' && (
              <motion.div
                key="position-sizing"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <MatrixCard className="p-6">
                  <h3 className="text-lg font-bold text-green-400 mb-6">
                    Position Sizing Configuration
                  </h3>
                  {renderPositionSizingConfig()}
                </MatrixCard>
              </motion.div>
            )}

            {activeTab === 'stop-loss' && (
              <motion.div
                key="stop-loss"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold text-green-400">
                      Stop Loss Configuration
                    </h3>
                    <MatrixButton onClick={() => addStopLoss('fixed_percentage')}>
                      <Plus className="w-4 h-4 mr-2" />
                      Add Stop Loss
                    </MatrixButton>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {config.stopLoss.map((stopLoss) => {
                      const stopType = stopLossTypes[stopLoss.type];
                      return (
                        <MatrixCard key={stopLoss.id} className="p-6">
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className="text-green-400">
                                  {stopType.icon}
                                </div>
                                <div>
                                  <div className="text-sm font-bold text-green-400">
                                    {stopLoss.name}
                                  </div>
                                  <div className="text-xs text-green-600">
                                    {stopType.name}
                                  </div>
                                </div>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => removeStopLoss(stopLoss.id)}
                                  className="text-red-400 hover:text-red-300"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </div>

                            {renderStopLossConfig(stopLoss)}
                          </div>
                        </MatrixCard>
                      );
                    })}

                    {config.stopLoss.length === 0 && (
                      <div className="col-span-2 text-center py-12">
                        <Shield className="w-16 h-16 mx-auto mb-4 opacity-50 text-green-400" />
                        <h3 className="text-lg font-bold text-green-400 mb-2">
                          No Stop Loss Rules
                        </h3>
                        <p className="text-green-600 mb-4">
                          Add stop loss rules to protect your positions
                        </p>
                        <MatrixButton onClick={() => addStopLoss('fixed_percentage')}>
                          <Plus className="w-4 h-4 mr-2" />
                          Add First Stop Loss
                        </MatrixButton>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'limits' && (
              <motion.div
                key="limits"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold text-green-400">
                      Risk Limits Configuration
                    </h3>
                    <MatrixButton onClick={() => addRiskLimit('position_size')}>
                      <Plus className="w-4 h-4 mr-2" />
                      Add Risk Limit
                    </MatrixButton>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {config.riskLimits.map((limit) => {
                      const limitType = riskLimitTypes[limit.type];
                      return (
                        <MatrixCard key={limit.id} className="p-6">
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className="text-green-400">
                                  {limitType.icon}
                                </div>
                                <div>
                                  <div className="text-sm font-bold text-green-400">
                                    {limit.name}
                                  </div>
                                  <div className="text-xs text-green-600">
                                    {limitType.name}
                                  </div>
                                </div>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => removeRiskLimit(limit.id)}
                                  className="text-red-400 hover:text-red-300"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </div>

                            {renderRiskLimitConfig(limit)}
                          </div>
                        </MatrixCard>
                      );
                    })}

                    {config.riskLimits.length === 0 && (
                      <div className="col-span-2 text-center py-12">
                        <Shield className="w-16 h-16 mx-auto mb-4 opacity-50 text-green-400" />
                        <h3 className="text-lg font-bold text-green-400 mb-2">
                          No Risk Limits Configured
                        </h3>
                        <p className="text-green-600 mb-4">
                          Add risk limits to protect your portfolio
                        </p>
                        <MatrixButton onClick={() => addRiskLimit('position_size')}>
                          <Plus className="w-4 h-4 mr-2" />
                          Add First Risk Limit
                        </MatrixButton>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};