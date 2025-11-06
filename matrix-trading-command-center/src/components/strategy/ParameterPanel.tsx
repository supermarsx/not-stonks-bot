import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Settings, 
  Save, 
  RotateCcw, 
  Copy, 
  Download,
  Upload,
  Plus,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';

interface ParameterConfig {
  id: string;
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select' | 'range' | 'time';
  value: any;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  unit?: string;
  description: string;
  validation?: {
    required: boolean;
    pattern?: string;
    custom?: (value: any) => string | null;
  };
  preset?: string;
}

interface ParameterTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  parameters: Record<string, any>;
}

interface ParameterPanelProps {
  strategyType: string;
  config: Record<string, any>;
  onChange: (config: Record<string, any>) => void;
  onSave?: () => void;
  onReset?: () => void;
}

export const ParameterPanel: React.FC<ParameterPanelProps> = ({
  strategyType,
  config,
  onChange,
  onSave,
  onReset
}) => {
  const [parameters, setParameters] = useState<ParameterConfig[]>(getParameterConfig(strategyType));
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState<'parameters' | 'presets' | 'validation'>('parameters');
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');

  const templates: ParameterTemplate[] = [
    {
      id: 'conservative',
      name: 'Conservative',
      description: 'Low risk, steady returns',
      category: 'Risk Management',
      parameters: {
        stopLoss: 0.02,
        takeProfit: 0.04,
        positionSize: 0.05,
        maxDrawdown: 0.05
      }
    },
    {
      id: 'aggressive',
      name: 'Aggressive',
      description: 'High risk, high reward',
      category: 'Risk Management',
      parameters: {
        stopLoss: 0.05,
        takeProfit: 0.15,
        positionSize: 0.2,
        maxDrawdown: 0.15
      }
    },
    {
      id: 'scalping',
      name: 'Scalping',
      description: 'High frequency, quick trades',
      category: 'Trading Style',
      parameters: {
        timeframe: 1,
        profitTarget: 0.002,
        stopLoss: 0.001,
        minVolume: 1000000
      }
    },
    {
      id: 'swing',
      name: 'Swing Trading',
      description: 'Medium-term positions',
      category: 'Trading Style',
      parameters: {
        timeframe: 240,
        lookbackPeriod: 50,
        entryThreshold: 2.5,
        exitThreshold: 1.0
      }
    }
  ];

  function getParameterConfig(strategyType: string): ParameterConfig[] {
    const configs: Record<string, ParameterConfig[]> = {
      mean_reversion: [
        {
          id: 'lookback_period',
          name: 'Lookback Period',
          type: 'number',
          value: config.lookbackPeriod || 20,
          min: 5,
          max: 100,
          step: 1,
          unit: 'periods',
          description: 'Number of periods to calculate moving average',
          validation: { required: true, custom: (value) => value > 0 ? null : 'Must be positive' }
        },
        {
          id: 'entry_threshold',
          name: 'Entry Threshold',
          type: 'number',
          value: config.entryThreshold || 2.0,
          min: 0.1,
          max: 5.0,
          step: 0.1,
          unit: 'σ',
          description: 'Number of standard deviations for entry signal',
          validation: { required: true, custom: (value) => value > 0 ? null : 'Must be positive' }
        },
        {
          id: 'exit_threshold',
          name: 'Exit Threshold',
          type: 'number',
          value: config.exitThreshold || 0.5,
          min: 0.1,
          max: 3.0,
          step: 0.1,
          unit: 'σ',
          description: 'Number of standard deviations for exit signal',
          validation: { required: true, custom: (value) => value > 0 ? null : 'Must be positive' }
        },
        {
          id: 'stop_loss',
          name: 'Stop Loss',
          type: 'number',
          value: config.stopLoss || 0.03,
          min: 0.005,
          max: 0.1,
          step: 0.005,
          unit: '%',
          description: 'Maximum loss per trade',
          validation: { required: true, custom: (value) => value > 0 && value < 0.5 ? null : 'Must be between 0.5% and 50%' }
        },
        {
          id: 'take_profit',
          name: 'Take Profit',
          type: 'number',
          value: config.takeProfit || 0.06,
          min: 0.01,
          max: 0.2,
          step: 0.01,
          unit: '%',
          description: 'Target profit per trade'
        },
        {
          id: 'position_size',
          name: 'Position Size',
          type: 'number',
          value: config.positionSize || 0.1,
          min: 0.01,
          max: 0.5,
          step: 0.01,
          unit: '× portfolio',
          description: 'Fraction of portfolio to risk per trade'
        },
        {
          id: 'enable_trailing_stop',
          name: 'Enable Trailing Stop',
          type: 'boolean',
          value: config.enableTrailingStop || false,
          description: 'Use trailing stop loss'
        }
      ],
      momentum: [
        {
          id: 'momentum_period',
          name: 'Momentum Period',
          type: 'number',
          value: config.momentumPeriod || 14,
          min: 5,
          max: 50,
          step: 1,
          unit: 'periods',
          description: 'Period for momentum calculation'
        },
        {
          id: 'entry_threshold',
          name: 'Entry Threshold',
          type: 'number',
          value: config.entryThreshold || 0.02,
          min: 0.001,
          max: 0.1,
          step: 0.001,
          unit: '%',
          description: 'Minimum momentum for entry'
        },
        {
          id: 'stop_loss',
          name: 'Stop Loss',
          type: 'number',
          value: config.stopLoss || 0.05,
          min: 0.01,
          max: 0.15,
          step: 0.005,
          unit: '%',
          description: 'Stop loss percentage'
        },
        {
          id: 'take_profit',
          name: 'Take Profit',
          type: 'number',
          value: config.takeProfit || 0.1,
          min: 0.02,
          max: 0.3,
          step: 0.01,
          unit: '%',
          description: 'Take profit percentage'
        },
        {
          id: 'max_holding_period',
          name: 'Max Holding Period',
          type: 'number',
          value: config.maxHoldingPeriod || 20,
          min: 1,
          max: 100,
          step: 1,
          unit: 'periods',
          description: 'Maximum time to hold position'
        }
      ],
      scalping: [
        {
          id: 'timeframe',
          name: 'Timeframe',
          type: 'select',
          value: config.timeframe || '1m',
          options: ['1m', '5m', '15m'],
          description: 'Chart timeframe for scalping'
        },
        {
          id: 'profit_target',
          name: 'Profit Target',
          type: 'number',
          value: config.profitTarget || 0.001,
          min: 0.0001,
          max: 0.01,
          step: 0.0001,
          unit: '%',
          description: 'Target profit per trade'
        },
        {
          id: 'stop_loss',
          name: 'Stop Loss',
          type: 'number',
          value: config.stopLoss || 0.0005,
          min: 0.0001,
          max: 0.005,
          step: 0.0001,
          unit: '%',
          description: 'Stop loss percentage'
        },
        {
          id: 'min_volume',
          name: 'Minimum Volume',
          type: 'number',
          value: config.minVolume || 1000000,
          min: 100000,
          max: 10000000,
          step: 100000,
          unit: 'shares',
          description: 'Minimum volume for trade consideration'
        },
        {
          id: 'max_trades_per_day',
          name: 'Max Trades Per Day',
          type: 'number',
          value: config.maxTradesPerDay || 50,
          min: 1,
          max: 200,
          step: 1,
          unit: 'trades',
          description: 'Maximum trades per day'
        }
      ]
    };

    return configs[strategyType] || [];
  }

  const updateParameter = (id: string, value: any) => {
    const updated = parameters.map(param => 
      param.id === id ? { ...param, value } : param
    );
    setParameters(updated);
    
    // Update config object
    const newConfig = { ...config };
    newConfig[id] = value;
    onChange(newConfig);
    
    // Clear validation error if exists
    if (validationErrors[id]) {
      const newErrors = { ...validationErrors };
      delete newErrors[id];
      setValidationErrors(newErrors);
    }
  };

  const validateParameter = (param: ParameterConfig): string | null => {
    if (param.validation?.required && (param.value === undefined || param.value === '')) {
      return 'This parameter is required';
    }
    
    if (param.validation?.custom) {
      return param.validation.custom(param.value);
    }
    
    if (param.type === 'number' && typeof param.value === 'number') {
      if (param.min !== undefined && param.value < param.min) {
        return `Must be at least ${param.min}`;
      }
      if (param.max !== undefined && param.value > param.max) {
        return `Must be at most ${param.max}`;
      }
    }
    
    return null;
  };

  const applyTemplate = (template: ParameterTemplate) => {
    const updated = parameters.map(param => ({
      ...param,
      value: template.parameters[param.id] !== undefined ? template.parameters[param.id] : param.value
    }));
    setParameters(updated);
    
    const newConfig = { ...config, ...template.parameters };
    onChange(newConfig);
    setSelectedTemplate(template.id);
  };

  const resetParameters = () => {
    const defaults = getParameterConfig(strategyType);
    setParameters(defaults);
    onChange({});
    setValidationErrors({});
    setSelectedTemplate('');
  };

  const exportConfig = () => {
    const exportData = {
      strategyType,
      config,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${strategyType}_config.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const validateAll = () => {
    const errors: Record<string, string> = {};
    
    parameters.forEach(param => {
      const error = validateParameter(param);
      if (error) {
        errors[param.id] = error;
      }
    });
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const renderParameterInput = (param: ParameterConfig) => {
    const baseInputClass = "matrix-input w-full px-3 py-2 text-sm";
    const errorClass = validationErrors[param.id] ? "border-red-500" : "border-green-800/30";

    switch (param.type) {
      case 'number':
        return (
          <div className="space-y-2">
            <div className="relative">
              <MatrixInput
                type="number"
                value={param.value}
                onChange={(e) => updateParameter(param.id, parseFloat(e.target.value))}
                min={param.min}
                max={param.max}
                step={param.step}
                className={`${baseInputClass} ${errorClass}`}
                placeholder={param.description}
              />
              {param.unit && (
                <span className="absolute right-3 top-2 text-xs text-green-600">
                  {param.unit}
                </span>
              )}
            </div>
            {param.min !== undefined && param.max !== undefined && (
              <input
                type="range"
                min={param.min}
                max={param.max}
                step={param.step}
                value={param.value}
                onChange={(e) => updateParameter(param.id, parseFloat(e.target.value))}
                className="w-full accent-green-500"
              />
            )}
          </div>
        );

      case 'boolean':
        return (
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={param.value}
              onChange={(e) => updateParameter(param.id, e.target.checked)}
              className="w-4 h-4 accent-green-500"
            />
            <span className="text-sm text-green-400">{param.description}</span>
          </label>
        );

      case 'select':
        return (
          <select
            value={param.value}
            onChange={(e) => updateParameter(param.id, e.target.value)}
            className={`${baseInputClass} ${errorClass}`}
          >
            {param.options?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );

      case 'string':
      default:
        return (
          <MatrixInput
            type="text"
            value={param.value}
            onChange={(e) => updateParameter(param.id, e.target.value)}
            className={`${baseInputClass} ${errorClass}`}
            placeholder={param.description}
          />
        );
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            PARAMETER CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure strategy parameters and presets</p>
        </div>
        
        <div className="flex items-center gap-2">
          <MatrixButton size="sm" variant="secondary" onClick={exportConfig}>
            <Download className="w-4 h-4 mr-2" />
            Export
          </MatrixButton>
          <MatrixButton size="sm" variant="secondary" onClick={resetParameters}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </MatrixButton>
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'parameters', label: 'Parameters', icon: <Settings className="w-4 h-4" /> },
          { id: 'presets', label: 'Presets', icon: <Copy className="w-4 h-4" /> },
          { id: 'validation', label: 'Validation', icon: <CheckCircle className="w-4 h-4" /> }
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

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          {activeTab === 'parameters' && (
            <motion.div
              key="parameters"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="p-6"
            >
              <div className="space-y-6">
                {parameters.map((param) => (
                  <MatrixCard key={param.id} className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <label className="text-sm font-bold text-green-400">
                              {param.name}
                            </label>
                            {param.validation?.required && (
                              <span className="text-red-400 text-xs">*</span>
                            )}
                            {validationErrors[param.id] && (
                              <AlertTriangle className="w-4 h-4 text-red-400" />
                            )}
                          </div>
                          <p className="text-xs text-green-600">{param.description}</p>
                        </div>
                      </div>
                      
                      {renderParameterInput(param)}
                      
                      {validationErrors[param.id] && (
                        <div className="flex items-center gap-2 text-red-400 text-xs">
                          <AlertTriangle className="w-3 h-3" />
                          {validationErrors[param.id]}
                        </div>
                      )}
                    </div>
                  </MatrixCard>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'presets' && (
            <motion.div
              key="presets"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="p-6"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {templates.map((template) => (
                  <MatrixCard 
                    key={template.id}
                    className={`p-4 cursor-pointer transition-all ${
                      selectedTemplate === template.id 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'hover:border-green-700'
                    }`}
                    onClick={() => applyTemplate(template)}
                  >
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-bold text-green-400">{template.name}</h3>
                          <p className="text-sm text-green-600">{template.category}</p>
                        </div>
                        {selectedTemplate === template.id && (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        )}
                      </div>
                      
                      <p className="text-xs text-green-600">{template.description}</p>
                      
                      <div className="space-y-1">
                        {Object.entries(template.parameters).slice(0, 3).map(([key, value]) => (
                          <div key={key} className="flex justify-between text-xs">
                            <span className="text-green-600">{key}:</span>
                            <span className="text-green-400 font-mono">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </MatrixCard>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'validation' && (
            <motion.div
              key="validation"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="p-6"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-green-400">Parameter Validation</h3>
                  <MatrixButton size="sm" onClick={validateAll}>
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Validate All
                  </MatrixButton>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {parameters.map((param) => {
                    const error = validateParameter(param);
                    const hasError = !!error;
                    
                    return (
                      <MatrixCard key={param.id} className="p-4">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-green-400">
                              {param.name}
                            </span>
                            {hasError ? (
                              <AlertTriangle className="w-4 h-4 text-red-400" />
                            ) : (
                              <CheckCircle className="w-4 h-4 text-green-400" />
                            )}
                          </div>
                          
                          <div className="text-xs text-green-600">
                            {param.description}
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-green-600">Current Value:</span>
                            <span className="text-xs font-mono text-green-400">
                              {String(param.value)}
                            </span>
                          </div>
                          
                          {hasError && (
                            <div className="flex items-center gap-2 text-red-400 text-xs">
                              <AlertTriangle className="w-3 h-3" />
                              {error}
                            </div>
                          )}
                        </div>
                      </MatrixCard>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};