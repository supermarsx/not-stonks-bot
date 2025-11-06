import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Play, 
  BarChart3, 
  Calendar, 
  Settings,
  Save,
  Clock,
  TrendingUp,
  TrendingDown,
  Target,
  Shield,
  Zap,
  AlertTriangle,
  CheckCircle,
  Download,
  Upload,
  RotateCcw
} from 'lucide-react';

interface BacktestConfig {
  id: string;
  name: string;
  description: string;
  strategy: {
    id: string;
    name: string;
    type: string;
    config: Record<string, any>;
  };
  data: {
    startDate: string;
    endDate: string;
    frequency: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
    symbols: string[];
    dataSource: 'yahoo' | 'alpha_vantage' | 'polygon' | 'iex' | 'custom';
    adjusted: boolean; // adjusted for splits/dividends
    maxGapBars: number; // maximum number of missing bars to allow
  };
  execution: {
    initialCapital: number;
    commission: number; // per trade
    slippage: number; // percentage
    marketHours: boolean;
    allowShortSelling: boolean;
    positionSizing: 'fixed' | 'percentage' | 'kelly' | 'optimal_f';
    maxPositions: number;
    rebalanceFrequency: 'never' | 'daily' | 'weekly' | 'monthly';
  };
  risk: {
    stopLoss?: number;
    takeProfit?: number;
    maxDrawdown: number;
    maxPositionSize: number;
    maxLeverage: number;
    correlationLimit: number;
    sectorExposureLimit: number;
  };
  performance: {
    benchmark?: string;
    riskFreeRate: number;
    metrics: {
      totalReturn: boolean;
      annualizedReturn: boolean;
      volatility: boolean;
      sharpeRatio: boolean;
      sortinoRatio: boolean;
      maxDrawdown: boolean;
      calmarRatio: boolean;
      winRate: boolean;
      profitFactor: boolean;
      avgTradeDuration: boolean;
      avgWin: boolean;
      avgLoss: boolean;
      largestWin: boolean;
      largestLoss: boolean;
      consecutiveWins: boolean;
      consecutiveLosses: boolean;
      recoveryFactor: boolean;
      ulcerIndex: boolean;
      sterlingRatio: boolean;
    };
  };
  reports: {
    detailedTrades: boolean;
    performanceChart: boolean;
    drawdownChart: boolean;
    monthlyReturns: boolean;
    rollingMetrics: boolean;
    attribution: boolean;
    riskAnalysis: boolean;
    exportFormat: 'html' | 'pdf' | 'excel' | 'json';
  };
  optimization: {
    enabled: boolean;
    parameters: string[];
    ranges: Record<string, { min: number; max: number; step: number }>;
    objective: 'totalReturn' | 'sharpeRatio' | 'sortinoRatio' | 'calmarRatio' | 'profitFactor';
    constraint?: Record<string, { operator: '>' | '<' | '>=' | '<='; value: number }>;
    parallel: boolean;
    maxWorkers: number;
  };
  status: 'draft' | 'running' | 'completed' | 'failed';
  results?: {
    summary: Record<string, number>;
    trades: any[];
    equity: any[];
    drawdown: any[];
    monthlyReturns: Record<string, number>;
    benchmark: any[];
    optimization?: {
      bestParams: Record<string, any>;
      bestScore: number;
      parameterSensitivity: Record<string, any[]>;
    };
  };
  createdAt: Date;
  updatedAt: Date;
}

interface BacktestConfigProps {
  config: BacktestConfig;
  onChange: (config: BacktestConfig) => void;
  onSave?: () => void;
  onRun?: () => void;
}

export const BacktestConfig: React.FC<BacktestConfigProps> = ({
  config,
  onChange,
  onSave,
  onRun
}) => {
  const [activeTab, setActiveTab] = useState<'data' | 'execution' | 'risk' | 'performance' | 'reports' | 'optimization' | 'results'>('data');
  const [selectedSymbol, setSelectedSymbol] = useState('');

  const dataFrequencies = [
    { value: '1m', label: '1 Minute', days: 7 },
    { value: '5m', label: '5 Minutes', days: 30 },
    { value: '15m', label: '15 Minutes', days: 60 },
    { value: '1h', label: '1 Hour', days: 365 },
    { value: '4h', label: '4 Hours', days: 730 },
    { value: '1d', label: '1 Day', days: 1825 },
    { value: '1w', label: '1 Week', days: 3650 }
  ];

  const dataSources = [
    { value: 'yahoo', label: 'Yahoo Finance', free: true, coverage: 'Global stocks, ETFs, crypto' },
    { value: 'alpha_vantage', label: 'Alpha Vantage', free: true, coverage: 'US stocks, forex, crypto' },
    { value: 'polygon', label: 'Polygon.io', free: false, coverage: 'US stocks, options, forex' },
    { value: 'iex', label: 'IEX Cloud', free: false, coverage: 'US stocks, forex, crypto' },
    { value: 'custom', label: 'Custom API', free: false, coverage: 'User-provided data' }
  ];

  const positionSizingMethods = {
    fixed: {
      name: 'Fixed Size',
      description: 'Trade fixed number of shares',
      config: ['fixedAmount']
    },
    percentage: {
      name: 'Percentage of Capital',
      description: 'Risk fixed percentage of capital',
      config: ['percentage']
    },
    kelly: {
      name: 'Kelly Criterion',
      description: 'Optimal position sizing using Kelly formula',
      config: ['kellyFraction', 'maxKelly']
    },
    optimal_f: {
      name: 'Optimal F',
      description: 'Position sizing based on Optimal F',
      config: ['optimalF', 'maxRisk']
    }
  };

  const commonSymbols = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'EEM',
    'GLD', 'SLV', 'USO', 'UNG',
    'BTC-USD', 'ETH-USD', 'LTC-USD'
  ];

  const backtestMetrics = [
    { key: 'totalReturn', name: 'Total Return', description: 'Overall return percentage' },
    { key: 'annualizedReturn', name: 'Annualized Return', description: 'CAGR over the period' },
    { key: 'volatility', name: 'Volatility', description: 'Standard deviation of returns' },
    { key: 'sharpeRatio', name: 'Sharpe Ratio', description: 'Risk-adjusted return' },
    { key: 'sortinoRatio', name: 'Sortino Ratio', description: 'Downside risk-adjusted return' },
    { key: 'maxDrawdown', name: 'Maximum Drawdown', description: 'Largest peak-to-trough decline' },
    { key: 'calmarRatio', name: 'Calmar Ratio', description: 'Annualized return / max drawdown' },
    { key: 'winRate', name: 'Win Rate', description: 'Percentage of profitable trades' },
    { key: 'profitFactor', name: 'Profit Factor', description: 'Gross profit / gross loss' },
    { key: 'avgTradeDuration', name: 'Average Trade Duration', description: 'Average holding period' },
    { key: 'avgWin', name: 'Average Win', description: 'Average winning trade return' },
    { key: 'avgLoss', name: 'Average Loss', description: 'Average losing trade return' },
    { key: 'largestWin', name: 'Largest Win', description: 'Maximum single trade gain' },
    { key: 'largestLoss', name: 'Largest Loss', description: 'Maximum single trade loss' },
    { key: 'consecutiveWins', name: 'Consecutive Wins', description: 'Maximum winning streak' },
    { key: 'consecutiveLosses', name: 'Consecutive Losses', description: 'Maximum losing streak' },
    { key: 'recoveryFactor', name: 'Recovery Factor', description: 'Net profit / max drawdown' },
    { key: 'ulcerIndex', name: 'Ulcer Index', description: 'Depth and duration of drawdowns' },
    { key: 'sterlingRatio', name: 'Sterling Ratio', description: '3-year return / avg drawdown' }
  ];

  const optimizationObjectives = {
    totalReturn: { name: 'Total Return', description: 'Maximize overall return' },
    sharpeRatio: { name: 'Sharpe Ratio', description: 'Maximize risk-adjusted return' },
    sortinoRatio: { name: 'Sortino Ratio', description: 'Maximize downside-adjusted return' },
    calmarRatio: { name: 'Calmar Ratio', description: 'Maximize return per unit of drawdown' },
    profitFactor: { name: 'Profit Factor', description: 'Maximize gross profit/gross loss ratio' }
  };

  const updateConfig = (section: string, field: string, value: any) => {
    onChange({
      ...config,
      [section]: {
        ...config[section as keyof BacktestConfig],
        [field]: value
      }
    });
  };

  const updateMetrics = (metric: string, enabled: boolean) => {
    onChange({
      ...config,
      performance: {
        ...config.performance,
        metrics: {
          ...config.performance.metrics,
          [metric]: enabled
        }
      }
    });
  };

  const addSymbol = () => {
    if (!selectedSymbol || config.data.symbols.includes(selectedSymbol)) return;
    
    updateConfig('data', 'symbols', [...config.data.symbols, selectedSymbol]);
    setSelectedSymbol('');
  };

  const removeSymbol = (symbol: string) => {
    updateConfig('data', 'symbols', config.data.symbols.filter(s => s !== symbol));
  };

  const getMaxDateRange = (frequency: string) => {
    const freqData = dataFrequencies.find(f => f.value === frequency);
    return freqData ? freqData.days : 365;
  };

  const renderDataConfig = () => {
    const maxDays = getMaxDateRange(config.data.frequency);
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Data Source</label>
            <select
              value={config.data.dataSource}
              onChange={(e) => updateConfig('data', 'dataSource', e.target.value)}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {dataSources.map(source => (
                <option key={source.value} value={source.value}>
                  {source.label} {source.free ? '(Free)' : '(Paid)'}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Frequency</label>
            <select
              value={config.data.frequency}
              onChange={(e) => updateConfig('data', 'frequency', e.target.value)}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {dataFrequencies.map(freq => (
                <option key={freq.value} value={freq.value}>
                  {freq.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Start Date</label>
            <input
              type="date"
              value={config.data.startDate}
              onChange={(e) => updateConfig('data', 'startDate', e.target.value)}
              className="matrix-input w-full px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">End Date</label>
            <input
              type="date"
              value={config.data.endDate}
              onChange={(e) => updateConfig('data', 'endDate', e.target.value)}
              className="matrix-input w-full px-3 py-2 text-sm"
            />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.data.adjusted}
              onChange={(e) => updateConfig('data', 'adjusted', e.target.checked)}
              className="w-4 h-4 accent-green-500"
            />
            <span className="text-sm text-green-400">Adjusted for splits/dividends</span>
          </label>
          
          <div>
            <label className="block text-xs text-green-400 mb-1">Max Gap Bars</label>
            <MatrixInput
              type="number"
              min="0"
              max="100"
              value={config.data.maxGapBars}
              onChange={(e) => updateConfig('data', 'maxGapBars', parseInt(e.target.value))}
              className="w-20"
            />
          </div>
        </div>

        {/* Symbol Selection */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Symbols ({config.data.symbols.length})</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Add Symbol</label>
              <div className="flex gap-2">
                <MatrixInput
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                  placeholder="e.g., AAPL"
                  className="flex-1"
                />
                <MatrixButton size="sm" onClick={addSymbol}>
                  Add
                </MatrixButton>
              </div>
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Common Symbols</label>
              <div className="grid grid-cols-4 gap-1 max-h-32 overflow-y-auto">
                {commonSymbols.map(symbol => (
                  <button
                    key={symbol}
                    onClick={() => setSelectedSymbol(symbol)}
                    className="text-xs bg-green-900/20 hover:bg-green-900/40 px-2 py-1 rounded text-green-400"
                  >
                    {symbol}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="matrix-card p-4 bg-black/30">
            <div className="flex flex-wrap gap-2">
              {config.data.symbols.map(symbol => (
                <div
                  key={symbol}
                  className="flex items-center gap-1 bg-green-800/30 px-2 py-1 rounded text-green-400 text-sm"
                >
                  {symbol}
                  <button
                    onClick={() => removeSymbol(symbol)}
                    className="text-red-400 hover:text-red-300 ml-1"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderExecutionConfig = () => {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Initial Capital ($)</label>
            <MatrixInput
              type="number"
              min="1000"
              step="1000"
              value={config.execution.initialCapital}
              onChange={(e) => updateConfig('execution', 'initialCapital', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Commission per Trade ($)</label>
            <MatrixInput
              type="number"
              min="0"
              step="0.01"
              value={config.execution.commission}
              onChange={(e) => updateConfig('execution', 'commission', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Slippage (%)</label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={config.execution.slippage}
              onChange={(e) => updateConfig('execution', 'slippage', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Max Positions</label>
            <MatrixInput
              type="number"
              min="1"
              max="100"
              value={config.execution.maxPositions}
              onChange={(e) => updateConfig('execution', 'maxPositions', parseInt(e.target.value))}
              className="text-sm"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Position Sizing Method</label>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(positionSizingMethods).map(([key, method]) => (
              <label
                key={key}
                className={`matrix-card p-3 cursor-pointer transition-all ${
                  config.execution.positionSizing === key 
                    ? 'border-green-500 bg-green-900/20' 
                    : 'border-green-800/30 hover:border-green-700'
                }`}
              >
                <input
                  type="radio"
                  name="positionSizing"
                  value={key}
                  checked={config.execution.positionSizing === key}
                  onChange={(e) => updateConfig('execution', 'positionSizing', e.target.value)}
                  className="sr-only"
                />
                <div className="text-sm font-medium text-green-400">{method.name}</div>
                <div className="text-xs text-green-600">{method.description}</div>
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.execution.marketHours}
              onChange={(e) => updateConfig('execution', 'marketHours', e.target.checked)}
              className="w-4 h-4 accent-green-500"
            />
            <span className="text-sm text-green-400">Restrict to market hours</span>
          </label>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.execution.allowShortSelling}
              onChange={(e) => updateConfig('execution', 'allowShortSelling', e.target.checked)}
              className="w-4 h-4 accent-green-500"
            />
            <span className="text-sm text-green-400">Allow short selling</span>
          </label>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Rebalance Frequency</label>
          <select
            value={config.execution.rebalanceFrequency}
            onChange={(e) => updateConfig('execution', 'rebalanceFrequency', e.target.value)}
            className="matrix-input w-full px-3 py-2 text-sm"
          >
            <option value="never">Never</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
      </div>
    );
  };

  const renderRiskConfig = () => {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Stop Loss (%)</label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="50"
              value={config.risk.stopLoss || ''}
              onChange={(e) => updateConfig('risk', 'stopLoss', e.target.value ? parseFloat(e.target.value) : undefined)}
              className="text-sm"
              placeholder="Optional"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Take Profit (%)</label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="100"
              value={config.risk.takeProfit || ''}
              onChange={(e) => updateConfig('risk', 'takeProfit', e.target.value ? parseFloat(e.target.value) : undefined)}
              className="text-sm"
              placeholder="Optional"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Max Drawdown (%)</label>
            <MatrixInput
              type="number"
              step="0.1"
              min="1"
              max="100"
              value={config.risk.maxDrawdown}
              onChange={(e) => updateConfig('risk', 'maxDrawdown', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Max Position Size (%)</label>
            <MatrixInput
              type="number"
              step="0.1"
              min="1"
              max="100"
              value={config.risk.maxPositionSize}
              onChange={(e) => updateConfig('risk', 'maxPositionSize', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Max Leverage (x)</label>
            <MatrixInput
              type="number"
              step="0.1"
              min="1"
              max="10"
              value={config.risk.maxLeverage}
              onChange={(e) => updateConfig('risk', 'maxLeverage', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Correlation Limit</label>
            <MatrixInput
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={config.risk.correlationLimit}
              onChange={(e) => updateConfig('risk', 'correlationLimit', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-green-400 mb-1">Sector Exposure Limit (%)</label>
          <MatrixInput
            type="number"
            step="1"
            min="0"
            max="100"
            value={config.risk.sectorExposureLimit}
            onChange={(e) => updateConfig('risk', 'sectorExposureLimit', parseFloat(e.target.value))}
            className="text-sm"
          />
        </div>
      </div>
    );
  };

  const renderPerformanceConfig = () => {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Benchmark Symbol</label>
            <MatrixInput
              value={config.performance.benchmark || ''}
              onChange={(e) => updateConfig('performance', 'benchmark', e.target.value)}
              className="text-sm"
              placeholder="e.g., SPY"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Risk-Free Rate (%)</label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="10"
              value={config.performance.riskFreeRate}
              onChange={(e) => updateConfig('performance', 'riskFreeRate', parseFloat(e.target.value))}
              className="text-sm"
            />
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold text-green-400 mb-3">Performance Metrics</h4>
          
          <div className="grid grid-cols-2 gap-2">
            {backtestMetrics.map(metric => (
              <label key={metric.key} className="flex items-center gap-2 cursor-pointer p-2 hover:bg-green-900/20 rounded">
                <input
                  type="checkbox"
                  checked={config.performance.metrics[metric.key as keyof typeof config.performance.metrics]}
                  onChange={(e) => updateMetrics(metric.key, e.target.checked)}
                  className="w-4 h-4 accent-green-500"
                />
                <div>
                  <div className="text-sm text-green-400">{metric.name}</div>
                  <div className="text-xs text-green-600">{metric.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderOptimizationConfig = () => {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.optimization.enabled}
              onChange={(e) => updateConfig('optimization', 'enabled', e.target.checked)}
              className="w-4 h-4 accent-green-500"
            />
            <span className="text-sm text-green-400">Enable Parameter Optimization</span>
          </label>
        </div>

        {config.optimization.enabled && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-green-400 mb-1">Optimization Objective</label>
                <select
                  value={config.optimization.objective}
                  onChange={(e) => updateConfig('optimization', 'objective', e.target.value)}
                  className="matrix-input w-full px-3 py-2 text-sm"
                >
                  {Object.entries(optimizationObjectives).map(([key, obj]) => (
                    <option key={key} value={key}>
                      {obj.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-green-400 mb-1">Max Workers</label>
                <MatrixInput
                  type="number"
                  min="1"
                  max="16"
                  value={config.optimization.maxWorkers}
                  onChange={(e) => updateConfig('optimization', 'maxWorkers', parseInt(e.target.value))}
                  className="text-sm"
                />
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-sm font-bold text-green-400">Parameters to Optimize</h4>
              
              <div className="matrix-card p-4 bg-black/30">
                <div className="space-y-2">
                  {config.optimization.parameters.length === 0 ? (
                    <div className="text-green-600 text-sm">No parameters selected for optimization</div>
                  ) : (
                    config.optimization.parameters.map((param, index) => (
                      <div key={param} className="flex items-center gap-4 p-2 bg-green-800/20 rounded">
                        <div className="flex-1">
                          <div className="text-sm text-green-400 font-mono">{param}</div>
                          {config.optimization.ranges[param] && (
                            <div className="text-xs text-green-600">
                              Range: {config.optimization.ranges[param].min} - {config.optimization.ranges[param].max} (step: {config.optimization.ranges[param].step})
                            </div>
                          )}
                        </div>
                        <button
                          onClick={() => {
                            const newParams = config.optimization.parameters.filter(p => p !== param);
                            const newRanges = { ...config.optimization.ranges };
                            delete newRanges[param];
                            updateConfig('optimization', 'parameters', newParams);
                            updateConfig('optimization', 'ranges', newRanges);
                          }}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
                
                <div className="mt-3">
                  <MatrixButton size="sm" variant="secondary">
                    <Plus className="w-3 h-3 mr-2" />
                    Add Parameter
                  </MatrixButton>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.optimization.parallel}
                  onChange={(e) => updateConfig('optimization', 'parallel', e.target.checked)}
                  className="w-4 h-4 accent-green-500"
                />
                <span className="text-sm text-green-400">Parallel processing</span>
              </label>
            </div>
          </>
        )}
      </div>
    );
  };

  const renderResults = () => {
    if (!config.results) {
      return (
        <div className="text-center py-12">
          <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50 text-green-400" />
          <h3 className="text-lg font-bold text-green-400 mb-2">No Results Available</h3>
          <p className="text-green-600 mb-4">Run a backtest to see results</p>
          {onRun && (
            <MatrixButton onClick={onRun}>
              <Play className="w-4 h-4 mr-2" />
              Run Backtest
            </MatrixButton>
          )}
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {/* Summary */}
        <MatrixCard className="p-6">
          <h3 className="text-lg font-bold text-green-400 mb-4">Backtest Summary</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(config.results.summary).map(([key, value]) => (
              <div key={key} className="matrix-card p-4">
                <div className="text-xs text-green-600 mb-1">
                  {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </div>
                <div className={`text-lg font-bold ${
                  typeof value === 'number' && value > 0 ? 'text-green-400' : 
                  typeof value === 'number' && value < 0 ? 'text-red-400' : 'text-green-400'
                }`}>
                  {typeof value === 'number' ? value.toFixed(2) : value}
                  {(key.includes('Return') || key.includes('Drawdown')) && '%'}
                </div>
              </div>
            ))}
          </div>
        </MatrixCard>

        {/* Optimization Results */}
        {config.results.optimization && (
          <MatrixCard className="p-6">
            <h3 className="text-lg font-bold text-green-400 mb-4">Optimization Results</h3>
            
            <div className="space-y-4">
              <div>
                <div className="text-sm text-green-600">Best Score</div>
                <div className="text-xl font-bold text-green-400">
                  {config.results.optimization.bestScore.toFixed(4)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-green-600 mb-2">Best Parameters</div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {Object.entries(config.results.optimization.bestParams).map(([param, value]) => (
                    <div key={param} className="matrix-card p-2">
                      <div className="text-xs text-green-600">{param}</div>
                      <div className="text-sm text-green-400 font-mono">{String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </MatrixCard>
        )}

        {/* Export Options */}
        <MatrixCard className="p-6">
          <h3 className="text-lg font-bold text-green-400 mb-4">Export Results</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {['html', 'pdf', 'excel', 'json'].map(format => (
              <MatrixButton
                key={format}
                size="sm"
                variant="secondary"
                className="justify-start"
              >
                <Download className="w-3 h-3 mr-2" />
                {format.toUpperCase()}
              </MatrixButton>
            ))}
          </div>
        </MatrixCard>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            BACKTEST CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure and run strategy backtests</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Status</div>
            <div className={`text-lg font-bold ${
              config.status === 'completed' ? 'text-green-400' :
              config.status === 'running' ? 'text-yellow-400' :
              config.status === 'failed' ? 'text-red-400' : 'text-gray-400'
            }`}>
              {config.status.toUpperCase()}
            </div>
          </div>
          
          <MatrixButton variant="secondary" onClick={() => updateConfig('data', 'startDate', '2023-01-01')}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </MatrixButton>
          
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Config
            </MatrixButton>
          )}
          
          {onRun && config.status !== 'running' && (
            <MatrixButton onClick={onRun}>
              <Play className="w-4 h-4 mr-2" />
              Run Backtest
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30 overflow-x-auto">
        {[
          { id: 'data', label: 'Data', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'execution', label: 'Execution', icon: <Zap className="w-4 h-4" /> },
          { id: 'risk', label: 'Risk', icon: <Shield className="w-4 h-4" /> },
          { id: 'performance', label: 'Performance', icon: <TrendingUp className="w-4 h-4" /> },
          { id: 'reports', label: 'Reports', icon: <Download className="w-4 h-4" /> },
          { id: 'optimization', label: 'Optimization', icon: <Target className="w-4 h-4" /> },
          { id: 'results', label: 'Results', icon: <BarChart3 className="w-4 h-4" /> }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
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
      <div className="flex-1 overflow-y-auto p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {activeTab === 'data' && renderDataConfig()}
            {activeTab === 'execution' && renderExecutionConfig()}
            {activeTab === 'risk' && renderRiskConfig()}
            {activeTab === 'performance' && renderPerformanceConfig()}
            {activeTab === 'optimization' && renderOptimizationConfig()}
            {activeTab === 'results' && renderResults()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};