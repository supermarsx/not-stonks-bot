import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Network, 
  Settings, 
  DollarSign, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Wifi,
  WifiOff,
  Zap,
  Shield,
  Plus,
  Trash2,
  Save,
  RotateCcw,
  BarChart3,
  Clock,
  Target
} from 'lucide-react';

interface Broker {
  id: string;
  name: string;
  type: 'traditional' | 'robo' | 'crypto' | 'forex' | 'options';
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  latency: number; // milliseconds
  capabilities: string[];
  fees: {
    trading: number; // per trade
    percentage: number; // percentage of trade value
    account_minimum: number;
    monthly_fee?: number;
  };
  limits: {
    max_position_size: number;
    max_daily_volume: number;
    max_trades_per_day: number;
    supported_order_types: string[];
  };
  performance: {
    uptime: number; // percentage
    fill_rate: number; // percentage
    slippage: number; // average slippage
    reliability_score: number; // 1-10
  };
  allocation: {
    target_percentage: number;
    current_percentage: number;
    min_allocation: number;
    max_allocation: number;
    priority: number; // 1-10
  };
  strategies: {
    [strategyName: string]: {
      enabled: boolean;
      allocation: number;
      max_trades: number;
    };
  };
  features: {
    real_time_data: boolean;
    extended_hours: boolean;
    fractional_shares: boolean;
    options_trading: boolean;
    margin_trading: boolean;
    api_access: boolean;
  };
}

interface BrokerRoutingConfig {
  id: string;
  name: string;
  description: string;
  routing_strategy: 'round_robin' | 'performance_based' | 'cost_optimized' | 'capacity_based' | 'custom';
  brokers: Broker[];
  global_settings: {
    auto_failover: boolean;
    health_monitoring: boolean;
    performance_tracking: boolean;
    cost_optimization: boolean;
    load_balancing: boolean;
  };
  risk_controls: {
    max_broker_exposure: number;
    min_broker_balance: number;
    stop_loss_allocation: boolean;
    emergency_failover: boolean;
  };
}

interface BrokerRoutingProps {
  config: BrokerRoutingConfig;
  onChange: (config: BrokerRoutingConfig) => void;
  onSave?: () => void;
}

export const BrokerRouting: React.FC<BrokerRoutingProps> = ({
  config,
  onChange,
  onSave
}) => {
  const [selectedBroker, setSelectedBroker] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'brokers' | 'routing' | 'performance' | 'settings'>('overview');

  const brokerTypes = {
    traditional: { name: 'Traditional', color: 'border-blue-500', icon: <BarChart3 className="w-4 h-4" /> },
    robo: { name: 'Robo-Advisor', color: 'border-green-500', icon: <Target className="w-4 h-4" /> },
    crypto: { name: 'Cryptocurrency', color: 'border-orange-500', icon: <Zap className="w-4 h-4" /> },
    forex: { name: 'Forex', color: 'border-yellow-500', icon: <DollarSign className="w-4 h-4" /> },
    options: { name: 'Options', color: 'border-purple-500', icon: <TrendingUp className="w-4 h-4" /> }
  };

  const routingStrategies = {
    round_robin: {
      name: 'Round Robin',
      description: 'Distribute trades evenly across all brokers',
      icon: <RotateCcw className="w-4 h-4" />
    },
    performance_based: {
      name: 'Performance Based',
      description: 'Route trades to best performing brokers',
      icon: <TrendingUp className="w-4 h-4" />
    },
    cost_optimized: {
      name: 'Cost Optimized',
      description: 'Minimize trading costs and fees',
      icon: <DollarSign className="w-4 h-4" />
    },
    capacity_based: {
      name: 'Capacity Based',
      description: 'Route based on available capacity',
      icon: <BarChart3 className="w-4 h-4" />
    },
    custom: {
      name: 'Custom Rules',
      description: 'Use custom routing logic',
      icon: <Settings className="w-4 h-4" />
    }
  };

  const defaultBrokers: Omit<Broker, 'id'>[] = [
    {
      name: 'Alpaca Markets',
      type: 'traditional',
      status: 'connected',
      latency: 45,
      capabilities: ['Stocks', 'ETFs', 'Extended Hours'],
      fees: { trading: 0, percentage: 0, account_minimum: 0 },
      limits: { max_position_size: 1000000, max_daily_volume: 10000000, max_trades_per_day: 10000, supported_order_types: ['market', 'limit', 'stop'] },
      performance: { uptime: 99.8, fill_rate: 98.5, slippage: 0.02, reliability_score: 9 },
      allocation: { target_percentage: 40, current_percentage: 38, min_allocation: 10, max_allocation: 80, priority: 1 },
      strategies: { mean_reversion: { enabled: true, allocation: 100, max_trades: 50 } },
      features: { real_time_data: true, extended_hours: true, fractional_shares: true, options_trading: false, margin_trading: true, api_access: true }
    },
    {
      name: 'Interactive Brokers',
      type: 'traditional',
      status: 'connected',
      latency: 120,
      capabilities: ['Stocks', 'Options', 'Futures', 'Forex'],
      fees: { trading: 0.005, percentage: 0.001, account_minimum: 1000, monthly_fee: 10 },
      limits: { max_position_size: 10000000, max_daily_volume: 100000000, max_trades_per_day: 50000, supported_order_types: ['market', 'limit', 'stop', 'stop_limit'] },
      performance: { uptime: 99.9, fill_rate: 99.2, slippage: 0.01, reliability_score: 9.5 },
      allocation: { target_percentage: 30, current_percentage: 32, min_allocation: 20, max_allocation: 60, priority: 2 },
      strategies: { momentum: { enabled: true, allocation: 100, max_trades: 100 }, pairs_trading: { enabled: true, allocation: 100, max_trades: 20 } },
      features: { real_time_data: true, extended_hours: true, fractional_shares: true, options_trading: true, margin_trading: true, api_access: true }
    },
    {
      name: 'Binance',
      type: 'crypto',
      status: 'connected',
      latency: 25,
      capabilities: ['Crypto Spot', 'Crypto Futures', 'DeFi'],
      fees: { trading: 0.001, percentage: 0.001, account_minimum: 0 },
      limits: { max_position_size: 1000000, max_daily_volume: 50000000, max_trades_per_day: 100000, supported_order_types: ['market', 'limit'] },
      performance: { uptime: 99.5, fill_rate: 97.8, slippage: 0.05, reliability_score: 8 },
      allocation: { target_percentage: 20, current_percentage: 18, min_allocation: 5, max_allocation: 40, priority: 3 },
      strategies: { crypto_momentum: { enabled: true, allocation: 100, max_trades: 200 } },
      features: { real_time_data: true, extended_hours: false, fractional_shares: false, options_trading: false, margin_trading: true, api_access: true }
    },
    {
      name: 'E*TRADE',
      type: 'traditional',
      status: 'connected',
      latency: 85,
      capabilities: ['Stocks', 'ETFs', 'Options'],
      fees: { trading: 0, percentage: 0, account_minimum: 500 },
      limits: { max_position_size: 500000, max_daily_volume: 5000000, max_trades_per_day: 5000, supported_order_types: ['market', 'limit', 'stop'] },
      performance: { uptime: 99.2, fill_rate: 96.5, slippage: 0.03, reliability_score: 7 },
      allocation: { target_percentage: 10, current_percentage: 12, min_allocation: 5, max_allocation: 30, priority: 4 },
      strategies: { long_term: { enabled: true, allocation: 100, max_trades: 25 } },
      features: { real_time_data: true, extended_hours: true, fractional_shares: true, options_trading: true, margin_trading: true, api_access: true }
    }
  ];

  const addBroker = (type: keyof typeof brokerTypes) => {
    const brokerType = brokerTypes[type];
    const newBroker: Broker = {
      id: `broker_${Date.now()}`,
      name: `${brokerType.name} Broker`,
      type,
      status: 'disconnected',
      latency: 100,
      capabilities: [],
      fees: { trading: 0, percentage: 0, account_minimum: 0 },
      limits: { max_position_size: 100000, max_daily_volume: 1000000, max_trades_per_day: 1000, supported_order_types: ['market', 'limit'] },
      performance: { uptime: 0, fill_rate: 0, slippage: 0, reliability_score: 5 },
      allocation: { target_percentage: 0, current_percentage: 0, min_allocation: 0, max_allocation: 100, priority: config.brokers.length + 1 },
      strategies: {},
      features: { real_time_data: false, extended_hours: false, fractional_shares: false, options_trading: false, margin_trading: false, api_access: false }
    };

    onChange({
      ...config,
      brokers: [...config.brokers, newBroker]
    });
    setSelectedBroker(newBroker.id);
  };

  const updateBroker = (brokerId: string, updates: Partial<Broker>) => {
    onChange({
      ...config,
      brokers: config.brokers.map(broker =>
        broker.id === brokerId ? { ...broker, ...updates } : broker
      )
    });
  };

  const removeBroker = (brokerId: string) => {
    onChange({
      ...config,
      brokers: config.brokers.filter(broker => broker.id !== brokerId)
    });
    if (selectedBroker === brokerId) {
      setSelectedBroker(null);
    }
  };

  const getBrokerTypeInfo = (type: Broker['type']) => {
    return brokerTypes[type];
  };

  const getStatusIcon = (status: Broker['status']) => {
    switch (status) {
      case 'connected':
        return <Wifi className="w-4 h-4 text-green-400" />;
      case 'connecting':
        return <RotateCcw className="w-4 h-4 text-yellow-400 animate-spin" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default:
        return <WifiOff className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: Broker['status']) => {
    switch (status) {
      case 'connected':
        return 'text-green-400';
      case 'connecting':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const renderBrokerConfig = (broker: Broker) => {
    const typeInfo = getBrokerTypeInfo(broker.type);

    return (
      <div className="space-y-6">
        {/* Basic Information */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Basic Information</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Broker Name</label>
              <MatrixInput
                value={broker.name}
                onChange={(e) => updateBroker(broker.id, { name: e.target.value })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Broker Type</label>
              <select
                value={broker.type}
                onChange={(e) => updateBroker(broker.id, { type: e.target.value as Broker['type'] })}
                className="matrix-input w-full px-3 py-2 text-sm"
              >
                {Object.entries(brokerTypes).map(([key, info]) => (
                  <option key={key} value={key}>
                    {info.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Connection Status</label>
              <select
                value={broker.status}
                onChange={(e) => updateBroker(broker.id, { status: e.target.value as Broker['status'] })}
                className="matrix-input w-full px-3 py-2 text-sm"
              >
                <option value="connected">Connected</option>
                <option value="disconnected">Disconnected</option>
                <option value="connecting">Connecting</option>
                <option value="error">Error</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Latency (ms)</label>
              <MatrixInput
                type="number"
                min="0"
                value={broker.latency}
                onChange={(e) => updateBroker(broker.id, { latency: parseInt(e.target.value) })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        {/* Allocation Settings */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Allocation Settings</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Target Allocation (%)</label>
              <MatrixInput
                type="number"
                min="0"
                max="100"
                step="0.1"
                value={broker.allocation.target_percentage}
                onChange={(e) => updateBroker(broker.id, {
                  allocation: { ...broker.allocation, target_percentage: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Priority (1-10)</label>
              <MatrixInput
                type="number"
                min="1"
                max="10"
                value={broker.allocation.priority}
                onChange={(e) => updateBroker(broker.id, {
                  allocation: { ...broker.allocation, priority: parseInt(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Min Allocation (%)</label>
              <MatrixInput
                type="number"
                min="0"
                max="100"
                value={broker.allocation.min_allocation}
                onChange={(e) => updateBroker(broker.id, {
                  allocation: { ...broker.allocation, min_allocation: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Max Allocation (%)</label>
              <MatrixInput
                type="number"
                min="0"
                max="100"
                value={broker.allocation.max_allocation}
                onChange={(e) => updateBroker(broker.id, {
                  allocation: { ...broker.allocation, max_allocation: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Performance Metrics</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Uptime (%)</label>
              <MatrixInput
                type="number"
                min="0"
                max="100"
                step="0.1"
                value={broker.performance.uptime}
                onChange={(e) => updateBroker(broker.id, {
                  performance: { ...broker.performance, uptime: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Fill Rate (%)</label>
              <MatrixInput
                type="number"
                min="0"
                max="100"
                step="0.1"
                value={broker.performance.fill_rate}
                onChange={(e) => updateBroker(broker.id, {
                  performance: { ...broker.performance, fill_rate: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Avg Slippage (%)</label>
              <MatrixInput
                type="number"
                step="0.001"
                value={broker.performance.slippage}
                onChange={(e) => updateBroker(broker.id, {
                  performance: { ...broker.performance, slippage: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Reliability Score (1-10)</label>
              <MatrixInput
                type="number"
                min="1"
                max="10"
                step="0.1"
                value={broker.performance.reliability_score}
                onChange={(e) => updateBroker(broker.id, {
                  performance: { ...broker.performance, reliability_score: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        {/* Fee Structure */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Fee Structure</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Per Trade Fee ($)</label>
              <MatrixInput
                type="number"
                step="0.001"
                value={broker.fees.trading}
                onChange={(e) => updateBroker(broker.id, {
                  fees: { ...broker.fees, trading: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Percentage Fee (%)</label>
              <MatrixInput
                type="number"
                step="0.001"
                value={broker.fees.percentage}
                onChange={(e) => updateBroker(broker.id, {
                  fees: { ...broker.fees, percentage: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-green-400 mb-1">Account Minimum ($)</label>
              <MatrixInput
                type="number"
                value={broker.fees.account_minimum}
                onChange={(e) => updateBroker(broker.id, {
                  fees: { ...broker.fees, account_minimum: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-green-400 mb-1">Monthly Fee ($)</label>
              <MatrixInput
                type="number"
                value={broker.fees.monthly_fee || 0}
                onChange={(e) => updateBroker(broker.id, {
                  fees: { ...broker.fees, monthly_fee: parseFloat(e.target.value) }
                })}
                className="text-sm"
              />
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="space-y-4">
          <h4 className="text-sm font-bold text-green-400">Features</h4>
          
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(broker.features).map(([feature, enabled]) => (
              <label key={feature} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => updateBroker(broker.id, {
                    features: { ...broker.features, [feature]: e.target.checked }
                  })}
                  className="w-4 h-4 accent-green-500"
                />
                <span className="text-sm text-green-400 capitalize">
                  {feature.replace(/_/g, ' ')}
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const totalAllocation = config.brokers.reduce((sum, broker) => sum + broker.allocation.target_percentage, 0);
  const avgLatency = config.brokers.filter(b => b.status === 'connected').reduce((sum, broker) => sum + broker.latency, 0) / 
                    Math.max(1, config.brokers.filter(b => b.status === 'connected').length);
  const totalPerformance = config.brokers.reduce((sum, broker) => sum + broker.performance.reliability_score, 0) / 
                          Math.max(1, config.brokers.length);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            BROKER ROUTING CONFIGURATION
          </h1>
          <p className="text-green-600 text-sm">Configure broker allocation and routing strategies</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Total Brokers</div>
            <div className="text-lg font-bold text-green-400">
              {config.brokers.length}
            </div>
          </div>
          
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Connected</div>
            <div className="text-lg font-bold text-green-400">
              {config.brokers.filter(b => b.status === 'connected').length}
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
          { id: 'brokers', label: 'Brokers', icon: <Network className="w-4 h-4" /> },
          { id: 'routing', label: 'Routing Strategy', icon: <Settings className="w-4 h-4" /> },
          { id: 'performance', label: 'Performance', icon: <TrendingUp className="w-4 h-4" /> },
          { id: 'settings', label: 'Settings', icon: <Shield className="w-4 h-4" /> }
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
        {/* Left Panel - Broker List */}
        <div className="w-80 border-r border-green-800/30 flex flex-col">
          <div className="p-4 border-b border-green-800/30">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-green-400">BROKERS</h3>
              <MatrixButton 
                size="sm" 
                onClick={() => addBroker('traditional')}
                className="flex items-center gap-1"
              >
                <Plus className="w-3 h-3" />
                Add
              </MatrixButton>
            </div>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {config.brokers.map((broker) => {
                const typeInfo = getBrokerTypeInfo(broker.type);
                const allocationDiff = broker.allocation.current_percentage - broker.allocation.target_percentage;
                
                return (
                  <MatrixCard
                    key={broker.id}
                    className={`p-3 cursor-pointer transition-all ${
                      selectedBroker === broker.id 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-green-800/30 hover:border-green-700'
                    }`}
                    onClick={() => setSelectedBroker(broker.id)}
                  >
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={typeInfo.color}>
                            {typeInfo.icon}
                          </div>
                          <div>
                            <div className="text-sm font-bold text-green-400">
                              {broker.name}
                            </div>
                            <div className="text-xs text-green-600">
                              {typeInfo.name}
                            </div>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className="flex items-center gap-1">
                            {getStatusIcon(broker.status)}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-xs">
                        <div className="text-green-600">
                          {broker.allocation.target_percentage}% allocated
                        </div>
                        <div className={`${
                          Math.abs(allocationDiff) > 5 ? 'text-yellow-400' : 'text-green-600'
                        }`}>
                          {allocationDiff > 0 ? '+' : ''}{allocationDiff.toFixed(1)}%
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="text-xs text-green-600">
                          Latency: {broker.latency}ms
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              removeBroker(broker.id);
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

          {/* Quick Add */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">QUICK ADD</h3>
            <div className="grid grid-cols-1 gap-2">
              {defaultBrokers.map((defaultBroker, index) => {
                const typeInfo = getBrokerTypeInfo(defaultBroker.type);
                return (
                  <MatrixCard
                    key={index}
                    className="p-3 cursor-pointer hover:bg-green-900/20 transition-colors"
                    onClick={() => {
                      const newBroker: Broker = {
                        ...defaultBroker,
                        id: `broker_${Date.now()}_${index}`
                      };
                      onChange({
                        ...config,
                        brokers: [...config.brokers, newBroker]
                      });
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <div className={typeInfo.color}>
                        {typeInfo.icon}
                      </div>
                      <div className="flex-1">
                        <div className="text-xs font-bold text-green-400">
                          {defaultBroker.name}
                        </div>
                        <div className="text-xs text-green-600">
                          {typeInfo.name}
                        </div>
                      </div>
                    </div>
                  </MatrixCard>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Panel - Configuration */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {selectedBroker ? (
              <motion.div
                key="config"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                {(() => {
                  const broker = config.brokers.find(b => b.id === selectedBroker);
                  if (!broker) return null;
                  
                  const typeInfo = getBrokerTypeInfo(broker.type);
                  return (
                    <MatrixCard className="p-6">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-6">
                          <div className={`${typeInfo.color} p-2 rounded`}>
                            {typeInfo.icon}
                          </div>
                          <div>
                            <h2 className="text-lg font-bold text-green-400">
                              {broker.name}
                            </h2>
                            <p className="text-sm text-green-600">
                              {typeInfo.name} Broker Configuration
                            </p>
                          </div>
                        </div>

                        {renderBrokerConfig(broker)}
                      </div>
                    </MatrixCard>
                  );
                })()}
              </motion.div>
            ) : (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  {/* System Overview */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      System Overview
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-xs text-green-600">Total Brokers</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.brokers.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Connected</div>
                        <div className="text-2xl font-bold text-green-400">
                          {config.brokers.filter(b => b.status === 'connected').length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Avg Latency</div>
                        <div className="text-2xl font-bold text-yellow-400">
                          {avgLatency.toFixed(0)}ms
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Reliability Score</div>
                        <div className="text-2xl font-bold text-green-400">
                          {totalPerformance.toFixed(1)}/10
                        </div>
                      </div>
                    </div>
                  </MatrixCard>

                  {/* Allocation Overview */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Allocation Overview
                    </h3>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-green-600">Total Allocation</span>
                        <span className={`text-sm font-bold ${
                          Math.abs(totalAllocation - 100) < 1 ? 'text-green-400' : 'text-yellow-400'
                        }`}>
                          {totalAllocation.toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="w-full bg-gray-700 rounded-full h-3">
                        <div 
                          className={`h-3 rounded-full transition-all ${
                            Math.abs(totalAllocation - 100) < 1 ? 'bg-green-500' : 'bg-yellow-500'
                          }`}
                          style={{ width: `${Math.min(100, totalAllocation)}%` }}
                        />
                      </div>
                      
                      {Math.abs(totalAllocation - 100) >= 1 && (
                        <div className="text-xs text-yellow-400">
                          Allocation should sum to 100%
                        </div>
                      )}
                    </div>
                  </MatrixCard>

                  {/* Broker Performance */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Broker Performance
                    </h3>
                    
                    <div className="space-y-3">
                      {config.brokers.map((broker) => {
                        const typeInfo = getBrokerTypeInfo(broker.type);
                        return (
                          <div key={broker.id} className="matrix-card p-4">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <div className={typeInfo.color}>
                                  {typeInfo.icon}
                                </div>
                                <span className="text-sm font-bold text-green-400">
                                  {broker.name}
                                </span>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                {getStatusIcon(broker.status)}
                                <span className={`text-xs ${getStatusColor(broker.status)}`}>
                                  {broker.status.toUpperCase()}
                                </span>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-4 gap-4 text-xs">
                              <div>
                                <div className="text-green-600">Latency</div>
                                <div className="text-green-400 font-bold">{broker.latency}ms</div>
                              </div>
                              <div>
                                <div className="text-green-600">Uptime</div>
                                <div className="text-green-400 font-bold">{broker.performance.uptime.toFixed(1)}%</div>
                              </div>
                              <div>
                                <div className="text-green-600">Fill Rate</div>
                                <div className="text-green-400 font-bold">{broker.performance.fill_rate.toFixed(1)}%</div>
                              </div>
                              <div>
                                <div className="text-green-600">Reliability</div>
                                <div className="text-green-400 font-bold">{broker.performance.reliability_score}/10</div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
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