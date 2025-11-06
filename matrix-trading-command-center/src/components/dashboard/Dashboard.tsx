import React from 'react';
import { motion } from 'framer-motion';
import { useTradingStore, type Strategy } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { MatrixTable } from '@/components/ui/MatrixTable';
import { MarketTicker } from '@/components/market/MarketTicker';
import { BrokerStatus } from '@/components/trading/BrokerStatus';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Activity,
  AlertTriangle,
  Clock,
  BarChart3,
  Brain,
  Shield,
  Zap,
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const { 
    portfolio, 
    marketTicker, 
    riskMetrics, 
    brokerConnections,
    strategies,
    alerts 
  } = useTradingStore();
  
  // Calculate stats
  const activePositions = portfolio.positions.length;
  const pendingOrders = portfolio.orders.filter(o => o.status === 'pending').length;
  const runningStrategies = strategies.filter(s => s.status === 'running').length;
  const connectedBrokers = brokerConnections.filter(b => b.status === 'connected').length;
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };
  
  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold matrix-text-glow text-green-400 mb-2">
          MATRIX TRADING DASHBOARD
        </h1>
        <p className="text-green-600 font-mono">
          Real-time Trading Orchestrator Command Center
        </p>
      </motion.div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Portfolio Value */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Portfolio Value</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {formatCurrency(portfolio.totalValue)}
                </p>
                <p className={`text-sm font-mono flex items-center gap-1 ${
                  portfolio.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolio.totalPnL >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  {formatCurrency(portfolio.totalPnL)} ({formatPercentage(portfolio.totalPnLPercentage)})
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-green-500" />
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* Active Positions */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Active Positions</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {activePositions}
                </p>
                <p className="text-sm text-green-600 font-mono flex items-center gap-1">
                  <Activity className="w-4 h-4" />
                  {pendingOrders} Pending Orders
                </p>
              </div>
              <Target className="w-8 h-8 text-green-500" />
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* Risk Metrics */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Risk Level</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {riskMetrics.riskUtilization}%
                </p>
                <p className="text-sm text-green-600 font-mono flex items-center gap-1">
                  <AlertTriangle className="w-4 h-4" />
                  VaR: {formatCurrency(riskMetrics.var95)}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-500" />
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* System Status */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">System Status</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {connectedBrokers}/{brokerConnections.length}
                </p>
                <p className="text-sm text-green-600 font-mono flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {runningStrategies} Strategies Running
                </p>
              </div>
              <Activity className="w-8 h-8 text-green-500" />
            </div>
          </MatrixCard>
        </motion.div>
      </div>
      
      {/* Market Ticker Section */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <MatrixCard title="Live Market Data" subtitle="Real-time trading feeds and market conditions">
          <MarketTicker />
        </MatrixCard>
      </motion.div>

      {/* Key Metrics Row */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        <MatrixCard title="AI Insights" glow>
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-400" />
            <div>
              <div className="text-xl font-bold matrix-text-glow">2</div>
              <div className="text-sm text-green-600">Active recommendations</div>
            </div>
          </div>
        </MatrixCard>
        
        <MatrixCard title="Risk Score" glow>
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8 text-yellow-400" />
            <div>
              <div className="text-xl font-bold matrix-text-glow">
                {riskMetrics.riskUtilization < 70 ? 'LOW' : 
                 riskMetrics.riskUtilization < 85 ? 'MED' : 'HIGH'}
              </div>
              <div className="text-sm text-green-600">
                {riskMetrics.riskUtilization}% utilized
              </div>
            </div>
          </div>
        </MatrixCard>
        
        <MatrixCard title="System Performance" glow>
          <div className="flex items-center gap-3">
            <Zap className="w-8 h-8 text-green-400" />
            <div>
              <div className="text-xl font-bold matrix-text-glow">OPTIMAL</div>
              <div className="text-sm text-green-600">All systems online</div>
            </div>
          </div>
        </MatrixCard>
      </motion.div>

      {/* Secondary Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Broker Status */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
        >
          <BrokerStatus />
        </motion.div>

        {/* Recent Alerts */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.7 }}
        >
          <MatrixCard title="Recent Alerts" subtitle="System notifications and warnings">
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {alerts.slice(0, 8).map((alert, index) => (
                <motion.div
                  key={alert.id}
                  className="flex items-start gap-3 p-3 rounded border border-green-800/30 bg-green-900/10"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 + index * 0.1 }}
                >
                  <StatusIndicator
                    status={
                      alert.type === 'error' ? 'error' :
                      alert.type === 'warning' ? 'warning' :
                      alert.type === 'success' ? 'positive' : 'neutral'
                    }
                    size="sm"
                  />
                  <div className="flex-1">
                    <p className="font-mono text-sm text-green-400">{alert.title}</p>
                    <p className="text-xs text-green-600 mt-1">{alert.message}</p>
                    <p className="text-xs text-green-700 mt-1">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>
              ))}
              
              {alerts.length === 0 && (
                <div className="text-center py-8">
                  <p className="text-green-600 font-mono">No recent alerts</p>
                </div>
              )}
            </div>
          </MatrixCard>
        </motion.div>
      </div>
      
      {/* Active Strategies */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <MatrixCard title="Active Strategies" subtitle="Trading strategy performance">
          <MatrixTable
            data={strategies.filter(s => s.status === 'running')}
            columns={[
              {
                key: 'name',
                title: 'Strategy',
                width: '30%',
              },
              {
                key: 'status',
                title: 'Status',
                width: '20%',
                render: (value) => (
                  <StatusIndicator
                    status={value === 'running' ? 'online' : 'offline'}
                    pulse={value === 'running'}
                  >
                    {value}
                  </StatusIndicator>
                ),
              },
              {
                key: 'performance' as keyof Strategy,
                title: 'P&L',
                width: '20%',
                align: 'right',
                render: (value: Strategy['performance']) => (
                  <span className={value.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {formatCurrency(value.totalPnL)}
                  </span>
                ),
              },
              {
                key: 'performance' as keyof Strategy,
                title: 'Win Rate',
                width: '15%',
                align: 'right',
                render: (value: Strategy['performance']) => `${value.winRate.toFixed(1)}%`,
              },
              {
                key: 'performance' as keyof Strategy,
                title: 'Trades',
                width: '15%',
                align: 'right',
                render: (value: Strategy['performance']) => value.totalTrades,
              },
            ]}
            emptyMessage="No active strategies"
          />
        </MatrixCard>
      </motion.div>
    </div>
  );
};