import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useTradingStore } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixTable } from '@/components/ui/MatrixTable';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import {
  Plus,
  Minus,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  RefreshCw,
} from 'lucide-react';

export const Portfolio: React.FC = () => {
  const { portfolio, marketData } = useTradingStore();
  const [showClosedPositions, setShowClosedPositions] = useState(false);
  
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
  
  // Calculate portfolio allocation
  const totalPositionValue = portfolio.positions.reduce((sum, pos) => 
    sum + (pos.size * pos.currentPrice), 0
  );
  
  const allocation = portfolio.positions.map(pos => ({
    ...pos,
    allocation: (pos.size * pos.currentPrice / totalPositionValue) * 100,
    marketValue: pos.size * pos.currentPrice,
  })).sort((a, b) => b.allocation - a.allocation);
  
  // Calculate daily P&L
  const dailyPnL = portfolio.positions.reduce((sum, pos) => {
    // Mock daily change calculation (in real app, this would come from market data)
    const dailyChange = (Math.random() - 0.5) * 100;
    return sum + dailyChange;
  }, 0);
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400 mb-2">
            PORTFOLIO OVERVIEW
          </h1>
          <p className="text-green-600 font-mono">
            Real-time position tracking and performance analytics
          </p>
        </div>
        
        <div className="flex gap-3">
          <MatrixButton variant="secondary" size="sm">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </MatrixButton>
          <MatrixButton size="sm">
            <Plus className="w-4 h-4" />
            New Position
          </MatrixButton>
        </div>
      </motion.div>
      
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Value */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Total Value</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {formatCurrency(portfolio.totalValue)}
                </p>
                <p className="text-sm text-green-600 font-mono">
                  Available: {formatCurrency(portfolio.availableCash)}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-green-500" />
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* Total P&L */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Total P&L</p>
                <p className={`text-2xl font-bold matrix-text-glow ${
                  portfolio.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatCurrency(portfolio.totalPnL)}
                </p>
                <p className={`text-sm font-mono flex items-center gap-1 ${
                  portfolio.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolio.totalPnL >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  {formatPercentage(portfolio.totalPnLPercentage)}
                </p>
              </div>
              {portfolio.totalPnL >= 0 ? (
                <TrendingUp className="w-8 h-8 text-green-500" />
              ) : (
                <TrendingDown className="w-8 h-8 text-red-500" />
              )}
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* Daily P&L */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Daily P&L</p>
                <p className={`text-2xl font-bold matrix-text-glow ${
                  dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatCurrency(dailyPnL)}
                </p>
                <p className="text-sm text-green-600 font-mono">
                  {formatPercentage((dailyPnL / portfolio.totalValue) * 100)}
                </p>
              </div>
              {dailyPnL >= 0 ? (
                <TrendingUp className="w-8 h-8 text-green-500" />
              ) : (
                <TrendingDown className="w-8 h-8 text-red-500" />
              )}
            </div>
          </MatrixCard>
        </motion.div>
        
        {/* Active Positions */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
        >
          <MatrixCard glow>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 font-mono text-sm">Active Positions</p>
                <p className="text-2xl font-bold matrix-text-glow text-green-400">
                  {portfolio.positions.length}
                </p>
                <p className="text-sm text-green-600 font-mono">
                  {portfolio.positions.filter(p => p.side === 'long').length} Long,{' '}
                  {portfolio.positions.filter(p => p.side === 'short').length} Short
                </p>
              </div>
              <Target className="w-8 h-8 text-green-500" />
            </div>
          </MatrixCard>
        </motion.div>
      </div>
      
      {/* Portfolio Allocation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <MatrixCard 
          title="Portfolio Allocation" 
          subtitle="Position distribution by value"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Allocation Chart Placeholder */}
            <div className="matrix-terminal h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="text-green-400 font-mono text-lg mb-2">
                  [ ALLOCATION CHART ]
                </div>
                <div className="space-y-2">
                  {allocation.slice(0, 5).map((pos, index) => (
                    <div key={pos.id} className="flex justify-between items-center text-sm">
                      <span className="text-green-400 font-mono">{pos.symbol}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-green-900 rounded">
                          <div 
                            className="h-full bg-green-500 rounded"
                            style={{ width: `${Math.min(pos.allocation, 100)}%` }}
                          />
                        </div>
                        <span className="text-green-600 font-mono w-12 text-right">
                          {pos.allocation.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Allocation Summary */}
            <div className="space-y-4">
              <h4 className="font-bold text-green-400 font-mono">Top Holdings</h4>
              {allocation.slice(0, 5).map((pos, index) => (
                <motion.div
                  key={pos.id}
                  className="flex justify-between items-center p-3 rounded border border-green-800/30 bg-green-900/10"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + index * 0.1 }}
                >
                  <div>
                    <p className="font-mono text-green-400">{pos.symbol}</p>
                    <p className="text-xs text-green-600">
                      {pos.size} @ {formatCurrency(pos.entryPrice)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-green-400">
                      {formatCurrency(pos.marketValue)}
                    </p>
                    <p className="text-xs text-green-600">
                      {pos.allocation.toFixed(1)}%
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </MatrixCard>
      </motion.div>
      
      {/* Positions Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <MatrixCard 
          title="Active Positions" 
          subtitle="Current market positions and P&L"
        >
          <MatrixTable
            data={portfolio.positions}
            columns={[
              {
                key: 'symbol',
                title: 'Symbol',
                width: '15%',
                render: (value, row) => (
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-green-400">{value}</span>
                    <StatusIndicator 
                      status={row.side === 'long' ? 'positive' : 'negative'} 
                      size="sm"
                    />
                  </div>
                ),
              },
              {
                key: 'size',
                title: 'Size',
                width: '15%',
                align: 'right',
                render: (value) => (
                  <span className="font-mono text-green-400">{value.toFixed(4)}</span>
                ),
              },
              {
                key: 'entryPrice',
                title: 'Entry Price',
                width: '15%',
                align: 'right',
                render: (value) => formatCurrency(value),
              },
              {
                key: 'currentPrice',
                title: 'Current Price',
                width: '15%',
                align: 'right',
                render: (value, row) => {
                  const currentMarketData = marketData[row.symbol];
                  const price = currentMarketData?.price || value;
                  return (
                    <span className="font-mono text-green-400 data-updated">
                      {formatCurrency(price)}
                    </span>
                  );
                },
              },
              {
                key: 'pnl',
                title: 'P&L',
                width: '15%',
                align: 'right',
                render: (value, row) => (
                  <div>
                    <div className={`font-mono ${value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatCurrency(value)}
                    </div>
                    <div className={`text-xs ${row.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercentage(row.pnlPercentage)}
                    </div>
                  </div>
                ),
              },
              {
                key: 'timestamp',
                title: 'Open Time',
                width: '15%',
                render: (value) => (
                  <span className="text-green-600 font-mono text-xs">
                    {new Date(value).toLocaleTimeString()}
                  </span>
                ),
              },
              {
                key: 'broker',
                title: 'Broker',
                width: '10%',
                render: (value) => (
                  <span className="text-green-500 font-mono text-xs">{value}</span>
                ),
              },
            ]}
            emptyMessage="No active positions"
          />
        </MatrixCard>
      </motion.div>
    </div>
  );
};