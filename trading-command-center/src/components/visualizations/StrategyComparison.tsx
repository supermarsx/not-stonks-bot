import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { TrendingUp, TrendingDown, Target, Activity, Star, BarChart3 } from 'lucide-react';

interface StrategyPerformance {
  id: string;
  name: string;
  color: string;
  totalReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  avgTrade: number;
  profitFactor: number;
  performance: {
    date: string;
    value: number;
    benchmark: number;
    trades: number;
  }[];
  monthlyReturns: number[];
  returns: number[];
}

interface StrategyComparisonProps {
  strategies: StrategyPerformance[];
  title?: string;
  className?: string;
  benchmarkData?: number[];
  showScatterPlot?: boolean;
  showCorrelation?: boolean;
  onStrategySelect?: (strategy: StrategyPerformance) => void;
  onStrategyRemove?: (strategyId: string) => void;
}

export const StrategyComparison: React.FC<StrategyComparisonProps> = ({
  strategies,
  title = "Strategy Performance Comparison",
  className = "",
  benchmarkData = [],
  showScatterPlot = true,
  showCorrelation = true,
  onStrategySelect,
  onStrategyRemove,
}) => {
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['totalReturn', 'sharpeRatio', 'maxDrawdown']);
  const [chartType, setChartType] = useState<'performance' | 'returns' | 'scatter' | 'correlation'>('performance');
  const [timeframe, setTimeframe] = useState<'1M' | '3M' | '6M' | '1Y' | 'ALL'>('1Y');
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(
    strategies.slice(0, 3).map(s => s.id) // Select first 3 strategies by default
  );

  // Filter strategies based on timeframe
  const filteredStrategies = useMemo(() => {
    return strategies.filter(strategy => selectedStrategies.includes(strategy.id));
  }, [strategies, selectedStrategies]);

  // Calculate portfolio-level metrics
  const portfolioMetrics = useMemo(() => {
    if (filteredStrategies.length === 0) return null;

    const totalValue = filteredStrategies.reduce((sum, s) => sum + s.totalReturn, 0) / filteredStrategies.length;
    const totalVolatility = filteredStrategies.reduce((sum, s) => sum + s.volatility, 0) / filteredStrategies.length;
    const totalSharpe = filteredStrategies.reduce((sum, s) => sum + s.sharpeRatio, 0) / filteredStrategies.length;
    const totalDrawdown = Math.max(...filteredStrategies.map(s => s.maxDrawdown));
    const totalWinRate = filteredStrategies.reduce((sum, s) => sum + s.winRate, 0) / filteredStrategies.length;

    return {
      totalReturn: totalValue,
      volatility: totalVolatility,
      sharpeRatio: totalSharpe,
      maxDrawdown: totalDrawdown,
      winRate: totalWinRate,
    };
  }, [filteredStrategies]);

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!filteredStrategies.length || !filteredStrategies[0].performance.length) return [];

    return filteredStrategies[0].performance.map((_, index) => {
      const dataPoint: any = { date: filteredStrategies[0].performance[index].date };
      
      filteredStrategies.forEach(strategy => {
        dataPoint[strategy.name] = strategy.performance[index]?.value || 0;
      });
      
      return dataPoint;
    });
  }, [filteredStrategies]);

  // Risk-Return scatter data
  const riskReturnData = useMemo(() => {
    return filteredStrategies.map(strategy => ({
      name: strategy.name,
      risk: strategy.volatility,
      return: strategy.totalReturn,
      sharpe: strategy.sharpeRatio,
      drawdown: strategy.maxDrawdown,
    }));
  }, [filteredStrategies]);

  // Correlation matrix calculation (simplified)
  const correlationMatrix = useMemo(() => {
    if (filteredStrategies.length < 2) return [];

    const correlations = [];
    for (let i = 0; i < filteredStrategies.length; i++) {
      for (let j = 0; j < filteredStrategies.length; j++) {
        if (i === j) {
          correlations.push(1);
        } else {
          // Simplified correlation calculation
          const returns1 = filteredStrategies[i].returns;
          const returns2 = filteredStrategies[j].returns;
          const correlation = calculateCorrelation(returns1, returns2);
          correlations.push(correlation);
        }
      }
    }
    return correlations;
  }, [filteredStrategies]);

  // Helper function to calculate correlation
  function calculateCorrelation(arr1: number[], arr2: number[]): number {
    const n = Math.min(arr1.length, arr2.length);
    const mean1 = arr1.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    const mean2 = arr2.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    
    let numerator = 0;
    let sum1Sq = 0;
    let sum2Sq = 0;
    
    for (let i = 0; i < n; i++) {
      const diff1 = arr1[i] - mean1;
      const diff2 = arr2[i] - mean2;
      numerator += diff1 * diff2;
      sum1Sq += diff1 * diff1;
      sum2Sq += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(sum1Sq * sum2Sq);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Format currency/percentage
  const formatValue = (value: number, type: 'currency' | 'percentage' | 'number' = 'percentage') => {
    switch (type) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
        }).format(value);
      case 'percentage':
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
      default:
        return value.toFixed(2);
    }
  };

  // Strategy selection
  const toggleStrategy = (strategyId: string) => {
    setSelectedStrategies(prev => {
      if (prev.includes(strategyId)) {
        return prev.filter(id => id !== strategyId);
      } else {
        return [...prev, strategyId].slice(0, 6); // Max 6 strategies
      }
    });
  };

  // Render performance chart
  const renderPerformanceChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
        <XAxis 
          dataKey="date" 
          stroke="#00ff00"
          tick={{ fontSize: 12, fontFamily: 'monospace' }}
        />
        <YAxis 
          stroke="#00ff00"
          tickFormatter={(value) => formatValue(value)}
          tick={{ fontSize: 12, fontFamily: 'monospace' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#0a0a0a',
            border: '1px solid #00ff00',
            borderRadius: '4px',
            color: '#00ff00'
          }}
          formatter={(value: number, name: string) => [formatValue(value), name]}
        />
        <Legend />
        {filteredStrategies.map(strategy => (
          <Line
            key={strategy.id}
            type="monotone"
            dataKey={strategy.name}
            stroke={strategy.color}
            strokeWidth={2}
            dot={false}
            name={strategy.name}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );

  // Render scatter plot
  const renderScatterPlot = () => (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart data={riskReturnData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
        <XAxis 
          type="number" 
          dataKey="risk" 
          name="Risk (Volatility)"
          stroke="#00ff00"
          tickFormatter={(value) => formatValue(value)}
          tick={{ fontSize: 12, fontFamily: 'monospace' }}
        />
        <YAxis 
          type="number" 
          dataKey="return" 
          name="Return"
          stroke="#00ff00"
          tickFormatter={(value) => formatValue(value)}
          tick={{ fontSize: 12, fontFamily: 'monospace' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#0a0a0a',
            border: '1px solid #00ff00',
            borderRadius: '4px',
            color: '#00ff00'
          }}
          formatter={(value: number, name: string) => [formatValue(value), name]}
          labelFormatter={(label) => `Strategy: ${label}`}
        />
        {filteredStrategies.map(strategy => (
          <Scatter
            key={strategy.id}
            data={[riskReturnData.find(d => d.name === strategy.name)]}
            fill={strategy.color}
            name={strategy.name}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );

  // Render correlation matrix
  const renderCorrelationMatrix = () => {
    const size = Math.sqrt(correlationMatrix.length);
    return (
      <div className="grid grid-cols-4 gap-2">
        {correlationMatrix.map((corr, index) => {
          const row = Math.floor(index / size);
          const col = index % size;
          const strategy1 = filteredStrategies[row]?.name || '';
          const strategy2 = filteredStrategies[col]?.name || '';
          const intensity = Math.abs(corr);
          
          return (
            <div
              key={index}
              className="aspect-square flex items-center justify-center rounded border border-matrix-green/30"
              style={{
                backgroundColor: corr > 0 
                  ? `rgba(0, 255, 0, ${intensity * 0.5})` 
                  : `rgba(255, 0, 0, ${intensity * 0.5})`,
              }}
              title={`${strategy1} vs ${strategy2}: ${corr.toFixed(3)}`}
            >
              <span className="text-xs font-mono text-matrix-black font-bold">
                {corr.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Strategy Selection */}
        <div className="flex flex-wrap gap-2 mb-4">
          {strategies.map(strategy => (
            <MatrixButton
              key={strategy.id}
              size="sm"
              variant={selectedStrategies.includes(strategy.id) ? 'primary' : 'secondary'}
              onClick={() => toggleStrategy(strategy.id)}
              onContextMenu={(e) => {
                e.preventDefault();
                onStrategyRemove?.(strategy.id);
              }}
            >
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: strategy.color }}
                />
                <span className="font-mono">{strategy.name}</span>
              </div>
            </MatrixButton>
          ))}
        </div>

        {/* Chart Controls */}
        <div className="flex flex-wrap gap-2">
          <div className="flex gap-1">
            {(['performance', 'returns', 'scatter', 'correlation'] as const).map(type => (
              <MatrixButton
                key={type}
                size="sm"
                variant={chartType === type ? 'primary' : 'secondary'}
                onClick={() => setChartType(type)}
              >
                {type.toUpperCase()}
              </MatrixButton>
            ))}
          </div>
          
          {chartType === 'performance' && (
            <div className="flex gap-1">
              {(['1M', '3M', '6M', '1Y', 'ALL'] as const).map(tf => (
                <MatrixButton
                  key={tf}
                  size="sm"
                  variant={timeframe === tf ? 'primary' : 'secondary'}
                  onClick={() => setTimeframe(tf)}
                >
                  {tf}
                </MatrixButton>
              ))}
            </div>
          )}
        </div>

        {/* Portfolio Summary */}
        {portfolioMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center">
              <TrendingUp className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Portfolio Return</p>
              <p className={`font-bold ${portfolioMetrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatValue(portfolioMetrics.totalReturn)}
              </p>
            </div>
            <div className="text-center">
              <Activity className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Avg Volatility</p>
              <p className="text-yellow-400 font-bold">
                {formatValue(portfolioMetrics.volatility)}
              </p>
            </div>
            <div className="text-center">
              <Target className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Avg Sharpe</p>
              <p className={`font-bold ${portfolioMetrics.sharpeRatio >= 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                {portfolioMetrics.sharpeRatio.toFixed(2)}
              </p>
            </div>
            <div className="text-center">
              <TrendingDown className="w-6 h-6 text-red-400 mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Max Drawdown</p>
              <p className="text-red-400 font-bold">
                {formatValue(portfolioMetrics.maxDrawdown)}
              </p>
            </div>
            <div className="text-center">
              <div className="w-6 h-6 mx-auto mb-1 flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-green-400 mr-1" />
                <div className="w-2 h-2 rounded-full bg-red-400" />
              </div>
              <p className="text-matrix-green/70 text-sm font-mono">Avg Win Rate</p>
              <p className="text-green-400 font-bold">
                {formatValue(portfolioMetrics.winRate)}
              </p>
            </div>
          </div>
        )}

        {/* Charts */}
        <div className="h-96">
          {chartType === 'performance' && renderPerformanceChart()}
          {chartType === 'scatter' && showScatterPlot && renderScatterPlot()}
          {chartType === 'correlation' && showCorrelation && renderCorrelationMatrix()}
        </div>

        {/* Strategy Details Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-matrix-green/30">
                <th className="text-left py-2 text-matrix-green font-mono">Strategy</th>
                <th className="text-right py-2 text-matrix-green font-mono">Return</th>
                <th className="text-right py-2 text-matrix-green font-mono">Volatility</th>
                <th className="text-right py-2 text-matrix-green font-mono">Sharpe</th>
                <th className="text-right py-2 text-matrix-green font-mono">Max DD</th>
                <th className="text-right py-2 text-matrix-green font-mono">Win Rate</th>
                <th className="text-right py-2 text-matrix-green font-mono">Trades</th>
              </tr>
            </thead>
            <tbody>
              {filteredStrategies.map(strategy => (
                <tr
                  key={strategy.id}
                  className="border-b border-matrix-green/10 cursor-pointer hover:bg-matrix-green/5"
                  onClick={() => onStrategySelect?.(strategy)}
                >
                  <td className="py-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: strategy.color }}
                      />
                      <span className="text-matrix-green font-mono">{strategy.name}</span>
                    </div>
                  </td>
                  <td className={`text-right py-2 font-mono ${
                    strategy.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatValue(strategy.totalReturn)}
                  </td>
                  <td className="text-right py-2 text-yellow-400 font-mono">
                    {formatValue(strategy.volatility)}
                  </td>
                  <td className={`text-right py-2 font-mono ${
                    strategy.sharpeRatio >= 1 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {strategy.sharpeRatio.toFixed(2)}
                  </td>
                  <td className="text-right py-2 text-red-400 font-mono">
                    {formatValue(strategy.maxDrawdown)}
                  </td>
                  <td className="text-right py-2 text-green-400 font-mono">
                    {formatValue(strategy.winRate)}
                  </td>
                  <td className="text-right py-2 text-matrix-green font-mono">
                    {strategy.totalTrades}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </MatrixCard>
  );
};

export default StrategyComparison;