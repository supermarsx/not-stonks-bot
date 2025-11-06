import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ComposedChart, Bar, ReferenceLine } from 'recharts';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { TrendingUp, TrendingDown, Calendar, Activity, Target } from 'lucide-react';

interface PerformanceDataPoint {
  date: string;
  portfolio: number;
  benchmark?: number;
  drawdown?: number;
  volatility?: number;
  returns?: number;
  volume?: number;
}

interface PerformanceChartProps {
  data: PerformanceDataPoint[];
  title?: string;
  className?: string;
  showBenchmark?: boolean;
  showDrawdown?: boolean;
  showVolume?: boolean;
  showReturns?: boolean;
  chartType?: 'line' | 'area' | 'composed';
  timeframe?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | 'ALL';
  onTimeframeChange?: (timeframe: string) => void;
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data,
  title = "Performance Analysis",
  className = "",
  showBenchmark = true,
  showDrawdown = false,
  showVolume = false,
  showReturns = false,
  chartType = 'area',
  timeframe = '1Y',
  onTimeframeChange,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<PerformanceDataPoint | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<'portfolio' | 'drawdown' | 'returns'>('portfolio');

  // Calculate metrics
  const metrics = useMemo(() => {
    if (!data.length) return null;

    const firstValue = data[0].portfolio;
    const lastValue = data[data.length - 1].portfolio;
    const totalReturn = ((lastValue - firstValue) / firstValue) * 100;
    
    const maxDrawdown = Math.min(...data.map(d => d.drawdown || 0)) * 100;
    const maxValue = Math.max(...data.map(d => d.portfolio));
    const minValue = Math.min(...data.map(d => d.portfolio));
    
    const returns = data.map((d, i) => 
      i === 0 ? 0 : ((d.portfolio - data[i - 1].portfolio) / data[i - 1].portfolio) * 100
    );
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const volatility = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    ) * Math.sqrt(252) * 100; // Annualized volatility

    return {
      totalReturn,
      maxDrawdown,
      maxValue,
      minValue,
      volatility,
      sharpeRatio: totalReturn / volatility,
    };
  }, [data]);

  // Filter data by timeframe
  const filteredData = useMemo(() => {
    if (!data.length) return [];
    
    const now = new Date();
    const periods = {
      '1D': 1,
      '1W': 7,
      '1M': 30,
      '3M': 90,
      '6M': 180,
      '1Y': 365,
      'ALL': Infinity,
    };
    
    const daysBack = periods[timeframe];
    if (daysBack === Infinity) return data;
    
    const cutoffDate = new Date(now.getTime() - (daysBack * 24 * 60 * 60 * 1000));
    return data.filter(d => new Date(d.date) >= cutoffDate);
  }, [data, timeframe]);

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: timeframe === '1Y' || timeframe === 'ALL' ? 'numeric' : undefined
    });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      setHoveredPoint(data);
      
      return (
        <div className="bg-matrix-black border border-matrix-green p-4 rounded shadow-lg max-w-xs">
          <p className="text-matrix-green font-mono text-sm mb-2">
            {formatDate(label)}
          </p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex justify-between items-center gap-4">
              <span className="text-matrix-green font-mono text-xs">{entry.name}:</span>
              <span className="text-matrix-green font-mono font-bold">
                {entry.name.includes('%') || entry.name === 'Returns' || entry.name === 'Drawdown'
                  ? formatPercentage(entry.value)
                  : formatCurrency(entry.value)
                }
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  // Render chart based on type
  const renderChart = () => {
    const commonProps = {
      data: filteredData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chartType) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
            <XAxis 
              dataKey="date" 
              stroke="#00ff00"
              tickFormatter={formatDate}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            <YAxis 
              stroke="#00ff00"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="portfolio"
              stroke="#00ff00"
              strokeWidth={2}
              dot={false}
              name="Portfolio"
            />
            {showBenchmark && (
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="#ffff00"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Benchmark"
              />
            )}
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
            <XAxis 
              dataKey="date" 
              stroke="#00ff00"
              tickFormatter={formatDate}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            <YAxis 
              stroke="#00ff00"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              type="monotone"
              dataKey="portfolio"
              stroke="#00ff00"
              fill="url(#portfolioGradient)"
              strokeWidth={2}
              name="Portfolio"
            />
            {showBenchmark && (
              <Area
                type="monotone"
                dataKey="benchmark"
                stroke="#ffff00"
                fill="url(#benchmarkGradient)"
                strokeWidth={2}
                name="Benchmark"
              />
            )}
            <defs>
              <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00ff00" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#00ff00" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ffff00" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#ffff00" stopOpacity={0}/>
              </linearGradient>
            </defs>
          </AreaChart>
        );

      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
            <XAxis 
              dataKey="date" 
              stroke="#00ff00"
              tickFormatter={formatDate}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            <YAxis 
              yAxisId="left"
              stroke="#00ff00"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 12, fontFamily: 'monospace' }}
            />
            {showVolume && (
              <YAxis 
                yAxisId="right"
                orientation="right"
                stroke="#00cc00"
                tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
                tick={{ fontSize: 12, fontFamily: 'monospace' }}
              />
            )}
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="portfolio"
              stroke="#00ff00"
              strokeWidth={2}
              dot={false}
              name="Portfolio"
            />
            {showBenchmark && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="benchmark"
                stroke="#ffff00"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Benchmark"
              />
            )}
            {showVolume && (
              <Bar
                yAxisId="right"
                dataKey="volume"
                fill="#00ff0033"
                name="Volume"
              />
            )}
            {showDrawdown && (
              <ReferenceLine 
                yAxisId="left"
                y={0}
                stroke="#ff0000"
                strokeDasharray="3 3"
              />
            )}
          </ComposedChart>
        );
    }
  };

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Timeframe Controls */}
        <div className="flex flex-wrap gap-2 mb-4">
          {(['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'] as const).map((tf) => (
            <MatrixButton
              key={tf}
              size="sm"
              variant={timeframe === tf ? 'primary' : 'secondary'}
              onClick={() => onTimeframeChange?.(tf)}
            >
              {tf}
            </MatrixButton>
          ))}
        </div>

        {/* Chart Type Controls */}
        <div className="flex gap-2 mb-4">
          {(['line', 'area', 'composed'] as const).map((type) => (
            <MatrixButton
              key={type}
              size="sm"
              variant={chartType === type ? 'primary' : 'secondary'}
              onClick={() => {/* setChartType(type) */}}
            >
              {type.toUpperCase()}
            </MatrixButton>
          ))}
        </div>

        {/* Metrics Summary */}
        {metrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <TrendingUp className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Total Return</p>
              <p className={`font-bold ${
                metrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercentage(metrics.totalReturn)}
              </p>
            </div>
            <div className="text-center">
              <TrendingDown className="w-6 h-6 text-red-400 mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Max Drawdown</p>
              <p className="text-red-400 font-bold">
                {formatPercentage(metrics.maxDrawdown)}
              </p>
            </div>
            <div className="text-center">
              <Activity className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Volatility</p>
              <p className="text-yellow-400 font-bold">
                {formatPercentage(metrics.volatility)}
              </p>
            </div>
            <div className="text-center">
              <Target className="w-6 h-6 text-matrix-green mx-auto mb-1" />
              <p className="text-matrix-green/70 text-sm font-mono">Sharpe Ratio</p>
              <p className={`font-bold ${
                metrics.sharpeRatio >= 1 ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {metrics.sharpeRatio.toFixed(2)}
              </p>
            </div>
          </div>
        )}

        {/* Chart */}
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </div>

        {/* Hover Info */}
        {hoveredPoint && (
          <div className="mt-4 p-3 bg-matrix-green/10 border border-matrix-green/30 rounded">
            <p className="text-matrix-green font-mono text-sm">
              {formatDate(hoveredPoint.date)}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2">
              <div>
                <p className="text-matrix-green/70 text-xs font-mono">Portfolio</p>
                <p className="text-matrix-green font-bold">
                  {formatCurrency(hoveredPoint.portfolio)}
                </p>
              </div>
              {hoveredPoint.benchmark && (
                <div>
                  <p className="text-matrix-green/70 text-xs font-mono">Benchmark</p>
                  <p className="text-yellow-400 font-bold">
                    {formatCurrency(hoveredPoint.benchmark)}
                  </p>
                </div>
              )}
              {hoveredPoint.drawdown && (
                <div>
                  <p className="text-matrix-green/70 text-xs font-mono">Drawdown</p>
                  <p className="text-red-400 font-bold">
                    {formatPercentage(hoveredPoint.drawdown * 100)}
                  </p>
                </div>
              )}
              {hoveredPoint.returns && (
                <div>
                  <p className="text-matrix-green/70 text-xs font-mono">Returns</p>
                  <p className={`font-bold ${
                    hoveredPoint.returns >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatPercentage(hoveredPoint.returns)}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </MatrixCard>
  );
};

export default PerformanceChart;