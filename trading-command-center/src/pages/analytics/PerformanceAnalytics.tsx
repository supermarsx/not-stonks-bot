import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart } from 'recharts';
import { Calendar, TrendingUp, TrendingDown, Activity, Target, DollarSign } from 'lucide-react';
import { MatrixCard } from '../../components/MatrixCard';
import { StatCard } from '../../components/StatCard';
import { GlowingButton } from '../../components/GlowingButton';
import { analyticsApi } from '../../services/analyticsApi';

interface PerformanceMetrics {
  annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  win_rate: number;
  best_day: number;
  worst_day: number;
}

interface PerformanceData {
  dates: string[];
  portfolio_returns: number[];
  benchmark_returns: number[];
  stats: {
    portfolio_annual_return: number;
    portfolio_annual_volatility: number;
    benchmark_annual_return: number;
    benchmark_annual_volatility: number;
    correlation: number;
  };
}

export default function PerformanceAnalytics() {
  const [timeframe, setTimeframe] = useState('1Y');
  const [rollingWindow, setRollingWindow] = useState(252);

  // Fetch demo performance data
  const { data: performanceData, isLoading: isLoadingData } = useQuery<PerformanceData>({
    queryKey: ['performance-demo-data'],
    queryFn: analyticsApi.getPerformanceDemoData,
  });

  // Fetch performance metrics
  const { data: metrics, isLoading: isLoadingMetrics } = useQuery<PerformanceMetrics>({
    queryKey: ['performance-metrics', performanceData],
    queryFn: () => {
      if (!performanceData) return null;
      return analyticsApi.calculatePerformanceMetrics({
        returns: performanceData.portfolio_returns,
        dates: performanceData.dates,
        risk_free_rate: 0.02,
        benchmark_returns: performanceData.benchmark_returns,
      });
    },
    enabled: !!performanceData,
  });

  // Fetch rolling metrics
  const { data: rollingMetrics, isLoading: isLoadingRolling } = useQuery({
    queryKey: ['rolling-metrics', performanceData, rollingWindow],
    queryFn: () => {
      if (!performanceData) return null;
      return analyticsApi.calculateRollingMetrics({
        returns: performanceData.portfolio_returns,
        dates: performanceData.dates,
        risk_free_rate: 0.02,
      }, rollingWindow);
    },
    enabled: !!performanceData,
  });

  // Process data for charts
  const chartData = performanceData ? performanceData.dates.map((date, index) => ({
    date,
    portfolio: ((1 + performanceData.portfolio_returns.slice(0, index + 1).reduce((acc, ret) => acc * (1 + ret), 1)) - 1) * 100,
    benchmark: ((1 + performanceData.benchmark_returns.slice(0, index + 1).reduce((acc, ret) => acc * (1 + ret), 1)) - 1) * 100,
    portfolioDaily: performanceData.portfolio_returns[index] * 100,
    benchmarkDaily: performanceData.benchmark_returns[index] * 100,
  })) : [];

  const monthlyReturns = chartData.length > 0 ? chartData.filter((_, index) => index % 21 === 0) : [];

  if (isLoadingData || isLoadingMetrics) {
    return (
      <div className="p-6 space-y-6">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-matrix-green"></div>
          <p className="mt-4 text-matrix-green">Loading Performance Analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">Performance Analytics</h1>
          <p className="text-matrix-green/70 mt-2">Comprehensive portfolio performance analysis and metrics</p>
        </div>
        <div className="flex space-x-4">
          {['1M', '3M', '6M', '1Y', '3Y'].map((period) => (
            <GlowingButton
              key={period}
              variant={timeframe === period ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setTimeframe(period)}
            >
              {period}
            </GlowingButton>
          ))}
        </div>
      </div>

      {/* Key Performance Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            label="Annual Return"
            value={`${(metrics.annual_return * 100).toFixed(2)}%`}
            icon={TrendingUp}
          />
          <StatCard
            label="Volatility"
            value={`${(metrics.annual_volatility * 100).toFixed(2)}%`}
            icon={Activity}
          />
          <StatCard
            label="Max Drawdown"
            value={`${(metrics.max_drawdown * 100).toFixed(2)}%`}
            icon={TrendingDown}
          />
          <StatCard
            label="Win Rate"
            value={`${(metrics.win_rate * 100).toFixed(1)}%`}
            icon={Target}
          />
        </div>
      )}

      {/* Portfolio vs Benchmark Performance */}
      <MatrixCard title="Cumulative Performance" className="p-6">
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
              <XAxis 
                dataKey="date" 
                stroke="#00ff00"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis 
                stroke="#00ff00"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${value.toFixed(1)}%`}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#0a0a0a',
                  border: '1px solid #00ff00',
                  borderRadius: '4px',
                  color: '#00ff00'
                }}
                formatter={(value: number, name: string) => [
                  `${value.toFixed(2)}%`,
                  name === 'portfolio' ? 'Portfolio' : 'Benchmark'
                ]}
                titleFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="portfolio" 
                stroke="#00ff00" 
                strokeWidth={2}
                name="Portfolio"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="benchmark" 
                stroke="#ffff00" 
                strokeWidth={2}
                name="Benchmark"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </MatrixCard>

      {/* Daily Returns Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Daily Returns Distribution" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData.slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="date" 
                  stroke="#00ff00"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(1)}%`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number) => [`${value.toFixed(2)}%`, 'Daily Return']}
                />
                <Bar 
                  dataKey="portfolioDaily" 
                  fill="#00ff00"
                  opacity={0.8}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>

        <MatrixCard title="Risk-Return Profile" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-matrix-green/70">Risk-Adjusted Metrics</span>
              <Calendar className="h-4 w-4 text-matrix-green/70" />
            </div>
            {metrics && (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Sharpe Ratio</span>
                  <span className={`font-mono ${metrics.sharpe_ratio > 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {metrics.sharpe_ratio.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Sortino Ratio</span>
                  <span className={`font-mono ${metrics.sortino_ratio > 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {metrics.sortino_ratio.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Calmar Ratio</span>
                  <span className={`font-mono ${metrics.calmar_ratio > 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {metrics.calmar_ratio.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Best Daily Return</span>
                  <span className="font-mono text-green-400">
                    {(metrics.best_day * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Worst Daily Return</span>
                  <span className="font-mono text-red-400">
                    {(metrics.worst_day * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </MatrixCard>
      </div>

      {/* Rolling Performance Metrics */}
      {rollingMetrics && (
        <MatrixCard title="Rolling Performance Metrics" className="p-6">
          <div className="mb-4 flex space-x-4">
            <title className="text-sm text-matrix-green/70">
              Rolling Window:
              <select 
                value={rollingWindow}
                onChange={(e) => setRollingWindow(Number(e.target.value))}
                className="ml-2 bg-matrix-black border border-matrix-green text-matrix-green px-2 py-1 rounded"
              >
                <option value={63}>3 Months</option>
                <option value={126}>6 Months</option>
                <option value={252}>1 Year</option>
              </select>
            </title>
          </div>
          <div className="text-center py-4 text-matrix-green/70">
            <p>Rolling metrics: {rollingMetrics.data_points} data points</p>
            <p className="text-sm">Window: {rollingMetrics.window_days} days</p>
          </div>
        </MatrixCard>
      )}

      {/* Attribution Analysis */}
      <MatrixCard title="Performance Attribution" className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <DollarSign className="h-8 w-8 text-matrix-green mx-auto mb-2" />
            <h3 className="text-lg font-semibold">Alpha Generation</h3>
            <p className="text-2xl font-mono text-green-400">
              {performanceData ? ((performanceData.stats.portfolio_annual_return - performanceData.stats.benchmark_annual_return) * 100).toFixed(2) : '0.00'}%
            </p>
            <p className="text-sm text-matrix-green/70">vs Benchmark</p>
          </div>
          <div className="text-center">
            <Activity className="h-8 w-8 text-matrix-green mx-auto mb-2" />
            <h3 className="text-lg font-semibold">Correlation</h3>
            <p className="text-2xl font-mono text-yellow-400">
              {performanceData ? performanceData.stats.correlation.toFixed(3) : '0.000'}
            </p>
            <p className="text-sm text-matrix-green/70">to Benchmark</p>
          </div>
          <div className="text-center">
            <Target className="h-8 w-8 text-matrix-green mx-auto mb-2" />
            <h3 className="text-lg font-semibold">Tracking Error</h3>
            <p className="text-2xl font-mono text-blue-400">
              {performanceData ? (Math.abs(performanceData.stats.portfolio_annual_volatility - performanceData.stats.benchmark_annual_volatility) * 100).toFixed(2) : '0.00'}%
            </p>
            <p className="text-sm text-matrix-green/70">Volatility Diff</p>
          </div>
        </div>
      </MatrixCard>
    </div>
  );
}