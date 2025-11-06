import { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, PieChart, Pie, Cell, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Target, TrendingUp, Activity, PieChart as PieChartIcon, Settings, RefreshCw } from 'lucide-react';
import { MatrixCard } from '../../components/MatrixCard';
import { StatCard } from '../../components/StatCard';
import { GlowingButton } from '../../components/GlowingButton';
import { analyticsApi } from '../../services/analyticsApi';

interface OptimizationResult {
  optimal_weights: number[];
  portfolio_return: number;
  portfolio_volatility: number;
  sharpe_ratio: number;
  optimization_success: boolean;
}

interface EfficientFrontierPoint {
  return: number;
  volatility: number;
  sharpe_ratio: number;
  weights: number[];
}

const demoAssets = [
  'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'
];

const demoExpectedReturns = [
  0.12, 0.15, 0.11, 0.25, 0.14, 0.28, 0.16, 0.18
];

const demoCovariance = [
  [0.0225, 0.0108, 0.0162, 0.0270, 0.0144, 0.0324, 0.0189, 0.0216],
  [0.0108, 0.0256, 0.0128, 0.0256, 0.0154, 0.0307, 0.0179, 0.0205],
  [0.0162, 0.0128, 0.0196, 0.0245, 0.0147, 0.0294, 0.0172, 0.0196],
  [0.0270, 0.0256, 0.0245, 0.0625, 0.0300, 0.0500, 0.0350, 0.0375],
  [0.0144, 0.0154, 0.0147, 0.0300, 0.0225, 0.0315, 0.0189, 0.0225],
  [0.0324, 0.0307, 0.0294, 0.0500, 0.0315, 0.0784, 0.0392, 0.0441],
  [0.0189, 0.0179, 0.0172, 0.0350, 0.0189, 0.0392, 0.0289, 0.0306],
  [0.0216, 0.0205, 0.0196, 0.0375, 0.0225, 0.0441, 0.0306, 0.0324],
];

const currentPortfolio = {
  'AAPL': 0.25,
  'GOOGL': 0.20,
  'MSFT': 0.15,
  'TSLA': 0.10,
  'AMZN': 0.10,
  'NVDA': 0.08,
  'META': 0.07,
  'NFLX': 0.05,
};

export default function PortfolioOptimization() {
  const [optimizationMethod, setOptimizationMethod] = useState('mean-variance');
  const [riskAversion, setRiskAversion] = useState(1.0);
  const [targetReturn, setTargetReturn] = useState(0.15);
  const [targetVolatility, setTargetVolatility] = useState(0.20);

  // Mean-Variance Optimization
  const { data: mvOptimization, isLoading: isLoadingMV, refetch: refetchMV } = useQuery<OptimizationResult>({
    queryKey: ['mean-variance-optimization', riskAversion, targetReturn],
    queryFn: () => analyticsApi.optimizeMeanVariance({
      expected_returns: demoExpectedReturns,
      covariance_matrix: demoCovariance,
      asset_names: demoAssets,
      risk_aversion: riskAversion,
      target_return: targetReturn,
    }),
  });

  // Efficient Frontier
  const { data: efficientFrontier, isLoading: isLoadingEF } = useQuery({
    queryKey: ['efficient-frontier'],
    queryFn: () => analyticsApi.generateEfficientFrontier({
      expected_returns: demoExpectedReturns,
      covariance_matrix: demoCovariance,
      asset_names: demoAssets,
      num_portfolios: 50,
    }),
  });

  // Risk Parity Optimization
  const { data: riskParity, isLoading: isLoadingRP } = useQuery({
    queryKey: ['risk-parity-optimization', targetVolatility],
    queryFn: () => analyticsApi.optimizeRiskParity({
      covariance_matrix: demoCovariance,
      asset_names: demoAssets,
      target_volatility: targetVolatility,
    }),
  });

  // Black-Litterman (simplified demo data)
  const { data: blackLitterman, isLoading: isLoadingBL } = useQuery({
    queryKey: ['black-litterman-optimization'],
    queryFn: () => analyticsApi.optimizeBlackLitterman({
      market_caps: [2800, 1800, 2500, 800, 1700, 1200, 900, 200], // in billions
      returns_data: demoAssets.reduce((acc, asset, i) => {
        acc[asset] = Array.from({ length: 252 }, () => 
          Math.random() * 0.04 - 0.02 + demoExpectedReturns[i] / 252
        );
        return acc;
      }, {} as Record<string, number[]>),
      asset_names: demoAssets,
      risk_aversion: 3.0,
    }),
  });

  // Process optimization results for display
  const optimizedWeights = mvOptimization?.optimal_weights || [];
  const optimizedPortfolio = demoAssets.map((asset, index) => ({
    asset,
    current: currentPortfolio[asset] * 100,
    optimized: optimizedWeights[index] * 100 || 0,
    change: (optimizedWeights[index] || 0) * 100 - currentPortfolio[asset] * 100,
  }));

  // Efficient frontier data for chart
  const frontierData = efficientFrontier?.efficient_portfolios?.map((portfolio: any) => ({
    volatility: portfolio.volatility * 100,
    return: portfolio.return * 100,
    sharpe: portfolio.sharpe_ratio,
  })) || [];

  // Current portfolio point
  const currentPortfolioMetrics = {
    volatility: Math.sqrt(demoAssets.reduce((sum, asset1, i) => 
      sum + demoAssets.reduce((innerSum, asset2, j) => 
        innerSum + currentPortfolio[asset1] * currentPortfolio[asset2] * demoCovariance[i][j],
        0
      ), 0
    )) * Math.sqrt(252) * 100,
    return: demoAssets.reduce((sum, asset, i) => 
      sum + currentPortfolio[asset] * demoExpectedReturns[i], 0
    ) * 100,
  };

  // Risk decomposition for risk parity
  const riskContributions = riskParity?.optimization_result?.risk_contributions || [];
  const riskParityData = demoAssets.map((asset, index) => ({
    asset,
    weight: riskParity?.optimization_result?.optimal_weights?.[index] * 100 || 0,
    riskContribution: riskContributions[index] * 100 || 0,
  }));

  // Factor exposure analysis (simulated)
  const factorExposure = [
    { factor: 'Market Beta', current: 1.15, optimized: 1.05, benchmark: 1.00 },
    { factor: 'Size', current: 0.25, optimized: 0.15, benchmark: 0.00 },
    { factor: 'Value', current: -0.35, optimized: -0.15, benchmark: 0.00 },
    { factor: 'Momentum', current: 0.45, optimized: 0.25, benchmark: 0.00 },
    { factor: 'Quality', current: 0.55, optimized: 0.65, benchmark: 0.00 },
    { factor: 'Volatility', current: -0.25, optimized: -0.35, benchmark: 0.00 },
  ];

  const COLORS = ['#00ff00', '#ffff00', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'];

  if (isLoadingMV || isLoadingEF || isLoadingRP) {
    return (
      <div className="p-6 space-y-6">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-matrix-green"></div>
          <p className="mt-4 text-matrix-green">Loading Portfolio Optimization...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">Portfolio Optimization</h1>
          <p className="text-matrix-green/70 mt-2">Advanced portfolio construction and allocation strategies</p>
        </div>
        <div className="flex space-x-4">
          {['mean-variance', 'risk-parity', 'black-litterman'].map((method) => (
            <GlowingButton
              key={method}
              variant={optimizationMethod === method ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setOptimizationMethod(method)}
            >
              {method.replace('-', ' ').toUpperCase()}
            </GlowingButton>
          ))}
        </div>
      </div>

      {/* Optimization Controls */}
      <MatrixCard title="Optimization Parameters" className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <title className="block text-sm text-matrix-green/70 mb-2">Risk Aversion</title>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={riskAversion}
              onChange={(e) => setRiskAversion(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-sm font-mono">{riskAversion.toFixed(1)}</span>
          </div>
          <div>
            <title className="block text-sm text-matrix-green/70 mb-2">Target Return</title>
            <input
              type="range"
              min="0.05"
              max="0.30"
              step="0.01"
              value={targetReturn}
              onChange={(e) => setTargetReturn(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-sm font-mono">{(targetReturn * 100).toFixed(1)}%</span>
          </div>
          <div>
            <title className="block text-sm text-matrix-green/70 mb-2">Target Volatility</title>
            <input
              type="range"
              min="0.10"
              max="0.40"
              step="0.01"
              value={targetVolatility}
              onChange={(e) => setTargetVolatility(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-sm font-mono">{(targetVolatility * 100).toFixed(1)}%</span>
          </div>
          <div className="flex items-end">
            <GlowingButton 
              onClick={() => {
                refetchMV();
              }}
              className="w-full"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Optimize
            </GlowingButton>
          </div>
        </div>
      </MatrixCard>

      {/* Key Metrics */}
      {mvOptimization && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            label="Expected Return"
            value={`${(mvOptimization.portfolio_return * 100).toFixed(2)}%`}
            icon={TrendingUp}
          />
          <StatCard
            label="Portfolio Risk"
            value={`${(mvOptimization.portfolio_volatility * 100).toFixed(2)}%`}
            icon={Activity}
          />
          <StatCard
            label="Sharpe Ratio"
            value={mvOptimization.sharpe_ratio.toFixed(3)}
            icon={Target}
          />
          <StatCard
            label="Optimization"
            value={mvOptimization.optimization_success ? "SUCCESS" : "FAILED"}
            icon={Settings}
          />
        </div>
      )}

      {/* Efficient Frontier and Current vs Optimized */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Efficient Frontier" className="p-6">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  type="number"
                  dataKey="volatility"
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(1)}%`}
                  name="Volatility"
                />
                <YAxis 
                  type="number"
                  dataKey="return"
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(1)}%`}
                  name="Expected Return"
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
                    name === 'return' ? 'Expected Return' : 'Volatility'
                  ]}
                />
                <Scatter 
                  name="Efficient Frontier" 
                  data={frontierData} 
                  fill="#00ff00"
                  opacity={0.8}
                />
                <Scatter 
                  name="Current Portfolio" 
                  data={[currentPortfolioMetrics]} 
                  fill="#ff6b6b"
                  opacity={1}
                />
                {mvOptimization && (
                  <Scatter 
                    name="Optimized Portfolio" 
                    data={[{
                      volatility: mvOptimization.portfolio_volatility * 100,
                      return: mvOptimization.portfolio_return * 100,
                    }]} 
                    fill="#ffff00"
                    opacity={1}
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>

        <MatrixCard title="Portfolio Allocation Comparison" className="p-6">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={optimizedPortfolio}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="asset" 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(0)}%`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)}%`,
                    name === 'current' ? 'Current Weight' : 'Optimized Weight'
                  ]}
                />
                <Legend />
                <Bar dataKey="current" fill="#ff6b6b" opacity={0.8} name="Current" />
                <Bar dataKey="optimized" fill="#00ff00" opacity={0.8} name="Optimized" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>
      </div>

      {/* Risk Parity and Factor Exposure */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Risk Parity Allocation" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskParityData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ asset, weight }) => `${asset}: ${weight.toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="weight"
                >
                  {riskParityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Risk Contribution']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>

        <MatrixCard title="Factor Exposure Analysis" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={factorExposure}>
                <PolarGrid stroke="#00ff0033" />
                <PolarAngleAxis 
                  dataKey="factor" 
                  tick={{ fontSize: 10, fill: '#00ff00' }}
                />
                <PolarRadiusAxis 
                  angle={30} 
                  domain={[-1, 1]}
                  tick={{ fontSize: 10, fill: '#00ff00' }}
                />
                <Radar
                  name="Current"
                  dataKey="current"
                  stroke="#ff6b6b"
                  fill="#ff6b6b"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <Radar
                  name="Optimized"
                  dataKey="optimized"
                  stroke="#00ff00"
                  fill="#00ff00"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>
      </div>

      {/* Optimization Results Summary */}
      <MatrixCard title="Optimization Results Summary" className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-matrix-green mb-4">Mean-Variance Optimization</h3>
            {mvOptimization && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Expected Return</span>
                  <span className="font-mono text-green-400">
                    {(mvOptimization.portfolio_return * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Portfolio Risk</span>
                  <span className="font-mono text-yellow-400">
                    {(mvOptimization.portfolio_volatility * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Sharpe Ratio</span>
                  <span className="font-mono text-blue-400">
                    {mvOptimization.sharpe_ratio.toFixed(3)}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-lg font-semibold text-matrix-green mb-4">Risk Parity</h3>
            {riskParity && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Equal Risk Contrib</span>
                  <span className="font-mono text-green-400">
                    {riskParity.optimization_result?.optimization_success ? 'YES' : 'NO'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Portfolio Risk</span>
                  <span className="font-mono text-yellow-400">
                    {riskParity.optimization_result?.portfolio_volatility ? 
                      `${(riskParity.optimization_result.portfolio_volatility * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Max Risk Contrib</span>
                  <span className="font-mono text-blue-400">
                    {riskContributions.length > 0 ? 
                      `${(Math.max(...riskContributions) * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-lg font-semibold text-matrix-green mb-4">Implementation</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Total Turnover</span>
                <span className="font-mono text-green-400">
                  {optimizedPortfolio.reduce((sum, asset) => sum + Math.abs(asset.change), 0).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Largest Change</span>
                <span className="font-mono text-yellow-400">
                  {Math.max(...optimizedPortfolio.map(asset => Math.abs(asset.change))).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Rebalance Cost</span>
                <span className="font-mono text-blue-400">Est. $2,450</span>
              </div>
            </div>
          </div>
        </div>
      </MatrixCard>

      {/* Detailed Asset Allocation Table */}
      <MatrixCard title="Detailed Asset Allocation" className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-matrix-green/30">
                <th className="text-left py-2">Asset</th>
                <th className="text-left py-2">Current %</th>
                <th className="text-left py-2">Optimized %</th>
                <th className="text-left py-2">Change</th>
                <th className="text-left py-2">Expected Return</th>
                <th className="text-left py-2">Risk Contrib</th>
                <th className="text-left py-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {optimizedPortfolio.map((asset, index) => (
                <tr key={asset.asset} className="border-b border-matrix-green/10">
                  <td className="py-2 font-mono font-semibold">{asset.asset}</td>
                  <td className="py-2 font-mono">{asset.current.toFixed(1)}%</td>
                  <td className="py-2 font-mono">{asset.optimized.toFixed(1)}%</td>
                  <td className="py-2 font-mono">
                    <span className={asset.change > 0 ? 'text-green-400' : asset.change < 0 ? 'text-red-400' : 'text-matrix-green'}>
                      {asset.change > 0 ? '+' : ''}{asset.change.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 font-mono">{(demoExpectedReturns[index] * 100).toFixed(1)}%</td>
                  <td className="py-2 font-mono">
                    {riskContributions[index] ? `${(riskContributions[index] * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td className="py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      Math.abs(asset.change) > 5 ? 'bg-red-900 text-red-300' :
                      Math.abs(asset.change) > 2 ? 'bg-yellow-900 text-yellow-300' :
                      'bg-green-900 text-green-300'
                    }`}>
                      {Math.abs(asset.change) > 5 ? 'REBALANCE' :
                       Math.abs(asset.change) > 2 ? 'ADJUST' : 'HOLD'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </MatrixCard>
    </div>
  );
}