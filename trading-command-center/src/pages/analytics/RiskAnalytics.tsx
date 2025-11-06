import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ComposedChart, Line, ScatterChart, Scatter } from 'recharts';
import { Shield, AlertTriangle, TrendingDown, Activity, Zap, Target } from 'lucide-react';
import { MatrixCard } from '../../components/MatrixCard';
import { StatCard } from '../../components/StatCard';
import { GlowingButton } from '../../components/GlowingButton';
import { analyticsApi } from '../../services/analyticsApi';

interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  portfolio_volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  beta: number;
}

interface StressTestResult {
  scenario_name: string;
  portfolio_loss: number;
  probability: number;
  description: string;
}

interface CorrelationData {
  correlation_matrix: Record<string, Record<string, number>>;
  eigenvalues: number[];
  max_correlation: number;
  min_correlation: number;
}

const demoPortfolio = {
  'AAPL': 0.25,
  'GOOGL': 0.20,
  'MSFT': 0.15,
  'TSLA': 0.10,
  'AMZN': 0.10,
  'NVDA': 0.08,
  'META': 0.07,
  'NFLX': 0.05,
};

const demoReturns = Array.from({ length: 252 }, (_, i) => {
  return Math.random() * 0.04 - 0.02; // Daily returns between -2% and 2%
});

export default function RiskAnalytics() {
  const [confidenceLevel, setConfidenceLevel] = useState(95);
  const [timeHorizon, setTimeHorizon] = useState(1);
  const [portfolioValue] = useState(1000000);

  // Fetch Monte Carlo VaR
  const { data: varResults, isLoading: isLoadingVar } = useQuery({
    queryKey: ['monte-carlo-var', confidenceLevel, timeHorizon],
    queryFn: () => analyticsApi.calculateMonteCarloVar({
      returns: demoReturns,
      confidence_levels: [confidenceLevel / 100],
      time_horizon: timeHorizon,
      num_simulations: 10000,
    }),
  });

  // Fetch parametric VaR
  const { data: parametricVar, isLoading: isLoadingParametric } = useQuery({
    queryKey: ['parametric-var', confidenceLevel, timeHorizon],
    queryFn: () => analyticsApi.calculateParametricVar({
      portfolio_value: portfolioValue,
      expected_return: 0.0008, // Daily expected return
      volatility: 0.015, // Daily volatility
      confidence_levels: [confidenceLevel / 100],
      time_horizon: timeHorizon,
    }),
  });

  // Fetch default stress scenarios
  const { data: stressScenarios, isLoading: isLoadingScenarios } = useQuery({
    queryKey: ['default-stress-scenarios'],
    queryFn: analyticsApi.getDefaultStressScenarios,
  });

  // Fetch tail risk analysis
  const { data: tailRisk, isLoading: isLoadingTailRisk } = useQuery({
    queryKey: ['tail-risk-analysis'],
    queryFn: () => analyticsApi.analyzeTailRisk({
      returns: demoReturns,
      tail_percentile: 0.05,
    }),
  });

  // Fetch concentration risk
  const { data: concentrationRisk, isLoading: isLoadingConcentration } = useQuery({
    queryKey: ['concentration-risk'],
    queryFn: () => analyticsApi.calculateConcentrationRisk({
      portfolio_weights: demoPortfolio,
    }),
  });

  // Process VaR distribution data
  const varDistribution = varResults?.var_distribution ? 
    varResults.var_distribution.map((value: number, index: number) => ({
      percentile: index / varResults.var_distribution.length * 100,
      loss: value * portfolioValue,
    })) : [];

  // Risk decomposition data
  const riskDecomposition = Object.entries(demoPortfolio).map(([asset, weight]) => ({
    asset,
    weight: weight * 100,
    contribution: weight * 100 * (1 + Math.random() * 0.5), // Simulated risk contribution
    marginalVar: Math.random() * 0.02 + 0.005, // Simulated marginal VaR
  }));

  // Stress test results
  const stressTestResults: StressTestResult[] = [
    { scenario_name: '2008 Financial Crisis', portfolio_loss: -0.35, probability: 0.02, description: 'Severe market crash scenario' },
    { scenario_name: 'Tech Bubble Burst', portfolio_loss: -0.28, probability: 0.03, description: 'Technology sector collapse' },
    { scenario_name: 'Interest Rate Shock', portfolio_loss: -0.15, probability: 0.08, description: 'Sharp rate increase' },
    { scenario_name: 'Geopolitical Crisis', portfolio_loss: -0.22, probability: 0.05, description: 'Major geopolitical event' },
    { scenario_name: 'Pandemic Scenario', portfolio_loss: -0.31, probability: 0.02, description: 'Global pandemic impact' },
  ];

  // Correlation heatmap data
  const assets = Object.keys(demoPortfolio);
  const correlationMatrix = assets.map(asset1 => 
    assets.reduce((acc, asset2) => {
      acc[asset2] = asset1 === asset2 ? 1 : Math.random() * 0.8 + 0.1; // Simulated correlations
      return acc;
    }, {} as Record<string, number>)
  );

  const heatmapData = assets.flatMap((asset1, i) =>
    assets.map((asset2, j) => ({
      x: asset1,
      y: asset2,
      value: correlationMatrix[i][asset2],
    }))
  );

  if (isLoadingVar || isLoadingParametric || isLoadingScenarios) {
    return (
      <div className="p-6 space-y-6">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-matrix-green"></div>
          <p className="mt-4 text-matrix-green">Loading Risk Analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">Risk Analytics</h1>
          <p className="text-matrix-green/70 mt-2">Comprehensive portfolio risk assessment and monitoring</p>
        </div>
        <div className="flex space-x-4">
          <select 
            value={confidenceLevel}
            onChange={(e) => setConfidenceLevel(Number(e.target.value))}
            className="bg-matrix-black border border-matrix-green text-matrix-green px-3 py-1 rounded"
          >
            <option value={90}>90% Confidence</option>
            <option value={95}>95% Confidence</option>
            <option value={99}>99% Confidence</option>
          </select>
          <select 
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(Number(e.target.value))}
            className="bg-matrix-black border border-matrix-green text-matrix-green px-3 py-1 rounded"
          >
            <option value={1}>1 Day</option>
            <option value={5}>5 Days</option>
            <option value={10}>10 Days</option>
            <option value={21}>1 Month</option>
          </select>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          label={`VaR ${confidenceLevel}%`}
          value={varResults ? `$${(varResults.var * portfolioValue).toLocaleString()}` : '$0'}
          icon={Shield}
        />
        <StatCard
          label={`CVaR ${confidenceLevel}%`}
          value={varResults ? `$${(varResults.cvar * portfolioValue).toLocaleString()}` : '$0'}
          icon={AlertTriangle}
        />
        <StatCard
          label="Portfolio Volatility"
          value={parametricVar ? `${(parametricVar.portfolio_volatility * 100).toFixed(2)}%` : '0.00%'}
          icon={Activity}
        />
        <StatCard
          label="Max Drawdown"
          value={tailRisk ? `${(tailRisk.max_drawdown * 100).toFixed(2)}%` : '0.00%'}
          icon={TrendingDown}
        />
      </div>

      {/* VaR Distribution and Risk Decomposition */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="VaR Distribution (Monte Carlo)" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={varDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="percentile" 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(0)}%`}
                />
                <YAxis 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Portfolio Loss']}
                  titleFormatter={(value) => `${value}th Percentile`}
                />
                <Area 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#00ff00" 
                  fill="#00ff0020"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>

        <MatrixCard title="Risk Contribution by Asset" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDecomposition}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="asset" 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
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
                    name === 'weight' ? 'Portfolio Weight' : 'Risk Contribution'
                  ]}
                />
                <Bar dataKey="weight" fill="#00ff0060" name="Portfolio Weight" />
                <Bar dataKey="contribution" fill="#00ff00" name="Risk Contribution" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>
      </div>

      {/* Stress Testing Results */}
      <MatrixCard title="Stress Testing Scenarios" className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stressTestResults}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="scenario_name" 
                  stroke="#00ff00"
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Portfolio Loss']}
                />
                <Bar 
                  dataKey="portfolio_loss" 
                  fill="#ff6b6b"
                  opacity={0.8}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-matrix-green">Scenario Impact Analysis</h3>
            {stressTestResults.map((scenario, index) => (
              <div key={index} className="border border-matrix-green/20 rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold">{scenario.scenario_name}</span>
                  <span className={`font-mono ${scenario.portfolio_loss < -0.3 ? 'text-red-400' : 'text-yellow-400'}`}>
                    {(scenario.portfolio_loss * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-matrix-green/70">{scenario.description}</p>
                <p className="text-xs text-matrix-green/50 mt-1">
                  Probability: {(scenario.probability * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      </MatrixCard>

      {/* Correlation Analysis and Tail Risk */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Asset Correlation Matrix" className="p-6">
          <div className="space-y-4">
            <div className="grid grid-cols-9 gap-1 text-xs">
              <div></div>
              {assets.map(asset => (
                <div key={asset} className="text-center font-mono">{asset}</div>
              ))}
              {assets.map((asset1, i) => (
                <div key={asset1}>
                  <div className="font-mono mb-1">{asset1}</div>
                  {assets.map((asset2, j) => (
                    <div 
                      key={`${asset1}-${asset2}`}
                      className={`h-6 flex items-center justify-center text-xs font-mono ${
                        correlationMatrix[i][asset2] > 0.7 ? 'bg-red-900' :
                        correlationMatrix[i][asset2] > 0.4 ? 'bg-yellow-900' :
                        'bg-green-900'
                      }`}
                      style={{ 
                        opacity: Math.abs(correlationMatrix[i][asset2])
                      }}
                    >
                      {correlationMatrix[i][asset2].toFixed(2)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </MatrixCard>

        <MatrixCard title="Tail Risk Analysis" className="p-6">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <Zap className="h-8 w-8 text-yellow-400 mx-auto mb-2" />
                <h3 className="text-lg font-semibold">Expected Shortfall</h3>
                <p className="text-2xl font-mono text-yellow-400">
                  {tailRisk ? `${(tailRisk.expected_shortfall * 100).toFixed(2)}%` : '0.00%'}
                </p>
                <p className="text-sm text-matrix-green/70">5% Tail Average</p>
              </div>
              <div className="text-center">
                <Target className="h-8 w-8 text-red-400 mx-auto mb-2" />
                <h3 className="text-lg font-semibold">Tail Ratio</h3>
                <p className="text-2xl font-mono text-red-400">
                  {tailRisk ? tailRisk.tail_ratio.toFixed(3) : '0.000'}
                </p>
                <p className="text-sm text-matrix-green/70">Risk Asymmetry</p>
              </div>
            </div>
            
            {concentrationRisk && (
              <div className="border-t border-matrix-green/30 pt-4">
                <h4 className="font-semibold mb-3">Concentration Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Herfindahl Index</span>
                    <span className="font-mono">{concentrationRisk.herfindahl_index?.toFixed(3) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Effective Assets</span>
                    <span className="font-mono">{concentrationRisk.effective_assets?.toFixed(1) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Max Weight</span>
                    <span className="font-mono">{concentrationRisk.max_weight ? `${(concentrationRisk.max_weight * 100).toFixed(1)}%` : 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </MatrixCard>
      </div>

      {/* Risk Limits and Alerts */}
      <MatrixCard title="Risk Limits & Monitoring" className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-matrix-green">VaR Limits</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span>Daily VaR Limit</span>
                <span className="font-mono">$50,000</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Current VaR</span>
                <span className={`font-mono ${varResults && Math.abs(varResults.var * portfolioValue) > 50000 ? 'text-red-400' : 'text-green-400'}`}>
                  ${varResults ? Math.abs(varResults.var * portfolioValue).toLocaleString() : '0'}
                </span>
              </div>
              <div className="w-full bg-matrix-black border border-matrix-green rounded">
                <div 
                  className={`h-2 rounded ${varResults && Math.abs(varResults.var * portfolioValue) > 50000 ? 'bg-red-600' : 'bg-green-600'}`}
                  style={{ 
                    width: `${varResults ? Math.min((Math.abs(varResults.var * portfolioValue) / 50000) * 100, 100) : 0}%` 
                  }}
                ></div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-matrix-green">Concentration Limits</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span>Single Asset Limit</span>
                <span className="font-mono">30%</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Max Current Position</span>
                <span className={`font-mono ${Math.max(...Object.values(demoPortfolio)) > 0.3 ? 'text-red-400' : 'text-green-400'}`}>
                  {(Math.max(...Object.values(demoPortfolio)) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-matrix-black border border-matrix-green rounded">
                <div 
                  className={`h-2 rounded ${Math.max(...Object.values(demoPortfolio)) > 0.3 ? 'bg-red-600' : 'bg-green-600'}`}
                  style={{ 
                    width: `${(Math.max(...Object.values(demoPortfolio)) / 0.3) * 100}%` 
                  }}
                ></div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-matrix-green">Risk Alerts</h3>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm">VaR within limits</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                <span className="text-sm">High correlation detected</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm">Concentration within limits</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                <span className="text-sm">Stress test breach: 2008 scenario</span>
              </div>
            </div>
          </div>
        </div>
      </MatrixCard>
    </div>
  );
}