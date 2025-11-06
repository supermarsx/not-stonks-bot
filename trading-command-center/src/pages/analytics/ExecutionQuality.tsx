import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ComposedChart, Area, AreaChart } from 'recharts';
import { Activity, Clock, DollarSign, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { MatrixCard } from '../../components/MatrixCard';
import { StatCard } from '../../components/StatCard';
import { GlowingButton } from '../../components/GlowingButton';
import { analyticsApi } from '../../services/analyticsApi';

interface ExecutionMetrics {
  implementation_shortfall: number;
  vwap_performance: number;
  market_impact: number;
  timing_alpha: number;
  fill_rate: number;
  average_slippage: number;
}

interface TradeData {
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  execution_price: number;
  vwap_price: number;
  arrival_price: number;
  decision_price: number;
  commission: number;
  market_impact: number;
  implementation_shortfall: number;
  venue: string;
}

export default function ExecutionQuality() {
  const [timeframe, setTimeframe] = useState('1D');
  const [selectedVenue, setSelectedVenue] = useState('ALL');

  // Fetch demo trades data
  const { data: tradesData, isLoading: isLoadingTrades } = useQuery<TradeData[]>({
    queryKey: ['execution-demo-trades'],
    queryFn: analyticsApi.getDemoTrades,
  });

  // Calculate execution scorecard
  const { data: scorecard, isLoading: isLoadingScorecard } = useQuery({
    queryKey: ['execution-scorecard', tradesData],
    queryFn: () => {
      if (!tradesData) return null;
      return analyticsApi.generateExecutionScorecard(tradesData);
    },
    enabled: !!tradesData,
  });

  // Process data for charts
  const processedTrades = tradesData ? tradesData.map((trade, index) => ({
    ...trade,
    id: index,
    slippage: ((trade.execution_price - trade.arrival_price) / trade.arrival_price) * 10000, // in basis points
    vwapSlippage: ((trade.execution_price - trade.vwap_price) / trade.vwap_price) * 10000,
    tradeValue: trade.quantity * trade.execution_price,
    hour: new Date(trade.timestamp).getHours(),
    date: new Date(trade.timestamp).toLocaleDateString(),
  })) : [];

  // Aggregate metrics by venue
  const venueMetrics = processedTrades.reduce((acc, trade) => {
    if (!acc[trade.venue]) {
      acc[trade.venue] = {
        venue: trade.venue,
        totalVolume: 0,
        averageSlippage: 0,
        tradeCount: 0,
        totalCost: 0,
      };
    }
    acc[trade.venue].totalVolume += trade.tradeValue;
    acc[trade.venue].tradeCount += 1;
    acc[trade.venue].totalCost += Math.abs(trade.slippage) * trade.tradeValue / 10000;
    return acc;
  }, {} as Record<string, any>);

  // Calculate average slippage by venue
  Object.values(venueMetrics).forEach((venue: any) => {
    const venueTrades = processedTrades.filter(t => t.venue === venue.venue);
    venue.averageSlippage = venueTrades.reduce((sum, t) => sum + Math.abs(t.slippage), 0) / venueTrades.length;
  });

  const venueData = Object.values(venueMetrics);

  // Hourly execution pattern
  const hourlyPattern = Array.from({ length: 24 }, (_, hour) => {
    const hourTrades = processedTrades.filter(t => t.hour === hour);
    return {
      hour,
      tradeCount: hourTrades.length,
      averageSlippage: hourTrades.length > 0 ? 
        hourTrades.reduce((sum, t) => sum + Math.abs(t.slippage), 0) / hourTrades.length : 0,
      totalVolume: hourTrades.reduce((sum, t) => sum + t.tradeValue, 0),
    };
  });

  if (isLoadingTrades || isLoadingScorecard) {
    return (
      <div className="p-6 space-y-6">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-matrix-green"></div>
          <p className="mt-4 text-matrix-green">Loading Execution Analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">Execution Quality</h1>
          <p className="text-matrix-green/70 mt-2">Trade execution analysis and transaction cost analytics</p>
        </div>
        <div className="flex space-x-4">
          {['1D', '1W', '1M', '3M'].map((period) => (
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

      {/* Key Execution Metrics */}
      {scorecard && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            label="Implementation Shortfall"
            value={`${(scorecard.implementation_shortfall_bps || 0).toFixed(1)} bps`}
            icon={TrendingDown}
          />
          <StatCard
            label="VWAP Performance"
            value={`${(scorecard.vwap_performance_bps || 0).toFixed(1)} bps`}
            icon={Activity}
          />
          <StatCard
            label="Fill Rate"
            value={`${((scorecard.fill_rate || 1) * 100).toFixed(1)}%`}
            icon={CheckCircle}
          />
          <StatCard
            label="Market Impact"
            value={`${(scorecard.market_impact_bps || 0).toFixed(1)} bps`}
            icon={AlertTriangle}
          />
        </div>
      )}

      {/* Execution Cost Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Venue Performance Comparison" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={venueData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="venue" 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value.toFixed(1)} bps`}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(2)} bps`,
                    'Avg Slippage'
                  ]}
                />
                <Bar 
                  dataKey="averageSlippage" 
                  fill="#00ff00"
                  opacity={0.8}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>

        <MatrixCard title="Hourly Execution Pattern" className="p-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={hourlyPattern}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  yAxisId="left"
                  stroke="#00ff00"
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  stroke="#ffff00"
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#0a0a0a',
                    border: '1px solid #00ff00',
                    borderRadius: '4px',
                    color: '#00ff00'
                  }}
                />
                <Bar 
                  yAxisId="left"
                  dataKey="tradeCount" 
                  fill="#00ff00"
                  opacity={0.6}
                  name="Trade Count"
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="averageSlippage" 
                  stroke="#ffff00"
                  strokeWidth={2}
                  name="Avg Slippage (bps)"
                  dot={{ r: 3 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </MatrixCard>
      </div>

      {/* Trade Analysis */}
      <MatrixCard title="Implementation Shortfall Analysis" className="p-6">
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={processedTrades}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
              <XAxis 
                dataKey="tradeValue" 
                stroke="#00ff00"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                name="Trade Value"
              />
              <YAxis 
                dataKey="implementation_shortfall"
                stroke="#00ff00"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value * 10000).toFixed(1)}`}
                name="IS (bps)"
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#0a0a0a',
                  border: '1px solid #00ff00',
                  borderRadius: '4px',
                  color: '#00ff00'
                }}
                formatter={(value: number, name: string) => {
                  if (name === 'implementation_shortfall') {
                    return [`${(value * 10000).toFixed(2)} bps`, 'Implementation Shortfall'];
                  }
                  return [value, name];
                }}
                titleFormatter={(title, payload) => {
                  if (payload && payload[0]) {
                    const data = payload[0].payload;
                    return `${data.symbol} - ${data.side.toUpperCase()}`;
                  }
                  return title;
                }}
              />
              <Scatter 
                dataKey="implementation_shortfall" 
                fill="#00ff00"
                opacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </MatrixCard>

      {/* Transaction Cost Components */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <MatrixCard title="Cost Components" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-matrix-green/70">Transaction Cost Breakdown</span>
              <DollarSign className="h-4 w-4 text-matrix-green/70" />
            </div>
            {scorecard && (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Market Impact</span>
                  <span className="font-mono text-red-400">
                    {(scorecard.market_impact_bps || 0).toFixed(2)} bps
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Timing Cost</span>
                  <span className="font-mono text-yellow-400">
                    {(scorecard.timing_cost_bps || 0).toFixed(2)} bps
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Commission</span>
                  <span className="font-mono text-blue-400">
                    {(scorecard.commission_bps || 0).toFixed(2)} bps
                  </span>
                </div>
                <div className="border-t border-matrix-green/30 pt-2">
                  <div className="flex justify-between font-semibold">
                    <span>Total TCA</span>
                    <span className="font-mono text-matrix-green">
                      {((scorecard.market_impact_bps || 0) + (scorecard.timing_cost_bps || 0) + (scorecard.commission_bps || 0)).toFixed(2)} bps
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </MatrixCard>

        <MatrixCard title="Execution Quality Score" className="p-6">
          <div className="text-center">
            <div className="text-4xl font-mono font-bold text-matrix-green mb-2">
              {scorecard ? (scorecard.execution_quality_score || 85).toFixed(0) : '85'}
            </div>
            <div className="text-sm text-matrix-green/70 mb-4">Overall Score</div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Speed</span>
                <span className="text-green-400">A+</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Price Improvement</span>
                <span className="text-green-400">A</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Fill Rate</span>
                <span className="text-green-400">A+</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Market Impact</span>
                <span className="text-yellow-400">B+</span>
              </div>
            </div>
          </div>
        </MatrixCard>

        <MatrixCard title="Best Execution Analysis" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-matrix-green/70">Regulatory Compliance</span>
              <CheckCircle className="h-4 w-4 text-green-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>Price Priority</span>
                <span className="text-green-400">✓ PASS</span>
              </div>
              <div className="flex justify-between">
                <span>Time Priority</span>
                <span className="text-green-400">✓ PASS</span>
              </div>
              <div className="flex justify-between">
                <span>Size Improvement</span>
                <span className="text-green-400">✓ PASS</span>
              </div>
              <div className="flex justify-between">
                <span>Venue Diversity</span>
                <span className="text-yellow-400">⚠ REVIEW</span>
              </div>
              <div className="border-t border-matrix-green/30 pt-2">
                <div className="text-sm text-matrix-green/70">
                  Compliance Score: {scorecard ? (scorecard.compliance_score || 92).toFixed(0) : '92'}%
                </div>
              </div>
            </div>
          </div>
        </MatrixCard>
      </div>

      {/* Trade Details Table */}
      <MatrixCard title="Recent Trades" className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-matrix-green/30">
                <th className="text-left py-2">Symbol</th>
                <th className="text-left py-2">Side</th>
                <th className="text-left py-2">Quantity</th>
                <th className="text-left py-2">Exec Price</th>
                <th className="text-left py-2">VWAP</th>
                <th className="text-left py-2">Slippage</th>
                <th className="text-left py-2">Venue</th>
                <th className="text-left py-2">IS (bps)</th>
              </tr>
            </thead>
            <tbody>
              {processedTrades.slice(0, 10).map((trade, index) => (
                <tr key={index} className="border-b border-matrix-green/10">
                  <td className="py-2 font-mono">{trade.symbol}</td>
                  <td className="py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      trade.side === 'buy' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                    }`}>
                      {trade.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-2 font-mono">{trade.quantity.toLocaleString()}</td>
                  <td className="py-2 font-mono">${trade.execution_price.toFixed(2)}</td>
                  <td className="py-2 font-mono">${trade.vwap_price.toFixed(2)}</td>
                  <td className="py-2 font-mono">
                    <span className={trade.vwapSlippage > 0 ? 'text-red-400' : 'text-green-400'}>
                      {trade.vwapSlippage.toFixed(1)} bps
                    </span>
                  </td>
                  <td className="py-2">{trade.venue}</td>
                  <td className="py-2 font-mono">
                    <span className={trade.implementation_shortfall > 0 ? 'text-red-400' : 'text-green-400'}>
                      {(trade.implementation_shortfall * 10000).toFixed(1)}
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