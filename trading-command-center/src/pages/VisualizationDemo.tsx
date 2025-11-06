import React, { useState, useEffect, useRef } from 'react';
import { PortfolioAllocation } from '../components/visualizations/PortfolioAllocation';
import { PerformanceChart } from '../components/visualizations/PerformanceChart';
import { PnLWaterfall } from '../components/visualizations/PLWaterfall';
import { RiskHeatmap } from '../components/visualizations/RiskHeatmap';
import { StrategyComparison } from '../components/visualizations/StrategyComparison';
import { CorrelationMatrix } from '../components/visualizations/CorrelationMatrix';
import { CandlestickChart } from '../components/charts/CandlestickChart';
import { TradingViewChart } from '../components/charts/TradingViewChart';
import { ChartJSCanvas } from '../components/charts/ChartJSCanvas';
import { DateRangePicker } from '../components/filters/DateRangePicker';
import { SymbolSearch } from '../components/filters/SymbolSearch';
import { MatrixCard } from '../components/MatrixCard';
import { MatrixButton } from '../components/MatrixButton';
import { exportUtils } from '../utils/visualizationExporter';
import { Download, Share, Filter, TrendingUp, Activity } from 'lucide-react';

// Demo data generators
const generateMockData = () => {
  // Portfolio positions
  const positions = [
    {
      id: '1',
      symbol: 'AAPL',
      name: 'Apple Inc.',
      size: 100,
      marketValue: 17500,
      allocation: 25.5,
      pnl: 1200,
      pnlPercentage: 7.35,
      side: 'long' as const,
      entryPrice: 163.00,
      currentPrice: 175.00,
      timestamp: new Date().toISOString(),
      broker: 'ALPACA'
    },
    {
      id: '2',
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      size: 50,
      marketValue: 7140,
      allocation: 18.2,
      pnl: -300,
      pnlPercentage: -4.02,
      side: 'long' as const,
      entryPrice: 146.80,
      currentPrice: 142.80,
      timestamp: new Date().toISOString(),
      broker: 'IBKR'
    },
    {
      id: '3',
      symbol: 'BTC-USD',
      name: 'Bitcoin USD',
      size: 0.15,
      marketValue: 6428,
      allocation: 15.8,
      pnl: 1250,
      pnlPercentage: 24.15,
      side: 'long' as const,
      entryPrice: 34500,
      currentPrice: 42850,
      timestamp: new Date().toISOString(),
      broker: 'BINANCE'
    }
  ];

  // Performance data
  const performanceData = Array.from({ length: 252 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (252 - i));
    return {
      date: date.toISOString().split('T')[0],
      portfolio: 100000 * (1 + Math.sin(i / 20) * 0.1 + (Math.random() - 0.5) * 0.05),
      benchmark: 100000 * (1 + Math.sin(i / 25) * 0.08 + (Math.random() - 0.5) * 0.04),
      drawdown: Math.min(0, (Math.random() - 0.7) * 0.1),
      volume: Math.floor(Math.random() * 1000000) + 500000,
      returns: (Math.random() - 0.5) * 0.04,
    };
  });

  // P&L data
  const pnlData = [
    { id: '1', label: 'Trading Gains', category: 'trading' as const, value: 2500, description: 'Successful trades', symbol: 'AAPL' },
    { id: '2', label: 'Trading Losses', category: 'trading' as const, value: -800, description: 'Unsuccessful trades', symbol: 'GOOGL' },
    { id: '3', label: 'Commission Fees', category: 'fees' as const, value: -45, description: 'Trading commissions', symbol: 'ALL' },
    { id: '4', label: 'Dividends', category: 'dividends' as const, value: 120, description: 'Quarterly dividends', symbol: 'AAPL' },
    { id: '5', label: 'Unrealized P&L', category: 'unrealized' as const, value: 1500, description: 'Current positions', symbol: 'BTC-USD' },
  ];

  // Risk data
  const riskData = [
    { symbol: 'AAPL', name: 'Apple Inc.', volatility: 0.25, beta: 1.2, var95: 0.035, var99: 0.055, expectedShortfall: 0.065, maxDrawdown: 0.15, sharpeRatio: 1.15, category: 'equity' as const, allocation: 25.5 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', volatility: 0.30, beta: 1.1, var95: 0.042, var99: 0.065, expectedShortfall: 0.078, maxDrawdown: 0.22, sharpeRatio: 0.85, category: 'equity' as const, allocation: 18.2 },
    { symbol: 'BTC-USD', name: 'Bitcoin USD', volatility: 0.85, beta: 0.8, var95: 0.125, var99: 0.185, expectedShortfall: 0.215, maxDrawdown: 0.65, sharpeRatio: 0.75, category: 'crypto' as const, allocation: 15.8 },
    { symbol: 'SPY', name: 'S&P 500 ETF', volatility: 0.18, beta: 1.0, var95: 0.025, var99: 0.040, expectedShortfall: 0.048, maxDrawdown: 0.12, sharpeRatio: 0.95, category: 'etf' as const, allocation: 20.0 },
    { symbol: 'GLD', name: 'Gold ETF', volatility: 0.22, beta: -0.1, var95: 0.030, var99: 0.048, expectedShortfall: 0.058, maxDrawdown: 0.18, sharpeRatio: 0.65, category: 'commodity' as const, allocation: 10.5 },
  ];

  // Strategy data
  const strategies = [
    {
      id: '1',
      name: 'Momentum Strategy',
      color: '#00ff00',
      totalReturn: 15.2,
      volatility: 22.5,
      sharpeRatio: 1.15,
      maxDrawdown: 12.8,
      winRate: 68.5,
      totalTrades: 245,
      avgTrade: 0.85,
      profitFactor: 1.45,
      performance: performanceData.map(d => ({ ...d })),
      monthlyReturns: Array.from({ length: 12 }, () => (Math.random() - 0.3) * 0.1),
      returns: Array.from({ length: 252 }, () => (Math.random() - 0.45) * 0.04),
    },
    {
      id: '2',
      name: 'Mean Reversion',
      color: '#ffff00',
      totalReturn: 8.7,
      volatility: 18.2,
      sharpeRatio: 0.85,
      maxDrawdown: 8.5,
      winRate: 72.1,
      totalTrades: 156,
      avgTrade: 0.65,
      profitFactor: 1.28,
      performance: performanceData.map(d => ({ ...d, value: d.portfolio * 0.95 })),
      monthlyReturns: Array.from({ length: 12 }, () => (Math.random() - 0.4) * 0.08),
      returns: Array.from({ length: 252 }, () => (Math.random() - 0.48) * 0.035),
    },
    {
      id: '3',
      name: 'Value Investing',
      color: '#ff6600',
      totalReturn: 12.3,
      volatility: 25.8,
      sharpeRatio: 0.75,
      maxDrawdown: 18.2,
      winRate: 65.8,
      totalTrades: 89,
      avgTrade: 1.15,
      profitFactor: 1.35,
      performance: performanceData.map(d => ({ ...d, value: d.portfolio * 1.05 })),
      monthlyReturns: Array.from({ length: 12 }, () => (Math.random() - 0.35) * 0.12),
      returns: Array.from({ length: 252 }, () => (Math.random) - 0.42 * 0.045),
    },
  ];

  // Correlation data
  const correlationData = [
    { asset1: 'AAPL', asset2: 'GOOGL', correlation: 0.75, pValue: 0.001, sampleSize: 252 },
    { asset1: 'AAPL', asset2: 'BTC-USD', correlation: 0.45, pValue: 0.01, sampleSize: 252 },
    { asset1: 'GOOGL', asset2: 'BTC-USD', correlation: 0.35, pValue: 0.05, sampleSize: 252 },
    { asset1: 'AAPL', asset2: 'SPY', correlation: 0.85, pValue: 0.001, sampleSize: 252 },
    { asset1: 'GOOGL', asset2: 'SPY', correlation: 0.78, pValue: 0.001, sampleSize: 252 },
    { asset1: 'BTC-USD', asset2: 'SPY', correlation: 0.25, pValue: 0.1, sampleSize: 252 },
    { asset1: 'AAPL', asset2: 'GLD', correlation: -0.15, pValue: 0.15, sampleSize: 252 },
    { asset1: 'GOOGL', asset2: 'GLD', correlation: -0.08, pValue: 0.2, sampleSize: 252 },
    { asset1: 'BTC-USD', asset2: 'GLD', correlation: 0.12, pValue: 0.15, sampleSize: 252 },
    { asset1: 'SPY', asset2: 'GLD', correlation: -0.22, pValue: 0.05, sampleSize: 252 },
  ];

  // Candlestick data
  const candlestickData = Array.from({ length: 100 }, (_, i) => {
    const basePrice = 150;
    const volatility = 0.02;
    const change = (Math.random() - 0.5) * volatility;
    const open = basePrice * (1 + change);
    const close = open * (1 + (Math.random() - 0.5) * volatility);
    const high = Math.max(open, close) * (1 + Math.random() * volatility);
    const low = Math.min(open, close) * (1 - Math.random() * volatility);
    
    return {
      time: Date.now() - (100 - i) * 24 * 60 * 60 * 1000,
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 1000000) + 500000,
    };
  });

  return {
    positions,
    performanceData,
    pnlData,
    riskData,
    strategies,
    correlationData,
    candlestickData,
  };
};

export default function VisualizationDemo() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1Y');
  const [selectedSymbol, setSelectedSymbol] = useState<any>(null);
  const [dateRange, setDateRange] = useState({ startDate: new Date(), endDate: new Date() });
  const [activeTab, setActiveTab] = useState('overview');
  
  const portfolioRef = useRef<HTMLDivElement>(null);
  const performanceRef = useRef<HTMLDivElement>(null);
  const riskRef = useRef<HTMLDivElement>(null);
  
  const mockData = generateMockData();

  // Handle symbol selection
  const handleSymbolSelect = (symbol: any) => {
    setSelectedSymbol(symbol);
    console.log('Selected symbol:', symbol);
  };

  // Handle date range change
  const handleDateRangeChange = (range: any) => {
    setDateRange(range);
    console.log('Date range changed:', range);
  };

  // Export functions
  const exportChart = async (ref: React.RefObject<HTMLDivElement>, filename: string) => {
    if (ref.current) {
      await exportUtils.quickExport(ref.current, 'png', filename);
    }
  };

  const exportReport = async () => {
    const elements = [
      { element: portfolioRef.current!, title: 'Portfolio Allocation', description: 'Current portfolio distribution' },
      { element: performanceRef.current!, title: 'Performance Analysis', description: 'Historical performance metrics' },
      { element: riskRef.current!, title: 'Risk Analysis', description: 'Portfolio risk assessment' },
    ];
    
    const exporter = exportUtils.quickExport;
    // Implementation would use the full exporter here
    console.log('Exporting report...', elements);
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: TrendingUp },
    { id: 'portfolio', label: 'Portfolio', icon: Activity },
    { id: 'analysis', label: 'Analysis', icon: Filter },
  ];

  return (
    <div className="min-h-screen bg-matrix-black text-matrix-green p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold matrix-glow-text mb-2">
              ADVANCED TRADING VISUALIZATIONS
            </h1>
            <p className="text-matrix-green/70 font-mono">
              Comprehensive visualization suite for trading analytics
            </p>
          </div>
          
          <div className="flex gap-3">
            <MatrixButton onClick={exportReport}>
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </MatrixButton>
            <MatrixButton variant="secondary">
              <Share className="w-4 h-4 mr-2" />
              Share
            </MatrixButton>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DateRangePicker
            onDateRangeChange={handleDateRangeChange}
          />
          <SymbolSearch
            onSymbolSelect={handleSymbolSelect}
            placeholder="Search symbols, ETFs, crypto..."
          />
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-2 border-b border-matrix-green/30">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-2 font-mono transition-colors
                  ${activeTab === tab.id 
                    ? 'bg-matrix-green text-matrix-black border-b-2 border-matrix-green' 
                    : 'text-matrix-green hover:bg-matrix-green/10'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Portfolio Summary */}
            <div ref={portfolioRef}>
              <PortfolioAllocation
                positions={mockData.positions}
                totalValue={mockData.positions.reduce((sum, p) => sum + p.marketValue, 0)}
                title="Portfolio Overview"
                showPnL={true}
              />
            </div>

            {/* Performance Chart */}
            <div ref={performanceRef}>
              <PerformanceChart
                data={mockData.performanceData}
                title="Performance Analysis"
                showBenchmark={true}
                showDrawdown={true}
                showVolume={true}
                chartType="area"
                onTimeframeChange={setSelectedTimeframe}
              />
            </div>

            {/* Strategy Comparison */}
            <StrategyComparison
              strategies={mockData.strategies}
              title="Strategy Performance Comparison"
              showScatterPlot={true}
              showCorrelation={true}
            />
          </div>
        )}

        {/* Portfolio Tab */}
        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            {/* Allocation and Performance Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <PortfolioAllocation
                  positions={mockData.positions}
                  totalValue={mockData.positions.reduce((sum, p) => sum + p.marketValue, 0)}
                  title="Asset Allocation"
                  viewType="donut"
                />
              </div>
              <div>
                <PerformanceChart
                  data={mockData.performanceData}
                  title="Portfolio Value"
                  chartType="line"
                  showBenchmark={false}
                />
              </div>
            </div>

            {/* P&L Waterfall */}
            <div ref={riskRef}>
              <PnLWaterfall
                data={mockData.pnlData}
                title="P&L Attribution Analysis"
                showBreakdown={true}
                currency="USD"
              />
            </div>
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-6">
            {/* Risk Heatmap */}
            <RiskHeatmap
              data={mockData.riskData}
              title="Portfolio Risk Analysis"
              riskMetric="volatility"
              interactive={true}
              showTooltip={true}
            />

            {/* Correlation Matrix */}
            <CorrelationMatrix
              correlationData={mockData.correlationData}
              title="Asset Correlation Matrix"
              assetNames={{
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet',
                'BTC-USD': 'Bitcoin',
                'SPY': 'S&P 500 ETF',
                'GLD': 'Gold ETF'
              }}
              interactive={true}
              showValues={true}
            />

            {/* Candlestick Chart */}
            <CandlestickChart
              data={mockData.candlestickData}
              title="Real-time Price Chart"
              showVolume={true}
              height={400}
            />

            {/* Chart.js Demo */}
            <ChartJSCanvas
              type="doughnut"
              data={{
                labels: mockData.positions.map(p => p.symbol),
                datasets: [{
                  data: mockData.positions.map(p => p.allocation),
                  backgroundColor: ['#00ff00', '#00cc00', '#009900'],
                }]
              }}
              title="Alternative Allocation View"
              height={300}
            />

            {/* TradingView Chart Demo */}
            <TradingViewChart
              data={mockData.candlestickData}
              chartType="candlestick"
              indicators={{
                sma: [{ period: 20, color: '#00ff00' }, { period: 50, color: '#ffff00' }],
                ema: [{ period: 12, color: '#ff6600' }],
              }}
              height={350}
              showVolume={true}
            />
          </div>
        )}

        {/* Export Controls */}
        <div className="flex justify-center gap-4 pt-6 border-t border-matrix-green/30">
          <MatrixButton onClick={() => exportChart(portfolioRef, 'portfolio-allocation')}>
            Export Portfolio Chart
          </MatrixButton>
          <MatrixButton onClick={() => exportChart(performanceRef, 'performance-analysis')}>
            Export Performance Chart
          </MatrixButton>
          <MatrixButton onClick={() => exportChart(riskRef, 'risk-analysis')}>
            Export Risk Chart
          </MatrixButton>
        </div>
      </div>
    </div>
  );
}