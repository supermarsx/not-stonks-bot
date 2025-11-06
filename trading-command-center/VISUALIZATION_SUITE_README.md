# Advanced Trading Visualization Suite

A comprehensive React-based visualization suite for trading applications, featuring advanced charting capabilities, portfolio analytics, risk assessment, and strategy comparison tools.

## 🚀 Features

### Chart Library Integration
- **TradingView Lightweight Charts**: Professional-grade financial charts with real-time data
- **Chart.js Integration**: Flexible charting for various data visualizations
- **Recharts**: React-native charts with excellent performance

### Portfolio Visualizations
- **Portfolio Allocation**: Interactive pie/donut charts with real-time P&L tracking
- **Performance Chart**: Multi-timeframe performance analysis with benchmarks
- **Asset Class Breakdown**: Detailed allocation analysis by asset type

### Risk & Analytics
- **Risk Heatmap**: Visual risk assessment with percentile-based color coding
- **P&L Waterfall**: Attribution analysis with categorical breakdowns
- **Correlation Matrix**: Interactive asset correlation visualization
- **Strategy Comparison**: Multi-strategy performance analysis

### Interactive Filters
- **Date Range Picker**: Flexible timeframe selection with presets
- **Symbol Search**: Advanced symbol search with watchlist integration
- **Real-time Data**: WebSocket integration for live updates

### Export Capabilities
- **PNG/PDF Export**: High-quality chart exports
- **SVG Export**: Vector graphics for publications
- **CSV Data Export**: Raw data export for analysis
- **Report Generation**: Multi-chart PDF reports

## 📦 Installation

```bash
# Install dependencies
pnpm add lightweight-charts chart.js react-chartjs-2 recharts html2canvas jspdf

# Install types
pnpm add -D @types/html2canvas
```

## 🎯 Quick Start

### 1. Portfolio Allocation Component

```tsx
import { PortfolioAllocation } from '@/components/visualizations/PortfolioAllocation';

const positions = [
  {
    symbol: 'AAPL',
    size: 100,
    marketValue: 17500,
    allocation: 25.5,
    pnl: 1200,
    pnlPercentage: 7.35,
    side: 'long',
  }
];

<PortfolioAllocation
  positions={positions}
  totalValue={68668}
  title="My Portfolio"
  showPnL={true}
  viewType="donut"
/>
```

### 2. Performance Chart

```tsx
import { PerformanceChart } from '@/components/visualizations/PerformanceChart';

const performanceData = [
  { date: '2024-01-01', portfolio: 100000, benchmark: 100000 },
  { date: '2024-01-02', portfolio: 101500, benchmark: 100800 },
  // ... more data
];

<PerformanceChart
  data={performanceData}
  title="Portfolio Performance"
  showBenchmark={true}
  showDrawdown={true}
  chartType="area"
/>
```

### 3. Candlestick Chart with Indicators

```tsx
import { CandlestickChart } from '@/components/charts/CandlestickChart';

const candlestickData = [
  { time: 1640995200000, open: 150, high: 155, low: 148, close: 152, volume: 1000000 },
  // ... more data
];

<CandlestickChart
  data={candlestickData}
  title="AAPL Price Chart"
  showVolume={true}
  height={500}
/>
```

### 4. Risk Heatmap

```tsx
import { RiskHeatmap } from '@/components/visualizations/RiskHeatmap';

const riskData = [
  {
    symbol: 'AAPL',
    volatility: 0.25,
    var95: 0.035,
    maxDrawdown: 0.15,
    sharpeRatio: 1.15,
    allocation: 25.5
  }
];

<RiskHeatmap
  data={riskData}
  title="Portfolio Risk Analysis"
  riskMetric="volatility"
  interactive={true}
/>
```

### 5. Export Functionality

```tsx
import { exportUtils } from '@/utils/visualizationExporter';

// Export chart to PNG
await exportUtils.quickExport(chartElement, 'png', 'my-chart');

// Export data to CSV
exportUtils.exportData(chartData, 'portfolio-data');

// Create export buttons
const exportButton = exportUtils.createExportButtons(container, 'chart-name');
```

## 🧩 Component Library

### Chart Components

| Component | Description | Chart Type |
|-----------|-------------|------------|
| `TradingViewChart` | Professional financial charts | Candlestick, Line, Bar, Histogram |
| `ChartJSCanvas` | Flexible charting with Chart.js | Line, Bar, Doughnut, Pie, Scatter |
| `CandlestickChart` | Advanced candlestick with indicators | Candlestick with SMA, EMA, RSI, MACD |

### Visualization Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| `PortfolioAllocation` | Portfolio allocation analysis | Pie/Donut/Bar views, P&L tracking |
| `PerformanceChart` | Performance tracking | Multi-timeframe, benchmarks, drawdowns |
| `PnLWaterfall` | P&L attribution | Category breakdown, cumulative view |
| `RiskHeatmap` | Risk visualization | Percentile-based color coding |
| `StrategyComparison` | Multi-strategy analysis | Performance, correlation, scatter plots |
| `CorrelationMatrix` | Asset correlation analysis | Interactive matrix, filtering |

### Filter Components

| Component | Description | Features |
|-----------|-------------|----------|
| `DateRangePicker` | Flexible date selection | Presets, calendar, quick ranges |
| `SymbolSearch` | Advanced symbol search | Type filtering, watchlist integration |

## 📊 Data Formats

### Portfolio Position Data
```typescript
interface Position {
  id: string;
  symbol: string;
  name?: string;
  size: number;
  marketValue: number;
  allocation: number;
  pnl: number;
  pnlPercentage: number;
  side: 'long' | 'short';
  entryPrice?: number;
  currentPrice?: number;
  timestamp?: string;
  broker?: string;
}
```

### Performance Data
```typescript
interface PerformanceDataPoint {
  date: string;
  portfolio: number;
  benchmark?: number;
  drawdown?: number;
  volatility?: number;
  returns?: number;
  volume?: number;
}
```

### Risk Data
```typescript
interface AssetRiskData {
  symbol: string;
  name: string;
  volatility: number;
  beta: number;
  var95: number;
  var99: number;
  expectedShortfall: number;
  maxDrawdown: number;
  sharpeRatio: number;
  category: 'equity' | 'bond' | 'crypto' | 'commodity' | 'currency';
  allocation: number;
}
```

## 🎨 Styling & Theming

The visualization suite uses a Matrix-inspired dark theme with green accents:

```css
/* Primary colors */
--matrix-black: #0a0a0a;
--matrix-green: #00ff00;
--matrix-green-dark: #00cc00;

/* Usage */
.chart-container {
  background-color: var(--matrix-black);
  color: var(--matrix-green);
}
```

## 🔧 Configuration

### TradingView Charts
```tsx
const tradingViewConfig = {
  theme: 'dark',
  showGrid: true,
  showVolume: true,
  indicators: {
    sma: [{ period: 20, color: '#00ff00' }],
    ema: [{ period: 12, color: '#ffff00' }],
    rsi: { period: 14, color: '#ff6600' }
  }
};
```

### Export Options
```tsx
const exportOptions = {
  format: 'png' | 'pdf' | 'svg' | 'csv',
  filename: 'chart-name',
  quality: 1.0,
  width: 1920,
  height: 1080,
  includeData: true
};
```

## 🌟 Advanced Features

### Technical Indicators
- **SMA/EMA**: Moving average overlays
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **Volume Analysis**: Volume indicators

### Interactive Features
- **Zoom & Pan**: Chart navigation
- **Hover Tooltips**: Detailed data on hover
- **Click Actions**: Drill-down capabilities
- **Real-time Updates**: WebSocket integration

### Risk Metrics
- **VaR**: Value at Risk (95%, 99%)
- **CVaR**: Conditional Value at Risk
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough analysis
- **Beta**: Market correlation

## 🚀 Performance Optimization

- **Lazy Loading**: Components load on demand
- **Memoization**: React.useMemo for expensive calculations
- **Virtualization**: Large datasets handled efficiently
- **WebGL Rendering**: Hardware-accelerated charts
- **Debounced Updates**: Reduced re-renders

## 📱 Responsive Design

All components are fully responsive with mobile-optimized layouts:
- Grid layouts adapt to screen size
- Touch-friendly interactions
- Optimized for tablets and phones

## 🔒 Security

- **Safe Data Handling**: No XSS vulnerabilities
- **Input Sanitization**: All user inputs validated
- **CORS Support**: Cross-origin requests handled safely

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🎯 Demo

Check out the complete demo at `src/pages/VisualizationDemo.tsx` to see all components in action with mock data.

## 🆘 Support

For questions and support:
- Check the demo code for examples
- Review component prop types
- Consult TradingView Lightweight Charts and Chart.js documentation

---

**Built with ❤️ for the trading community**