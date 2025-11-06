import React, { useState, useMemo } from 'react';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { AlertTriangle, Activity, TrendingDown, Target } from 'lucide-react';

interface AssetRiskData {
  symbol: string;
  name: string;
  volatility: number;
  beta: number;
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number; // CVaR
  maxDrawdown: number;
  sharpeRatio: number;
  category: 'equity' | 'bond' | 'crypto' | 'commodity' | 'currency';
  allocation: number;
}

interface RiskHeatmapProps {
  data: AssetRiskData[];
  title?: string;
  className?: string;
  riskMetric?: 'volatility' | 'var95' | 'var99' | 'expectedShortfall' | 'maxDrawdown' | 'beta';
  showTooltip?: boolean;
  interactive?: boolean;
  onAssetClick?: (asset: AssetRiskData) => void;
  onRiskAlert?: (asset: AssetRiskData, riskLevel: 'low' | 'medium' | 'high' | 'extreme') => void;
}

export const RiskHeatmap: React.FC<RiskHeatmapProps> = ({
  data,
  title = "Risk Analysis Heatmap",
  className = "",
  riskMetric = 'volatility',
  showTooltip = true,
  interactive = true,
  onAssetClick,
  onRiskAlert,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<typeof riskMetric>(riskMetric);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [hoveredAsset, setHoveredAsset] = useState<AssetRiskData | null>(null);
  const [showPercentiles, setShowPercentiles] = useState(false);

  // Calculate risk levels and thresholds
  const riskAnalysis = useMemo(() => {
    if (!data.length) return { low: [], medium: [], high: [], extreme: [] };

    const metricValues = data.map(asset => asset[selectedMetric]);
    const sorted = [...metricValues].sort((a, b) => a - b);
    
    const percentiles = {
      p25: sorted[Math.floor(sorted.length * 0.25)],
      p50: sorted[Math.floor(sorted.length * 0.50)],
      p75: sorted[Math.floor(sorted.length * 0.75)],
      p90: sorted[Math.floor(sorted.length * 0.90)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
    };

    // Classify assets by risk level
    const classified = {
      low: data.filter(asset => asset[selectedMetric] <= percentiles.p25),
      medium: data.filter(asset => asset[selectedMetric] > percentiles.p25 && asset[selectedMetric] <= percentiles.p50),
      high: data.filter(asset => asset[selectedMetric] > percentiles.p50 && asset[selectedMetric] <= percentiles.p75),
      extreme: data.filter(asset => asset[selectedMetric] > percentiles.p75),
    };

    // Check for risk alerts
    classified.extreme.forEach(asset => {
      if (onRiskAlert) {
        const riskLevel = asset[selectedMetric] > percentiles.p95 ? 'extreme' : 'high';
        onRiskAlert(asset, riskLevel);
      }
    });

    return { ...classified, percentiles };
  }, [data, selectedMetric, onRiskAlert]);

  // Get risk color based on value and percentiles
  const getRiskColor = (value: number) => {
    const { percentiles } = riskAnalysis;
    if (value <= percentiles.p25) return '#00ff00'; // Green - Low risk
    if (value <= percentiles.p50) return '#88ff00'; // Light green - Low-Medium
    if (value <= percentiles.p75) return '#ffdd00'; // Yellow - Medium
    if (value <= percentiles.p90) return '#ff8800'; // Orange - High
    return '#ff0000'; // Red - Extreme risk
  };

  // Get risk intensity (opacity)
  const getRiskIntensity = (value: number, asset: AssetRiskData) => {
    const { percentiles } = riskAnalysis;
    const maxValue = Math.max(...data.map(d => d[selectedMetric]));
    const normalizedValue = value / maxValue;
    
    // Adjust opacity based on allocation (higher allocation = more visible)
    const allocationWeight = Math.min(asset.allocation / 20, 1); // Cap at 20% allocation
    return Math.max(0.3, Math.min(1, normalizedValue + allocationWeight * 0.3));
  };

  // Format risk metric value
  const formatRiskValue = (value: number, metric: string) => {
    switch (metric) {
      case 'volatility':
      case 'var95':
      case 'var99':
      case 'expectedShortfall':
      case 'maxDrawdown':
        return `${(value * 100).toFixed(2)}%`;
      case 'beta':
        return value.toFixed(2);
      default:
        return value.toFixed(2);
    }
  };

  // Get metric icon
  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'volatility':
        return <Activity className="w-4 h-4" />;
      case 'var95':
      case 'var99':
        return <TrendingDown className="w-4 h-4" />;
      case 'expectedShortfall':
        return <AlertTriangle className="w-4 h-4" />;
      case 'maxDrawdown':
        return <TrendingDown className="w-4 h-4" />;
      case 'beta':
        return <Target className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  // Sort data
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => {
      const aValue = a[selectedMetric];
      const bValue = b[selectedMetric];
      return sortOrder === 'desc' ? bValue - aValue : aValue - bValue;
    });
  }, [data, selectedMetric, sortOrder]);

  // Risk metrics configuration
  const riskMetrics = [
    { key: 'volatility', label: 'Volatility', description: 'Price volatility' },
    { key: 'var95', label: 'VaR 95%', description: 'Value at Risk 95%' },
    { key: 'var99', label: 'VaR 99%', description: 'Value at Risk 99%' },
    { key: 'expectedShortfall', label: 'CVaR', description: 'Conditional Value at Risk' },
    { key: 'maxDrawdown', label: 'Max DD', description: 'Maximum Drawdown' },
    { key: 'beta', label: 'Beta', description: 'Market correlation' },
  ] as const;

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-2 items-center">
          <div className="flex gap-1">
            {riskMetrics.map((metric) => (
              <MatrixButton
                key={metric.key}
                size="sm"
                variant={selectedMetric === metric.key ? 'primary' : 'secondary'}
                onClick={() => setSelectedMetric(metric.key)}
                title={metric.description}
              >
                <div className="flex items-center gap-1">
                  {getMetricIcon(metric.key)}
                  <span className="font-mono text-xs">{metric.label}</span>
                </div>
              </MatrixButton>
            ))}
          </div>

          <div className="flex gap-1 ml-auto">
            <MatrixButton
              size="sm"
              variant={sortOrder === 'desc' ? 'primary' : 'secondary'}
              onClick={() => setSortOrder('desc')}
            >
              HIGH TO LOW
            </MatrixButton>
            <MatrixButton
              size="sm"
              variant={sortOrder === 'asc' ? 'primary' : 'secondary'}
              onClick={() => setSortOrder('asc')}
            >
              LOW TO HIGH
            </MatrixButton>
            <MatrixButton
              size="sm"
              variant={showPercentiles ? 'primary' : 'secondary'}
              onClick={() => setShowPercentiles(!showPercentiles)}
            >
              PERCENTILES
            </MatrixButton>
          </div>
        </div>

        {/* Risk Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-green-900/20 border border-green-500/30 rounded">
            <p className="text-green-400 font-mono text-sm">Low Risk</p>
            <p className="text-green-400 font-bold text-xl">{riskAnalysis.low.length}</p>
          </div>
          <div className="text-center p-3 bg-yellow-900/20 border border-yellow-500/30 rounded">
            <p className="text-yellow-400 font-mono text-sm">Medium Risk</p>
            <p className="text-yellow-400 font-bold text-xl">{riskAnalysis.medium.length}</p>
          </div>
          <div className="text-center p-3 bg-orange-900/20 border border-orange-500/30 rounded">
            <p className="text-orange-400 font-mono text-sm">High Risk</p>
            <p className="text-orange-400 font-bold text-xl">{riskAnalysis.high.length}</p>
          </div>
          <div className="text-center p-3 bg-red-900/20 border border-red-500/30 rounded">
            <p className="text-red-400 font-mono text-sm">Extreme Risk</p>
            <p className="text-red-400 font-bold text-xl">{riskAnalysis.extreme.length}</p>
          </div>
        </div>

        {/* Percentiles */}
        {showPercentiles && (
          <div className="bg-matrix-green/10 border border-matrix-green/30 rounded p-4">
            <h4 className="text-matrix-green font-mono text-sm mb-3">Risk Percentiles - {selectedMetric.toUpperCase()}</h4>
            <div className="grid grid-cols-5 gap-4 text-center">
              {Object.entries(riskAnalysis.percentiles).map(([percentile, value]) => (
                <div key={percentile}>
                  <p className="text-matrix-green/70 font-mono text-xs">{percentile.toUpperCase()}</p>
                  <p className="text-matrix-green font-mono font-bold">
                    {formatRiskValue(value, selectedMetric)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Heatmap Grid */}
        <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-7 gap-2">
          {sortedData.map((asset, index) => {
            const riskValue = asset[selectedMetric];
            const riskColor = getRiskColor(riskValue);
            const riskIntensity = getRiskIntensity(riskValue, asset);
            
            return (
              <div
                key={asset.symbol}
                className={`
                  relative aspect-square rounded-lg border-2 transition-all cursor-pointer
                  ${interactive ? 'hover:scale-105 hover:z-10' : ''}
                `}
                style={{
                  backgroundColor: riskColor,
                  opacity: riskIntensity,
                  borderColor: riskColor,
                }}
                onMouseEnter={() => setHoveredAsset(asset)}
                onMouseLeave={() => setHoveredAsset(null)}
                onClick={() => interactive && onAssetClick?.(asset)}
                title={`${asset.symbol}: ${formatRiskValue(riskValue, selectedMetric)}`}
              >
                {/* Asset Symbol */}
                <div className="absolute inset-0 flex flex-col items-center justify-center text-black font-mono font-bold text-xs p-1">
                  <span className="text-center leading-tight">{asset.symbol}</span>
                  <span className="text-xs opacity-80">
                    {formatRiskValue(riskValue, selectedMetric)}
                  </span>
                </div>

                {/* Allocation indicator */}
                <div 
                  className="absolute top-1 right-1 w-2 h-2 rounded-full bg-black/30"
                  style={{
                    width: `${Math.min(Math.sqrt(asset.allocation) * 4, 8)}px`,
                    height: `${Math.min(Math.sqrt(asset.allocation) * 4, 8)}px`,
                  }}
                />

                {/* Risk level indicator */}
                <div className="absolute bottom-1 left-1 flex gap-1">
                  {riskAnalysis.extreme.includes(asset) && (
                    <div className="w-1 h-1 bg-black/50 rounded-full" />
                  )}
                  {riskAnalysis.high.includes(asset) && (
                    <div className="w-1 h-1 bg-black/50 rounded-full" />
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Hover Tooltip */}
        {showTooltip && hoveredAsset && (
          <div className="bg-matrix-black border border-matrix-green rounded-lg p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-matrix-green font-mono font-bold">{hoveredAsset.symbol}</p>
                <p className="text-matrix-green/70 text-sm font-mono">{hoveredAsset.name}</p>
                <p className="text-matrix-green/50 text-xs font-mono">{hoveredAsset.category}</p>
              </div>
              
              <div>
                <p className="text-matrix-green/70 font-mono text-sm">Selected Metric</p>
                <p className="text-matrix-green font-mono font-bold">
                  {formatRiskValue(hoveredAsset[selectedMetric], selectedMetric)}
                </p>
              </div>
              
              <div>
                <p className="text-matrix-green/70 font-mono text-sm">Allocation</p>
                <p className="text-matrix-green font-mono font-bold">
                  {hoveredAsset.allocation.toFixed(2)}%
                </p>
              </div>
              
              <div>
                <p className="text-matrix-green/70 font-mono text-sm">Sharpe Ratio</p>
                <p className={`font-mono font-bold ${
                  hoveredAsset.sharpeRatio >= 1 ? 'text-green-400' : 
                  hoveredAsset.sharpeRatio >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {hoveredAsset.sharpeRatio.toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Risk Legend */}
        <div className="flex justify-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span className="text-green-400">Low Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span className="text-yellow-400">Medium Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-500 rounded"></div>
            <span className="text-orange-400">High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span className="text-red-400">Extreme Risk</span>
          </div>
        </div>
      </div>
    </MatrixCard>
  );
};

export default RiskHeatmap;