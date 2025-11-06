import React, { useState, useEffect } from 'react';
import { TradingViewChart } from './TradingViewChart';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';

interface CandlestickData {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface Indicator {
  id: string;
  name: string;
  enabled: boolean;
  type: 'sma' | 'ema' | 'rsi' | 'macd' | 'bollinger' | 'volume';
  period?: number;
  color?: string;
}

interface CandlestickChartProps {
  data: CandlestickData[];
  title?: string;
  className?: string;
  height?: number;
  showVolume?: boolean;
  initialIndicators?: Indicator[];
  onIndicatorChange?: (indicators: Indicator[]) => void;
  onPriceAlert?: (price: number, type: 'resistance' | 'support') => void;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  title = "Price Chart",
  className = "",
  height = 500,
  showVolume = true,
  initialIndicators = [],
  onIndicatorChange,
  onPriceAlert,
}) => {
  const [indicators, setIndicators] = useState<Indicator[]>([
    { id: 'sma20', name: 'SMA 20', enabled: true, type: 'sma', period: 20, color: '#00ff00' },
    { id: 'sma50', name: 'SMA 50', enabled: true, type: 'sma', period: 50, color: '#ffff00' },
    { id: 'ema12', name: 'EMA 12', enabled: false, type: 'ema', period: 12, color: '#ff6600' },
    { id: 'ema26', name: 'EMA 26', enabled: false, type: 'ema', period: 26, color: '#ff0066' },
    { id: 'rsi14', name: 'RSI 14', enabled: false, type: 'rsi', period: 14, color: '#00ffff' },
    { id: 'macd', name: 'MACD', enabled: false, type: 'macd', period: 12, color: '#ff00ff' },
    { id: 'bollinger', name: 'Bollinger Bands', enabled: false, type: 'bollinger', period: 20, color: '#6600ff' },
    { id: 'volume', name: 'Volume', enabled: true, type: 'volume', color: '#00ff0033' },
  ]);

  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePercent, setPriceChangePercent] = useState(0);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1m' | '5m' | '15m' | '1h' | '1d' | '1w'>('1d');

  // Calculate current price and change
  useEffect(() => {
    if (data.length > 0) {
      const latest = data[data.length - 1];
      const previous = data[data.length - 2];
      
      setCurrentPrice(latest.close);
      
      if (previous) {
        const change = latest.close - previous.close;
        const changePercent = (change / previous.close) * 100;
        
        setPriceChange(change);
        setPriceChangePercent(changePercent);
      }
    }
  }, [data]);

  // Handle indicator toggle
  const toggleIndicator = (indicatorId: string) => {
    const newIndicators = indicators.map(indicator =>
      indicator.id === indicatorId
        ? { ...indicator, enabled: !indicator.enabled }
        : indicator
    );
    setIndicators(newIndicators);
    onIndicatorChange?.(newIndicators);
  };

  // Convert indicators to TradingView format
  const getTradingViewIndicators = () => {
    const tvIndicators: any = {};
    
    indicators.filter(ind => ind.enabled).forEach(indicator => {
      switch (indicator.type) {
        case 'sma':
          if (!tvIndicators.sma) tvIndicators.sma = [];
          tvIndicators.sma.push({
            period: indicator.period,
            color: indicator.color,
          });
          break;
        case 'ema':
          if (!tvIndicators.ema) tvIndicators.ema = [];
          tvIndicators.ema.push({
            period: indicator.period,
            color: indicator.color,
          });
          break;
        case 'rsi':
          tvIndicators.rsi = {
            period: indicator.period,
            color: indicator.color,
          };
          break;
        case 'macd':
          tvIndicators.macd = {
            fast: 12,
            slow: 26,
            signal: 9,
          };
          break;
        case 'bollinger':
          tvIndicators.bollinger = {
            period: indicator.period,
            deviation: 2,
          };
          break;
      }
    });

    return tvIndicators;
  };

  // Calculate support and resistance levels
  const supportResistanceLevels = React.useMemo(() => {
    if (data.length < 20) return { support: [], resistance: [] };
    
    const highs = data.slice(-20).map(d => d.high);
    const lows = data.slice(-20).map(d => d.low);
    
    // Simple support/resistance calculation
    const support = Math.min(...lows);
    const resistance = Math.max(...highs);
    
    return {
      support: [support * 0.99, support, support * 1.01],
      resistance: [resistance * 0.99, resistance, resistance * 1.01],
    };
  }, [data]);

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Price Info Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <h3 className="text-2xl font-bold text-matrix-green font-mono">
                {formatCurrency(currentPrice)}
              </h3>
              <div className={`flex items-center gap-1 ${
                priceChange >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {priceChange >= 0 ? (
                  <TrendingUp className="w-4 h-4" />
                ) : (
                  <TrendingDown className="w-4 h-4" />
                )}
                <span className="font-mono">
                  {formatCurrency(Math.abs(priceChange))} ({formatPercentage(priceChangePercent)})
                </span>
              </div>
            </div>
          </div>

          {/* Timeframe Selector */}
          <div className="flex gap-1">
            {(['1m', '5m', '15m', '1h', '1d', '1w'] as const).map((tf) => (
              <MatrixButton
                key={tf}
                size="sm"
                variant={selectedTimeframe === tf ? 'primary' : 'secondary'}
                onClick={() => setSelectedTimeframe(tf)}
              >
                {tf.toUpperCase()}
              </MatrixButton>
            ))}
          </div>
        </div>

        {/* Indicator Controls */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {indicators.map((indicator) => (
            <MatrixButton
              key={indicator.id}
              size="sm"
              variant={indicator.enabled ? 'primary' : 'secondary'}
              onClick={() => toggleIndicator(indicator.id)}
              className="text-left justify-start"
            >
              <div className="flex items-center gap-2">
                {indicator.type === 'sma' && <Activity className="w-3 h-3" />}
                {indicator.type === 'ema' && <TrendingUp className="w-3 h-3" />}
                {indicator.type === 'rsi' && <BarChart3 className="w-3 h-3" />}
                {indicator.type === 'macd' && <Activity className="w-3 h-3" />}
                {indicator.type === 'bollinger' && <BarChart3 className="w-3 h-3" />}
                {indicator.type === 'volume' && <BarChart3 className="w-3 h-3" />}
                <span className="font-mono text-xs">{indicator.name}</span>
              </div>
            </MatrixButton>
          ))}
        </div>

        {/* Chart */}
        <div>
          <TradingViewChart
            data={data}
            chartType="candlestick"
            indicators={getTradingViewIndicators()}
            height={height}
            showVolume={showVolume && indicators.find(ind => ind.type === 'volume' && ind.enabled)?.enabled}
            theme="dark"
            onPriceChange={(price) => {
              setCurrentPrice(price);
              // Check for support/resistance alerts
              if (onPriceAlert) {
                const nearestSupport = supportResistanceLevels.support.find(level => Math.abs(price - level) / level < 0.001);
                const nearestResistance = supportResistanceLevels.resistance.find(level => Math.abs(price - level) / level < 0.001);
                
                if (nearestSupport) {
                  onPriceAlert(price, 'support');
                } else if (nearestResistance) {
                  onPriceAlert(price, 'resistance');
                }
              }
            }}
          />
        </div>

        {/* Support/Resistance Levels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-matrix-green/10 border border-matrix-green/30 rounded p-3">
            <h4 className="text-matrix-green font-mono text-sm mb-2">Support Levels</h4>
            <div className="space-y-1">
              {supportResistanceLevels.support.map((level, index) => (
                <div key={index} className="flex justify-between items-center">
                  <span className="text-matrix-green/70 text-xs font-mono">
                    {index === 1 ? 'Strong' : index === 0 ? 'Weak' : 'Minor'}
                  </span>
                  <span className="text-matrix-green font-mono text-sm">
                    {formatCurrency(level)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-red-900/10 border border-red-500/30 rounded p-3">
            <h4 className="text-red-400 font-mono text-sm mb-2">Resistance Levels</h4>
            <div className="space-y-1">
              {supportResistanceLevels.resistance.map((level, index) => (
                <div key={index} className="flex justify-between items-center">
                  <span className="text-red-400/70 text-xs font-mono">
                    {index === 1 ? 'Strong' : index === 0 ? 'Weak' : 'Minor'}
                  </span>
                  <span className="text-red-400 font-mono text-sm">
                    {formatCurrency(level)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Volume Analysis */}
        {showVolume && (
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-matrix-green/70 text-sm font-mono">Volume</p>
              <p className="text-matrix-green font-bold">
                {data.length > 0 ? (data[data.length - 1].volume || 0).toLocaleString() : '0'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-matrix-green/70 text-sm font-mono">Avg Volume</p>
              <p className="text-matrix-green font-bold">
                {data.slice(-20).reduce((sum, d) => sum + (d.volume || 0), 0) / Math.min(20, data.length)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-matrix-green/70 text-sm font-mono">Volume Trend</p>
              <p className={`font-bold ${
                data.length > 1 && (data[data.length - 1].volume || 0) > (data[data.length - 2].volume || 0)
                  ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.length > 1 && (data[data.length - 1].volume || 0) > (data[data.length - 2].volume || 0)
                  ? 'Increasing' : 'Decreasing'}
              </p>
            </div>
          </div>
        )}
      </div>
    </MatrixCard>
  );
};

export default CandlestickChart;
