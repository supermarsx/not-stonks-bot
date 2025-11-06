import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { TrendingUp, TrendingDown, DollarSign, Target } from 'lucide-react';

interface PnLItem {
  id: string;
  label: string;
  category: 'trading' | 'fees' | 'dividends' | 'interest' | 'realized' | 'unrealized' | 'total';
  value: number;
  description?: string;
  symbol?: string;
}

interface PnLWaterfallProps {
  data: PnLItem[];
  title?: string;
  className?: string;
  showBreakdown?: boolean;
  showPercentage?: boolean;
  currency?: string;
  onItemClick?: (item: PnLItem) => void;
}

export const PnLWaterfall: React.FC<PnLWaterfallProps> = ({
  data,
  title = "P&L Waterfall Analysis",
  className = "",
  showBreakdown = true,
  showPercentage = true,
  currency = 'USD',
  onItemClick,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'value' | 'category'>('value');

  // Calculate cumulative values for waterfall effect
  const waterfallData = useMemo(() => {
    let cumulative = 0;
    const categories = ['trading', 'fees', 'dividends', 'interest', 'realized', 'unrealized'];
    
    // Filter data by category
    const filteredData = selectedCategory === 'all' 
      ? data 
      : data.filter(item => item.category === selectedCategory);

    // Sort data
    const sortedData = [...filteredData].sort((a, b) => {
      if (sortBy === 'value') {
        return Math.abs(b.value) - Math.abs(a.value);
      } else {
        return categories.indexOf(a.category) - categories.indexOf(b.category);
      }
    });

    return sortedData.map((item, index) => {
      const startValue = cumulative;
      cumulative += item.value;
      
      return {
        ...item,
        startValue,
        endValue: cumulative,
        displayValue: item.value,
        isTotal: item.category === 'total',
      };
    });
  }, [data, selectedCategory, sortBy]);

  // Calculate summary metrics
  const summary = useMemo(() => {
    const total = data.reduce((sum, item) => sum + item.value, 0);
    const positive = data.filter(item => item.value > 0).reduce((sum, item) => sum + item.value, 0);
    const negative = data.filter(item => item.value < 0).reduce((sum, item) => sum + item.value, 0);
    const tradingPnL = data.filter(item => item.category === 'trading').reduce((sum, item) => sum + item.value, 0);
    const fees = data.filter(item => item.category === 'fees').reduce((sum, item) => sum + item.value, 0);

    return {
      total,
      positive,
      negative,
      netPnL: tradingPnL,
      fees,
      winRate: data.filter(item => item.value > 0).length / data.length * 100,
    };
  }, [data]);

  // Get category colors
  const getCategoryColor = (category: string) => {
    const colors = {
      trading: summary.netPnL >= 0 ? '#00ff00' : '#ff0000',
      fees: '#ff6600',
      dividends: '#00ffff',
      interest: '#ffff00',
      realized: '#0066ff',
      unrealized: '#6600ff',
      total: '#ffffff',
    };
    return colors[category as keyof typeof colors] || '#00ff00';
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-matrix-black border border-matrix-green p-4 rounded shadow-lg max-w-xs">
          <p className="text-matrix-green font-mono font-bold mb-2">{label}</p>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-matrix-green/70 font-mono text-sm">Value:</span>
              <span className={`font-mono font-bold ${
                data.value >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(data.value)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-matrix-green/70 font-mono text-sm">Cumulative:</span>
              <span className={`font-mono font-bold ${
                data.endValue >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(data.endValue)}
              </span>
            </div>
            {data.symbol && (
              <div className="flex justify-between">
                <span className="text-matrix-green/70 font-mono text-sm">Symbol:</span>
                <span className="text-matrix-green font-mono">{data.symbol}</span>
              </div>
            )}
            {data.description && (
              <div className="mt-2 pt-2 border-t border-matrix-green/30">
                <p className="text-matrix-green/70 font-mono text-xs">{data.description}</p>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom bar component for waterfall effect
  const CustomBar = (props: any) => {
    const { payload, ...rest } = props;
    const isPositive = payload.value >= 0;
    const color = getCategoryColor(payload.category);
    
    return <Cell {...rest} fill={color} />;
  };

  const categories = ['all', 'trading', 'fees', 'dividends', 'interest', 'realized', 'unrealized'];

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center">
            <DollarSign className="w-6 h-6 text-matrix-green mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Total P&L</p>
            <p className={`font-bold ${summary.total >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatCurrency(summary.total)}
            </p>
          </div>
          
          <div className="text-center">
            <TrendingUp className="w-6 h-6 text-green-400 mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Gains</p>
            <p className="text-green-400 font-bold">
              {formatCurrency(summary.positive)}
            </p>
          </div>
          
          <div className="text-center">
            <TrendingDown className="w-6 h-6 text-red-400 mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Losses</p>
            <p className="text-red-400 font-bold">
              {formatCurrency(Math.abs(summary.negative))}
            </p>
          </div>
          
          <div className="text-center">
            <Target className="w-6 h-6 text-matrix-green mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Trading P&L</p>
            <p className={`font-bold ${summary.netPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatCurrency(summary.netPnL)}
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-6 h-6 mx-auto mb-1 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-green-400 mr-1" />
              <div className="w-2 h-2 rounded-full bg-red-400" />
            </div>
            <p className="text-matrix-green/70 text-sm font-mono">Win Rate</p>
            <p className="text-yellow-400 font-bold">
              {summary.winRate.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-2">
          <div className="flex gap-1">
            {categories.map((category) => (
              <MatrixButton
                key={category}
                size="sm"
                variant={selectedCategory === category ? 'primary' : 'secondary'}
                onClick={() => setSelectedCategory(category)}
              >
                {category.toUpperCase()}
              </MatrixButton>
            ))}
          </div>
          
          <div className="flex gap-1 ml-auto">
            <MatrixButton
              size="sm"
              variant={sortBy === 'value' ? 'primary' : 'secondary'}
              onClick={() => setSortBy('value')}
            >
              BY VALUE
            </MatrixButton>
            <MatrixButton
              size="sm"
              variant={sortBy === 'category' ? 'primary' : 'secondary'}
              onClick={() => setSortBy('category')}
            >
              BY CATEGORY
            </MatrixButton>
          </div>
        </div>

        {/* Waterfall Chart */}
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={waterfallData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#00ff0033" />
              <XAxis 
                dataKey="label" 
                stroke="#00ff00"
                tick={{ fontSize: 12, fontFamily: 'monospace' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                stroke="#00ff00"
                tickFormatter={(value) => formatCurrency(value)}
                tick={{ fontSize: 12, fontFamily: 'monospace' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="displayValue" 
                radius={[2, 2, 0, 0]}
                onClick={(data) => onItemClick?.(data)}
              >
                {waterfallData.map((entry, index) => (
                  <CustomBar key={`cell-${index}`} {...entry} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed Breakdown */}
        {showBreakdown && (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            <h4 className="text-matrix-green font-mono text-sm mb-3">Detailed Breakdown</h4>
            {waterfallData.map((item, index) => (
              <div
                key={item.id}
                onClick={() => onItemClick?.(item)}
                className="flex items-center justify-between p-3 bg-matrix-green/5 border border-matrix-green/20 rounded cursor-pointer hover:bg-matrix-green/10 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded"
                    style={{ backgroundColor: getCategoryColor(item.category) }}
                  />
                  <div>
                    <p className="text-matrix-green font-mono font-bold">{item.label}</p>
                    {item.symbol && (
                      <p className="text-matrix-green/70 font-mono text-xs">{item.symbol}</p>
                    )}
                  </div>
                </div>
                
                <div className="text-right">
                  <p className={`font-mono font-bold ${
                    item.value >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(item.value)}
                  </p>
                  {showPercentage && (
                    <p className="text-matrix-green/70 font-mono text-xs">
                      {formatPercentage((item.value / summary.total) * 100)}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </MatrixCard>
  );
};

export default PnLWaterfall;