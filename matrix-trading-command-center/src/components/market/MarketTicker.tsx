import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore, MarketData } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  BarChart3,
  DollarSign,
  Volume2,
  Clock,
  Zap,
  RefreshCw,
  Star,
  Filter
} from 'lucide-react';

export const MarketTicker: React.FC = () => {
  const { marketData, updateMarketData } = useTradingStore();
  const [sortBy, setSortBy] = useState<'symbol' | 'change' | 'volume'>('change');
  const [filterType, setFilterType] = useState<'all' | 'gainers' | 'losers'>('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [selectedStocks, setSelectedStocks] = useState<Set<string>>(new Set());

  // Simulate real-time market data updates
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      Object.values(marketData).forEach((data) => {
        // Simulate price movement
        const volatility = 0.02; // 2% max change
        const changePercent = (Math.random() - 0.5) * volatility * 100;
        const newPrice = data.price * (1 + changePercent / 100);
        const volume = data.volume + Math.floor((Math.random() - 0.5) * 1000000);

        const updatedData: MarketData = {
          ...data,
          price: Math.max(0.01, newPrice),
          change: newPrice - data.open,
          changePercent: ((newPrice - data.open) / data.open) * 100,
          volume: Math.max(0, volume),
          timestamp: Date.now()
        };

        updateMarketData(data.symbol, updatedData);
      });
      
      setLastUpdate(Date.now());
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, [marketData, autoRefresh, updateMarketData]);

  const marketArray = Object.values(marketData);

  const sortedAndFilteredData = marketArray
    .filter(data => {
      if (filterType === 'gainers') return data.changePercent > 0;
      if (filterType === 'losers') return data.changePercent < 0;
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'symbol':
          return a.symbol.localeCompare(b.symbol);
        case 'volume':
          return b.volume - a.volume;
        case 'change':
        default:
          return b.changePercent - a.changePercent;
      }
    });

  const toggleStockSelection = (symbol: string) => {
    const newSelection = new Set(selectedStocks);
    if (newSelection.has(symbol)) {
      newSelection.delete(symbol);
    } else {
      newSelection.add(symbol);
    }
    setSelectedStocks(newSelection);
  };

  const formatNumber = (num: number, decimals = 2) => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(decimals);
  };

  const getChangeColor = (change: number, changePercent: number) => {
    if (change > 0) return 'text-green-400';
    if (change < 0) return 'text-red-400';
    return 'text-yellow-400';
  };

  const getVolumeColor = (volume: number) => {
    const avgVolume = marketArray.reduce((sum, data) => sum + data.volume, 0) / marketArray.length;
    if (volume > avgVolume * 1.5) return 'text-green-400';
    if (volume > avgVolume) return 'text-yellow-400';
    return 'text-green-600';
  };

  const marketStats = {
    gainers: marketArray.filter(d => d.changePercent > 0).length,
    losers: marketArray.filter(d => d.changePercent < 0).length,
    unchanged: marketArray.filter(d => d.changePercent === 0).length,
    totalVolume: marketArray.reduce((sum, d) => sum + d.volume, 0),
    avgChange: marketArray.reduce((sum, d) => sum + d.changePercent, 0) / marketArray.length
  };

  return (
    <div className="space-y-4">
      {/* Market Stats Header */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MatrixCard title="Gainers" glow>
          <div className="text-xl font-bold text-green-400">{marketStats.gainers}</div>
        </MatrixCard>
        <MatrixCard title="Losers" glow>
          <div className="text-xl font-bold text-red-400">{marketStats.losers}</div>
        </MatrixCard>
        <MatrixCard title="Unchanged" glow>
          <div className="text-xl font-bold text-yellow-400">{marketStats.unchanged}</div>
        </MatrixCard>
        <MatrixCard title="Total Volume" glow>
          <div className="text-xl font-bold matrix-text-glow">
            {formatNumber(marketStats.totalVolume, 0)}
          </div>
        </MatrixCard>
        <MatrixCard title="Avg Change" glow>
          <div className={`text-xl font-bold ${
            marketStats.avgChange > 0 ? 'text-green-400' : 
            marketStats.avgChange < 0 ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {marketStats.avgChange > 0 ? '+' : ''}{marketStats.avgChange.toFixed(2)}%
          </div>
        </MatrixCard>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-green-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="matrix-input px-3 py-1 rounded text-sm"
            >
              <option value="all">All</option>
              <option value="gainers">Gainers</option>
              <option value="losers">Losers</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-green-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="matrix-input px-3 py-1 rounded text-sm"
            >
              <option value="change">Change %</option>
              <option value="symbol">Symbol</option>
              <option value="volume">Volume</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-green-600">
            <Clock className="w-4 h-4" />
            Last Update: {new Date(lastUpdate).toLocaleTimeString()}
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-green-400">Auto Refresh</span>
            <div 
              className={`w-10 h-5 rounded-full p-1 cursor-pointer transition-colors ${
                autoRefresh ? 'bg-green-600' : 'bg-gray-600'
              }`}
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              <div className={`w-3 h-3 bg-white rounded-full transition-transform ${
                autoRefresh ? 'translate-x-5' : ''
              }`} />
            </div>
          </div>
          
          <MatrixButton
            variant="secondary"
            size="sm"
            onClick={() => setLastUpdate(Date.now())}
          >
            <RefreshCw className="w-4 h-4" />
          </MatrixButton>
        </div>
      </div>

      {/* Market Data Table */}
      <MatrixCard title="Live Market Data" glow>
        <div className="overflow-x-auto">
          <table className="matrix-table">
            <thead>
              <tr>
                <th className="w-12">★</th>
                <th>Symbol</th>
                <th>Price</th>
                <th>Change</th>
                <th>Change %</th>
                <th>Volume</th>
                <th>High</th>
                <th>Low</th>
                <th>Open</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {sortedAndFilteredData.map((data) => (
                  <motion.tr
                    key={data.symbol}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className={`hover:bg-green-900/10 cursor-pointer ${
                      selectedStocks.has(data.symbol) ? 'bg-green-900/20' : ''
                    }`}
                    onClick={() => toggleStockSelection(data.symbol)}
                  >
                    <td>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleStockSelection(data.symbol);
                        }}
                        className="text-green-400 hover:text-green-300 transition-colors"
                      >
                        <Star 
                          className={`w-4 h-4 ${
                            selectedStocks.has(data.symbol) ? 'fill-current' : ''
                          }`} 
                        />
                      </button>
                    </td>
                    
                    <td className="font-bold text-green-400">{data.symbol}</td>
                    
                    <td className="font-mono">
                      ${data.price.toFixed(2)}
                    </td>
                    
                    <td className={`font-mono ${getChangeColor(data.change, data.changePercent)}`}>
                      <div className="flex items-center gap-1">
                        {data.change > 0 ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : data.change < 0 ? (
                          <TrendingDown className="w-3 h-3" />
                        ) : (
                          <Activity className="w-3 h-3" />
                        )}
                        {data.change > 0 ? '+' : ''}${data.change.toFixed(2)}
                      </div>
                    </td>
                    
                    <td className={`font-mono font-bold ${
                      data.changePercent > 0 ? 'text-green-400' : 
                      data.changePercent < 0 ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {data.changePercent > 0 ? '+' : ''}{data.changePercent.toFixed(2)}%
                    </td>
                    
                    <td className={`font-mono ${getVolumeColor(data.volume)}`}>
                      {formatNumber(data.volume, 0)}
                    </td>
                    
                    <td className="font-mono text-green-400">
                      ${data.high.toFixed(2)}
                    </td>
                    
                    <td className="font-mono text-red-400">
                      ${data.low.toFixed(2)}
                    </td>
                    
                    <td className="font-mono text-green-600">
                      ${data.open.toFixed(2)}
                    </td>
                    
                    <td className="font-mono text-xs text-green-600">
                      {new Date(data.timestamp).toLocaleTimeString('en-US', { 
                        hour12: false,
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      </MatrixCard>

      {/* Selected Stocks Summary */}
      {selectedStocks.size > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
        >
          <MatrixCard title="Watchlist Summary" glow>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-green-600">Selected Stocks</div>
                <div className="text-xl font-bold matrix-text-glow">
                  {selectedStocks.size} symbols
                </div>
              </div>
              
              <div>
                <div className="text-sm text-green-600">Total Value</div>
                <div className="text-xl font-bold text-green-400">
                  ${formatNumber(
                    Array.from(selectedStocks).reduce((sum, symbol) => {
                      const data = marketData[symbol];
                      return sum + (data ? data.price : 0);
                    }, 0)
                  )}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-green-600">Combined Change</div>
                <div className={`text-xl font-bold ${
                  Array.from(selectedStocks).reduce((sum, symbol) => {
                    const data = marketData[symbol];
                    return sum + (data ? data.changePercent : 0);
                  }, 0) / selectedStocks.size > 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(
                    Array.from(selectedStocks).reduce((sum, symbol) => {
                      const data = marketData[symbol];
                      return sum + (data ? data.changePercent : 0);
                    }, 0) / selectedStocks.size
                  ).toFixed(2)}%
                </div>
              </div>
            </div>
          </MatrixCard>
        </motion.div>
      )}
    </div>
  );
};