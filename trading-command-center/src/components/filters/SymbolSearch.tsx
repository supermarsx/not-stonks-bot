import React, { useState, useEffect, useRef } from 'react';
import { Search, Star, Plus, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { MatrixButton } from '../ui/MatrixButton';
import { MatrixInput } from '../ui/MatrixInput';
import { MatrixCard } from '../ui/MatrixCard';
import { Badge } from '../ui/Badge';
import { useDebounce } from '../../hooks/useDebounce';

export interface Symbol {
  symbol: string;
  name: string;
  exchange: string;
  type: 'stock' | 'etf' | 'crypto' | 'forex' | 'commodity';
  price?: number;
  change?: number;
  changePercent?: number;
  isWatched?: boolean;
  isFavorite?: boolean;
  marketCap?: number;
  volume?: number;
}

interface SymbolSearchProps {
  onSymbolSelect: (symbol: Symbol) => void;
  watchlist?: Symbol[];
  favorites?: Symbol[];
  onWatchlistToggle?: (symbol: Symbol) => void;
  onFavoriteToggle?: (symbol: Symbol) => void;
  placeholder?: string;
  allowMultiple?: boolean;
  selectedSymbols?: string[];
}

const SymbolSearch: React.FC<SymbolSearchProps> = ({
  onSymbolSelect,
  watchlist = [],
  favorites = [],
  onWatchlistToggle,
  onFavoriteToggle,
  placeholder = "Search for symbols...",
  allowMultiple = false,
  selectedSymbols = []
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState<'all' | Symbol['type']>('all');
  const searchRef = useRef<HTMLDivElement>(null);
  const debouncedSearchTerm = useDebounce(searchTerm, 300);

  // Mock data for demonstration
  const mockSymbols: Symbol[] = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      exchange: 'NASDAQ',
      type: 'stock',
      price: 150.25,
      change: 2.15,
      changePercent: 1.45,
      isWatched: true,
      isFavorite: false,
      marketCap: 2500000000000,
      volume: 65000000
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      exchange: 'NASDAQ',
      type: 'stock',
      price: 2750.80,
      change: -15.20,
      changePercent: -0.55,
      isWatched: false,
      isFavorite: true,
      marketCap: 1800000000000,
      volume: 1200000
    },
    {
      symbol: 'BTC-USD',
      name: 'Bitcoin',
      exchange: 'BINANCE',
      type: 'crypto',
      price: 43250.75,
      change: 1250.30,
      changePercent: 2.98,
      isWatched: true,
      isFavorite: true,
      volume: 28000000000
    },
    {
      symbol: 'EURUSD=X',
      name: 'Euro / US Dollar',
      exchange: 'FX',
      type: 'forex',
      price: 1.0850,
      change: -0.0025,
      changePercent: -0.23,
      isWatched: false,
      isFavorite: false
    },
    {
      symbol: 'GLD',
      name: 'SPDR Gold Trust',
      exchange: 'NYSEARCA',
      type: 'etf',
      price: 180.45,
      change: 1.25,
      changePercent: 0.70,
      isWatched: false,
      isFavorite: false,
      volume: 8500000
    },
    {
      symbol: 'XAUUSD',
      name: 'Gold Spot',
      exchange: 'COMEX',
      type: 'commodity',
      price: 2015.50,
      change: -12.75,
      changePercent: -0.63,
      isWatched: true,
      isFavorite: false
    }
  ];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (debouncedSearchTerm) {
      setIsLoading(true);
      // Simulate API call
      setTimeout(() => {
        const filtered = mockSymbols.filter(symbol =>
          symbol.symbol.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
          symbol.name.toLowerCase().includes(debouncedSearchTerm.toLowerCase())
        ).filter(symbol => selectedFilter === 'all' || symbol.type === selectedFilter);
        setSearchResults(filtered);
        setIsLoading(false);
        setIsOpen(true);
      }, 200);
    } else {
      setSearchResults([]);
      setIsOpen(false);
    }
  }, [debouncedSearchTerm, selectedFilter]);

  const handleSymbolClick = (symbol: Symbol) => {
    onSymbolSelect(symbol);
    if (!allowMultiple) {
      setSearchTerm('');
      setIsOpen(false);
    }
  };

  const formatPrice = (price?: number) => {
    if (!price) return '-';
    if (price < 1) return price.toFixed(4);
    if (price < 100) return price.toFixed(2);
    return price.toFixed(2);
  };

  const formatChange = (change?: number, changePercent?: number) => {
    if (!change || !changePercent) return { value: '-', isPositive: false };
    return {
      value: `${change > 0 ? '+' : ''}${change.toFixed(2)} (${changePercent > 0 ? '+' : ''}${changePercent.toFixed(2)}%)`,
      isPositive: change > 0
    };
  };

  const formatMarketCap = (marketCap?: number) => {
    if (!marketCap) return '-';
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
    return `$${marketCap.toLocaleString()}`;
  };

  const getTypeColor = (type: Symbol['type']) => {
    switch (type) {
      case 'stock': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'etf': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'crypto': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'forex': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      case 'commodity': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getSymbolFromMockData = (symbol: string): Symbol | undefined => {
    return mockSymbols.find(s => s.symbol === symbol);
  };

  const isSymbolWatched = (symbol: string): boolean => {
    const mockSymbol = getSymbolFromMockData(symbol);
    return mockSymbol?.isWatched || watchlist.some(w => w.symbol === symbol);
  };

  const isSymbolFavorite = (symbol: string): boolean => {
    const mockSymbol = getSymbolFromMockData(symbol);
    return mockSymbol?.isFavorite || favorites.some(f => f.symbol === symbol);
  };

  return (
    <div ref={searchRef} className="relative w-full max-w-2xl">
      <div className="flex gap-2 mb-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-matrix-green w-4 h-4" />
          <MatrixInput
            type="text"
            placeholder={placeholder}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 w-full"
            onFocus={() => searchTerm && setIsOpen(true)}
          />
        </div>
        <select
          value={selectedFilter}
          onChange={(e) => setSelectedFilter(e.target.value as typeof selectedFilter)}
          className="bg-matrix-darker border border-matrix-green/30 rounded px-3 py-2 text-matrix-green text-sm"
        >
          <option value="all">All Types</option>
          <option value="stock">Stocks</option>
          <option value="etf">ETFs</option>
          <option value="crypto">Crypto</option>
          <option value="forex">Forex</option>
          <option value="commodity">Commodities</option>
        </select>
      </div>

      {isOpen && (
        <MatrixCard className="absolute top-full left-0 right-0 z-50 mt-1 max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 text-center text-matrix-green">
              <div className="animate-spin w-6 h-6 border-2 border-matrix-green border-t-transparent rounded-full mx-auto mb-2"></div>
              Searching...
            </div>
          ) : searchResults.length > 0 ? (
            <div className="py-2">
              {searchResults.map((symbol) => {
                const change = formatChange(symbol.change, symbol.changePercent);
                const isSelected = selectedSymbols.includes(symbol.symbol);
                
                return (
                  <div
                    key={symbol.symbol}
                    className={`px-4 py-3 hover:bg-matrix-green/10 cursor-pointer transition-colors ${
                      isSelected ? 'bg-matrix-green/20' : ''
                    }`}
                    onClick={() => handleSymbolClick(symbol)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex flex-col">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-matrix-green font-medium">
                              {symbol.symbol}
                            </span>
                            <Badge 
                              variant="outline" 
                              className={`text-xs ${getTypeColor(symbol.type)}`}
                            >
                              {symbol.type.toUpperCase()}
                            </Badge>
                            {isSymbolFavorite(symbol.symbol) && (
                              <Star className="w-3 h-3 text-yellow-400 fill-current" />
                            )}
                            {isSymbolWatched(symbol.symbol) && (
                              <Plus className="w-3 h-3 text-matrix-green" />
                            )}
                          </div>
                          <span className="text-xs text-matrix-green/70">
                            {symbol.name} • {symbol.exchange}
                          </span>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="font-mono text-matrix-green">
                          ${formatPrice(symbol.price)}
                        </div>
                        <div className={`text-xs flex items-center gap-1 ${
                          change.isPositive ? 'text-matrix-green' : 'text-red-400'
                        }`}>
                          {change.isPositive ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : change.value === '-' ? (
                            <Minus className="w-3 h-3" />
                          ) : (
                            <TrendingDown className="w-3 h-3" />
                          )}
                          {change.value}
                        </div>
                        {symbol.marketCap && (
                          <div className="text-xs text-matrix-green/70">
                            {formatMarketCap(symbol.marketCap)}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {(onWatchlistToggle || onFavoriteToggle) && (
                      <div className="flex gap-2 mt-2">
                        {onWatchlistToggle && (
                          <MatrixButton
                            variant="outline"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              onWatchlistToggle(symbol);
                            }}
                            className={`text-xs ${
                              isSymbolWatched(symbol.symbol)
                                ? 'bg-matrix-green/20 border-matrix-green text-matrix-green'
                                : 'border-matrix-green/30 text-matrix-green/70 hover:border-matrix-green/50'
                            }`}
                          >
                            {isSymbolWatched(symbol.symbol) ? 'Watched' : 'Watch'}
                          </MatrixButton>
                        )}
                        {onFavoriteToggle && (
                          <MatrixButton
                            variant="outline"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              onFavoriteToggle(symbol);
                            }}
                            className={`text-xs ${
                              isSymbolFavorite(symbol.symbol)
                                ? 'bg-yellow-400/20 border-yellow-400 text-yellow-400'
                                : 'border-matrix-green/30 text-matrix-green/70 hover:border-matrix-green/50'
                            }`}
                          >
                            <Star className={`w-3 h-3 mr-1 ${
                              isSymbolFavorite(symbol.symbol) ? 'fill-current' : ''
                            }`} />
                            {isSymbolFavorite(symbol.symbol) ? 'Favorited' : 'Favorite'}
                          </MatrixButton>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : searchTerm ? (
            <div className="p-4 text-center text-matrix-green/70">
              No symbols found for "{searchTerm}"
            </div>
          ) : null}
        </MatrixCard>
      )}
    </div>
  );
};

export default SymbolSearch;