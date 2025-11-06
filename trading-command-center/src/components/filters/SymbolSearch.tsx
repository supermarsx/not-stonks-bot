import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Filter, Star, TrendingUp, TrendingDown } from 'lucide-react';
import { MatrixButton } from '../MatrixButton';
import { MatrixInput } from '../MatrixInput';

interface Symbol {
  symbol: string;
  name: string;
  exchange: string;
  type: 'stock' | 'etf' | 'crypto' | 'forex' | 'commodity';
  price?: number;
  change?: number;
  changePercent?: number;
  isWatched?: boolean;
  isFavorite?: boolean;
}

interface SymbolSearchProps {
  onSymbolSelect: (symbol: Symbol) => void;
  onWatchlistToggle?: (symbol: string) => void;
  onFavoritesToggle?: (symbol: string) => void;
  placeholder?: string;
  className?: string;
  showFilters?: boolean;
  showWatchlist?: boolean;
  showFavorites?: boolean;
  initialValue?: string;
  disabled?: boolean;
}

export const SymbolSearch: React.FC<SymbolSearchProps> = ({
  onSymbolSelect,
  onWatchlistToggle,
  onFavoritesToggle,
  placeholder = "Search symbols...",
  className = "",
  showFilters = true,
  showWatchlist = true,
  showFavorites = true,
  initialValue = "",
  disabled = false,
}) => {
  const [searchTerm, setSearchTerm] = useState(initialValue);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<Symbol | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [watchlistOnly, setWatchlistOnly] = useState(false);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [searchResults, setSearchResults] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Mock data - in real app, this would come from an API
  const mockSymbols: Symbol[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ', type: 'stock', price: 175.50, change: 2.30, changePercent: 1.33, isWatched: true, isFavorite: true },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ', type: 'stock', price: 142.80, change: -1.20, changePercent: -0.83, isWatched: true, isFavorite: false },
    { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ', type: 'stock', price: 378.90, change: 5.20, changePercent: 1.39, isWatched: false, isFavorite: true },
    { symbol: 'TSLA', name: 'Tesla, Inc.', exchange: 'NASDAQ', type: 'stock', price: 248.75, change: -3.45, changePercent: -1.37, isWatched: true, isFavorite: false },
    { symbol: 'SPY', name: 'SPDR S&P 500 ETF', exchange: 'NYSE', type: 'etf', price: 456.80, change: 1.80, changePercent: 0.40, isWatched: false, isFavorite: false },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust', exchange: 'NASDAQ', type: 'etf', price: 389.25, change: 2.15, changePercent: 0.56, isWatched: true, isFavorite: true },
    { symbol: 'BTC-USD', name: 'Bitcoin USD', exchange: 'COINBASE', type: 'crypto', price: 42850.75, change: 1250.30, changePercent: 3.01, isWatched: true, isFavorite: false },
    { symbol: 'ETH-USD', name: 'Ethereum USD', exchange: 'COINBASE', type: 'crypto', price: 2580.40, change: -45.20, changePercent: -1.72, isWatched: false, isFavorite: false },
    { symbol: 'EUR/USD', name: 'Euro to US Dollar', exchange: 'FOREX', type: 'forex', price: 1.0875, change: 0.0025, changePercent: 0.23, isWatched: false, isFavorite: false },
    { symbol: 'GLD', name: 'SPDR Gold Shares', exchange: 'NYSE', type: 'commodity', price: 189.45, change: -1.25, changePercent: -0.66, isWatched: false, isFavorite: true },
  ];

  // Filter symbols based on search term and filters
  const filterSymbols = (symbols: Symbol[], term: string, filter: string, watchlistOnly: boolean, favoritesOnly: boolean) => {
    return symbols.filter(symbol => {
      const matchesSearch = term === "" || 
        symbol.symbol.toLowerCase().includes(term.toLowerCase()) ||
        symbol.name.toLowerCase().includes(term.toLowerCase());
      
      const matchesFilter = filter === 'all' || symbol.type === filter;
      
      const matchesWatchlist = !watchlistOnly || symbol.isWatched;
      
      const matchesFavorites = !favoritesOnly || symbol.isFavorite;

      return matchesSearch && matchesFilter && matchesWatchlist && matchesFavorites;
    });
  };

  // Handle search
  useEffect(() => {
    const results = filterSymbols(mockSymbols, searchTerm, activeFilter, watchlistOnly, favoritesOnly);
    setSearchResults(results);
  }, [searchTerm, activeFilter, watchlistOnly, favoritesOnly]);

  // Handle input focus
  const handleInputFocus = () => {
    setIsOpen(true);
    setIsLoading(false);
  };

  // Handle input blur
  const handleInputBlur = (e: React.FocusEvent) => {
    // Delay closing to allow for result clicks
    setTimeout(() => {
      if (!resultsRef.current?.contains(document.activeElement)) {
        setIsOpen(false);
      }
    }, 200);
  };

  // Handle symbol selection
  const handleSymbolSelect = (symbol: Symbol) => {
    setSelectedSymbol(symbol);
    setSearchTerm(symbol.symbol);
    setIsOpen(false);
    onSymbolSelect(symbol);
  };

  // Handle watchlist toggle
  const handleWatchlistToggle = (e: React.MouseEvent, symbol: string) => {
    e.stopPropagation();
    onWatchlistToggle?.(symbol);
  };

  // Handle favorites toggle
  const handleFavoritesToggle = (e: React.MouseEvent, symbol: string) => {
    e.stopPropagation();
    onFavoritesToggle?.(symbol);
  };

  // Clear search
  const clearSearch = () => {
    setSearchTerm('');
    setSelectedSymbol(null);
    inputRef.current?.focus();
  };

  // Get symbol type color
  const getTypeColor = (type: Symbol['type']) => {
    const colors = {
      stock: '#00ff00',
      etf: '#00cc00',
      crypto: '#ffff00',
      forex: '#ff9900',
      commodity: '#ff6600',
    };
    return colors[type] || '#00ff00';
  };

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-4 w-4 text-matrix-green" />
        </div>
        
        <MatrixInput
          ref={inputRef}
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          placeholder={placeholder}
          disabled={disabled}
          className="pl-10 pr-10"
        />

        {searchTerm && (
          <button
            onClick={clearSearch}
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
          >
            <X className="h-4 w-4 text-matrix-green hover:text-matrix-green/70" />
          </button>
        )}
      </div>

      {/* Filters */}
      {showFilters && isOpen && (
        <div className="mt-2 flex flex-wrap gap-2 p-3 bg-matrix-black border border-matrix-green/30 rounded-lg">
          {/* Type Filter */}
          <div className="flex gap-1">
            {['all', 'stock', 'etf', 'crypto', 'forex', 'commodity'].map((filter) => (
              <MatrixButton
                key={filter}
                size="sm"
                variant={activeFilter === filter ? 'primary' : 'secondary'}
                onClick={() => setActiveFilter(filter)}
              >
                {filter.toUpperCase()}
              </MatrixButton>
            ))}
          </div>

          {/* Toggle Filters */}
          {showWatchlist && (
            <MatrixButton
              size="sm"
              variant={watchlistOnly ? 'primary' : 'secondary'}
              onClick={() => setWatchlistOnly(!watchlistOnly)}
            >
              <Star className="w-3 h-3 mr-1" />
              WATCHLIST
            </MatrixButton>
          )}

          {showFavorites && (
            <MatrixButton
              size="sm"
              variant={favoritesOnly ? 'primary' : 'secondary'}
              onClick={() => setFavoritesOnly(!favoritesOnly)}
            >
              <Filter className="w-3 h-3 mr-1" />
              FAVORITES
            </MatrixButton>
          )}
        </div>
      )}

      {/* Search Results */}
      {isOpen && (
        <div
          ref={resultsRef}
          className="absolute top-full left-0 right-0 mt-1 bg-matrix-black border border-matrix-green rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto"
        >
          {isLoading ? (
            <div className="p-4 text-center text-matrix-green font-mono">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-matrix-green mx-auto"></div>
              <p className="mt-2">Searching...</p>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="py-2">
              {searchResults.map((symbol) => (
                <div
                  key={symbol.symbol}
                  onClick={() => handleSymbolSelect(symbol)}
                  className="px-4 py-3 hover:bg-matrix-green/10 cursor-pointer border-b border-matrix-green/10 last:border-b-0"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <div>
                          <span className="text-matrix-green font-mono font-bold">
                            {symbol.symbol}
                          </span>
                          <span
                            className="ml-2 text-xs px-2 py-1 rounded font-mono"
                            style={{ 
                              backgroundColor: `${getTypeColor(symbol.type)}22`,
                              color: getTypeColor(symbol.type)
                            }}
                          >
                            {symbol.type.toUpperCase()}
                          </span>
                        </div>
                        {symbol.price && (
                          <div className="text-right">
                            <p className="text-matrix-green font-mono">
                              ${symbol.price.toFixed(2)}
                            </p>
                            {symbol.change && (
                              <div className={`flex items-center gap-1 text-xs ${
                                symbol.change >= 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                {symbol.change >= 0 ? (
                                  <TrendingUp className="w-3 h-3" />
                                ) : (
                                  <TrendingDown className="w-3 h-3" />
                                )}
                                {symbol.changePercent?.toFixed(2)}%
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <p className="text-matrix-green/70 text-sm font-mono mt-1">
                        {symbol.name} • {symbol.exchange}
                      </p>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-1 ml-4">
                      {showWatchlist && (
                        <button
                          onClick={(e) => handleWatchlistToggle(e, symbol.symbol)}
                          className={`p-1 rounded ${
                            symbol.isWatched 
                              ? 'text-yellow-400 hover:text-yellow-300' 
                              : 'text-matrix-green/50 hover:text-matrix-green'
                          }`}
                        >
                          <Star className="w-4 h-4" fill={symbol.isWatched ? 'currentColor' : 'none'} />
                        </button>
                      )}
                      
                      {showFavorites && (
                        <button
                          onClick={(e) => handleFavoritesToggle(e, symbol.symbol)}
                          className={`p-1 rounded ${
                            symbol.isFavorite 
                              ? 'text-matrix-green hover:text-matrix-green/70' 
                              : 'text-matrix-green/50 hover:text-matrix-green'
                          }`}
                        >
                          <Filter className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : searchTerm ? (
            <div className="p-4 text-center text-matrix-green/70 font-mono">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No symbols found for "{searchTerm}"</p>
            </div>
          ) : (
            <div className="p-4 text-center text-matrix-green/70 font-mono">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Type to search symbols...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SymbolSearch;