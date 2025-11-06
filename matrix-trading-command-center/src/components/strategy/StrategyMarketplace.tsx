import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Store, 
  Search,
  Filter,
  Star,
  Download,
  Upload,
  Eye,
  Heart,
  Share2,
  Copy,
  Plus,
  Settings,
  Save,
  Play,
  Pause,
  CheckCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  Shield,
  Zap,
  Target,
  Award,
  Users,
  Globe,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
  Bookmark,
  Flag,
  Tag,
  Calendar,
  User,
  Award as AwardIcon,
  GitBranch,
  RefreshCw,
  Grid,
  List,
  MapPin,
  Clock4,
  Activity,
  TrendingUp as TrendingUpIcon,
  DollarSign as DollarIcon,
  BarChart2,
  PieChart,
  LineChart,
  Layers,
  Sparkles
} from 'lucide-react';

interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: 'momentum' | 'mean_reversion' | 'arbitrage' | 'scalping' | 'ai_ml' | 'multi_asset' | 'commodities' | 'crypto';
  author: {
    id: string;
    name: string;
    avatar?: string;
    verified: boolean;
    reputation: number;
  };
  rating: {
    average: number;
    count: number;
    distribution: number[]; // 5-star to 1-star
  };
  downloads: number;
  likes: number;
  views: number;
  price: {
    amount: number;
    currency: string;
    isFree: boolean;
  };
  tags: string[];
  parameters: Record<string, any>;
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    avgReturn: number;
    volatility: number;
  };
  timeframe: {
    start: Date;
    end: Date;
    period: string;
  };
  assets: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'very_high';
  complexity: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  createdAt: Date;
  updatedAt: Date;
  featured: boolean;
  trending: boolean;
  images: string[];
  documentation: {
    hasGuide: boolean;
    hasVideo: boolean;
    hasBacktest: boolean;
  };
  license: {
    type: 'free' | 'premium' | 'subscription';
    terms: string;
  };
  compatibility: {
    brokers: string[];
    dataFeeds: string[];
    platforms: string[];
  };
}

interface UserReview {
  id: string;
  userId: string;
  userName: string;
  avatar?: string;
  rating: number;
  title: string;
  comment: string;
  helpful: number;
  createdAt: Date;
  verified: boolean;
}

const StrategyMarketplace: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'browse' | 'my_strategies' | 'community' | 'featured'>('browse');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');
  const [selectedComplexity, setSelectedComplexity] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'popularity' | 'rating' | 'downloads' | 'newest' | 'price'>('popularity');
  const [filterPrice, setFilterPrice] = useState<'all' | 'free' | 'premium'>('all');
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyTemplate | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const [strategies] = useState<StrategyTemplate[]>([
    {
      id: '1',
      name: 'Alpha Momentum Scalper',
      description: 'High-frequency momentum strategy with machine learning price prediction and rapid entry/exit execution',
      category: 'scalping',
      author: {
        id: 'user1',
        name: 'Quantum Trader',
        verified: true,
        reputation: 4.8
      },
      rating: {
        average: 4.7,
        count: 1247,
        distribution: [892, 245, 78, 22, 10]
      },
      downloads: 15847,
      likes: 2341,
      views: 45678,
      price: {
        amount: 0,
        currency: 'USD',
        isFree: true
      },
      tags: ['momentum', 'scalping', 'ai', 'high_frequency'],
      parameters: {
        'ma_period': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'position_size': 0.1
      },
      performance: {
        totalReturn: 0.324,
        sharpeRatio: 2.1,
        maxDrawdown: 0.087,
        winRate: 0.68,
        avgReturn: 0.0087,
        volatility: 0.156
      },
      timeframe: {
        start: new Date('2023-01-01'),
        end: new Date('2024-01-01'),
        period: '1 year'
      },
      assets: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
      riskLevel: 'high',
      complexity: 'expert',
      createdAt: new Date('2023-12-01'),
      updatedAt: new Date('2024-01-15'),
      featured: true,
      trending: true,
      images: ['/api/placeholder/300/200'],
      documentation: {
        hasGuide: true,
        hasVideo: true,
        hasBacktest: true
      },
      license: {
        type: 'free',
        terms: 'MIT License'
      },
      compatibility: {
        brokers: ['alpaca', 'ibkr', 'binance'],
        dataFeeds: ['polygon', 'alpha_vantage', 'yahoo'],
        platforms: ['desktop', 'web', 'api']
      }
    },
    {
      id: '2',
      name: 'Mean Reversion Arbitrage',
      description: 'Pairs trading strategy using statistical arbitrage techniques and cointegration analysis',
      category: 'arbitrage',
      author: {
        id: 'user2',
        name: 'StatsMaster',
        verified: true,
        reputation: 4.9
      },
      rating: {
        average: 4.8,
        count: 892,
        distribution: [734, 125, 23, 8, 2]
      },
      downloads: 12356,
      likes: 1876,
      views: 34521,
      price: {
        amount: 199,
        currency: 'USD',
        isFree: false
      },
      tags: ['arbitrage', 'pairs', 'statistical', 'mean_reversion'],
      parameters: {
        'lookback_period': 30,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'position_size': 0.05
      },
      performance: {
        totalReturn: 0.187,
        sharpeRatio: 1.7,
        maxDrawdown: 0.045,
        winRate: 0.72,
        avgReturn: 0.0034,
        volatility: 0.098
      },
      timeframe: {
        start: new Date('2022-06-01'),
        end: new Date('2024-01-01'),
        period: '1.5 years'
      },
      assets: ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
      riskLevel: 'medium',
      complexity: 'advanced',
      createdAt: new Date('2023-06-15'),
      updatedAt: new Date('2024-01-20'),
      featured: true,
      trending: false,
      images: ['/api/placeholder/300/200'],
      documentation: {
        hasGuide: true,
        hasVideo: false,
        hasBacktest: true
      },
      license: {
        type: 'premium',
        terms: 'Commercial License'
      },
      compatibility: {
        brokers: ['ibkr', 'alpaca', 'interactive_brokers'],
        dataFeeds: ['bloomberg', 'refinitiv', 'polygon'],
        platforms: ['desktop', 'api']
      }
    },
    {
      id: '3',
      name: 'Crypto Arbitrage Bot',
      description: 'Automated cryptocurrency arbitrage across multiple exchanges with real-time price monitoring',
      category: 'crypto',
      author: {
        id: 'user3',
        name: 'CryptoKing',
        verified: true,
        reputation: 4.6
      },
      rating: {
        average: 4.5,
        count: 2156,
        distribution: [1456, 523, 134, 32, 11]
      },
      downloads: 28934,
      likes: 3456,
      views: 67823,
      price: {
        amount: 0,
        currency: 'USD',
        isFree: true
      },
      tags: ['crypto', 'arbitrage', 'multi_exchange', 'automated'],
      parameters: {
        'min_profit_threshold': 0.005,
        'max_position_size': 10000,
        'exchange_fees': 0.001,
        'slippage_buffer': 0.0005
      },
      performance: {
        totalReturn: 0.412,
        sharpeRatio: 1.9,
        maxDrawdown: 0.123,
        winRate: 0.75,
        avgReturn: 0.0012,
        volatility: 0.245
      },
      timeframe: {
        start: new Date('2023-03-01'),
        end: new Date('2024-01-01'),
        period: '10 months'
      },
      assets: ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
      riskLevel: 'very_high',
      complexity: 'expert',
      createdAt: new Date('2023-02-10'),
      updatedAt: new Date('2024-01-18'),
      featured: false,
      trending: true,
      images: ['/api/placeholder/300/200'],
      documentation: {
        hasGuide: true,
        hasVideo: true,
        hasBacktest: true
      },
      license: {
        type: 'free',
        terms: 'GPL License'
      },
      compatibility: {
        brokers: ['binance', 'coinbase', 'kraken'],
        dataFeeds: ['binance_api', 'coinbase_api', 'ccxt'],
        platforms: ['api', 'cloud']
      }
    }
  ]);

  const [reviews] = useState<UserReview[]>([
    {
      id: '1',
      userId: 'user1',
      userName: 'AlgoTrader2024',
      rating: 5,
      title: 'Excellent performance and easy to use',
      comment: 'This strategy has been consistently profitable for the past 6 months. The documentation is thorough and the implementation is straightforward.',
      helpful: 23,
      createdAt: new Date('2024-01-20'),
      verified: true
    },
    {
      id: '2',
      userId: 'user2',
      userName: 'RiskManager',
      rating: 4,
      title: 'Good strategy but requires monitoring',
      comment: 'Works well in trending markets but needs careful risk management during volatile periods. Overall satisfied with the results.',
      helpful: 15,
      createdAt: new Date('2024-01-18'),
      verified: true
    }
  ]);

  const categories = [
    { id: 'all', label: 'All Categories', icon: Grid },
    { id: 'momentum', label: 'Momentum', icon: TrendingUpIcon },
    { id: 'mean_reversion', label: 'Mean Reversion', icon: BarChart2 },
    { id: 'arbitrage', label: 'Arbitrage', icon: GitBranch },
    { id: 'scalping', label: 'Scalping', icon: Zap },
    { id: 'ai_ml', label: 'AI/ML', icon: Sparkles },
    { id: 'multi_asset', label: 'Multi-Asset', icon: PieChart },
    { id: 'commodities', label: 'Commodities', icon: DollarIcon },
    { id: 'crypto', label: 'Cryptocurrency', icon: Activity }
  ];

  const tabs = [
    { id: 'browse', label: 'Browse Marketplace', icon: Store },
    { id: 'my_strategies', label: 'My Strategies', icon: User },
    { id: 'community', label: 'Community', icon: Users },
    { id: 'featured', label: 'Featured', icon: AwardIcon }
  ];

  const filteredStrategies = strategies.filter(strategy => {
    const matchesSearch = strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         strategy.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         strategy.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || strategy.category === selectedCategory;
    const matchesRisk = selectedRiskLevel === 'all' || strategy.riskLevel === selectedRiskLevel;
    const matchesComplexity = selectedComplexity === 'all' || strategy.complexity === selectedComplexity;
    const matchesPrice = filterPrice === 'all' || 
                        (filterPrice === 'free' && strategy.price.isFree) ||
                        (filterPrice === 'premium' && !strategy.price.isFree);

    return matchesSearch && matchesCategory && matchesRisk && matchesComplexity && matchesPrice;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'rating':
        return b.rating.average - a.rating.average;
      case 'downloads':
        return b.downloads - a.downloads;
      case 'newest':
        return b.createdAt.getTime() - a.createdAt.getTime();
      case 'price':
        return a.price.amount - b.price.amount;
      default:
        return b.popularity - a.popularity;
    }
  });

  const renderStrategyCard = (strategy: StrategyTemplate) => (
    <MatrixCard key={strategy.id} className="p-4 hover:bg-gray-800/50 transition-all duration-200 cursor-pointer"
                onClick={() => setSelectedStrategy(strategy)}>
      <div className="relative">
        {strategy.featured && (
          <div className="absolute top-2 right-2 bg-yellow-500 text-black px-2 py-1 rounded text-xs font-bold z-10">
            Featured
          </div>
        )}
        {strategy.trending && (
          <div className="absolute top-2 left-2 bg-red-500 text-white px-2 py-1 rounded text-xs font-bold z-10">
            Trending
          </div>
        )}
        <div className="w-full h-32 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg mb-3 flex items-center justify-center">
          <BarChart3 className="w-12 h-12 text-gray-400" />
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <h3 className="font-semibold text-white text-sm line-clamp-2">{strategy.name}</h3>
          <p className="text-xs text-gray-400 line-clamp-2 mt-1">{strategy.description}</p>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <Star className="w-3 h-3 text-yellow-400 fill-current" />
              <span className="text-xs text-gray-300">{strategy.rating.average}</span>
            </div>
            <span className="text-xs text-gray-500">({strategy.rating.count})</span>
          </div>
          <div className="text-xs text-gray-400">
            {strategy.price.isFree ? 'Free' : `$${strategy.price.amount}`}
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-gray-400">
          <span className="flex items-center">
            <Download className="w-3 h-3 mr-1" />
            {strategy.downloads.toLocaleString()}
          </span>
          <span className="flex items-center">
            <Heart className="w-3 h-3 mr-1" />
            {strategy.likes.toLocaleString()}
          </span>
          <span className={`
            px-2 py-1 rounded text-xs
            ${strategy.riskLevel === 'low' ? 'bg-green-900 text-green-200' :
              strategy.riskLevel === 'medium' ? 'bg-yellow-900 text-yellow-200' :
              strategy.riskLevel === 'high' ? 'bg-orange-900 text-orange-200' : 'bg-red-900 text-red-200'}
          `}>
            {strategy.riskLevel}
          </span>
        </div>

        <div className="flex flex-wrap gap-1">
          {strategy.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
              {tag}
            </span>
          ))}
          {strategy.tags.length > 3 && (
            <span className="text-xs text-gray-500">+{strategy.tags.length - 3}</span>
          )}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-gray-700">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full flex items-center justify-center">
              <span className="text-xs font-bold text-white">
                {strategy.author.name.charAt(0)}
              </span>
            </div>
            <div>
              <div className="flex items-center space-x-1">
                <span className="text-xs text-gray-300">{strategy.author.name}</span>
                {strategy.author.verified && (
                  <CheckCircle className="w-3 h-3 text-blue-400" />
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <MatrixButton size="sm" variant="secondary">
              <Eye className="w-3 h-3" />
            </MatrixButton>
            <MatrixButton size="sm" variant="secondary">
              <Download className="w-3 h-3" />
            </MatrixButton>
          </div>
        </div>
      </div>
    </MatrixCard>
  );

  const renderStrategyDetails = () => {
    if (!selectedStrategy) return null;

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
        onClick={() => setSelectedStrategy(null)}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6">
            <div className="flex justify-between items-start mb-6">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h2 className="text-2xl font-bold text-white">{selectedStrategy.name}</h2>
                  {selectedStrategy.featured && (
                    <span className="bg-yellow-500 text-black px-2 py-1 rounded text-sm font-bold">
                      Featured
                    </span>
                  )}
                  {selectedStrategy.trending && (
                    <span className="bg-red-500 text-white px-2 py-1 rounded text-sm font-bold">
                      Trending
                    </span>
                  )}
                </div>
                <p className="text-gray-300 mb-4">{selectedStrategy.description}</p>
                <div className="flex items-center space-x-4 text-sm text-gray-400">
                  <span className="flex items-center">
                    <Star className="w-4 h-4 text-yellow-400 fill-current mr-1" />
                    {selectedStrategy.rating.average} ({selectedStrategy.rating.count} reviews)
                  </span>
                  <span className="flex items-center">
                    <Download className="w-4 h-4 mr-1" />
                    {selectedStrategy.downloads.toLocaleString()} downloads
                  </span>
                  <span className="flex items-center">
                    <Eye className="w-4 h-4 mr-1" />
                    {selectedStrategy.views.toLocaleString()} views
                  </span>
                </div>
              </div>
              <MatrixButton
                variant="destructive"
                onClick={() => setSelectedStrategy(null)}
              >
                ✕
              </MatrixButton>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Performance Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-800 p-3 rounded">
                      <div className="text-sm text-gray-400">Total Return</div>
                      <div className="text-lg font-bold text-green-400">
                        {(selectedStrategy.performance.totalReturn * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-gray-800 p-3 rounded">
                      <div className="text-sm text-gray-400">Sharpe Ratio</div>
                      <div className="text-lg font-bold text-cyan-400">
                        {selectedStrategy.performance.sharpeRatio.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-gray-800 p-3 rounded">
                      <div className="text-sm text-gray-400">Max Drawdown</div>
                      <div className="text-lg font-bold text-red-400">
                        {(selectedStrategy.performance.maxDrawdown * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-gray-800 p-3 rounded">
                      <div className="text-sm text-gray-400">Win Rate</div>
                      <div className="text-lg font-bold text-blue-400">
                        {(selectedStrategy.performance.winRate * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Strategy Details</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Category</span>
                      <span className="text-white capitalize">{selectedStrategy.category.replace('_', ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Risk Level</span>
                      <span className={`capitalize ${selectedStrategy.riskLevel === 'low' ? 'text-green-400' :
                        selectedStrategy.riskLevel === 'medium' ? 'text-yellow-400' :
                        selectedStrategy.riskLevel === 'high' ? 'text-orange-400' : 'text-red-400'}`}>
                        {selectedStrategy.riskLevel}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Complexity</span>
                      <span className="text-white capitalize">{selectedStrategy.complexity}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Timeframe</span>
                      <span className="text-white">{selectedStrategy.timeframe.period}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Price</span>
                      <span className="text-white">
                        {selectedStrategy.price.isFree ? 'Free' : `$${selectedStrategy.price.amount}`}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Compatible Assets</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedStrategy.assets.map((asset) => (
                      <span key={asset} className="bg-cyan-600 text-white px-3 py-1 rounded-full text-sm">
                        {asset}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedStrategy.tags.map((tag) => (
                      <span key={tag} className="bg-gray-700 text-gray-300 px-2 py-1 rounded text-sm">
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Author</h3>
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full flex items-center justify-center">
                      <span className="text-lg font-bold text-white">
                        {selectedStrategy.author.name.charAt(0)}
                      </span>
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-white font-medium">{selectedStrategy.author.name}</span>
                        {selectedStrategy.author.verified && (
                          <CheckCircle className="w-4 h-4 text-blue-400" />
                        )}
                      </div>
                      <div className="text-sm text-gray-400">
                        Reputation: {selectedStrategy.author.reputation}/5.0
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Parameters</h3>
                  <div className="space-y-2">
                    {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                      <div key={key} className="flex justify-between bg-gray-800 p-2 rounded">
                        <span className="text-gray-400 capitalize">{key.replace('_', ' ')}</span>
                        <span className="text-white">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Documentation</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Setup Guide</span>
                      <CheckCircle className={`w-4 h-4 ${selectedStrategy.documentation.hasGuide ? 'text-green-400' : 'text-gray-600'}`} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Video Tutorial</span>
                      <CheckCircle className={`w-4 h-4 ${selectedStrategy.documentation.hasVideo ? 'text-green-400' : 'text-gray-600'}`} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Backtest Results</span>
                      <CheckCircle className={`w-4 h-4 ${selectedStrategy.documentation.hasBacktest ? 'text-green-400' : 'text-gray-600'}`} />
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <MatrixButton
                    className="w-full bg-cyan-600 hover:bg-cyan-700"
                    onClick={() => console.log('Download strategy:', selectedStrategy.id)}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    {selectedStrategy.price.isFree ? 'Download Free' : `Buy for $${selectedStrategy.price.amount}`}
                  </MatrixButton>
                  <div className="flex space-x-2">
                    <MatrixButton variant="secondary" className="flex-1">
                      <Heart className="w-4 h-4 mr-2" />
                      Like
                    </MatrixButton>
                    <MatrixButton variant="secondary" className="flex-1">
                      <Share2 className="w-4 h-4 mr-2" />
                      Share
                    </MatrixButton>
                    <MatrixButton variant="secondary" className="flex-1">
                      <Bookmark className="w-4 h-4 mr-2" />
                      Save
                    </MatrixButton>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <h3 className="text-lg font-semibold text-white mb-4">User Reviews</h3>
              <div className="space-y-4">
                {reviews.map((review) => (
                  <div key={review.id} className="bg-gray-800 p-4 rounded">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center">
                          <span className="text-sm font-bold text-white">
                            {review.userName.charAt(0)}
                          </span>
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-white font-medium">{review.userName}</span>
                            {review.verified && (
                              <CheckCircle className="w-3 h-3 text-blue-400" />
                            )}
                            <div className="flex">
                              {[...Array(5)].map((_, i) => (
                                <Star key={i} className={`w-3 h-3 ${i < review.rating ? 'text-yellow-400 fill-current' : 'text-gray-600'}`} />
                              ))}
                            </div>
                          </div>
                          <span className="text-xs text-gray-400">
                            {review.createdAt.toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                    <h4 className="text-white font-medium mb-2">{review.title}</h4>
                    <p className="text-gray-300 text-sm mb-3">{review.comment}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <button className="flex items-center space-x-1 text-gray-400 hover:text-white">
                          <ThumbsUp className="w-3 h-3" />
                          <span className="text-xs">{review.helpful}</span>
                        </button>
                        <button className="text-gray-400 hover:text-white">
                          <ThumbsDown className="w-3 h-3" />
                        </button>
                      </div>
                      <button className="text-gray-400 hover:text-white">
                        <MessageSquare className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    );
  };

  const renderBrowseTab = () => (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <MatrixInput
              type="text"
              placeholder="Search strategies, authors, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 w-full"
            />
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm"
          >
            <option value="popularity">Popularity</option>
            <option value="rating">Rating</option>
            <option value="downloads">Downloads</option>
            <option value="newest">Newest</option>
            <option value="price">Price</option>
          </select>
          <MatrixButton
            variant={viewMode === 'grid' ? 'default' : 'secondary'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            <Grid className="w-4 h-4" />
          </MatrixButton>
          <MatrixButton
            variant={viewMode === 'list' ? 'default' : 'secondary'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            <List className="w-4 h-4" />
          </MatrixButton>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {categories.map((category) => {
          const Icon = category.icon;
          return (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`
                flex items-center px-3 py-2 rounded-lg text-sm transition-all duration-200
                ${selectedCategory === category.id
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                }
              `}
            >
              <Icon className="w-4 h-4 mr-2" />
              {category.label}
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap gap-4">
        <select
          value={selectedRiskLevel}
          onChange={(e) => setSelectedRiskLevel(e.target.value)}
          className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm"
        >
          <option value="all">All Risk Levels</option>
          <option value="low">Low Risk</option>
          <option value="medium">Medium Risk</option>
          <option value="high">High Risk</option>
          <option value="very_high">Very High Risk</option>
        </select>

        <select
          value={selectedComplexity}
          onChange={(e) => setSelectedComplexity(e.target.value)}
          className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm"
        >
          <option value="all">All Complexity Levels</option>
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
          <option value="expert">Expert</option>
        </select>

        <select
          value={filterPrice}
          onChange={(e) => setFilterPrice(e.target.value as any)}
          className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm"
        >
          <option value="all">All Prices</option>
          <option value="free">Free Only</option>
          <option value="premium">Premium Only</option>
        </select>
      </div>

      <div className="text-sm text-gray-400">
        Showing {filteredStrategies.length} of {strategies.length} strategies
      </div>

      <div className={`
        ${viewMode === 'grid' 
          ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' 
          : 'space-y-4'
        }
      `}>
        {filteredStrategies.map((strategy) => 
          viewMode === 'grid' ? renderStrategyCard(strategy) : (
            <MatrixCard key={strategy.id} className="p-4">
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-8 h-8 text-gray-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-white">{strategy.name}</h3>
                    <div className="flex items-center space-x-2">
                      <span className="text-green-400 font-bold">
                        {(strategy.performance.totalReturn * 100).toFixed(1)}%
                      </span>
                      <span className="text-gray-400">
                        {strategy.price.isFree ? 'Free' : `$${strategy.price.amount}`}
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400 mt-1 line-clamp-2">{strategy.description}</p>
                  <div className="flex items-center justify-between mt-2">
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      <span className="flex items-center">
                        <Star className="w-3 h-3 text-yellow-400 fill-current mr-1" />
                        {strategy.rating.average}
                      </span>
                      <span className="flex items-center">
                        <Download className="w-3 h-3 mr-1" />
                        {strategy.downloads.toLocaleString()}
                      </span>
                      <span className={`
                        px-2 py-1 rounded text-xs
                        ${strategy.riskLevel === 'low' ? 'bg-green-900 text-green-200' :
                          strategy.riskLevel === 'medium' ? 'bg-yellow-900 text-yellow-200' :
                          strategy.riskLevel === 'high' ? 'bg-orange-900 text-orange-200' : 'bg-red-900 text-red-200'}
                      `}>
                        {strategy.riskLevel}
                      </span>
                    </div>
                    <MatrixButton size="sm">
                      <Eye className="w-3 h-3 mr-2" />
                      View
                    </MatrixButton>
                  </div>
                </div>
              </div>
            </MatrixCard>
          )
        )}
      </div>
    </div>
  );

  const renderMyStrategiesTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">My Published Strategies</h3>
        <MatrixButton onClick={() => setShowCreateModal(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Publish Strategy
        </MatrixButton>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {strategies.filter(s => s.author.id === 'user1').map((strategy) => renderStrategyCard(strategy))}
      </div>

      <div className="mt-8">
        <h3 className="text-lg font-semibold text-white mb-4">My Downloads</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {strategies.slice(0, 3).map((strategy) => renderStrategyCard(strategy))}
        </div>
      </div>
    </div>
  );

  const renderCommunityTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MatrixCard className="p-4 text-center">
          <Users className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
          <h3 className="text-lg font-semibold text-white">12,847</h3>
          <p className="text-sm text-gray-400">Active Traders</p>
        </MatrixCard>
        <MatrixCard className="p-4 text-center">
          <Store className="w-8 h-8 text-green-400 mx-auto mb-2" />
          <h3 className="text-lg font-semibold text-white">2,156</h3>
          <p className="text-sm text-gray-400">Strategies Available</p>
        </MatrixCard>
        <MatrixCard className="p-4 text-center">
          <Download className="w-8 h-8 text-blue-400 mx-auto mb-2" />
          <h3 className="text-lg font-semibold text-white">156K</h3>
          <p className="text-sm text-gray-400">Total Downloads</p>
        </MatrixCard>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Top Contributors</h3>
        <div className="space-y-3">
          {[
            { name: 'Quantum Trader', strategies: 47, reputation: 4.9, earnings: '$12,450' },
            { name: 'AlgoMaster', strategies: 32, reputation: 4.8, earnings: '$8,920' },
            { name: 'RiskGuru', strategies: 28, reputation: 4.7, earnings: '$6,780' }
          ].map((contributor, index) => (
            <MatrixCard key={index} className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
                    <span className="text-sm font-bold text-white">#{index + 1}</span>
                  </div>
                  <div>
                    <div className="text-white font-medium">{contributor.name}</div>
                    <div className="text-sm text-gray-400">
                      {contributor.strategies} strategies • {contributor.reputation}/5.0 rating
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-green-400 font-semibold">{contributor.earnings}</div>
                  <div className="text-xs text-gray-400">Total earnings</div>
                </div>
              </div>
            </MatrixCard>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
        <div className="space-y-3">
          {[
            { user: 'AlgoTrader2024', action: 'downloaded', strategy: 'Mean Reversion Arbitrage', time: '5 min ago' },
            { user: 'QuantMaster', action: 'published', strategy: 'ML Price Predictor', time: '12 min ago' },
            { user: 'RiskManager', action: 'reviewed', strategy: 'Alpha Momentum Scalper', time: '25 min ago' },
            { user: 'CryptoKing', action: 'updated', strategy: 'Crypto Arbitrage Bot', time: '1 hour ago' }
          ].map((activity, index) => (
            <div key={index} className="flex items-center space-x-3 text-sm">
              <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
              <span className="text-gray-300">
                <strong>{activity.user}</strong> {activity.action} <strong>{activity.strategy}</strong>
              </span>
              <span className="text-gray-500 ml-auto">{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderFeaturedTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MatrixCard className="p-6 bg-gradient-to-br from-yellow-600/20 to-orange-600/20 border-yellow-500/30">
          <div className="flex items-center justify-between mb-4">
            <Award className="w-8 h-8 text-yellow-400" />
            <span className="bg-yellow-500 text-black px-3 py-1 rounded-full text-sm font-bold">
              Editor's Choice
            </span>
          </div>
          <h3 className="text-xl font-bold text-white mb-2">Alpha Momentum Scalper</h3>
          <p className="text-gray-300 mb-4">
            High-frequency momentum strategy with ML price prediction and rapid execution
          </p>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-green-400 font-bold">+32.4%</div>
              <div className="flex items-center text-yellow-400">
                <Star className="w-4 h-4 fill-current mr-1" />
                <span>4.7</span>
              </div>
            </div>
            <MatrixButton>
              <Download className="w-4 h-4 mr-2" />
              Download Free
            </MatrixButton>
          </div>
        </MatrixCard>

        <MatrixCard className="p-6 bg-gradient-to-br from-red-600/20 to-pink-600/20 border-red-500/30">
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-red-400" />
            <span className="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-bold">
              Trending Now
            </span>
          </div>
          <h3 className="text-xl font-bold text-white mb-2">Crypto Arbitrage Bot</h3>
          <p className="text-gray-300 mb-4">
            Automated cryptocurrency arbitrage across multiple exchanges
          </p>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-green-400 font-bold">+41.2%</div>
              <div className="flex items-center text-yellow-400">
                <Star className="w-4 h-4 fill-current mr-1" />
                <span>4.5</span>
              </div>
            </div>
            <MatrixButton>
              <Download className="w-4 h-4 mr-2" />
              Download Free
            </MatrixButton>
          </div>
        </MatrixCard>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Featured Collections</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { name: 'Best for Beginners', icon: BookOpen, count: 24 },
            { name: 'High Frequency', icon: Zap, count: 12 },
            { name: 'Low Risk', icon: Shield, count: 18 },
            { name: 'Crypto Strategies', icon: Bitcoin, count: 15 }
          ].map((collection) => {
            const Icon = collection.icon;
            return (
              <MatrixCard key={collection.name} className="p-4 text-center cursor-pointer hover:bg-gray-800/50">
                <Icon className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h4 className="font-medium text-white mb-1">{collection.name}</h4>
                <p className="text-sm text-gray-400">{collection.count} strategies</p>
              </MatrixCard>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'browse':
        return renderBrowseTab();
      case 'my_strategies':
        return renderMyStrategiesTab();
      case 'community':
        return renderCommunityTab();
      case 'featured':
        return renderFeaturedTab();
      default:
        return renderBrowseTab();
    }
  };

  return (
    <div className="p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <MatrixCard className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Store className="w-6 h-6 text-cyan-400" />
              <h2 className="text-xl font-bold text-white">Strategy Marketplace</h2>
            </div>
            <div className="flex items-center space-x-2">
              <MatrixButton variant="secondary">
                <Upload className="w-4 h-4 mr-2" />
                Upload Strategy
              </MatrixButton>
              <MatrixButton variant="secondary" size="sm">
                <RefreshCw className="w-4 h-4" />
              </MatrixButton>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 mb-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    flex flex-col items-center p-3 rounded-lg transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400/30'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                    }
                  `}
                >
                  <Icon className="w-5 h-5 mb-1" />
                  <span className="text-xs text-center leading-tight">{tab.label}</span>
                </button>
              );
            })}
          </div>

          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <MatrixCard className="p-4 bg-gray-800/50">
              {renderTabContent()}
            </MatrixCard>
          </motion.div>
        </MatrixCard>
      </motion.div>

      <AnimatePresence>
        {selectedStrategy && renderStrategyDetails()}
      </AnimatePresence>
    </div>
  );
};

export default StrategyMarketplace;