import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore } from '@/stores/tradingStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { Navigation } from '@/components/common/Navigation';
import { SimpleMatrixRain } from '@/components/common/MatrixRain';
import { Dashboard } from '@/components/dashboard/Dashboard';
import { Portfolio } from '@/components/trading/Portfolio';
import { OrderManagement } from '@/components/trading/OrderManagement';
import { RiskManagement } from '@/components/trading/RiskManagement';
import { StrategyManager } from '@/components/trading/StrategyManager';
import { AIChat } from '@/components/trading/AIChat';
import { Settings } from '@/components/trading/Settings';
import { MarketTicker } from '@/components/market/MarketTicker';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { Menu, X } from 'lucide-react';
import './App.css';

// Mock data generators for demo
const generateMockData = () => {
  // Generate mock positions
  const mockPositions = [
    {
      id: '1',
      symbol: 'AAPL',
      side: 'long' as const,
      size: 100,
      entryPrice: 150.25,
      currentPrice: 155.80,
      pnl: 555,
      pnlPercentage: 3.7,
      timestamp: Date.now() - 3600000,
      broker: 'Alpaca',
    },
    {
      id: '2',
      symbol: 'TSLA',
      side: 'long' as const,
      size: 50,
      entryPrice: 220.50,
      currentPrice: 215.30,
      pnl: -260,
      pnlPercentage: -2.4,
      timestamp: Date.now() - 7200000,
      broker: 'Binance',
    },
    {
      id: '3',
      symbol: 'GOOGL',
      side: 'short' as const,
      size: 25,
      entryPrice: 2800.00,
      currentPrice: 2850.00,
      pnl: -1250,
      pnlPercentage: -1.8,
      timestamp: Date.now() - 1800000,
      broker: 'Interactive Brokers',
    },
  ];

  // Generate mock market data
  const mockMarketData = [
    { symbol: 'AAPL', price: 155.80, change: 2.1, changePercent: 1.37, volume: 45000000, high: 156.50, low: 153.20, open: 154.10, timestamp: Date.now() },
    { symbol: 'TSLA', price: 215.30, change: -5.2, changePercent: -2.36, volume: 62000000, high: 220.80, low: 214.50, open: 219.10, timestamp: Date.now() },
    { symbol: 'GOOGL', price: 2850.00, change: -25.5, changePercent: -0.89, volume: 12000000, high: 2875.20, low: 2840.30, open: 2865.80, timestamp: Date.now() },
    { symbol: 'MSFT', price: 380.25, change: 8.4, changePercent: 2.26, volume: 28000000, high: 382.10, low: 375.90, open: 376.20, timestamp: Date.now() },
    { symbol: 'AMZN', price: 3120.50, change: -45.2, changePercent: -1.43, volume: 3500000, high: 3165.80, low: 3115.60, open: 3142.30, timestamp: Date.now() },
  ];

  // Generate mock strategies
  const mockStrategies = [
    {
      id: '1',
      name: 'Mean Reversion',
      description: 'Statistical arbitrage strategy',
      status: 'running' as const,
      performance: {
        totalPnL: 15420.50,
        winRate: 68.5,
        totalTrades: 245,
        avgTradeDuration: 45, // minutes
      },
      config: {
        lookback_period: 20,
        entry_threshold: 2.0,
        exit_threshold: 0.5,
      },
    },
    {
      id: '2',
      name: 'Momentum Strategy',
      description: 'Trend following algorithm',
      status: 'running' as const,
      performance: {
        totalPnL: 8750.25,
        winRate: 72.1,
        totalTrades: 156,
        avgTradeDuration: 120,
      },
      config: {
        momentum_period: 14,
        entry_threshold: 0.02,
        stop_loss: 0.05,
      },
    },
    {
      id: '3',
      name: 'Pairs Trading',
      description: 'Statistical pairs strategy',
      status: 'stopped' as const,
      performance: {
        totalPnL: 3250.80,
        winRate: 64.2,
        totalTrades: 89,
        avgTradeDuration: 90,
      },
      config: {
        correlation_threshold: 0.8,
        z_score_entry: 2.0,
        z_score_exit: 0.5,
      },
    },
  ];

  // Generate mock broker connections
  const mockBrokers = [
    {
      id: '1',
      name: 'Alpaca',
      status: 'connected' as const,
      latency: 12,
      lastHeartbeat: Date.now(),
      capabilities: [' equities', ' options', ' crypto'],
    },
    {
      id: '2',
      name: 'Binance',
      status: 'connected' as const,
      latency: 8,
      lastHeartbeat: Date.now(),
      capabilities: [' spot', ' futures', ' options'],
    },
    {
      id: '3',
      name: 'Interactive Brokers',
      status: 'disconnected' as const,
      latency: 0,
      lastHeartbeat: Date.now() - 300000,
      capabilities: [' equities', ' futures', ' forex'],
    },
  ];

  return {
    positions: mockPositions,
    marketData: mockMarketData,
    strategies: mockStrategies,
    brokers: mockBrokers,
  };
};

function App() {
  const { 
    setActiveView, 
    sidebarOpen, 
    setSidebarOpen,
    updateMarketData,
    updatePortfolio,
    addStrategy,
    updateBrokerConnection,
  } = useTradingStore();

  const { connected } = useWebSocket();

  // Initialize mock data
  useEffect(() => {
    const mockData = generateMockData();
    
    // Update market data
    mockData.marketData.forEach((data) => {
      updateMarketData(data.symbol, data);
    });
    
    // Update portfolio with mock positions
    updatePortfolio({
      positions: mockData.positions,
    });
    
    // Add mock strategies
    mockData.strategies.forEach((strategy) => {
      addStrategy(strategy);
    });
    
    // Update broker connections
    mockData.brokers.forEach((broker) => {
      updateBrokerConnection(broker.id, broker);
    });
    
    // Add some mock alerts
    const alerts = [
      {
        type: 'info' as const,
        title: 'System Startup',
        message: 'Trading orchestrator initialized successfully',
      },
      {
        type: 'success' as const,
        title: 'Strategy Started',
        message: 'Mean Reversion strategy is now running',
      },
      {
        type: 'warning' as const,
        title: 'Risk Threshold',
        message: 'Portfolio risk utilization at 65%',
      },
    ];
    
    alerts.forEach((alert, index) => {
      setTimeout(() => {
        useTradingStore.getState().addAlert(alert);
      }, index * 1000);
    });
  }, []);

  const handleViewChange = (view: string) => {
    setActiveView(view as any);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const renderCurrentView = () => {
    const { activeView } = useTradingStore.getState();
    
    switch (activeView) {
      case 'dashboard':
        return <Dashboard />;
      case 'portfolio':
        return <Portfolio />;
      case 'orders':
        return <OrderManagement />;
      case 'risk':
        return <RiskManagement />;
      case 'strategies':
        return <StrategyManager />;
      case 'chat':
        return <AIChat />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Router>
      <div className="h-screen w-screen bg-black text-green-400 font-mono overflow-hidden relative">
        {/* Matrix Rain Background */}
        <SimpleMatrixRain density={0.2} />
        
        {/* Main Layout */}
        <div className="relative z-10 h-full flex">
          {/* Sidebar */}
          <AnimatePresence>
            {sidebarOpen && (
              <motion.div
                className="w-80 h-full bg-black/80 backdrop-blur-sm border-r border-green-800/50"
                initial={{ x: -320 }}
                animate={{ x: 0 }}
                exit={{ x: -320 }}
                transition={{ duration: 0.3 }}
              >
                <Navigation onViewChange={handleViewChange} />
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Main Content */}
          <div className="flex-1 h-full flex flex-col overflow-hidden">
            {/* Header */}
            <motion.header
              className="bg-black/60 backdrop-blur-sm border-b border-green-800/50 p-4"
              initial={{ y: -100 }}
              animate={{ y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <MatrixButton
                    variant="secondary"
                    size="sm"
                    onClick={toggleSidebar}
                  >
                    {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
                  </MatrixButton>
                  
                  <div>
                    <h2 className="text-xl font-bold text-green-400">
                      MATRIX TRADING CENTER
                    </h2>
                    <p className="text-sm text-green-600">
                      {useTradingStore.getState().activeView.toUpperCase()} VIEW
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4">
                  {/* Connection Status */}
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${
                      connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                    }`} />
                    <span className="text-sm text-green-400">
                      {connected ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                  
                  {/* Current Time */}
                  <div className="text-sm text-green-600">
                    {new Date().toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </motion.header>
            
            {/* Content Area */}
            <main className="flex-1 overflow-auto bg-black/40">
              <AnimatePresence mode="wait">
                <motion.div
                  key={useTradingStore.getState().activeView}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="h-full"
                >
                  {renderCurrentView()}
                </motion.div>
              </AnimatePresence>
            </main>
          </div>
        </div>
        
        {/* Terminal-style Footer */}
        <motion.footer
          className="absolute bottom-0 left-0 right-0 bg-black/80 backdrop-blur-sm border-t border-green-800/50 p-2"
          initial={{ y: 100 }}
          animate={{ y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="flex items-center justify-between text-xs text-green-600">
            <div className="flex items-center gap-4">
              <span>Status: OPERATIONAL</span>
              <span>Uptime: 99.9%</span>
              <span>Latency: 12ms</span>
            </div>
            <div className="flex items-center gap-4">
              <span>© 2025 MATRIX TRADING</span>
              <span className="animate-pulse">●</span>
            </div>
          </div>
        </motion.footer>
      </div>
    </Router>
  );
}

export default App;