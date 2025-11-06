// Core Trading Types
export interface Position {
  id: string;
  symbol: string;
  broker: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  realizedPnL?: number;
  timestamp: string;
}

export interface Order {
  id: string;
  symbol: string;
  broker: string;
  type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
  side: 'BUY' | 'SELL';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'PENDING' | 'ACTIVE' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED';
  filledQuantity?: number;
  averageFillPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  createdAt: string;
  updatedAt: string;
}

export interface Trade {
  id: string;
  orderId: string;
  symbol: string;
  broker: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  commission: number;
  pnl?: number;
  timestamp: string;
}

export interface Portfolio {
  totalValue: number;
  cashBalance: number;
  positionsValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  buyingPower: number;
  marginUsed: number;
  timestamp: string;
}

export interface Broker {
  id: string;
  name: 'binance' | 'ibkr' | 'alpaca';
  displayName: string;
  status: 'connected' | 'disconnected' | 'error' | 'connecting';
  accountType: 'live' | 'paper';
  balance: number;
  positionsCount: number;
  ordersCount: number;
  lastSync: string;
  config?: Record<string, unknown>;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'stopped' | 'error';
  type: 'momentum' | 'mean_reversion' | 'arbitrage' | 'pairs' | 'custom';
  broker: string;
  symbols: string[];
  totalPnL: number;
  dailyPnL: number;
  totalTrades: number;
  winRate: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  createdAt: string;
  updatedAt: string;
  parameters?: Record<string, unknown>;
}

export interface RiskMetrics {
  portfolioValue: number;
  currentDrawdown: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  winRate: number;
  profitFactor: number;
  dailyVaR: number;
  weeklyVaR: number;
  positionConcentration: number;
  leverageRatio: number;
  timestamp: string;
}

export interface MarketData {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  timestamp: string;
}

export interface AIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    toolCalls?: string[];
    confidence?: number;
    analysis?: Record<string, unknown>;
  };
}

export interface AIAnalysis {
  id: string;
  type: 'market_features' | 'strategy_backtest' | 'risk_check' | 'news_sentiment' | 'opportunity';
  symbol?: string;
  result: Record<string, unknown>;
  confidence: number;
  recommendations?: string[];
  timestamp: string;
}

// WebSocket Message Types
export interface WSMessage {
  type: 'portfolio' | 'position' | 'order' | 'trade' | 'market_data' | 'risk' | 'alert';
  data: unknown;
  timestamp: string;
}

export interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  source: string;
  timestamp: string;
  acknowledged: boolean;
}

// API Request/Response Types
export interface CreateOrderRequest {
  symbol: string;
  broker: string;
  type: Order['type'];
  side: Order['side'];
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: Order['timeInForce'];
}

export interface UpdateStrategyRequest {
  status?: Strategy['status'];
  parameters?: Record<string, unknown>;
}

export interface AIQueryRequest {
  message: string;
  context?: {
    portfolio?: boolean;
    positions?: boolean;
    marketData?: string[];
  };
}

export interface BacktestRequest {
  strategyType: string;
  symbols: string[];
  startDate: string;
  endDate: string;
  parameters: Record<string, unknown>;
}

// Configuration Types
export interface SystemConfig {
  trading: {
    defaultBroker: string;
    defaultTimeInForce: Order['timeInForce'];
    confirmOrders: boolean;
    maxOrderSize: number;
  };
  risk: {
    maxPositionSize: number;
    maxDailyLoss: number;
    maxDrawdown: number;
    enableStopLoss: boolean;
    enableTakeProfit: boolean;
  };
  ai: {
    provider: 'openai' | 'anthropic';
    model: string;
    temperature: number;
    enableAutoTrade: boolean;
  };
  display: {
    theme: 'matrix' | 'dark' | 'light';
    refreshInterval: number;
    showNotifications: boolean;
  };
}

// API Response Wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  timestamp: string;
}

// Pagination
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}
