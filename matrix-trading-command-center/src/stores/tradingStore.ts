import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Types
export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
  timestamp: number;
  broker: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop';
  size: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  timestamp: number;
  broker: string;
  filledSize?: number;
  filledPrice?: number;
}

export interface PortfolioData {
  totalValue: number;
  availableCash: number;
  totalPnL: number;
  totalPnLPercentage: number;
  positions: Position[];
  orders: Order[];
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  timestamp: number;
}

export interface RiskMetrics {
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  sharpeRatio: number;
  maxDrawdown: number;
  totalRisk: number;
  riskUtilization: number; // 0-100%
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'stopped' | 'error';
  performance: {
    totalPnL: number;
    winRate: number;
    totalTrades: number;
    avgTradeDuration: number;
  };
  config: Record<string, any>;
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
}

export interface BrokerConnection {
  id: string;
  name: string;
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  latency: number;
  lastHeartbeat: number;
  capabilities: string[];
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface TradingStore {
  // Portfolio Data
  portfolio: PortfolioData;
  
  // Market Data
  marketData: Record<string, MarketData>;
  marketTicker: MarketData[];
  
  // Risk Management
  riskMetrics: RiskMetrics;
  
  // Strategies
  strategies: Strategy[];
  activeStrategyId?: string;
  
  // Alerts
  alerts: Alert[];
  unreadAlerts: number;
  
  // Broker Connections
  brokerConnections: BrokerConnection[];
  
  // AI Chat
  chatMessages: ChatMessage[];
  isChatLoading: boolean;
  
  // Connection Status
  wsConnected: boolean;
  apiConnected: boolean;
  
  // UI State
  sidebarOpen: boolean;
  activeView: 'dashboard' | 'portfolio' | 'orders' | 'risk' | 'strategies' | 'chat' | 'settings';
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: number;
  }>;
  
  // Actions
  updatePortfolio: (portfolio: Partial<PortfolioData>) => void;
  updateMarketData: (symbol: string, data: MarketData) => void;
  updateRiskMetrics: (metrics: Partial<RiskMetrics>) => void;
  addPosition: (position: Position) => void;
  updatePosition: (id: string, position: Partial<Position>) => void;
  removePosition: (id: string) => void;
  addOrder: (order: Order) => void;
  updateOrder: (id: string, order: Partial<Order>) => void;
  removeOrder: (id: string) => void;
  
  // Strategy Actions
  addStrategy: (strategy: Strategy) => void;
  updateStrategy: (id: string, strategy: Partial<Strategy>) => void;
  removeStrategy: (id: string) => void;
  setActiveStrategy: (id?: string) => void;
  
  // Alert Actions
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp' | 'read'>) => void;
  markAlertAsRead: (id: string) => void;
  clearAlerts: () => void;
  
  // Broker Actions
  updateBrokerConnection: (id: string, connection: Partial<BrokerConnection>) => void;
  
  // Chat Actions
  addChatMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  setChatLoading: (loading: boolean) => void;
  
  // Connection Actions
  setWebSocketConnected: (connected: boolean) => void;
  setApiConnected: (connected: boolean) => void;
  
  // UI Actions
  setSidebarOpen: (open: boolean) => void;
  setActiveView: (view: TradingStore['activeView']) => void;
  addNotification: (notification: Omit<TradingStore['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
}

export const useTradingStore = create<TradingStore>()(
  devtools(
    (set, get) => ({
      // Initial State
      portfolio: {
        totalValue: 100000,
        availableCash: 50000,
        totalPnL: 5000,
        totalPnLPercentage: 5.0,
        positions: [],
        orders: [],
      },
      
      marketData: {},
      marketTicker: [],
      
      riskMetrics: {
        var95: -2500,
        var99: -4200,
        sharpeRatio: 1.8,
        maxDrawdown: -3.2,
        totalRisk: 15000,
        riskUtilization: 65,
      },
      
      strategies: [],
      activeStrategyId: undefined,
      
      alerts: [],
      unreadAlerts: 0,
      
      brokerConnections: [],
      
      chatMessages: [],
      isChatLoading: false,
      
      wsConnected: false,
      apiConnected: false,
      
      sidebarOpen: true,
      activeView: 'dashboard',
      notifications: [],
      
      // Portfolio Actions
      updatePortfolio: (portfolio) => {
        set((state) => ({
          portfolio: { ...state.portfolio, ...portfolio }
        }));
      },
      
      updateMarketData: (symbol, data) => {
        set((state) => ({
          marketData: { ...state.marketData, [symbol]: data },
          marketTicker: updateTicker(state.marketTicker, data)
        }));
      },
      
      updateRiskMetrics: (metrics) => {
        set((state) => ({
          riskMetrics: { ...state.riskMetrics, ...metrics }
        }));
      },
      
      // Position Actions
      addPosition: (position) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            positions: [...state.portfolio.positions, position]
          }
        }));
      },
      
      updatePosition: (id, position) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            positions: state.portfolio.positions.map((p) =>
              p.id === id ? { ...p, ...position } : p
            )
          }
        }));
      },
      
      removePosition: (id) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            positions: state.portfolio.positions.filter((p) => p.id !== id)
          }
        }));
      },
      
      // Order Actions
      addOrder: (order) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            orders: [...state.portfolio.orders, order]
          }
        }));
      },
      
      updateOrder: (id, order) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            orders: state.portfolio.orders.map((o) =>
              o.id === id ? { ...o, ...order } : o
            )
          }
        }));
      },
      
      removeOrder: (id) => {
        set((state) => ({
          portfolio: {
            ...state.portfolio,
            orders: state.portfolio.orders.filter((o) => o.id !== id)
          }
        }));
      },
      
      // Strategy Actions
      addStrategy: (strategy) => {
        set((state) => ({
          strategies: [...state.strategies, strategy]
        }));
      },
      
      updateStrategy: (id, strategy) => {
        set((state) => ({
          strategies: state.strategies.map((s) =>
            s.id === id ? { ...s, ...strategy } : s
          )
        }));
      },
      
      removeStrategy: (id) => {
        set((state) => ({
          strategies: state.strategies.filter((s) => s.id !== id)
        }));
      },
      
      setActiveStrategy: (id) => {
        set({ activeStrategyId: id });
      },
      
      // Alert Actions
      addAlert: (alert) => {
        const newAlert: Alert = {
          ...alert,
          id: generateId(),
          timestamp: Date.now(),
          read: false,
        };
        
        set((state) => ({
          alerts: [newAlert, ...state.alerts],
          unreadAlerts: state.unreadAlerts + 1
        }));
      },
      
      markAlertAsRead: (id) => {
        set((state) => ({
          alerts: state.alerts.map((a) =>
            a.id === id ? { ...a, read: true } : a
          ),
          unreadAlerts: Math.max(0, state.unreadAlerts - 1)
        }));
      },
      
      clearAlerts: () => {
        set({ alerts: [], unreadAlerts: 0 });
      },
      
      // Broker Actions
      updateBrokerConnection: (id, connection) => {
        set((state) => ({
          brokerConnections: state.brokerConnections.map((b) =>
            b.id === id ? { ...b, ...connection } : b
          )
        }));
      },
      
      // Chat Actions
      addChatMessage: (message) => {
        const newMessage: ChatMessage = {
          ...message,
          id: generateId(),
          timestamp: Date.now(),
        };
        
        set((state) => ({
          chatMessages: [...state.chatMessages, newMessage]
        }));
      },
      
      setChatLoading: (loading) => {
        set({ isChatLoading: loading });
      },
      
      // Connection Actions
      setWebSocketConnected: (connected) => {
        set({ wsConnected: connected });
      },
      
      setApiConnected: (connected) => {
        set({ apiConnected: connected });
      },
      
      // UI Actions
      setSidebarOpen: (open) => {
        set({ sidebarOpen: open });
      },
      
      setActiveView: (view) => {
        set({ activeView: view });
      },
      
      addNotification: (notification) => {
        const newNotification = {
          ...notification,
          id: generateId(),
          timestamp: Date.now(),
        };
        
        set((state) => ({
          notifications: [...state.notifications, newNotification]
        }));
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
          get().removeNotification(newNotification.id);
        }, 5000);
      },
      
      removeNotification: (id) => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id)
        }));
      },
    }),
    {
      name: 'trading-store',
    }
  )
);

// Helper functions
function updateTicker(current: MarketData[], newData: MarketData): MarketData[] {
  const existing = current.find(item => item.symbol === newData.symbol);
  if (existing) {
    return current.map(item => 
      item.symbol === newData.symbol ? newData : item
    );
  } else {
    return [newData, ...current].slice(0, 50); // Keep only latest 50
  }
}

function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}