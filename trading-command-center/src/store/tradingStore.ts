import { create } from 'zustand';
import type { Portfolio, Position, Order, MarketData, SystemConfig } from '../types';

interface TradingStore {
  // Portfolio state
  portfolio: Portfolio | null;
  setPortfolio: (portfolio: Portfolio) => void;

  // Positions state
  positions: Position[];
  setPositions: (positions: Position[]) => void;
  updatePosition: (position: Position) => void;
  removePosition: (id: string) => void;

  // Orders state
  orders: Order[];
  setOrders: (orders: Order[]) => void;
  addOrder: (order: Order) => void;
  updateOrder: (order: Order) => void;
  removeOrder: (id: string) => void;

  // Market data state
  marketData: Map<string, MarketData>;
  setMarketData: (symbol: string, data: MarketData) => void;
  getMarketData: (symbol: string) => MarketData | undefined;

  // Selected items
  selectedBroker: string | null;
  setSelectedBroker: (broker: string | null) => void;

  // System config
  config: SystemConfig | null;
  setConfig: (config: SystemConfig) => void;
  updateConfig: (updates: Partial<SystemConfig>) => void;

  // UI state
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // WebSocket connection state
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // Reset store
  reset: () => void;
}

const initialState = {
  portfolio: null,
  positions: [],
  orders: [],
  marketData: new Map<string, MarketData>(),
  selectedBroker: null,
  config: null,
  sidebarCollapsed: false,
  wsConnected: false,
};

export const useTradingStore = create<TradingStore>((set, get) => ({
  ...initialState,

  // Portfolio actions
  setPortfolio: (portfolio) => set({ portfolio }),

  // Positions actions
  setPositions: (positions) => set({ positions }),
  
  updatePosition: (position) =>
    set((state) => {
      const index = state.positions.findIndex((p) => p.id === position.id);
      if (index !== -1) {
        const newPositions = [...state.positions];
        newPositions[index] = position;
        return { positions: newPositions };
      }
      return { positions: [...state.positions, position] };
    }),

  removePosition: (id) =>
    set((state) => ({
      positions: state.positions.filter((p) => p.id !== id),
    })),

  // Orders actions
  setOrders: (orders) => set({ orders }),

  addOrder: (order) =>
    set((state) => ({
      orders: [order, ...state.orders],
    })),

  updateOrder: (order) =>
    set((state) => {
      const index = state.orders.findIndex((o) => o.id === order.id);
      if (index !== -1) {
        const newOrders = [...state.orders];
        newOrders[index] = order;
        return { orders: newOrders };
      }
      return { orders: [order, ...state.orders] };
    }),

  removeOrder: (id) =>
    set((state) => ({
      orders: state.orders.filter((o) => o.id !== id),
    })),

  // Market data actions
  setMarketData: (symbol, data) =>
    set((state) => {
      const newMarketData = new Map(state.marketData);
      newMarketData.set(symbol, data);
      return { marketData: newMarketData };
    }),

  getMarketData: (symbol) => get().marketData.get(symbol),

  // Selection actions
  setSelectedBroker: (broker) => set({ selectedBroker: broker }),

  // Config actions
  setConfig: (config) => set({ config }),
  
  updateConfig: (updates) =>
    set((state) => ({
      config: state.config ? { ...state.config, ...updates } : null,
    })),

  // UI actions
  toggleSidebar: () =>
    set((state) => ({
      sidebarCollapsed: !state.sidebarCollapsed,
    })),

  // WebSocket actions
  setWsConnected: (connected) => set({ wsConnected: connected }),

  // Reset
  reset: () => set(initialState),
}));
