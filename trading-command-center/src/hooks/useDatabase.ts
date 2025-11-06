/**
 * Enhanced Database Hooks
 * Provides React hooks with loading states, error handling, and real-time updates
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { dbService } from '../services/database';
import { wsService } from '../services/websocket';
import type {
  Portfolio,
  Position,
  Order,
  Trade,
  Broker,
  Strategy,
  RiskMetrics,
  MarketData,
  PaginatedResponse,
} from '../types';

// ==================== BASE HOOK ====================

interface BaseHookOptions {
  useCache?: boolean;
  refreshInterval?: number;
  enableRealtime?: boolean;
}

interface BaseHookState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// ==================== PORTFOLIO HOOK ====================

export function usePortfolio(options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval, enableRealtime = true } = options;
  
  const [state, setState] = useState<BaseHookState<Portfolio>>({
    data: null,
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const fetchPortfolio = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getPortfolio(useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch portfolio',
      }));
    }
  }, [useCache]);

  const refresh = useCallback(() => {
    fetchPortfolio();
  }, [fetchPortfolio]);

  useEffect(() => {
    fetchPortfolio();

    // Setup real-time updates
    if (enableRealtime) {
      unsubscribeRef.current = wsService.subscribePortfolio((data) => {
        setState(prev => ({
          ...prev,
          data,
          lastUpdated: new Date(),
        }));
      });
    }

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchPortfolio, refreshInterval);
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchPortfolio, refreshInterval, enableRealtime]);

  return {
    ...state,
    refresh,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== POSITIONS HOOK ====================

export function usePositions(broker?: string, options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval, enableRealtime = true } = options;
  
  const [state, setState] = useState<BaseHookState<Position[]>>({
    data: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const fetchPositions = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getPositions(broker, useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch positions',
      }));
    }
  }, [broker, useCache]);

  const refresh = useCallback(() => {
    fetchPositions();
  }, [fetchPositions]);

  const closePosition = useCallback(async (id: string) => {
    try {
      await dbService.closePosition(id);
      await fetchPositions(); // Refresh after close
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to close position',
      }));
    }
  }, [fetchPositions]);

  useEffect(() => {
    fetchPositions();

    // Setup real-time updates
    if (enableRealtime) {
      unsubscribeRef.current = wsService.subscribePositions((data) => {
        if (!broker || data.broker === broker) {
          setState(prev => ({
            ...prev,
            data,
            lastUpdated: new Date(),
          }));
        }
      });
    }

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchPositions, refreshInterval);
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchPositions, refreshInterval, enableRealtime, broker]);

  return {
    ...state,
    refresh,
    closePosition,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== ORDERS HOOK ====================

export function useOrders(params?: { broker?: string; status?: string; limit?: number }, options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval, enableRealtime = true } = options;
  
  const [state, setState] = useState<BaseHookState<Order[]>>({
    data: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const fetchOrders = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getOrders(params, useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch orders',
      }));
    }
  }, [params, useCache]);

  const refresh = useCallback(() => {
    fetchOrders();
  }, [fetchOrders]);

  const createOrder = useCallback(async (request: any) => {
    try {
      await dbService.createOrder(request);
      await fetchOrders(); // Refresh after creation
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to create order',
      }));
      throw error;
    }
  }, [fetchOrders]);

  const cancelOrder = useCallback(async (id: string) => {
    try {
      await dbService.cancelOrder(id);
      await fetchOrders(); // Refresh after cancellation
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to cancel order',
      }));
      throw error;
    }
  }, [fetchOrders]);

  const modifyOrder = useCallback(async (id: string, updates: any) => {
    try {
      await dbService.modifyOrder(id, updates);
      await fetchOrders(); // Refresh after modification
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to modify order',
      }));
      throw error;
    }
  }, [fetchOrders]);

  useEffect(() => {
    fetchOrders();

    // Setup real-time updates
    if (enableRealtime) {
      unsubscribeRef.current = wsService.subscribeOrders((data) => {
        if (!params?.broker || data.broker === params.broker) {
          fetchOrders(); // Refresh orders on any update
        }
      });
    }

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchOrders, refreshInterval);
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchOrders, refreshInterval, enableRealtime, params?.broker]);

  return {
    ...state,
    refresh,
    createOrder,
    cancelOrder,
    modifyOrder,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== TRADES HOOK WITH PAGINATION ====================

export function useTradesPaginated(initialPage = 1, pageSize = 50, options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval } = options;
  
  const [state, setState] = useState<BaseHookState<PaginatedResponse<Trade>> & {
    page: number;
    pageSize: number;
  }>({
    data: { items: [], total: 0, page: initialPage, pageSize, hasMore: false },
    loading: true,
    error: null,
    lastUpdated: null,
    page: initialPage,
    pageSize,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchTrades = useCallback(async (page: number, size: number) => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getTradesPaginated(page, size);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
        page,
        pageSize: size,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch trades',
      }));
    }
  }, []);

  const refresh = useCallback(() => {
    fetchTrades(state.page, state.pageSize);
  }, [fetchTrades, state.page, state.pageSize]);

  const nextPage = useCallback(() => {
    if (state.data?.hasMore) {
      fetchTrades(state.page + 1, state.pageSize);
    }
  }, [fetchTrades, state.page, state.pageSize, state.data?.hasMore]);

  const prevPage = useCallback(() => {
    if (state.page > 1) {
      fetchTrades(state.page - 1, state.pageSize);
    }
  }, [fetchTrades, state.page, state.pageSize]);

  const goToPage = useCallback((page: number) => {
    if (page >= 1 && page <= Math.ceil((state.data?.total || 0) / state.pageSize)) {
      fetchTrades(page, state.pageSize);
    }
  }, [fetchTrades, state.pageSize, state.data?.total]);

  const changePageSize = useCallback((size: number) => {
    fetchTrades(1, size);
  }, [fetchTrades]);

  useEffect(() => {
    fetchTrades(initialPage, pageSize);

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(() => {
        fetchTrades(state.page, state.pageSize);
      }, refreshInterval);
    }

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchTrades, refreshInterval, initialPage, pageSize]);

  return {
    ...state,
    refresh,
    nextPage,
    prevPage,
    goToPage,
    changePageSize,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== BROKERS HOOK ====================

export function useBrokers(options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval } = options;
  
  const [state, setState] = useState<BaseHookState<Broker[]>>({
    data: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchBrokers = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getBrokers(useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch brokers',
      }));
    }
  }, [useCache]);

  const refresh = useCallback(() => {
    fetchBrokers();
  }, [fetchBrokers]);

  const connectBroker = useCallback(async (id: string, credentials: Record<string, string>) => {
    try {
      await dbService.connectBroker(id, credentials);
      await fetchBrokers(); // Refresh after connection
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to connect broker',
      }));
      throw error;
    }
  }, [fetchBrokers]);

  const disconnectBroker = useCallback(async (id: string) => {
    try {
      await dbService.disconnectBroker(id);
      await fetchBrokers(); // Refresh after disconnection
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to disconnect broker',
      }));
      throw error;
    }
  }, [fetchBrokers]);

  const syncBroker = useCallback(async (id: string) => {
    try {
      await dbService.syncBroker(id);
      await fetchBrokers(); // Refresh after sync
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to sync broker',
      }));
      throw error;
    }
  }, [fetchBrokers]);

  useEffect(() => {
    fetchBrokers();

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchBrokers, refreshInterval);
    }

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchBrokers, refreshInterval]);

  return {
    ...state,
    refresh,
    connectBroker,
    disconnectBroker,
    syncBroker,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== STRATEGIES HOOK ====================

export function useStrategies(options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval } = options;
  
  const [state, setState] = useState<BaseHookState<Strategy[]>>({
    data: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchStrategies = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getStrategies(useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch strategies',
      }));
    }
  }, [useCache]);

  const refresh = useCallback(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  const createStrategy = useCallback(async (strategy: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>) => {
    try {
      await dbService.createStrategy(strategy);
      await fetchStrategies(); // Refresh after creation
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to create strategy',
      }));
      throw error;
    }
  }, [fetchStrategies]);

  const updateStrategy = useCallback(async (id: string, updates: any) => {
    try {
      await dbService.updateStrategy(id, updates);
      await fetchStrategies(); // Refresh after update
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to update strategy',
      }));
      throw error;
    }
  }, [fetchStrategies]);

  const deleteStrategy = useCallback(async (id: string) => {
    try {
      await dbService.deleteStrategy(id);
      await fetchStrategies(); // Refresh after deletion
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to delete strategy',
      }));
      throw error;
    }
  }, [fetchStrategies]);

  const backtestStrategy = useCallback(async (request: any) => {
    try {
      return await dbService.backtestStrategy(request);
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to backtest strategy',
      }));
      throw error;
    }
  }, []);

  useEffect(() => {
    fetchStrategies();

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchStrategies, refreshInterval);
    }

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchStrategies, refreshInterval]);

  return {
    ...state,
    refresh,
    createStrategy,
    updateStrategy,
    deleteStrategy,
    backtestStrategy,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== RISK METRICS HOOK ====================

export function useRiskMetrics(options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval, enableRealtime = true } = options;
  
  const [state, setState] = useState<BaseHookState<RiskMetrics>>({
    data: null,
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const fetchRiskMetrics = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getRiskMetrics(useCache);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch risk metrics',
      }));
    }
  }, [useCache]);

  const refresh = useCallback(() => {
    fetchRiskMetrics();
  }, [fetchRiskMetrics]);

  const checkRiskLimits = useCallback(async () => {
    try {
      return await dbService.checkRiskLimits();
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to check risk limits',
      }));
      throw error;
    }
  }, []);

  useEffect(() => {
    fetchRiskMetrics();

    // Setup real-time updates (risk metrics change frequently)
    if (enableRealtime) {
      // Note: You might want to create a specific risk websocket subscription
      // For now, we'll use general updates to trigger refresh
    }

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchRiskMetrics, refreshInterval);
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchRiskMetrics, refreshInterval, enableRealtime]);

  return {
    ...state,
    refresh,
    checkRiskLimits,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== MARKET DATA HOOK ====================

export function useMarketData(symbols: string[], options: BaseHookOptions = {}) {
  const { useCache = true, refreshInterval, enableRealtime = true } = options;
  
  const [state, setState] = useState<BaseHookState<MarketData[]>>({
    data: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRefs = useRef<(() => void)[]>([]);

  const fetchMarketData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await dbService.getMarketData(symbols);
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch market data',
      }));
    }
  }, [symbols, useCache]);

  const refresh = useCallback(() => {
    fetchMarketData();
  }, [fetchMarketData]);

  useEffect(() => {
    if (symbols.length === 0) {
      setState(prev => ({ ...prev, loading: false, data: [] }));
      return;
    }

    fetchMarketData();

    // Setup real-time updates for each symbol
    if (enableRealtime) {
      unsubscribeRefs.current = symbols.map(symbol =>
        wsService.subscribeMarketData((data) => {
          if (data.symbol === symbol) {
            fetchMarketData(); // Refresh all market data when any symbol updates
          }
        })
      );
    }

    // Setup refresh interval
    if (refreshInterval && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(fetchMarketData, refreshInterval);
    }

    return () => {
      unsubscribeRefs.current.forEach(unsubscribe => unsubscribe());
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchMarketData, refreshInterval, enableRealtime, symbols.join(',')]);

  return {
    ...state,
    refresh,
    isOnline: dbService.getOnlineStatus(),
    isConnected: dbService.isConnected(),
  };
}

// ==================== CONNECTION STATUS HOOK ====================

export function useConnectionStatus() {
  const [status, setStatus] = useState({
    isConnected: dbService.isConnected(),
    isOnline: dbService.getOnlineStatus(),
    wsReadyState: wsService.getReadyState(),
  });

  useEffect(() => {
    const updateStatus = () => {
      setStatus({
        isConnected: dbService.isConnected(),
        isOnline: dbService.getOnlineStatus(),
        wsReadyState: wsService.getReadyState(),
      });
    };

    const interval = setInterval(updateStatus, 1000); // Check every second

    return () => clearInterval(interval);
  }, []);

  return status;
}

// ==================== EXPORT HOOK ====================

export function useDataExport() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const exportData = useCallback(async (type: 'trades' | 'positions' | 'orders', format: 'csv' | 'json' = 'csv') => {
    try {
      setLoading(true);
      setError(null);
      
      const blob = await dbService.exportData(type, format);
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${type}_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      setLoading(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Export failed';
      setError(errorMessage);
      setLoading(false);
      throw new Error(errorMessage);
    }
  }, []);

  return {
    exportData,
    loading,
    error,
  };
}