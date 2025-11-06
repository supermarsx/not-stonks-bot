import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../services/api';
import type {
  Portfolio,
  Position,
  Order,
  Trade,
  Broker,
  Strategy,
  RiskMetrics,
  MarketData,
  CreateOrderRequest,
  SystemConfig,
} from '../types';

// Query keys
export const queryKeys = {
  portfolio: ['portfolio'] as const,
  portfolioHistory: (days: number) => ['portfolio', 'history', days] as const,
  positions: (broker?: string) => ['positions', broker] as const,
  position: (id: string) => ['position', id] as const,
  orders: (params?: Record<string, string>) => ['orders', params] as const,
  order: (id: string) => ['order', id] as const,
  trades: (params?: Record<string, string>) => ['trades', params] as const,
  brokers: ['brokers'] as const,
  broker: (id: string) => ['broker', id] as const,
  strategies: ['strategies'] as const,
  strategy: (id: string) => ['strategy', id] as const,
  riskMetrics: ['risk', 'metrics'] as const,
  riskHistory: (days: number) => ['risk', 'history', days] as const,
  marketData: (symbols: string[]) => ['market', 'data', symbols] as const,
  config: ['config'] as const,
};

// Portfolio hooks
export function usePortfolio() {
  return useQuery({
    queryKey: queryKeys.portfolio,
    queryFn: () => api.getPortfolio(),
    refetchInterval: 5000, // Refetch every 5 seconds
  });
}

export function usePortfolioHistory(days: number = 30) {
  return useQuery({
    queryKey: queryKeys.portfolioHistory(days),
    queryFn: () => api.getPortfolioHistory(days),
  });
}

// Positions hooks
export function usePositions(broker?: string) {
  return useQuery({
    queryKey: queryKeys.positions(broker),
    queryFn: () => api.getPositions(broker),
    refetchInterval: 3000,
  });
}

export function usePosition(id: string) {
  return useQuery({
    queryKey: queryKeys.position(id),
    queryFn: () => api.getPosition(id),
    enabled: !!id,
  });
}

export function useClosePosition() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.closePosition(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.positions() });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolio });
    },
  });
}

// Orders hooks
export function useOrders(params?: { broker?: string; status?: string; limit?: number }) {
  return useQuery({
    queryKey: queryKeys.orders(params as Record<string, string>),
    queryFn: () => api.getOrders(params),
    refetchInterval: 2000,
  });
}

export function useCreateOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateOrderRequest) => api.createOrder(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
    },
  });
}

export function useCancelOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.cancelOrder(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
    },
  });
}

// Trades hooks
export function useTrades(params?: { broker?: string; symbol?: string; limit?: number }) {
  return useQuery({
    queryKey: queryKeys.trades(params as Record<string, string>),
    queryFn: () => api.getTrades(params),
  });
}

// Brokers hooks
export function useBrokers() {
  return useQuery({
    queryKey: queryKeys.brokers,
    queryFn: () => api.getBrokers(),
    refetchInterval: 10000,
  });
}

export function useConnectBroker() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, credentials }: { id: string; credentials: Record<string, string> }) =>
      api.connectBroker(id, credentials),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.brokers });
    },
  });
}

export function useSyncBroker() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.syncBroker(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.brokers });
      queryClient.invalidateQueries({ queryKey: queryKeys.positions() });
      queryClient.invalidateQueries({ queryKey: queryKeys.orders() });
    },
  });
}

// Strategies hooks
export function useStrategies() {
  return useQuery({
    queryKey: queryKeys.strategies,
    queryFn: () => api.getStrategies(),
    refetchInterval: 5000,
  });
}

export function useUpdateStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: { status?: Strategy['status'] } }) =>
      api.updateStrategy(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.strategies });
    },
  });
}

// Risk hooks
export function useRiskMetrics() {
  return useQuery({
    queryKey: queryKeys.riskMetrics,
    queryFn: () => api.getRiskMetrics(),
    refetchInterval: 5000,
  });
}

export function useRiskHistory(days: number = 30) {
  return useQuery({
    queryKey: queryKeys.riskHistory(days),
    queryFn: () => api.getRiskHistory(days),
  });
}

// Market data hooks
export function useMarketData(symbols: string[]) {
  return useQuery({
    queryKey: queryKeys.marketData(symbols),
    queryFn: () => api.getMarketData(symbols),
    enabled: symbols.length > 0,
    refetchInterval: 5000,
  });
}

// Config hooks
export function useConfig() {
  return useQuery({
    queryKey: queryKeys.config,
    queryFn: () => api.getConfig(),
  });
}

export function useUpdateConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (updates: Partial<SystemConfig>) => api.updateConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.config });
    },
  });
}
