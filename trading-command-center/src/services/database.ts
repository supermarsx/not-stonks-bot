/**
 * Database Connection Service
 * Handles all database operations for the Trading Command Center
 * Provides connection pooling, caching, and optimization
 */

import { api } from './api';
import { wsService } from './websocket';
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
  ApiResponse,
} from '../types';

interface CacheConfig {
  ttl: number; // Time to live in milliseconds
  maxSize: number; // Maximum cache entries
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

class DatabaseService {
  private cache = new Map<string, CacheEntry<any>>();
  private cacheConfig: Record<string, CacheConfig> = {
    portfolio: { ttl: 30000, maxSize: 50 }, // 30 seconds
    positions: { ttl: 15000, maxSize: 100 }, // 15 seconds
    orders: { ttl: 10000, maxSize: 200 }, // 10 seconds
    trades: { ttl: 5000, maxSize: 500 }, // 5 seconds
    strategies: { ttl: 60000, maxSize: 50 }, // 1 minute
    risk: { ttl: 30000, maxSize: 50 }, // 30 seconds
    brokers: { ttl: 300000, maxSize: 20 }, // 5 minutes
    marketData: { ttl: 5000, maxSize: 1000 }, // 5 seconds
  };
  private connectionPool: Map<string, any> = new Map();
  private isOnline = true;

  constructor() {
    // Initialize WebSocket listeners for real-time updates
    this.setupWebSocketListeners();
    
    // Setup online/offline detection
    window.addEventListener('online', () => this.handleOnline());
    window.addEventListener('offline', () => this.handleOffline());
  }

  // ==================== CONNECTION MANAGEMENT ====================

  async connect(): Promise<void> {
    try {
      await wsService.connect();
      this.isOnline = true;
      console.log('[DB] Connected to database via WebSocket');
    } catch (error) {
      console.error('[DB] Failed to connect:', error);
      this.isOnline = false;
      throw error;
    }
  }

  disconnect(): void {
    wsService.disconnect();
    this.connectionPool.clear();
    this.cache.clear();
    console.log('[DB] Disconnected from database');
  }

  private setupWebSocketListeners(): void {
    // Portfolio updates
    wsService.subscribePortfolio((portfolio) => {
      this.updateCache('portfolio', portfolio);
    });

    // Position updates
    wsService.subscribePositions((positions) => {
      this.updateCache('positions', positions);
    });

    // Order updates
    wsService.subscribeOrders((order) => {
      this.updateCache('orders', null); // Invalidate orders cache
    });

    // Trade updates
    wsService.subscribeTrades((trade) => {
      this.updateCache('trades', null); // Invalidate trades cache
    });

    // Market data updates
    wsService.subscribeMarketData((data) => {
      this.updateMarketDataCache(data.symbol, data);
    });
  }

  private handleOnline(): void {
    this.isOnline = true;
    console.log('[DB] Back online - clearing cache');
    this.clearAllCache();
  }

  private handleOffline(): void {
    this.isOnline = false;
    console.warn('[DB] Gone offline - using cached data');
  }

  // ==================== CACHING LAYER ====================

  private getCacheKey(prefix: string, params?: Record<string, any>): string {
    if (!params) return prefix;
    const sortedParams = Object.keys(params).sort().reduce((result, key) => {
      result[key] = params[key];
      return result;
    }, {} as Record<string, any>);
    return `${prefix}_${JSON.stringify(sortedParams)}`;
  }

  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.data as T;
  }

  private setCache<T>(key: string, data: T, configKey: string): void {
    const config = this.cacheConfig[configKey];
    if (!config) return;

    // Enforce cache size limit
    const cacheKeys = Array.from(this.cache.keys()).filter(k => k.startsWith(configKey));
    if (cacheKeys.length >= config.maxSize) {
      const oldestKey = cacheKeys.sort((a, b) => 
        (this.cache.get(a)?.timestamp || 0) - (this.cache.get(b)?.timestamp || 0)
      )[0];
      this.cache.delete(oldestKey);
    }

    const now = Date.now();
    this.cache.set(key, {
      data,
      timestamp: now,
      expiresAt: now + config.ttl,
    });
  }

  private updateCache(configKey: string, data: any): void {
    const keys = Array.from(this.cache.keys()).filter(k => k.startsWith(configKey));
    if (data === null) {
      // Invalidate cache
      keys.forEach(key => this.cache.delete(key));
    } else {
      // Update cache entries
      keys.forEach(key => {
        const entry = this.cache.get(key);
        if (entry) {
          entry.data = data;
          entry.timestamp = Date.now();
        }
      });
    }
  }

  private updateMarketDataCache(symbol: string, data: MarketData): void {
    const key = this.getCacheKey('marketData', { symbol });
    this.setCache(key, data, 'marketData');
  }

  private clearAllCache(): void {
    this.cache.clear();
  }

  // ==================== PORTFOLIO OPERATIONS ====================

  async getPortfolio(useCache = true): Promise<Portfolio> {
    const cacheKey = this.getCacheKey('portfolio');
    if (useCache) {
      const cached = this.getFromCache<Portfolio>(cacheKey);
      if (cached) return cached;
    }

    try {
      const portfolio = await api.getPortfolio();
      this.setCache(cacheKey, portfolio, 'portfolio');
      return portfolio;
    } catch (error) {
      // Fallback to cached data if offline
      const cached = this.getFromCache<Portfolio>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getPortfolioHistory(days = 30): Promise<Portfolio[]> {
    const cacheKey = this.getCacheKey('portfolioHistory', { days });
    const cached = this.getFromCache<Portfolio[]>(cacheKey);
    if (cached) return cached;

    try {
      const history = await api.getPortfolioHistory(days);
      this.setCache(cacheKey, history, 'portfolio');
      return history;
    } catch (error) {
      const cached = this.getFromCache<Portfolio[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  // ==================== POSITION OPERATIONS ====================

  async getPositions(broker?: string, useCache = true): Promise<Position[]> {
    const cacheKey = this.getCacheKey('positions', { broker });
    if (useCache) {
      const cached = this.getFromCache<Position[]>(cacheKey);
      if (cached) return cached;
    }

    try {
      const positions = await api.getPositions(broker);
      this.setCache(cacheKey, positions, 'positions');
      return positions;
    } catch (error) {
      const cached = this.getFromCache<Position[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getPosition(id: string): Promise<Position> {
    const cacheKey = this.getCacheKey('position', { id });
    const cached = this.getFromCache<Position>(cacheKey);
    if (cached) return cached;

    try {
      const position = await api.getPosition(id);
      this.setCache(cacheKey, position, 'positions');
      return position;
    } catch (error) {
      const cached = this.getFromCache<Position>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async closePosition(id: string): Promise<void> {
    try {
      await api.closePosition(id);
      // Invalidate positions cache
      this.updateCache('positions', null);
    } catch (error) {
      throw error;
    }
  }

  // ==================== ORDER OPERATIONS ====================

  async getOrders(params?: { broker?: string; status?: string; limit?: number }, useCache = true): Promise<Order[]> {
    const cacheKey = this.getCacheKey('orders', params);
    if (useCache) {
      const cached = this.getFromCache<Order[]>(cacheKey);
      if (cached) return cached;
    }

    try {
      const orders = await api.getOrders(params);
      this.setCache(cacheKey, orders, 'orders');
      return orders;
    } catch (error) {
      const cached = this.getFromCache<Order[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getOrder(id: string): Promise<Order> {
    const cacheKey = this.getCacheKey('order', { id });
    const cached = this.getFromCache<Order>(cacheKey);
    if (cached) return cached;

    try {
      const order = await api.getOrder(id);
      this.setCache(cacheKey, order, 'orders');
      return order;
    } catch (error) {
      const cached = this.getFromCache<Order>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async createOrder(request: any): Promise<Order> {
    try {
      const order = await api.createOrder(request);
      // Invalidate orders cache
      this.updateCache('orders', null);
      return order;
    } catch (error) {
      throw error;
    }
  }

  async cancelOrder(id: string): Promise<void> {
    try {
      await api.cancelOrder(id);
      this.updateCache('orders', null);
    } catch (error) {
      throw error;
    }
  }

  async modifyOrder(id: string, updates: any): Promise<Order> {
    try {
      const order = await api.modifyOrder(id, updates);
      this.updateCache('orders', null);
      return order;
    } catch (error) {
      throw error;
    }
  }

  // ==================== TRADE OPERATIONS ====================

  async getTrades(params?: { broker?: string; symbol?: string; limit?: number }, useCache = true): Promise<Trade[]> {
    const cacheKey = this.getCacheKey('trades', params);
    if (useCache) {
      const cached = this.getFromCache<Trade[]>(cacheKey);
      if (cached) return cached;
    }

    try {
      const trades = await api.getTrades(params);
      this.setCache(cacheKey, trades, 'trades');
      return trades;
    } catch (error) {
      const cached = this.getFromCache<Trade[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getTradesPaginated(page = 1, pageSize = 50): Promise<PaginatedResponse<Trade>> {
    const cacheKey = this.getCacheKey('tradesPaginated', { page, pageSize });
    const cached = this.getFromCache<PaginatedResponse<Trade>>(cacheKey);
    if (cached) return cached;

    try {
      const trades = await api.getTradesPaginated(page, pageSize);
      this.setCache(cacheKey, trades, 'trades');
      return trades;
    } catch (error) {
      const cached = this.getFromCache<PaginatedResponse<Trade>>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  // ==================== BROKER OPERATIONS ====================

  async getBrokers(useCache = true): Promise<Broker[]> {
    const cacheKey = 'brokers';
    if (useCache) {
      const cached = this.getFromCache<Broker[]>(cacheKey);
      if (cached) return cached;
    }

    try {
      const brokers = await api.getBrokers();
      this.setCache(cacheKey, brokers, 'brokers');
      return brokers;
    } catch (error) {
      const cached = this.getFromCache<Broker[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getBroker(id: string): Promise<Broker> {
    const cacheKey = this.getCacheKey('broker', { id });
    const cached = this.getFromCache<Broker>(cacheKey);
    if (cached) return cached;

    try {
      const broker = await api.getBroker(id);
      this.setCache(cacheKey, broker, 'brokers');
      return broker;
    } catch (error) {
      const cached = this.getFromCache<Broker>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async connectBroker(id: string, credentials: Record<string, string>): Promise<Broker> {
    try {
      const broker = await api.connectBroker(id, credentials);
      // Invalidate brokers cache
      this.updateCache('brokers', null);
      return broker;
    } catch (error) {
      throw error;
    }
  }

  async disconnectBroker(id: string): Promise<void> {
    try {
      await api.disconnectBroker(id);
      this.updateCache('brokers', null);
    } catch (error) {
      throw error;
    }
  }

  async syncBroker(id: string): Promise<void> {
    try {
      await api.syncBroker(id);
      // Invalidate relevant caches
      this.updateCache('positions', null);
      this.updateCache('orders', null);
      this.updateCache('brokers', null);
    } catch (error) {
      throw error;
    }
  }

  // ==================== STRATEGY OPERATIONS ====================

  async getStrategies(useCache = true): Promise<Strategy[]> {
    const cacheKey = 'strategies';
    if (useCache) {
      const cached = this.getFromCache<Strategy[]>(cacheKey);
      if (cached) return cached;
    }

    try {
      const strategies = await api.getStrategies();
      this.setCache(cacheKey, strategies, 'strategies');
      return strategies;
    } catch (error) {
      const cached = this.getFromCache<Strategy[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getStrategy(id: string): Promise<Strategy> {
    const cacheKey = this.getCacheKey('strategy', { id });
    const cached = this.getFromCache<Strategy>(cacheKey);
    if (cached) return cached;

    try {
      const strategy = await api.getStrategy(id);
      this.setCache(cacheKey, strategy, 'strategies');
      return strategy;
    } catch (error) {
      const cached = this.getFromCache<Strategy>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async createStrategy(strategy: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>): Promise<Strategy> {
    try {
      const newStrategy = await api.createStrategy(strategy);
      this.updateCache('strategies', null);
      return newStrategy;
    } catch (error) {
      throw error;
    }
  }

  async updateStrategy(id: string, updates: any): Promise<Strategy> {
    try {
      const updated = await api.updateStrategy(id, updates);
      this.updateCache('strategies', null);
      return updated;
    } catch (error) {
      throw error;
    }
  }

  async deleteStrategy(id: string): Promise<void> {
    try {
      await api.deleteStrategy(id);
      this.updateCache('strategies', null);
    } catch (error) {
      throw error;
    }
  }

  async backtestStrategy(request: any): Promise<any> {
    try {
      return await api.backtestStrategy(request);
    } catch (error) {
      throw error;
    }
  }

  // ==================== RISK MANAGEMENT ====================

  async getRiskMetrics(useCache = true): Promise<RiskMetrics> {
    const cacheKey = 'riskMetrics';
    if (useCache) {
      const cached = this.getFromCache<RiskMetrics>(cacheKey);
      if (cached) return cached;
    }

    try {
      const metrics = await api.getRiskMetrics();
      this.setCache(cacheKey, metrics, 'risk');
      return metrics;
    } catch (error) {
      const cached = this.getFromCache<RiskMetrics>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async getRiskHistory(days = 30): Promise<RiskMetrics[]> {
    const cacheKey = this.getCacheKey('riskHistory', { days });
    const cached = this.getFromCache<RiskMetrics[]>(cacheKey);
    if (cached) return cached;

    try {
      const history = await api.getRiskHistory(days);
      this.setCache(cacheKey, history, 'risk');
      return history;
    } catch (error) {
      const cached = this.getFromCache<RiskMetrics[]>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  async checkRiskLimits(): Promise<{ passed: boolean; violations: string[] }> {
    try {
      return await api.checkRiskLimits();
    } catch (error) {
      throw error;
    }
  }

  // ==================== MARKET DATA ====================

  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    const cacheKey = this.getCacheKey('marketDataBulk', { symbols: symbols.sort() });
    
    // Check individual symbol caches first
    const results: MarketData[] = [];
    const missingSymbols: string[] = [];

    for (const symbol of symbols) {
      const symbolCacheKey = this.getCacheKey('marketData', { symbol });
      const cached = this.getFromCache<MarketData>(symbolCacheKey);
      
      if (cached) {
        results.push(cached);
      } else {
        missingSymbols.push(symbol);
      }
    }

    // Fetch missing symbols if any
    if (missingSymbols.length > 0) {
      try {
        const freshData = await api.getMarketData(missingSymbols);
        
        // Cache each symbol individually
        freshData.forEach(data => {
          const symbolCacheKey = this.getCacheKey('marketData', { symbol: data.symbol });
          this.setCache(symbolCacheKey, data, 'marketData');
          results.push(data);
        });

        return results;
      } catch (error) {
        // If offline, return cached data only
        if (!this.isOnline) {
          return results;
        }
        throw error;
      }
    }

    return results;
  }

  async getMarketQuote(symbol: string): Promise<MarketData> {
    const cacheKey = this.getCacheKey('marketData', { symbol });
    const cached = this.getFromCache<MarketData>(cacheKey);
    if (cached) return cached;

    try {
      const quote = await api.getMarketQuote(symbol);
      this.setCache(cacheKey, quote, 'marketData');
      return quote;
    } catch (error) {
      const cached = this.getFromCache<MarketData>(cacheKey);
      if (cached && !this.isOnline) {
        return cached;
      }
      throw error;
    }
  }

  // ==================== EXPORT FUNCTIONALITY ====================

  async exportData(type: 'trades' | 'positions' | 'orders', format: 'csv' | 'json' = 'csv'): Promise<Blob> {
    try {
      const endpoint = `/api/export/${type}`;
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
        body: JSON.stringify({ format }),
      });

      if (!response.ok) {
        throw new Error('Export failed');
      }

      return await response.blob();
    } catch (error) {
      throw error;
    }
  }

  // ==================== BULK OPERATIONS ====================

  async bulkUpdatePositions(positions: Position[]): Promise<void> {
    // Update cache in batches for performance
    const batchSize = 50;
    for (let i = 0; i < positions.length; i += batchSize) {
      const batch = positions.slice(i, i + batchSize);
      this.setCache(this.getCacheKey('positions'), batch, 'positions');
    }
  }

  async preloadEssentialData(): Promise<void> {
    // Preload critical data in parallel
    const promises = [
      this.getPortfolio(),
      this.getPositions(),
      this.getBrokers(),
      this.getStrategies(),
    ];

    try {
      await Promise.allSettled(promises);
    } catch (error) {
      console.warn('[DB] Some preloading failed:', error);
    }
  }

  // ==================== UTILITY METHODS ====================

  isConnected(): boolean {
    return wsService.isConnected();
  }

  getCacheStats(): { size: number; entries: string[] } {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys()),
    };
  }

  clearCache(pattern?: string): void {
    if (!pattern) {
      this.clearAllCache();
      return;
    }

    const keys = Array.from(this.cache.keys()).filter(key => key.includes(pattern));
    keys.forEach(key => this.cache.delete(key));
  }

  getOnlineStatus(): boolean {
    return this.isOnline;
  }
}

// Export singleton instance
export const dbService = new DatabaseService();
export default dbService;