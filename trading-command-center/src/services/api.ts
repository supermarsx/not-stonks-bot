import axios, { AxiosInstance, AxiosError } from 'axios';
import toast from 'react-hot-toast';
import type {
  Portfolio,
  Position,
  Order,
  Trade,
  Broker,
  Strategy,
  RiskMetrics,
  MarketData,
  AIAnalysis,
  CreateOrderRequest,
  UpdateStrategyRequest,
  AIQueryRequest,
  BacktestRequest,
  SystemConfig,
  ApiResponse,
  PaginatedResponse,
} from '../types';

// API Base URL - update this to match your FastAPI backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with defaults
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth tokens
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail: string }>) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred';
    
    // Don't show toast for 401/403 errors (handled by auth system)
    if (error.response?.status !== 401 && error.response?.status !== 403) {
      toast.error(message);
    }
    
    return Promise.reject(error);
  }
);

// API Service Class
class TradingAPI {
  // Portfolio & Account
  async getPortfolio(): Promise<Portfolio> {
    const { data } = await apiClient.get<ApiResponse<Portfolio>>('/api/portfolio');
    return data.data!;
  }

  async getPortfolioHistory(days: number = 30): Promise<Portfolio[]> {
    const { data } = await apiClient.get<ApiResponse<Portfolio[]>>(`/api/portfolio/history?days=${days}`);
    return data.data!;
  }

  // Positions
  async getPositions(broker?: string): Promise<Position[]> {
    const params = broker ? { broker } : {};
    const { data } = await apiClient.get<ApiResponse<Position[]>>('/api/positions', { params });
    return data.data!;
  }

  async getPosition(id: string): Promise<Position> {
    const { data } = await apiClient.get<ApiResponse<Position>>(`/api/positions/${id}`);
    return data.data!;
  }

  async closePosition(id: string): Promise<void> {
    await apiClient.post(`/api/positions/${id}/close`);
    toast.success('Position closed successfully');
  }

  // Orders
  async getOrders(params?: { broker?: string; status?: string; limit?: number }): Promise<Order[]> {
    const { data } = await apiClient.get<ApiResponse<Order[]>>('/api/orders', { params });
    return data.data!;
  }

  async getOrder(id: string): Promise<Order> {
    const { data } = await apiClient.get<ApiResponse<Order>>(`/api/orders/${id}`);
    return data.data!;
  }

  async createOrder(request: CreateOrderRequest): Promise<Order> {
    const { data } = await apiClient.post<ApiResponse<Order>>('/api/orders', request);
    toast.success('Order created successfully');
    return data.data!;
  }

  async cancelOrder(id: string): Promise<void> {
    await apiClient.delete(`/api/orders/${id}`);
    toast.success('Order cancelled');
  }

  async modifyOrder(id: string, updates: Partial<CreateOrderRequest>): Promise<Order> {
    const { data } = await apiClient.patch<ApiResponse<Order>>(`/api/orders/${id}`, updates);
    toast.success('Order modified');
    return data.data!;
  }

  // Trades
  async getTrades(params?: { broker?: string; symbol?: string; limit?: number }): Promise<Trade[]> {
    const { data } = await apiClient.get<ApiResponse<Trade[]>>('/api/trades', { params });
    return data.data!;
  }

  async getTradesPaginated(page: number = 1, pageSize: number = 50): Promise<PaginatedResponse<Trade>> {
    const { data } = await apiClient.get<ApiResponse<PaginatedResponse<Trade>>>('/api/trades/paginated', {
      params: { page, pageSize },
    });
    return data.data!;
  }

  // Brokers
  async getBrokers(): Promise<Broker[]> {
    const { data } = await apiClient.get<ApiResponse<Broker[]>>('/api/brokers');
    return data.data!;
  }

  async getBroker(id: string): Promise<Broker> {
    const { data } = await apiClient.get<ApiResponse<Broker>>(`/api/brokers/${id}`);
    return data.data!;
  }

  async connectBroker(id: string, credentials: Record<string, string>): Promise<Broker> {
    const { data } = await apiClient.post<ApiResponse<Broker>>(`/api/brokers/${id}/connect`, credentials);
    toast.success(`Connected to ${id}`);
    return data.data!;
  }

  async disconnectBroker(id: string): Promise<void> {
    await apiClient.post(`/api/brokers/${id}/disconnect`);
    toast.success(`Disconnected from ${id}`);
  }

  async syncBroker(id: string): Promise<void> {
    await apiClient.post(`/api/brokers/${id}/sync`);
    toast.success('Broker synced');
  }

  // Strategies
  async getStrategies(): Promise<Strategy[]> {
    const { data } = await apiClient.get<ApiResponse<Strategy[]>>('/api/strategies');
    return data.data!;
  }

  async getStrategy(id: string): Promise<Strategy> {
    const { data } = await apiClient.get<ApiResponse<Strategy>>(`/api/strategies/${id}`);
    return data.data!;
  }

  async createStrategy(strategy: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>): Promise<Strategy> {
    const { data } = await apiClient.post<ApiResponse<Strategy>>('/api/strategies', strategy);
    toast.success('Strategy created');
    return data.data!;
  }

  async updateStrategy(id: string, updates: UpdateStrategyRequest): Promise<Strategy> {
    const { data } = await apiClient.patch<ApiResponse<Strategy>>(`/api/strategies/${id}`, updates);
    toast.success('Strategy updated');
    return data.data!;
  }

  async deleteStrategy(id: string): Promise<void> {
    await apiClient.delete(`/api/strategies/${id}`);
    toast.success('Strategy deleted');
  }

  async backtestStrategy(request: BacktestRequest): Promise<AIAnalysis> {
    const { data } = await apiClient.post<ApiResponse<AIAnalysis>>('/api/strategies/backtest', request);
    return data.data!;
  }

  // Risk Management
  async getRiskMetrics(): Promise<RiskMetrics> {
    const { data } = await apiClient.get<ApiResponse<RiskMetrics>>('/api/risk/metrics');
    return data.data!;
  }

  async getRiskHistory(days: number = 30): Promise<RiskMetrics[]> {
    const { data } = await apiClient.get<ApiResponse<RiskMetrics[]>>(`/api/risk/history?days=${days}`);
    return data.data!;
  }

  async checkRiskLimits(): Promise<{ passed: boolean; violations: string[] }> {
    const { data } = await apiClient.get<ApiResponse<{ passed: boolean; violations: string[] }>>('/api/risk/check');
    return data.data!;
  }

  // Market Data
  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    const { data } = await apiClient.post<ApiResponse<MarketData[]>>('/api/market/data', { symbols });
    return data.data!;
  }

  async getMarketQuote(symbol: string): Promise<MarketData> {
    const { data } = await apiClient.get<ApiResponse<MarketData>>(`/api/market/quote/${symbol}`);
    return data.data!;
  }

  // AI Assistant
  async sendAIQuery(request: AIQueryRequest): Promise<{ response: string; analysis?: AIAnalysis }> {
    const { data } = await apiClient.post<ApiResponse<{ response: string; analysis?: AIAnalysis }>>('/api/ai/query', request);
    return data.data!;
  }

  async getAIAnalyses(limit: number = 20): Promise<AIAnalysis[]> {
    const { data } = await apiClient.get<ApiResponse<AIAnalysis[]>>(`/api/ai/analyses?limit=${limit}`);
    return data.data!;
  }

  async analyzeMarket(symbols: string[]): Promise<AIAnalysis> {
    const { data } = await apiClient.post<ApiResponse<AIAnalysis>>('/api/ai/analyze-market', { symbols });
    return data.data!;
  }

  // Configuration
  async getConfig(): Promise<SystemConfig> {
    const { data } = await apiClient.get<ApiResponse<SystemConfig>>('/api/config');
    return data.data!;
  }

  async updateConfig(updates: Partial<SystemConfig>): Promise<SystemConfig> {
    const { data } = await apiClient.patch<ApiResponse<SystemConfig>>('/api/config', updates);
    toast.success('Configuration updated');
    return data.data!;
  }

  // System Health
  async getSystemHealth(): Promise<{ status: string; services: Record<string, boolean> }> {
    const { data } = await apiClient.get<ApiResponse<{ status: string; services: Record<string, boolean> }>>('/api/health');
    return data.data!;
  }
}

// Export singleton instance
export const api = new TradingAPI();
export default api;
