import { Order, Position, Strategy, BrokerConnection } from '@/stores/tradingStore';

const API_BASE_URL = 'http://localhost:8000/api';

class ApiService {
  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Portfolio endpoints
  async getPortfolio(): Promise<any> {
    return this.request('/portfolio');
  }

  async getPositions(): Promise<Position[]> {
    return this.request('/positions');
  }

  async getOrders(): Promise<Order[]> {
    return this.request('/orders');
  }

  // Trading endpoints
  async submitOrder(order: Omit<Order, 'id' | 'timestamp' | 'status'>): Promise<Order> {
    return this.request('/orders', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async cancelOrder(orderId: string): Promise<void> {
    return this.request(`/orders/${orderId}`, {
      method: 'DELETE',
    });
  }

  // Market data endpoints
  async getMarketData(symbols: string[]): Promise<any> {
    return this.request(`/market-data?symbols=${symbols.join(',')}`);
  }

  // Strategy endpoints
  async getStrategies(): Promise<Strategy[]> {
    return this.request('/strategies');
  }

  async createStrategy(strategy: Omit<Strategy, 'id'>): Promise<Strategy> {
    return this.request('/strategies', {
      method: 'POST',
      body: JSON.stringify(strategy),
    });
  }

  async updateStrategy(strategyId: string, updates: Partial<Strategy>): Promise<Strategy> {
    return this.request(`/strategies/${strategyId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async startStrategy(strategyId: string): Promise<void> {
    return this.request(`/strategies/${strategyId}/start`, {
      method: 'POST',
    });
  }

  async stopStrategy(strategyId: string): Promise<void> {
    return this.request(`/strategies/${strategyId}/stop`, {
      method: 'POST',
    });
  }

  async deleteStrategy(strategyId: string): Promise<void> {
    return this.request(`/strategies/${strategyId}`, {
      method: 'DELETE',
    });
  }

  // Risk management endpoints
  async getRiskMetrics(): Promise<any> {
    return this.request('/risk/metrics');
  }

  async getRiskLimits(): Promise<any> {
    return this.request('/risk/limits');
  }

  async updateRiskLimits(limits: any): Promise<void> {
    return this.request('/risk/limits', {
      method: 'PUT',
      body: JSON.stringify(limits),
    });
  }

  // Broker endpoints
  async getBrokers(): Promise<BrokerConnection[]> {
    return this.request('/brokers');
  }

  async connectBroker(brokerId: string): Promise<void> {
    return this.request(`/brokers/${brokerId}/connect`, {
      method: 'POST',
    });
  }

  async disconnectBroker(brokerId: string): Promise<void> {
    return this.request(`/brokers/${brokerId}/disconnect`, {
      method: 'POST',
    });
  }

  async getBrokerConfig(brokerId: string): Promise<any> {
    return this.request(`/brokers/${brokerId}/config`);
  }

  async updateBrokerConfig(brokerId: string, config: any): Promise<void> {
    return this.request(`/brokers/${brokerId}/config`, {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  // AI Chat endpoints
  async sendChatMessage(message: string): Promise<any> {
    return this.request('/chat/message', {
      method: 'POST',
      body: JSON.stringify({ message }),
    });
  }

  async getChatHistory(): Promise<any[]> {
    return this.request('/chat/history');
  }

  // System endpoints
  async getSystemStatus(): Promise<any> {
    return this.request('/system/status');
  }

  async getHealthCheck(): Promise<any> {
    return this.request('/health');
  }

  // Configuration endpoints
  async getConfig(): Promise<any> {
    return this.request('/config');
  }

  async updateConfig(config: any): Promise<void> {
    return this.request('/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }
}

export const apiService = new ApiService();