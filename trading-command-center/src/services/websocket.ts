import toast from 'react-hot-toast';
import type { WSMessage, Portfolio, Position, Order, Trade, MarketData, Alert } from '../types';

type MessageHandler<T = unknown> = (data: T) => void;

interface Subscription {
  id: string;
  handler: MessageHandler;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  private subscriptions = new Map<string, Subscription[]>();
  private isConnecting = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private url: string;

  constructor() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = import.meta.env.VITE_WS_URL || 
                   `${wsProtocol}//${window.location.hostname}:8000`;
    this.url = `${wsHost}/ws`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        return;
      }

      this.isConnecting = true;

      try {
        const token = localStorage.getItem('auth_token');
        const wsUrl = token ? `${this.url}?token=${token}` : this.url;
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('[WS] Connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          toast.success('Real-time connection established');
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log('[WS] Disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.stopHeartbeat();
          
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[WS] Reconnecting... Attempt ${this.reconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay);
          } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            toast.error('Connection lost. Please refresh the page.');
          }
        };

        this.ws.onerror = (error) => {
          console.error('[WS] Error:', error);
          this.isConnecting = false;
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WSMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('[WS] Failed to parse message:', error);
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.subscriptions.clear();
    console.log('[WS] Disconnected by client');
  }

  private handleMessage(message: WSMessage): void {
    const subscriptions = this.subscriptions.get(message.type);
    
    if (subscriptions) {
      subscriptions.forEach((sub) => {
        try {
          sub.handler(message.data);
        } catch (error) {
          console.error('[WS] Handler error:', error);
        }
      });
    }

    // Handle alerts specially
    if (message.type === 'alert') {
      const alert = message.data as Alert;
      const toastFn = alert.severity === 'error' || alert.severity === 'critical' 
        ? toast.error 
        : alert.severity === 'warning' 
        ? toast 
        : toast.success;
      
      toastFn(`${alert.title}: ${alert.message}`);
    }
  }

  subscribe<T = unknown>(
    type: WSMessage['type'],
    handler: MessageHandler<T>
  ): () => void {
    const id = `${type}_${Date.now()}_${Math.random()}`;
    const subscription: Subscription = { id, handler: handler as MessageHandler };

    if (!this.subscriptions.has(type)) {
      this.subscriptions.set(type, []);
    }

    this.subscriptions.get(type)!.push(subscription);

    // Send subscription message to server (using portfolio type as placeholder)
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'subscribe',
        channel: type,
        timestamp: new Date().toISOString(),
      }));
    }

    // Return unsubscribe function
    return () => {
      const subs = this.subscriptions.get(type);
      if (subs) {
        const index = subs.findIndex((s) => s.id === id);
        if (index !== -1) {
          subs.splice(index, 1);
        }
        
        // If no more subscriptions for this type, unsubscribe from server
        if (subs.length === 0) {
          if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
              action: 'unsubscribe',
              channel: type,
              timestamp: new Date().toISOString(),
            }));
          }
          this.subscriptions.delete(type);
        }
      }
    };
  }

  // Convenience methods for specific data types
  subscribePortfolio(handler: MessageHandler<Portfolio>): () => void {
    return this.subscribe<Portfolio>('portfolio', handler);
  }

  subscribePositions(handler: MessageHandler<Position[]>): () => void {
    return this.subscribe<Position[]>('position', handler);
  }

  subscribeOrders(handler: MessageHandler<Order>): () => void {
    return this.subscribe<Order>('order', handler);
  }

  subscribeTrades(handler: MessageHandler<Trade>): () => void {
    return this.subscribe<Trade>('trade', handler);
  }

  subscribeMarketData(handler: MessageHandler<MarketData>): () => void {
    return this.subscribe<MarketData>('market_data', handler);
  }

  subscribeAlerts(handler: MessageHandler<Alert>): () => void {
    return this.subscribe<Alert>('alert', handler);
  }

  send(message: Partial<WSMessage>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        ...message,
        timestamp: new Date().toISOString(),
      }));
    } else {
      console.warn('[WS] Cannot send message - not connected');
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'ping', timestamp: new Date().toISOString() }));
      }
    }, 30000); // Ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getReadyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }
}

// Export singleton instance
export const wsService = new WebSocketService();
export default wsService;
