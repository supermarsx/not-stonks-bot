import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useTradingStore } from '@/stores/tradingStore';

interface UseWebSocketOptions {
  url?: string;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = ({
  url = 'ws://localhost:8000',
  reconnectAttempts = 5,
  reconnectInterval = 3000,
}: UseWebSocketOptions = {}) => {
  const socketRef = useRef<Socket | null>(null);
  const {
    setWebSocketConnected,
    updateMarketData,
    updatePortfolio,
    addAlert,
    addChatMessage,
    addNotification,
  } = useTradingStore();

  useEffect(() => {
    const connectSocket = () => {
      socketRef.current = io(url, {
        transports: ['websocket'],
        timeout: 20000,
        reconnection: true,
        reconnectionAttempts: reconnectAttempts,
        reconnectionDelay: reconnectInterval,
      });

      socketRef.current.on('connect', () => {
        console.log('WebSocket connected');
        setWebSocketConnected(true);
        addNotification({
          type: 'success',
          message: 'Connected to trading server',
        });
      });

      socketRef.current.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setWebSocketConnected(false);
        addNotification({
          type: 'warning',
          message: 'Disconnected from trading server',
        });
      });

      socketRef.current.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setWebSocketConnected(false);
        addAlert({
          type: 'error',
          title: 'Connection Error',
          message: 'Failed to connect to trading server',
        });
      });

      // Market data updates
      socketRef.current.on('market_data', (data) => {
        if (Array.isArray(data)) {
          data.forEach((item) => updateMarketData(item.symbol, item));
        } else {
          updateMarketData(data.symbol, data);
        }
      });

      // Portfolio updates
      socketRef.current.on('portfolio_update', (portfolioData) => {
        updatePortfolio(portfolioData);
      });

      // Order updates
      socketRef.current.on('order_filled', (orderData) => {
        addNotification({
          type: 'success',
          message: `Order filled: ${orderData.symbol} ${orderData.side} ${orderData.size}`,
        });
      });

      socketRef.current.on('order_rejected', (orderData) => {
        addAlert({
          type: 'error',
          title: 'Order Rejected',
          message: `Order rejected: ${orderData.reason}`,
        });
      });

      // Risk alerts
      socketRef.current.on('risk_alert', (alertData) => {
        addAlert({
          type: 'warning',
          title: 'Risk Alert',
          message: alertData.message,
        });
      });

      // AI Chat messages
      socketRef.current.on('ai_response', (message) => {
        addChatMessage({
          type: 'assistant',
          content: message.content,
        });
      });

      // System status
      socketRef.current.on('broker_status', (brokerData) => {
        addNotification({
          type: 'info',
          message: `Broker ${brokerData.name} ${brokerData.status}`,
        });
      });
    };

    connectSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [url, reconnectAttempts, reconnectInterval]);

  const emit = (event: string, data?: any) => {
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit(event, data);
    } else {
      addNotification({
        type: 'error',
        message: 'WebSocket not connected',
      });
    }
  };

  return {
    emit,
    connected: socketRef.current?.connected || false,
    socket: socketRef.current,
  };
};