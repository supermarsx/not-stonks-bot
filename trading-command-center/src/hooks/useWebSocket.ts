import { useEffect } from 'react';
import { wsService } from '../services/websocket';
import { useTradingStore } from '../store/tradingStore';
import type { Portfolio, Position, Order, Trade, MarketData, Alert } from '../types';

export function useWebSocket() {
  const { setWsConnected, setPortfolio, updatePosition, updateOrder } = useTradingStore();

  useEffect(() => {
    // Connect to WebSocket
    wsService.connect()
      .then(() => {
        setWsConnected(true);
      })
      .catch((error) => {
        console.error('WebSocket connection failed:', error);
        setWsConnected(false);
      });

    // Subscribe to portfolio updates
    const unsubPortfolio = wsService.subscribePortfolio((portfolio: Portfolio) => {
      setPortfolio(portfolio);
    });

    // Subscribe to position updates
    const unsubPositions = wsService.subscribePositions((positions: Position[]) => {
      positions.forEach((position) => updatePosition(position));
    });

    // Subscribe to order updates
    const unsubOrders = wsService.subscribeOrders((order: Order) => {
      updateOrder(order);
    });

    // Cleanup on unmount
    return () => {
      unsubPortfolio();
      unsubPositions();
      unsubOrders();
      wsService.disconnect();
      setWsConnected(false);
    };
  }, [setWsConnected, setPortfolio, updatePosition, updateOrder]);

  return {
    isConnected: wsService.isConnected(),
    subscribe: wsService.subscribe.bind(wsService),
    send: wsService.send.bind(wsService),
  };
}

// Specific hooks for different data types
export function useMarketDataSubscription(symbols: string[]) {
  const { setMarketData } = useTradingStore();

  useEffect(() => {
    if (symbols.length === 0) return;

    const unsubscribe = wsService.subscribeMarketData((data: MarketData) => {
      setMarketData(data.symbol, data);
    });

    // Request market data for symbols
    wsService.send({
      type: 'market_data' as const,
      data: { symbols, action: 'subscribe' },
    });

    return () => {
      // Unsubscribe from symbols
      wsService.send({
        type: 'market_data' as const,
        data: { symbols, action: 'unsubscribe' },
      });
      unsubscribe();
    };
  }, [symbols.join(','), setMarketData]);
}

export function useAlertsSubscription(callback: (alert: Alert) => void) {
  useEffect(() => {
    const unsubscribe = wsService.subscribeAlerts(callback);
    return unsubscribe;
  }, [callback]);
}
