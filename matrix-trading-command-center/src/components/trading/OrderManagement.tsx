import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore, Order } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { MatrixTable } from '@/components/ui/MatrixTable';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  Plus, 
  Search, 
  Filter, 
  RefreshCw, 
  TrendingUp, 
  TrendingDown, 
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';

export const OrderManagement: React.FC = () => {
  const { portfolio, addOrder, updateOrder } = useTradingStore();
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchSymbol, setSearchSymbol] = useState('');

  const [orderForm, setOrderForm] = useState({
    symbol: '',
    side: 'buy' as 'buy' | 'sell',
    type: 'limit' as 'market' | 'limit' | 'stop',
    size: 0,
    price: 0,
    stopPrice: 0,
    broker: 'Alpaca'
  });

  const handleSubmitOrder = () => {
    if (!orderForm.symbol || !orderForm.size) return;

    const newOrder: Order = {
      id: Date.now().toString(),
      symbol: orderForm.symbol.toUpperCase(),
      side: orderForm.side,
      type: orderForm.type,
      size: orderForm.size,
      price: orderForm.type === 'limit' ? orderForm.price : undefined,
      stopPrice: orderForm.type === 'stop' ? orderForm.stopPrice : undefined,
      status: 'pending',
      timestamp: Date.now(),
      broker: orderForm.broker
    };

    addOrder(newOrder);
    
    // Reset form
    setOrderForm({
      symbol: '',
      side: 'buy',
      type: 'limit',
      size: 0,
      price: 0,
      stopPrice: 0,
      broker: 'Alpaca'
    });
    setShowOrderForm(false);

    // Simulate order processing
    setTimeout(() => {
      updateOrder(newOrder.id, { status: 'filled' });
    }, 3000);
  };

  const filteredOrders = portfolio.orders.filter(order => {
    const matchesStatus = filterStatus === 'all' || order.status === filterStatus;
    const matchesSymbol = order.symbol.toLowerCase().includes(searchSymbol.toLowerCase());
    return matchesStatus && matchesSymbol;
  });

  const getStatusIcon = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'filled':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'rejected':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return null;
    }
  };

  const orderStats = {
    total: portfolio.orders.length,
    pending: portfolio.orders.filter(o => o.status === 'pending').length,
    filled: portfolio.orders.filter(o => o.status === 'filled').length,
    cancelled: portfolio.orders.filter(o => o.status === 'cancelled').length
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400">
            ORDER MANAGEMENT
          </h1>
          <p className="text-green-600 mt-1">Execute and monitor trading orders</p>
        </div>
        <MatrixButton
          onClick={() => setShowOrderForm(true)}
          className="flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Order
        </MatrixButton>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MatrixCard title="Total Orders" glow>
          <div className="text-2xl font-bold matrix-text-glow">{orderStats.total}</div>
        </MatrixCard>
        <MatrixCard title="Pending" glow>
          <div className="text-2xl font-bold text-yellow-400">{orderStats.pending}</div>
        </MatrixCard>
        <MatrixCard title="Filled" glow>
          <div className="text-2xl font-bold text-green-400">{orderStats.filled}</div>
        </MatrixCard>
        <MatrixCard title="Cancelled" glow>
          <div className="text-2xl font-bold text-red-400">{orderStats.cancelled}</div>
        </MatrixCard>
      </div>

      {/* Filters and Search */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-green-400" />
          <MatrixInput
            placeholder="Search symbols..."
            value={searchSymbol}
            onChange={(e) => setSearchSymbol(e.target.value)}
            className="w-48"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-green-400" />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="matrix-input px-3 py-1 rounded"
          >
            <option value="all">All Orders</option>
            <option value="pending">Pending</option>
            <option value="filled">Filled</option>
            <option value="cancelled">Cancelled</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>

        <MatrixButton variant="secondary" size="sm">
          <RefreshCw className="w-4 h-4" />
        </MatrixButton>
      </div>

      {/* Orders Table */}
      <MatrixCard title="Active Orders" glow>
        <div className="overflow-x-auto">
          <table className="matrix-table">
            <thead>
              <tr>
                <th>Order ID</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Type</th>
                <th>Size</th>
                <th>Price</th>
                <th>Status</th>
                <th>Broker</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {filteredOrders.map((order) => (
                  <motion.tr
                    key={order.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="hover:bg-green-900/10"
                  >
                    <td className="font-mono text-xs">{order.id.slice(-8)}</td>
                    <td className="font-bold">{order.symbol}</td>
                    <td className={`flex items-center gap-1 ${
                      order.side === 'buy' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {order.side === 'buy' ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      {order.side.toUpperCase()}
                    </td>
                    <td className="font-mono">{order.type.toUpperCase()}</td>
                    <td className="font-mono">{order.size}</td>
                    <td className="font-mono">
                      {order.price ? `$${order.price.toFixed(2)}` : 'Market'}
                    </td>
                    <td className="flex items-center gap-2">
                      {getStatusIcon(order.status)}
                      <span className={`capitalize ${
                        order.status === 'filled' ? 'text-green-400' :
                        order.status === 'pending' ? 'text-yellow-400' :
                        order.status === 'cancelled' ? 'text-red-400' : 'text-red-400'
                      }`}>
                        {order.status}
                      </span>
                    </td>
                    <td className="font-mono text-xs">{order.broker}</td>
                    <td className="font-mono text-xs">
                      {new Date(order.timestamp).toLocaleTimeString()}
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      </MatrixCard>

      {/* Order Form Modal */}
      <AnimatePresence>
        {showOrderForm && (
          <motion.div
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="matrix-card p-6 w-full max-w-md mx-4"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
            >
              <h2 className="text-xl font-bold matrix-text-glow mb-4">
                New Order
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-green-400 mb-1">Symbol</label>
                  <MatrixInput
                    placeholder="AAPL"
                    value={orderForm.symbol}
                    onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value })}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-green-400 mb-1">Side</label>
                    <select
                      value={orderForm.side}
                      onChange={(e) => setOrderForm({ ...orderForm, side: e.target.value as 'buy' | 'sell' })}
                      className="matrix-input w-full px-3 py-2 rounded"
                    >
                      <option value="buy">BUY</option>
                      <option value="sell">SELL</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-green-400 mb-1">Type</label>
                    <select
                      value={orderForm.type}
                      onChange={(e) => setOrderForm({ ...orderForm, type: e.target.value as any })}
                      className="matrix-input w-full px-3 py-2 rounded"
                    >
                      <option value="market">MARKET</option>
                      <option value="limit">LIMIT</option>
                      <option value="stop">STOP</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-green-400 mb-1">Size</label>
                  <MatrixInput
                    type="number"
                    placeholder="100"
                    value={orderForm.size || ''}
                    onChange={(e) => setOrderForm({ ...orderForm, size: parseFloat(e.target.value) || 0 })}
                  />
                </div>

                {orderForm.type === 'limit' && (
                  <div>
                    <label className="block text-sm text-green-400 mb-1">Price</label>
                    <MatrixInput
                      type="number"
                      placeholder="150.00"
                      value={orderForm.price || ''}
                      onChange={(e) => setOrderForm({ ...orderForm, price: parseFloat(e.target.value) || 0 })}
                    />
                  </div>
                )}

                {orderForm.type === 'stop' && (
                  <div>
                    <label className="block text-sm text-green-400 mb-1">Stop Price</label>
                    <MatrixInput
                      type="number"
                      placeholder="145.00"
                      value={orderForm.stopPrice || ''}
                      onChange={(e) => setOrderForm({ ...orderForm, stopPrice: parseFloat(e.target.value) || 0 })}
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm text-green-400 mb-1">Broker</label>
                  <select
                    value={orderForm.broker}
                    onChange={(e) => setOrderForm({ ...orderForm, broker: e.target.value })}
                    className="matrix-input w-full px-3 py-2 rounded"
                  >
                    <option value="Alpaca">Alpaca</option>
                    <option value="Binance">Binance</option>
                    <option value="Interactive Brokers">Interactive Brokers</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <MatrixButton
                  onClick={handleSubmitOrder}
                  className="flex-1"
                  disabled={!orderForm.symbol || !orderForm.size}
                >
                  Submit Order
                </MatrixButton>
                <MatrixButton
                  variant="secondary"
                  onClick={() => setShowOrderForm(false)}
                >
                  Cancel
                </MatrixButton>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};