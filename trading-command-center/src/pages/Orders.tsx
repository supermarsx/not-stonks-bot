import { useState, useEffect } from 'react';
import { Plus, X, RefreshCw, AlertCircle } from 'lucide-react';
import { MatrixCard } from '../components/MatrixCard';
import { GlowingButton } from '../components/GlowingButton';
import { DataTable } from '../components/DataTable';
import { StatusBadge } from '../components/StatusBadge';
import { useOrders, useBrokers, useDataExport } from '../hooks/useDatabase';
import toast from 'react-hot-toast';

export default function Orders() {
  const [showForm, setShowForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Fetch real data from database
  const {
    data: orders,
    loading: ordersLoading,
    error: ordersError,
    refresh: refreshOrders,
    createOrder,
    cancelOrder,
  } = useOrders(undefined, { useCache: false, refreshInterval: 10000, enableRealtime: true });

  const {
    data: brokers,
    loading: brokersLoading,
    error: brokersError,
  } = useBrokers({ useCache: true, refreshInterval: 300000 }); // 5 minutes

  const { exportData, loading: exportLoading } = useDataExport();

  const [formData, setFormData] = useState<{
    symbol: string;
    broker: string;
    type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
    side: 'BUY' | 'SELL';
    quantity: string;
    price: string;
    stopPrice: string;
    timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  }>({
    symbol: '',
    broker: 'alpaca',
    type: 'MARKET',
    side: 'BUY',
    quantity: '',
    price: '',
    stopPrice: '',
    timeInForce: 'GTC',
  });

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      await exportData('orders', format);
      toast.success(`Orders exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Export failed');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.symbol || !formData.quantity) {
      toast.error('Symbol and quantity are required');
      return;
    }

    if (formData.type === 'LIMIT' && !formData.price) {
      toast.error('Price is required for limit orders');
      return;
    }

    if (formData.type === 'STOP' && !formData.stopPrice) {
      toast.error('Stop price is required for stop orders');
      return;
    }

    setSubmitting(true);
    try {
      const orderRequest = {
        symbol: formData.symbol.toUpperCase(),
        broker: formData.broker,
        type: formData.type,
        side: formData.side,
        quantity: parseFloat(formData.quantity),
        ...(formData.price && { price: parseFloat(formData.price) }),
        ...(formData.stopPrice && { stopPrice: parseFloat(formData.stopPrice) }),
        timeInForce: formData.timeInForce,
      };

      await createOrder(orderRequest);
      
      setShowForm(false);
      setFormData({
        symbol: '',
        broker: 'alpaca',
        type: 'MARKET',
        side: 'BUY',
        quantity: '',
        price: '',
        stopPrice: '',
        timeInForce: 'GTC',
      });
      
      toast.success('Order submitted successfully!');
    } catch (error) {
      toast.error('Failed to submit order');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (orderId: string) => {
    try {
      await cancelOrder(orderId);
      toast.success(`Order ${orderId} cancelled`);
    } catch (error) {
      toast.error('Failed to cancel order');
    }
  };

  const hasErrors = ordersError || brokersError;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">ORDER MANAGEMENT</h1>
          <p className="mt-1 text-sm text-matrix-green/70">
            Create, modify, and monitor trading orders
          </p>
          {hasErrors && (
            <p className="mt-1 text-xs text-red-400">
              ⚠️ Some data may be outdated. Check your connection.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Export Dropdown */}
          <div className="relative group">
            <GlowingButton variant="secondary" size="sm">
              EXPORT
            </GlowingButton>
            <div className="absolute right-0 top-full mt-2 w-32 bg-matrix-black border border-matrix-green/30 rounded shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
              <button
                onClick={() => handleExport('csv')}
                disabled={exportLoading}
                className="w-full px-3 py-2 text-left text-sm text-matrix-green hover:bg-matrix-dark-green/50 transition-colors"
              >
                CSV
              </button>
              <button
                onClick={() => handleExport('json')}
                disabled={exportLoading}
                className="w-full px-3 py-2 text-left text-sm text-matrix-green hover:bg-matrix-dark-green/50 transition-colors"
              >
                JSON
              </button>
            </div>
          </div>
          
          <GlowingButton
            variant="secondary"
            size="sm"
            onClick={refreshOrders}
            disabled={ordersLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${ordersLoading ? 'animate-spin' : ''}`} />
            REFRESH
          </GlowingButton>
          
          <GlowingButton
            icon={showForm ? <X size={20} /> : <Plus size={20} />}
            onClick={() => setShowForm(!showForm)}
          >
            {showForm ? 'CLOSE' : 'NEW ORDER'}
          </GlowingButton>
        </div>
      </div>

      {/* Order Entry Form */}
      {showForm && (
        <MatrixCard title="PLACE NEW ORDER" glow>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {/* Symbol */}
              <div>
                <label className="mb-2 block text-sm text-matrix-green/70">SYMBOL</label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  placeholder="AAPL, BTC/USDT"
                  className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
                />
              </div>

              {/* Broker */}
              <div>
                <label className="mb-2 block text-sm text-matrix-green/70">BROKER</label>
                {brokersLoading ? (
                  <div className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green/50">
                    Loading brokers...
                  </div>
                ) : brokersError ? (
                  <div className="w-full border-2 border-red-500 bg-matrix-black px-3 py-2 text-red-400">
                    <AlertCircle className="h-4 w-4 inline mr-2" />
                    Failed to load brokers
                  </div>
                ) : (
                  <select
                    value={formData.broker}
                    onChange={(e) => setFormData({ ...formData, broker: e.target.value })}
                    className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
                  >
                    {brokers.map((broker) => (
                      <option key={broker.id} value={broker.id}>
                        {broker.displayName} ({broker.status})
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Order Type */}
              <div>
                <label className="mb-2 block text-sm text-matrix-green/70">TYPE</label>
                <select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value as any })}
                  className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green focus:outline-none focus:ring-2 focus:ring-matrix-green"
                >
                  <option value="MARKET">MARKET</option>
                  <option value="LIMIT">LIMIT</option>
                  <option value="STOP">STOP</option>
                  <option value="STOP_LIMIT">STOP LIMIT</option>
                </select>
              </div>

              {/* Side */}
              <div>
                <label className="mb-2 block text-sm text-matrix-green/70">SIDE</label>
                <div className="flex gap-2">
                  <GlowingButton
                    type="button"
                    variant={formData.side === 'BUY' ? 'primary' : 'secondary'}
                    onClick={() => setFormData({ ...formData, side: 'BUY' })}
                    className="flex-1"
                  >
                    BUY
                  </GlowingButton>
                  <GlowingButton
                    type="button"
                    variant={formData.side === 'SELL' ? 'danger' : 'secondary'}
                    onClick={() => setFormData({ ...formData, side: 'SELL' })}
                    className="flex-1"
                  >
                    SELL
                  </GlowingButton>
                </div>
              </div>

              {/* Quantity */}
              <div>
                <label className="mb-2 block text-sm text-matrix-green/70">QUANTITY</label>
                <input
                  type="number"
                  value={formData.quantity}
                  onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                  placeholder="100"
                  step="any"
                  className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
                />
              </div>

              {/* Price (for LIMIT orders) */}
              {(formData.type === 'LIMIT' || formData.type === 'STOP_LIMIT') && (
                <div>
                  <label className="mb-2 block text-sm text-matrix-green/70">LIMIT PRICE</label>
                  <input
                    type="number"
                    value={formData.price}
                    onChange={(e) => setFormData({ ...formData, price: e.target.value })}
                    placeholder="0.00"
                    step="0.01"
                    className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
                  />
                </div>
              )}

              {/* Stop Price (for STOP orders) */}
              {(formData.type === 'STOP' || formData.type === 'STOP_LIMIT') && (
                <div>
                  <label className="mb-2 block text-sm text-matrix-green/70">STOP PRICE</label>
                  <input
                    type="number"
                    value={formData.stopPrice}
                    onChange={(e) => setFormData({ ...formData, stopPrice: e.target.value })}
                    placeholder="0.00"
                    step="0.01"
                    className="w-full border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
                  />
                </div>
              )}
            </div>

            {/* Submit Button */}
            <div className="flex gap-4">
              <GlowingButton type="submit" className="flex-1" disabled={submitting}>
                {submitting ? 'SUBMITTING...' : 'SUBMIT ORDER'}
              </GlowingButton>
              <GlowingButton 
                type="button" 
                variant="secondary" 
                onClick={() => setShowForm(false)}
                disabled={submitting}
              >
                CANCEL
              </GlowingButton>
            </div>
          </form>
        </MatrixCard>
      )}

      {/* Orders Table */}
      <MatrixCard title="ACTIVE ORDERS" glow>
        {ordersLoading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-matrix-dark-green/30 rounded animate-pulse"></div>
            ))}
          </div>
        ) : orders.length > 0 ? (
          <>
            <DataTable
              data={orders}
              keyField="id"
              columns={[
                {
                  key: 'createdAt',
                  header: 'TIME',
                  render: (o) => new Date(o.createdAt).toLocaleTimeString(),
                },
                { key: 'symbol', header: 'SYMBOL', className: 'font-bold' },
                { key: 'broker', header: 'BROKER', render: (o) => o.broker.toUpperCase() },
                {
                  key: 'side',
                  header: 'SIDE',
                  render: (o) => (
                    <span className={o.side === 'BUY' ? 'text-matrix-green' : 'text-red-500'}>
                      {o.side}
                    </span>
                  ),
                },
                { key: 'type', header: 'TYPE' },
                { key: 'quantity', header: 'QTY' },
                {
                  key: 'price',
                  header: 'PRICE',
                  render: (o) => (o.price ? `$${o.price.toFixed(2)}` : 'MARKET'),
                },
                {
                  key: 'status',
                  header: 'STATUS',
                  render: (o) => {
                    const statusMap: Record<string, 'success' | 'warning' | 'error' | 'info'> = {
                      FILLED: 'success',
                      PARTIALLY_FILLED: 'warning',
                      ACTIVE: 'info',
                      PENDING: 'info',
                      CANCELLED: 'error',
                      REJECTED: 'error',
                    };
                    return <StatusBadge status={statusMap[o.status] || 'info'} />;
                  },
                },
                {
                  key: 'id',
                  header: 'ACTIONS',
                  render: (o) =>
                    o.status === 'ACTIVE' || o.status === 'PENDING' ? (
                      <GlowingButton
                        size="sm"
                        variant="danger"
                        onClick={() => handleCancel(o.id)}
                      >
                        CANCEL
                      </GlowingButton>
                    ) : (
                      '-'
                    ),
                },
              ]}
              emptyMessage="NO ORDERS FOUND"
            />
            {ordersError && (
              <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-sm">
                Error loading orders: {ordersError}
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8 text-matrix-green/70">
            <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No orders found</p>
            <p className="text-sm mt-2">Create your first order to get started</p>
          </div>
        )}
      </MatrixCard>
    </div>
  );
}
