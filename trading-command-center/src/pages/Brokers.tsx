import { useEffect } from 'react';
import { CheckCircle, XCircle, RefreshCw, DollarSign, Activity } from 'lucide-react';
import { MatrixCard } from '../components/MatrixCard';
import { GlowingButton } from '../components/GlowingButton';
import { StatCard } from '../components/StatCard';
import { StatusBadge } from '../components/StatusBadge';
import { demoBrokers } from '../utils/demoData';
import toast from 'react-hot-toast';

export default function Brokers() {
  const brokers = demoBrokers;

  const handleSync = (brokerName: string) => {
    toast.success(`Syncing ${brokerName}...`);
  };

  const handleReconnect = (brokerName: string) => {
    toast.success(`Reconnecting to ${brokerName}...`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold matrix-glow-text">BROKER CONNECTIONS</h1>
        <p className="mt-1 text-sm text-matrix-green/70">
          Manage connections to trading platforms
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          label="Total Brokers"
          value={brokers.length.toString()}
          icon={<Activity size={32} />}
        />
        <StatCard
          label="Connected"
          value={brokers.filter((b) => b.status === 'connected').length.toString()}
          icon={<CheckCircle size={32} />}
        />
        <StatCard
          label="Total Balance"
          value={`$${brokers.reduce((sum, b) => sum + b.balance, 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          icon={<DollarSign size={32} />}
        />
      </div>

      {/* Broker Cards */}
      <div className="grid gap-4 lg:grid-cols-2">
        {brokers.map((broker) => (
          <MatrixCard key={broker.id} glow>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="mb-3 flex items-center gap-3">
                  <h3 className="text-2xl font-bold matrix-glow-text">
                    {broker.displayName}
                  </h3>
                  {broker.status === 'connected' ? (
                    <CheckCircle className="text-matrix-green" size={24} />
                  ) : (
                    <XCircle className="text-red-500" size={24} />
                  )}
                </div>

                <div className="mb-4 flex gap-2">
                  <StatusBadge
                    status={broker.status === 'connected' ? 'success' : 'error'}
                  />
                  <StatusBadge
                    status={broker.accountType === 'live' ? 'warning' : 'info'}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-matrix-green/70">BALANCE</p>
                    <p className="text-xl font-bold">
                      ${broker.balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-matrix-green/70">POSITIONS</p>
                    <p className="text-xl font-bold">{broker.positionsCount}</p>
                  </div>
                  <div>
                    <p className="text-xs text-matrix-green/70">ORDERS</p>
                    <p className="text-xl font-bold">{broker.ordersCount}</p>
                  </div>
                  <div>
                    <p className="text-xs text-matrix-green/70">LAST SYNC</p>
                    <p className="text-sm">
                      {new Date(broker.lastSync).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 flex gap-2">
              <GlowingButton
                size="sm"
                onClick={() => handleSync(broker.displayName)}
                icon={<RefreshCw size={16} />}
              >
                SYNC
              </GlowingButton>
              {broker.status !== 'connected' && (
                <GlowingButton
                  size="sm"
                  variant="secondary"
                  onClick={() => handleReconnect(broker.displayName)}
                >
                  RECONNECT
                </GlowingButton>
              )}
            </div>
          </MatrixCard>
        ))}
      </div>

      {/* Connection Info */}
      <MatrixCard title="CONNECTION INFORMATION" glow>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between border-b border-matrix-dark-green pb-2">
            <span className="text-matrix-green/70">Binance API</span>
            <span>Crypto trading via REST + WebSocket</span>
          </div>
          <div className="flex justify-between border-b border-matrix-dark-green pb-2">
            <span className="text-matrix-green/70">Interactive Brokers</span>
            <span>Professional trading via TWS API</span>
          </div>
          <div className="flex justify-between border-b border-matrix-dark-green pb-2">
            <span className="text-matrix-green/70">Alpaca Markets</span>
            <span>US equities + crypto, commission-free</span>
          </div>
        </div>
      </MatrixCard>
    </div>
  );
}
