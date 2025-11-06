import { useState, useEffect } from 'react';
import { Play, Pause, TrendingUp, Target, RefreshCw, Plus, AlertCircle, Settings } from 'lucide-react';
import { MatrixCard } from '../components/MatrixCard';
import { GlowingButton } from '../components/GlowingButton';
import { StatusBadge } from '../components/StatusBadge';
import { PriceDisplay } from '../components/PriceDisplay';
import { DataTable } from '../components/DataTable';
import { useStrategies } from '../hooks/useDatabase';
import toast from 'react-hot-toast';

export default function Strategies() {
  const [showCreateForm, setShowCreateForm] = useState(false);
  
  // Fetch real data from database
  const {
    data: strategies,
    loading: strategiesLoading,
    error: strategiesError,
    refresh: refreshStrategies,
    updateStrategy,
    deleteStrategy,
    createStrategy,
    backtestStrategy,
  } = useStrategies({ useCache: true, refreshInterval: 60000 }); // 1 minute

  const handleToggle = async (id: string, currentStatus: string) => {
    const newStatus = currentStatus === 'active' ? 'paused' : 'active';
    
    try {
      await updateStrategy(id, { status: newStatus });
      toast.success(`Strategy ${newStatus === 'active' ? 'activated' : 'paused'}`);
    } catch (error) {
      toast.error('Failed to update strategy status');
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (window.confirm(`Are you sure you want to delete strategy "${name}"?`)) {
      try {
        await deleteStrategy(id);
        toast.success('Strategy deleted successfully');
      } catch (error) {
        toast.error('Failed to delete strategy');
      }
    }
  };

  const hasErrors = strategiesError;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">STRATEGY MANAGEMENT</h1>
          <p className="mt-1 text-sm text-matrix-green/70">
            Monitor and control automated trading strategies
          </p>
          {hasErrors && (
            <p className="mt-1 text-xs text-red-400">
              ⚠️ Some data may be outdated. Check your connection.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <GlowingButton
            variant="secondary"
            size="sm"
            onClick={refreshStrategies}
            disabled={strategiesLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${strategiesLoading ? 'animate-spin' : ''}`} />
            REFRESH
          </GlowingButton>
          <GlowingButton
            icon={<Plus size={20} />}
            onClick={() => setShowCreateForm(true)}
          >
            NEW STRATEGY
          </GlowingButton>
        </div>
      </div>

      {/* Strategy Cards */}
      <div className="grid gap-4">
        {strategiesLoading ? (
          [...Array(3)].map((_, i) => (
            <MatrixCard key={i} glow className="animate-pulse">
              <div className="h-48 bg-matrix-dark-green/30 rounded"></div>
            </MatrixCard>
          ))
        ) : strategies.length > 0 ? (
          strategies.map((strategy) => (
            <MatrixCard key={strategy.id} glow>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="mb-3 flex items-center gap-3">
                    <h3 className="text-2xl font-bold matrix-glow-text">{strategy.name}</h3>
                    <StatusBadge
                      status={
                        strategy.status === 'active'
                          ? 'success'
                          : strategy.status === 'paused'
                          ? 'warning'
                          : 'error'
                      }
                    />
                  </div>

                  <p className="mb-4 text-sm text-matrix-green/70">{strategy.description}</p>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                    <div>
                      <p className="text-xs text-matrix-green/70">TOTAL P&L</p>
                      <PriceDisplay value={strategy.totalPnL} showSign size="md" />
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">DAILY P&L</p>
                      <PriceDisplay value={strategy.dailyPnL} showSign size="md" />
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">TOTAL TRADES</p>
                      <p className="text-xl font-bold">{strategy.totalTrades}</p>
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">WIN RATE</p>
                      <p className="text-xl font-bold text-matrix-green">
                        {strategy.winRate.toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">SHARPE RATIO</p>
                      <p className="text-xl font-bold">
                        {strategy.sharpeRatio?.toFixed(2) || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">MAX DRAWDOWN</p>
                      <p className="text-xl font-bold text-red-500">
                        {strategy.maxDrawdown?.toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">BROKER</p>
                      <p className="text-xl font-bold">{strategy.broker.toUpperCase()}</p>
                    </div>
                    <div>
                      <p className="text-xs text-matrix-green/70">SYMBOLS</p>
                      <p className="text-sm">{strategy.symbols.join(', ')}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <GlowingButton
                  size="sm"
                  variant={strategy.status === 'active' ? 'danger' : 'primary'}
                  onClick={() => handleToggle(strategy.id, strategy.status)}
                  icon={strategy.status === 'active' ? <Pause size={16} /> : <Play size={16} />}
                >
                  {strategy.status === 'active' ? 'PAUSE' : 'START'}
                </GlowingButton>
                <GlowingButton size="sm" variant="secondary" icon={<Settings size={16} />}>
                  CONFIGURE
                </GlowingButton>
                <GlowingButton
                  size="sm"
                  variant="danger"
                  onClick={() => handleDelete(strategy.id, strategy.name)}
                >
                  DELETE
                </GlowingButton>
              </div>
            </MatrixCard>
          ))
        ) : (
          <MatrixCard title="NO STRATEGIES FOUND">
            <div className="text-center py-8 text-matrix-green/70">
              <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No trading strategies configured</p>
              <p className="text-sm mt-2">Create your first strategy to get started</p>
              <GlowingButton
                className="mt-4"
                icon={<Plus size={20} />}
                onClick={() => setShowCreateForm(true)}
              >
                CREATE STRATEGY
              </GlowingButton>
            </div>
          </MatrixCard>
        )}
        {strategiesError && (
          <MatrixCard title="ERROR">
            <div className="text-center py-8 text-red-400">
              <AlertCircle className="h-12 w-12 mx-auto mb-4" />
              <p>Failed to load strategies</p>
              <p className="text-sm mt-2">{strategiesError}</p>
              <GlowingButton
                className="mt-4"
                variant="secondary"
                onClick={refreshStrategies}
              >
                RETRY
              </GlowingButton>
            </div>
          </MatrixCard>
        )}
      </div>

      {/* Performance Summary */}
      <MatrixCard title="PERFORMANCE SUMMARY" glow>
        {strategiesLoading ? (
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-12 bg-matrix-dark-green/30 rounded animate-pulse"></div>
            ))}
          </div>
        ) : strategies.length > 0 ? (
          <DataTable
            data={strategies}
            keyField="id"
            columns={[
              { key: 'name', header: 'STRATEGY', className: 'font-bold' },
              { key: 'type', header: 'TYPE', render: (s) => s.type.replace('_', ' ').toUpperCase() },
              {
                key: 'status',
                header: 'STATUS',
                render: (s) => (
                  <StatusBadge
                    status={
                      s.status === 'active' ? 'success' : s.status === 'paused' ? 'warning' : 'error'
                    }
                  />
                ),
              },
              {
                key: 'totalPnL',
                header: 'TOTAL P&L',
                render: (s) => <PriceDisplay value={s.totalPnL} showSign size="sm" />,
              },
              { key: 'totalTrades', header: 'TRADES' },
              { key: 'winRate', header: 'WIN RATE', render: (s) => `${s.winRate.toFixed(1)}%` },
              {
                key: 'sharpeRatio',
                header: 'SHARPE',
                render: (s) => s.sharpeRatio?.toFixed(2) || '-',
              },
            ]}
            emptyMessage="NO STRATEGIES CONFIGURED"
          />
        ) : (
          <div className="text-center py-8 text-matrix-green/70">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No strategies to display</p>
          </div>
        )}
      </MatrixCard>
    </div>
  );
}
