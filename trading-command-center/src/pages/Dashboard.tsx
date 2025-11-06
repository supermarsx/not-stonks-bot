import { useEffect, useState } from 'react';
import { DollarSign, TrendingUp, Activity, AlertTriangle, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { MatrixCard } from '../components/MatrixCard';
import { StatCard } from '../components/StatCard';
import { DataTable } from '../components/DataTable';
import { PriceDisplay } from '../components/PriceDisplay';
import { StatusBadge } from '../components/StatusBadge';
import { usePortfolio, usePositions, useRiskMetrics, useTradesPaginated, useConnectionStatus } from '../hooks/useDatabase';
import { dbService } from '../services/database';

export default function Dashboard() {
  const { isConnected, isOnline } = useConnectionStatus();
  const [refreshing, setRefreshing] = useState(false);

  // Fetch real data from database
  const {
    data: portfolio,
    loading: portfolioLoading,
    error: portfolioError,
    refresh: refreshPortfolio,
  } = usePortfolio({ useCache: true, refreshInterval: 30000, enableRealtime: true });

  const {
    data: positions,
    loading: positionsLoading,
    error: positionsError,
    refresh: refreshPositions,
  } = usePositions(undefined, { useCache: true, refreshInterval: 15000, enableRealtime: true });

  const {
    data: riskMetrics,
    loading: riskLoading,
    error: riskError,
    refresh: refreshRisk,
  } = useRiskMetrics({ useCache: true, refreshInterval: 30000, enableRealtime: true });

  const {
    data: tradesData,
    loading: tradesLoading,
    error: tradesError,
    refresh: refreshTrades,
  } = useTradesPaginated(1, 10, { useCache: false, refreshInterval: 5000 });

  const handleRefreshAll = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        refreshPortfolio(),
        refreshPositions(),
        refreshRisk(),
        refreshTrades(),
      ]);
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  // Loading states
  const isLoading = portfolioLoading || positionsLoading || tradesLoading;

  // Error handling
  const hasErrors = portfolioError || positionsError || tradesError || riskError;

  // Use cached demo data as fallback if no real data and offline
  const displayPortfolio = portfolio || null;
  const displayPositions = positions || [];
  const displayRiskMetrics = riskMetrics;
  const recentTrades = tradesData?.items || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">COMMAND CENTER</h1>
          <p className="mt-1 text-sm text-matrix-green/70">
            Real-time trading orchestration and monitoring
          </p>
          {hasErrors && (
            <p className="mt-1 text-xs text-red-400">
              ⚠️ Some data may be outdated. Check your connection.
            </p>
          )}
        </div>
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            {isOnline ? (
              <Wifi className="h-4 w-4 text-matrix-green" />
            ) : (
              <WifiOff className="h-4 w-4 text-red-500" />
            )}
            <div className="flex items-center gap-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  isConnected && isOnline ? 'bg-matrix-green' : 'bg-red-500'
                } animate-pulse`}
              />
              <span className="text-sm">
                {isConnected && isOnline ? 'LIVE' : isOnline ? 'RECONNECTING' : 'OFFLINE'}
              </span>
            </div>
          </div>
          
          {/* Refresh Button */}
          <button
            onClick={handleRefreshAll}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-1 bg-matrix-dark-green hover:bg-matrix-green/20 border border-matrix-green/30 rounded text-sm transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            REFRESH
          </button>
        </div>
      </div>

      {/* Portfolio Stats Grid */}
      {portfolioLoading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <MatrixCard key={i} className="animate-pulse">
              <div className="h-16 bg-matrix-dark-green/50 rounded"></div>
            </MatrixCard>
          ))}
        </div>
      ) : displayPortfolio ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Value"
            value={`$${displayPortfolio.totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            change={displayPortfolio.dailyPnL}
            changePercent={displayPortfolio.dailyPnLPercent}
            icon={<DollarSign size={32} />}
            loading={portfolioLoading}
          />
          <StatCard
            label="Daily P&L"
            value={`$${displayPortfolio.dailyPnL.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            changePercent={displayPortfolio.dailyPnLPercent}
            icon={<TrendingUp size={32} />}
            loading={portfolioLoading}
          />
          <StatCard
            label="Cash Balance"
            value={`$${displayPortfolio.cashBalance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            icon={<Activity size={32} />}
            loading={portfolioLoading}
          />
          <StatCard
            label="Buying Power"
            value={`$${displayPortfolio.buyingPower.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            icon={<DollarSign size={32} />}
            loading={portfolioLoading}
          />
        </div>
      ) : (
        <MatrixCard title="PORTFOLIO UNAVAILABLE">
          <div className="text-center py-8 text-red-400">
            <AlertTriangle className="h-12 w-12 mx-auto mb-4" />
            <p>Unable to load portfolio data</p>
            <p className="text-sm mt-2">{portfolioError || 'Connection error'}</p>
          </div>
        </MatrixCard>
      )}

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Positions Table - 2 columns */}
        <div className="lg:col-span-2">
          <MatrixCard title="ACTIVE POSITIONS" glow>
            {positionsLoading ? (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-12 bg-matrix-dark-green/30 rounded animate-pulse"></div>
                ))}
              </div>
            ) : displayPositions.length > 0 ? (
              <DataTable
                data={displayPositions}
                keyField="id"
                columns={[
                  { key: 'symbol', header: 'SYMBOL', className: 'font-bold' },
                  { key: 'broker', header: 'BROKER', render: (p) => p.broker.toUpperCase() },
                  {
                    key: 'side',
                    header: 'SIDE',
                    render: (p) => (
                      <StatusBadge status={p.side === 'LONG' ? 'success' : 'warning'} />
                    ),
                  },
                  { key: 'quantity', header: 'QTY' },
                  {
                    key: 'avgPrice',
                    header: 'AVG PRICE',
                    render: (p) => `$${p.avgPrice.toFixed(2)}`,
                  },
                  {
                    key: 'currentPrice',
                    header: 'CURRENT',
                    render: (p) => `$${p.currentPrice.toFixed(2)}`,
                  },
                  {
                    key: 'unrealizedPnL',
                    header: 'UNREALIZED P&L',
                    render: (p) => (
                      <PriceDisplay
                        value={Math.abs(p.unrealizedPnL)}
                        changePercent={p.unrealizedPnLPercent}
                        showSign
                        size="sm"
                      />
                    ),
                  },
                ]}
              />
            ) : (
              <div className="text-center py-8 text-matrix-green/70">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No active positions</p>
                <p className="text-sm mt-2">Positions will appear here when trades are executed</p>
              </div>
            )}
            {positionsError && (
              <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-sm">
                Error loading positions: {positionsError}
              </div>
            )}
          </MatrixCard>
        </div>

        {/* Risk Metrics - 1 column */}
        <div>
          <MatrixCard title="RISK METRICS" glow>
            {riskLoading ? (
              <div className="space-y-4">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-6 bg-matrix-dark-green/30 rounded animate-pulse"></div>
                ))}
              </div>
            ) : displayRiskMetrics ? (
              <div className="space-y-4">
                <RiskMetricRow
                  label="Current Drawdown"
                  value={`${displayRiskMetrics.currentDrawdown.toFixed(2)}%`}
                  status={displayRiskMetrics.currentDrawdown < 3 ? 'success' : 'warning'}
                />
                <RiskMetricRow
                  label="Max Drawdown"
                  value={`${displayRiskMetrics.maxDrawdown.toFixed(2)}%`}
                  status={displayRiskMetrics.maxDrawdown < 5 ? 'success' : 'warning'}
                />
                <RiskMetricRow
                  label="Sharpe Ratio"
                  value={displayRiskMetrics.sharpeRatio.toFixed(2)}
                  status={displayRiskMetrics.sharpeRatio > 1.5 ? 'success' : 'warning'}
                />
                <RiskMetricRow
                  label="Win Rate"
                  value={`${displayRiskMetrics.winRate.toFixed(1)}%`}
                  status={displayRiskMetrics.winRate > 60 ? 'success' : 'warning'}
                />
                <RiskMetricRow
                  label="Profit Factor"
                  value={displayRiskMetrics.profitFactor.toFixed(2)}
                  status={displayRiskMetrics.profitFactor > 1.5 ? 'success' : 'warning'}
                />
                <RiskMetricRow
                  label="Daily VaR"
                  value={`$${displayRiskMetrics.dailyVaR.toLocaleString()}`}
                  status="info"
                />
                <div className="mt-4 pt-4 border-t border-matrix-dark-green">
                  <p className="text-xs text-matrix-green/50">
                    Last updated: {new Date(displayRiskMetrics.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-matrix-green/70">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Risk metrics unavailable</p>
              </div>
            )}
            {riskError && (
              <div className="mt-4 p-2 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-xs">
                {riskError}
              </div>
            )}
          </MatrixCard>
        </div>
      </div>

      {/* Recent Trades */}
      <MatrixCard title="RECENT TRADES" glow>
        {tradesLoading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-matrix-dark-green/30 rounded animate-pulse"></div>
            ))}
          </div>
        ) : recentTrades.length > 0 ? (
          <DataTable
            data={recentTrades}
            keyField="id"
            columns={[
              {
                key: 'timestamp',
                header: 'TIME',
                render: (t) => new Date(t.timestamp).toLocaleTimeString(),
              },
              { key: 'symbol', header: 'SYMBOL', className: 'font-bold' },
              {
                key: 'side',
                header: 'SIDE',
                render: (t) => (
                  <span className={t.side === 'BUY' ? 'text-matrix-green' : 'text-red-500'}>
                    {t.side}
                  </span>
                ),
              },
              { key: 'quantity', header: 'QUANTITY' },
              {
                key: 'price',
                header: 'PRICE',
                render: (t) => `$${t.price.toFixed(2)}`,
              },
              { key: 'broker', header: 'BROKER', render: (t) => t.broker.toUpperCase() },
              {
                key: 'pnl',
                header: 'P&L',
                render: (t) =>
                  t.pnl !== undefined ? (
                    <PriceDisplay value={t.pnl} showSign size="sm" />
                  ) : (
                    '-'
                  ),
              },
            ]}
            emptyMessage="NO TRADES EXECUTED YET"
          />
        ) : (
          <div className="text-center py-8 text-matrix-green/70">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No trades executed yet</p>
            <p className="text-sm mt-2">Trade history will appear here</p>
          </div>
        )}
        {tradesError && (
          <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-sm">
            Error loading trades: {tradesError}
          </div>
        )}
      </MatrixCard>
    </div>
  );
}

function RiskMetricRow({
  label,
  value,
  status,
}: {
  label: string;
  value: string;
  status: 'success' | 'warning' | 'info';
}) {
  const statusColor = {
    success: 'text-matrix-green',
    warning: 'text-yellow-500',
    info: 'text-blue-400',
  };

  return (
    <div className="flex items-center justify-between border-b border-matrix-dark-green pb-2">
      <span className="text-sm text-matrix-green/70">{label}</span>
      <span className={`font-bold ${statusColor[status]}`}>{value}</span>
    </div>
  );
}
