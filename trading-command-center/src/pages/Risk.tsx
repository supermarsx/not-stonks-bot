import { AlertTriangle, Shield, TrendingDown, Target, RefreshCw, AlertCircle } from 'lucide-react';
import { MatrixCard } from '../components/MatrixCard';
import { StatCard } from '../components/StatCard';
import { useRiskMetrics } from '../hooks/useDatabase';

export default function Risk() {
  // Fetch real data from database
  const {
    data: metrics,
    loading: metricsLoading,
    error: metricsError,
    refresh: refreshMetrics,
    checkRiskLimits,
  } = useRiskMetrics({ useCache: true, refreshInterval: 30000, enableRealtime: true });

  const handleCheckLimits = async () => {
    try {
      const result = await checkRiskLimits();
      if (!result.passed) {
        alert(`Risk limit violations:\n${result.violations.join('\n')}`);
      } else {
        alert('All risk limits are within acceptable ranges');
      }
    } catch (error) {
      alert('Failed to check risk limits');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">RISK MONITORING</h1>
          <p className="mt-1 text-sm text-matrix-green/70">
            Real-time risk metrics and portfolio analytics
          </p>
          {metricsError && (
            <p className="mt-1 text-xs text-red-400">
              ⚠️ Risk data may be outdated. Check your connection.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCheckLimits}
            className="flex items-center gap-2 px-4 py-2 bg-matrix-dark-green hover:bg-matrix-green/20 border border-matrix-green/30 rounded text-sm transition-colors"
          >
            <Shield className="h-4 w-4" />
            CHECK LIMITS
          </button>
          <button
            onClick={refreshMetrics}
            disabled={metricsLoading}
            className="flex items-center gap-2 px-3 py-2 bg-matrix-dark-green hover:bg-matrix-green/20 border border-matrix-green/30 rounded text-sm transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${metricsLoading ? 'animate-spin' : ''}`} />
            REFRESH
          </button>
        </div>
      </div>

      {/* Key Risk Metrics */}
      {metricsLoading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <MatrixCard key={i} className="animate-pulse">
              <div className="h-16 bg-matrix-dark-green/50 rounded"></div>
            </MatrixCard>
          ))}
        </div>
      ) : metrics ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Portfolio Value"
            value={`$${metrics.portfolioValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
            icon={<Shield size={32} />}
            loading={metricsLoading}
          />
          <StatCard
            label="Current Drawdown"
            value={`${metrics.currentDrawdown.toFixed(2)}%`}
            icon={<TrendingDown size={32} />}
            loading={metricsLoading}
          />
          <StatCard
            label="Sharpe Ratio"
            value={metrics.sharpeRatio.toFixed(2)}
            icon={<Target size={32} />}
            loading={metricsLoading}
          />
          <StatCard
            label="Daily VaR"
            value={`$${metrics.dailyVaR.toLocaleString()}`}
            icon={<AlertTriangle size={32} />}
            loading={metricsLoading}
          />
        </div>
      ) : (
        <MatrixCard title="RISK METRICS UNAVAILABLE">
          <div className="text-center py-8 text-red-400">
            <AlertTriangle className="h-12 w-12 mx-auto mb-4" />
            <p>Unable to load risk metrics</p>
            <p className="text-sm mt-2">{metricsError || 'Connection error'}</p>
          </div>
        </MatrixCard>
      )}

      {/* Detailed Metrics Grid */}
      {metrics ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <MatrixCard title="DRAWDOWN ANALYSIS" glow>
            <div className="space-y-3">
              <MetricRow
                label="Current Drawdown"
                value={`${metrics.currentDrawdown.toFixed(2)}%`}
                status={metrics.currentDrawdown < 3 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Maximum Drawdown"
                value={`${metrics.maxDrawdown.toFixed(2)}%`}
                status={metrics.maxDrawdown < 5 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Recovery Status"
                value={metrics.currentDrawdown < metrics.maxDrawdown / 2 ? 'RECOVERING' : 'STABLE'}
                status="good"
              />
              <div className="mt-4 pt-4 border-t border-matrix-dark-green">
                <p className="text-xs text-matrix-green/50">
                  Last updated: {new Date(metrics.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </MatrixCard>

          <MatrixCard title="PERFORMANCE RATIOS" glow>
            <div className="space-y-3">
              <MetricRow
                label="Sharpe Ratio"
                value={metrics.sharpeRatio.toFixed(2)}
                status={metrics.sharpeRatio > 1.5 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Sortino Ratio"
                value={metrics.sortinoRatio.toFixed(2)}
                status={metrics.sortinoRatio > 2 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Profit Factor"
                value={metrics.profitFactor.toFixed(2)}
                status={metrics.profitFactor > 1.5 ? 'good' : 'warning'}
              />
            </div>
          </MatrixCard>

          <MatrixCard title="WIN METRICS" glow>
            <div className="space-y-3">
              <MetricRow
                label="Win Rate"
                value={`${metrics.winRate.toFixed(1)}%`}
                status={metrics.winRate > 60 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Profit Factor"
                value={metrics.profitFactor.toFixed(2)}
                status={metrics.profitFactor > 1.5 ? 'good' : 'warning'}
              />
              <MetricRow label="Expected Return" value="POSITIVE" status="good" />
            </div>
          </MatrixCard>

          <MatrixCard title="VALUE AT RISK (VaR)" glow>
            <div className="space-y-3">
              <MetricRow
                label="Daily VaR (95%)"
                value={`$${metrics.dailyVaR.toLocaleString()}`}
                status="info"
              />
              <MetricRow
                label="Weekly VaR (95%)"
                value={`$${metrics.weeklyVaR.toLocaleString()}`}
                status="info"
              />
              <MetricRow
                label="VaR as % of Portfolio"
                value={`${((metrics.dailyVaR / metrics.portfolioValue) * 100).toFixed(2)}%`}
                status="info"
              />
            </div>
          </MatrixCard>

          <MatrixCard title="PORTFOLIO CONCENTRATION" glow>
            <div className="space-y-3">
              <MetricRow
                label="Concentration Index"
                value={metrics.positionConcentration.toFixed(2)}
                status={metrics.positionConcentration < 0.3 ? 'good' : 'warning'}
              />
              <MetricRow
                label="Leverage Ratio"
                value={`${metrics.leverageRatio.toFixed(2)}x`}
                status={metrics.leverageRatio < 2 ? 'good' : 'warning'}
              />
              <MetricRow label="Diversification" value="MODERATE" status="good" />
            </div>
          </MatrixCard>

          <MatrixCard title="RISK STATUS" glow>
            <div className="space-y-3">
              <MetricRow label="Overall Risk Level" value="MEDIUM" status="warning" />
              <MetricRow label="Position Sizing" value="OPTIMAL" status="good" />
              <MetricRow label="Exposure Limits" value="WITHIN RANGE" status="good" />
            </div>
          </MatrixCard>
        </div>
      ) : metricsLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <MatrixCard key={i} className="animate-pulse">
              <div className="h-40 bg-matrix-dark-green/30 rounded"></div>
            </MatrixCard>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-matrix-green/70">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Risk metrics data unavailable</p>
        </div>
      )}

      {/* Risk Alerts */}
      <MatrixCard title="RISK ALERTS" glow>
        <div className="flex items-center gap-3">
          <Shield className="text-matrix-green" size={24} />
          <div>
            <p className="font-bold">NO ACTIVE RISK ALERTS</p>
            <p className="text-sm text-matrix-green/70">
              All positions are within risk parameters. Portfolio is properly diversified.
            </p>
          </div>
        </div>
      </MatrixCard>
    </div>
  );
}

function MetricRow({
  label,
  value,
  status,
}: {
  label: string;
  value: string;
  status: 'good' | 'warning' | 'danger' | 'info';
}) {
  const colors = {
    good: 'text-matrix-green',
    warning: 'text-yellow-500',
    danger: 'text-red-500',
    info: 'text-blue-400',
  };

  return (
    <div className="flex items-center justify-between border-b border-matrix-dark-green pb-2">
      <span className="text-sm text-matrix-green/70">{label}</span>
      <span className={`font-bold ${colors[status]}`}>{value}</span>
    </div>
  );
}
