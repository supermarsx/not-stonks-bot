import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useTradingStore, RiskMetrics } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  TrendingUp,
  Settings,
  Target,
  Activity,
  Zap,
  Eye,
  Bell
} from 'lucide-react';

export const RiskManagement: React.FC = () => {
  const { riskMetrics, updateRiskMetrics, portfolio } = useTradingStore();
  const [alertThresholds, setAlertThresholds] = useState({
    maxDrawdown: -5, // -5%
    var95: -5000, // -$5000
    riskUtilization: 80, // 80%
    positionSize: 10 // 10% max per position
  });
  const [showConfig, setShowConfig] = useState(false);

  // Simulate real-time risk updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Add some random variation to simulate live data
      const variance = 0.05; // 5% variance
      updateRiskMetrics({
        var95: riskMetrics.var95 * (1 + (Math.random() - 0.5) * variance),
        var99: riskMetrics.var99 * (1 + (Math.random() - 0.5) * variance),
        sharpeRatio: riskMetrics.sharpeRatio + (Math.random() - 0.5) * 0.1,
        maxDrawdown: riskMetrics.maxDrawdown + (Math.random() - 0.5) * 0.5,
        riskUtilization: Math.max(0, Math.min(100, riskMetrics.riskUtilization + (Math.random() - 0.5) * 5))
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [riskMetrics.var95, riskMetrics.var99, riskMetrics.sharpeRatio, riskMetrics.maxDrawdown, riskMetrics.riskUtilization, updateRiskMetrics]);

  const getRiskLevel = (metric: number, type: 'drawdown' | 'var' | 'utilization' | 'position') => {
    switch (type) {
      case 'drawdown':
        if (metric > -2) return { level: 'low', color: 'text-green-400', icon: TrendingUp };
        if (metric > -5) return { level: 'medium', color: 'text-yellow-400', icon: AlertTriangle };
        return { level: 'high', color: 'text-red-400', icon: TrendingDown };
      
      case 'var':
        if (metric > -2000) return { level: 'low', color: 'text-green-400', icon: Shield };
        if (metric > -4000) return { level: 'medium', color: 'text-yellow-400', icon: AlertTriangle };
        return { level: 'high', color: 'text-red-400', icon: TrendingDown };
      
      case 'utilization':
        if (metric < 50) return { level: 'low', color: 'text-green-400', icon: Target };
        if (metric < 80) return { level: 'medium', color: 'text-yellow-400', icon: AlertTriangle };
        return { level: 'high', color: 'text-red-400', icon: Activity };
      
      case 'position':
        if (metric < 5) return { level: 'low', color: 'text-green-400', icon: Shield };
        if (metric < 10) return { level: 'medium', color: 'text-yellow-400', icon: AlertTriangle };
        return { level: 'high', color: 'text-red-400', icon: TrendingDown };
      
      default:
        return { level: 'low', color: 'text-green-400', icon: Shield };
    }
  };

  const riskDistribution = {
    var95: Math.abs(riskMetrics.var95),
    var99: Math.abs(riskMetrics.var99),
    totalRisk: riskMetrics.totalRisk
  };

  // Calculate position concentration
  const totalValue = portfolio.totalValue;
  const positionConcentrations = portfolio.positions.map(pos => ({
    symbol: pos.symbol,
    concentration: (pos.size * pos.currentPrice) / totalValue * 100
  }));

  const maxConcentration = Math.max(...positionConcentrations.map(p => p.concentration), 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400">
            RISK MANAGEMENT
          </h1>
          <p className="text-green-600 mt-1">Monitor and control portfolio risk</p>
        </div>
        <MatrixButton
          onClick={() => setShowConfig(!showConfig)}
          className="flex items-center gap-2"
        >
          <Settings className="w-4 h-4" />
          Configure Alerts
        </MatrixButton>
      </div>

      {/* Risk Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MatrixCard 
          title="Value at Risk (95%)" 
          glow={getRiskLevel(riskMetrics.var95, 'var').level === 'high'}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className={`text-2xl font-bold ${getRiskLevel(riskMetrics.var95, 'var').color}`}>
                ${riskMetrics.var95.toFixed(0)}
              </div>
              <div className="text-sm text-green-600">1-day VaR</div>
            </div>
            <Shield className={`w-8 h-8 ${getRiskLevel(riskMetrics.var95, 'var').color}`} />
          </div>
        </MatrixCard>

        <MatrixCard 
          title="Max Drawdown" 
          glow={getRiskLevel(riskMetrics.maxDrawdown, 'drawdown').level === 'high'}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className={`text-2xl font-bold ${getRiskLevel(riskMetrics.maxDrawdown, 'drawdown').color}`}>
                {riskMetrics.maxDrawdown.toFixed(1)}%
              </div>
              <div className="text-sm text-green-600">Peak to trough</div>
            </div>
            <TrendingDown className={`w-8 h-8 ${getRiskLevel(riskMetrics.maxDrawdown, 'drawdown').color}`} />
          </div>
        </MatrixCard>

        <MatrixCard 
          title="Risk Utilization" 
          glow={getRiskLevel(riskMetrics.riskUtilization, 'utilization').level === 'high'}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className={`text-2xl font-bold ${getRiskLevel(riskMetrics.riskUtilization, 'utilization').color}`}>
                {riskMetrics.riskUtilization.toFixed(0)}%
              </div>
              <div className="text-sm text-green-600">Of allocated capital</div>
            </div>
            <Activity className={`w-8 h-8 ${getRiskLevel(riskMetrics.riskUtilization, 'utilization').color}`} />
          </div>
          {/* Progress bar */}
          <div className="mt-3">
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${
                  getRiskLevel(riskMetrics.riskUtilization, 'utilization').level === 'high' ? 'bg-red-500' :
                  getRiskLevel(riskMetrics.riskUtilization, 'utilization').level === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${riskMetrics.riskUtilization}%` }}
              />
            </div>
          </div>
        </MatrixCard>

        <MatrixCard 
          title="Sharpe Ratio" 
          glow={riskMetrics.sharpeRatio < 1}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className={`text-2xl font-bold ${riskMetrics.sharpeRatio > 1.5 ? 'text-green-400' : riskMetrics.sharpeRatio > 1 ? 'text-yellow-400' : 'text-red-400'}`}>
                {riskMetrics.sharpeRatio.toFixed(2)}
              </div>
              <div className="text-sm text-green-600">Risk-adjusted return</div>
            </div>
            <Target className={`w-8 h-8 ${riskMetrics.sharpeRatio > 1.5 ? 'text-green-400' : riskMetrics.sharpeRatio > 1 ? 'text-yellow-400' : 'text-red-400'}`} />
          </div>
        </MatrixCard>
      </div>

      {/* Risk Breakdown and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Breakdown */}
        <MatrixCard title="Risk Breakdown" glow>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-green-400">Systematic Risk</span>
              <span className="font-mono">${riskDistribution.var95.toFixed(0)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-green-400">Total Risk Capital</span>
              <span className="font-mono">${riskDistribution.totalRisk.toFixed(0)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-green-400">Available Capacity</span>
              <span className="font-mono text-yellow-400">
                ${(riskDistribution.totalRisk - riskDistribution.var95).toFixed(0)}
              </span>
            </div>
            
            {/* Risk distribution chart placeholder */}
            <div className="mt-4 p-4 bg-black/50 rounded border border-green-800/30">
              <div className="text-xs text-green-600 mb-2">RISK DISTRIBUTION</div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded"></div>
                  <span className="text-xs">VaR (95%): {(riskDistribution.var95 / riskDistribution.totalRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                  <span className="text-xs">Available: {((riskDistribution.totalRisk - riskDistribution.var95) / riskDistribution.totalRisk * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </MatrixCard>

        {/* Position Concentration */}
        <MatrixCard title="Position Concentration" glow>
          <div className="space-y-4">
            <div className="text-sm text-green-600">
              Max position size: {maxConcentration.toFixed(1)}% of portfolio
            </div>
            
            {positionConcentrations.length > 0 ? (
              <div className="space-y-2">
                {positionConcentrations
                  .sort((a, b) => b.concentration - a.concentration)
                  .slice(0, 5)
                  .map((pos, index) => (
                    <div key={pos.symbol} className="flex items-center justify-between">
                      <span className="text-sm font-mono">{pos.symbol}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-gray-800 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              pos.concentration > 10 ? 'bg-red-500' :
                              pos.concentration > 5 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${Math.min(pos.concentration * 4, 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs">{pos.concentration.toFixed(1)}%</span>
                      </div>
                    </div>
                  ))
                }
              </div>
            ) : (
              <div className="text-center text-green-600 py-8">
                No open positions
              </div>
            )}
          </div>
        </MatrixCard>
      </div>

      {/* Alert Configuration */}
      {showConfig && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
        >
          <MatrixCard title="Alert Configuration" glow>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm text-green-400 mb-2">Max Drawdown Alert (%)</label>
                <MatrixInput
                  type="number"
                  value={alertThresholds.maxDrawdown}
                  onChange={(e) => setAlertThresholds({
                    ...alertThresholds,
                    maxDrawdown: parseFloat(e.target.value)
                  })}
                  placeholder="-5"
                />
              </div>
              
              <div>
                <label className="block text-sm text-green-400 mb-2">VaR Alert ($)</label>
                <MatrixInput
                  type="number"
                  value={alertThresholds.var95}
                  onChange={(e) => setAlertThresholds({
                    ...alertThresholds,
                    var95: parseFloat(e.target.value)
                  })}
                  placeholder="-5000"
                />
              </div>
              
              <div>
                <label className="block text-sm text-green-400 mb-2">Risk Utilization Alert (%)</label>
                <MatrixInput
                  type="number"
                  value={alertThresholds.riskUtilization}
                  onChange={(e) => setAlertThresholds({
                    ...alertThresholds,
                    riskUtilization: parseFloat(e.target.value)
                  })}
                  placeholder="80"
                />
              </div>
              
              <div>
                <label className="block text-sm text-green-400 mb-2">Max Position Size (%)</label>
                <MatrixInput
                  type="number"
                  value={alertThresholds.positionSize}
                  onChange={(e) => setAlertThresholds({
                    ...alertThresholds,
                    positionSize: parseFloat(e.target.value)
                  })}
                  placeholder="10"
                />
              </div>
            </div>
            
            <div className="flex gap-2 mt-6">
              <MatrixButton>
                <Bell className="w-4 h-4 mr-2" />
                Save Configuration
              </MatrixButton>
              <MatrixButton variant="secondary" onClick={() => setShowConfig(false)}>
                Cancel
              </MatrixButton>
            </div>
          </MatrixCard>
        </motion.div>
      )}

      {/* Risk Metrics Timeline */}
      <MatrixCard title="Risk Metrics Timeline" glow>
        <div className="h-64 bg-black/50 rounded border border-green-800/30 flex items-center justify-center">
          <div className="text-center text-green-600">
            <Eye className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Risk metrics visualization</p>
            <p className="text-xs mt-1">Real-time risk monitoring dashboard</p>
          </div>
        </div>
      </MatrixCard>

      {/* Risk Controls */}
      <MatrixCard title="Emergency Risk Controls" glow>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MatrixButton variant="secondary" className="p-4 flex items-center justify-center gap-2">
            <Zap className="w-4 h-4" />
            Emergency Stop
          </MatrixButton>
          <MatrixButton variant="secondary" className="p-4 flex items-center justify-center gap-2">
            <Shield className="w-4 h-4" />
            Close All Positions
          </MatrixButton>
          <MatrixButton variant="secondary" className="p-4 flex items-center justify-center gap-2">
            <Activity className="w-4 h-4" />
            Reduce Risk 50%
          </MatrixButton>
        </div>
      </MatrixCard>
    </div>
  );
};