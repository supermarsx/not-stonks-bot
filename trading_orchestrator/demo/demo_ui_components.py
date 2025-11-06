"""
Demo Mode UI Components - React components for demo mode interface
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .demo_mode_manager import get_demo_manager
from .demo_dashboard import get_demo_dashboard
from .virtual_portfolio import get_virtual_portfolio


class DemoModeUI:
    """
    UI component manager for demo mode
    
    Provides React component interfaces for:
    - Demo mode indicators
    - Virtual portfolio views
    - Demo trading interface
    - Configuration panels
    """
    
    def __init__(self):
        self.demo_manager = None
        self.dashboard = None
        self.portfolio = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize UI components"""
        try:
            self.demo_manager = await get_demo_manager()
            self.dashboard = await get_demo_dashboard()
            self.portfolio = await get_virtual_portfolio()
            self.is_initialized = True
        except Exception as e:
            print(f"Failed to initialize demo UI: {e}")
    
    async def get_demo_indicator_component(self) -> Dict[str, Any]:
        """Get demo mode indicator component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            status = await self.demo_manager.get_demo_status()
            
            return {
                "component": "DemoModeIndicator",
                "props": {
                    "isDemoActive": status.get("is_active", False),
                    "demoState": status.get("state", "disabled"),
                    "sessionId": status.get("session", {}).get("session_id"),
                    "environment": status.get("environment", "development"),
                    "demoBalance": status.get("config_summary", {}).get("demo_balance", 0),
                    "lastActivity": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoModeIndicator",
                "props": {
                    "isDemoActive": False,
                    "error": str(e)
                }
            }
    
    async def get_virtual_portfolio_component(self) -> Dict[str, Any]:
        """Get virtual portfolio view component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get portfolio data
            portfolio_value = await self.portfolio.get_portfolio_value()
            positions = await self.portfolio.get_positions_summary()
            metrics = await self.portfolio.get_portfolio_metrics()
            risk_metrics = await self.portfolio.get_risk_metrics()
            
            # Format positions for UI
            formatted_positions = []
            for pos in positions:
                if pos.quantity != 0:
                    formatted_positions.append({
                        "symbol": pos.symbol,
                        "side": "Long" if pos.quantity > 0 else "Short",
                        "quantity": abs(pos.quantity),
                        "avgPrice": pos.avg_entry_price,
                        "currentPrice": pos.current_price,
                        "marketValue": pos.market_value,
                        "unrealizedPnl": pos.unrealized_pnl,
                        "pnlPercentage": pos.pnl_percentage,
                        "weight": pos.weight,
                        "commission": pos.commission_paid
                    })
            
            return {
                "component": "VirtualPortfolio",
                "props": {
                    "totalValue": portfolio_value,
                    "cashBalance": metrics.get("cash_balance", 0),
                    "investedValue": metrics.get("invested_value", 0),
                    "totalReturn": metrics.get("total_return_pct", 0),
                    "dailyChange": 0,  # Would calculate from history
                    "positions": formatted_positions,
                    "performanceMetrics": {
                        "sharpeRatio": metrics.get("sharpe_ratio", 0),
                        "maxDrawdown": metrics.get("max_drawdown_pct", 0),
                        "winRate": metrics.get("win_rate", 0),
                        "totalTrades": metrics.get("trades_count", 0)
                    },
                    "riskMetrics": {
                        "var95": risk_metrics.var_95 * 100,
                        "volatility": risk_metrics.volatility * 100,
                        "sortinoRatio": risk_metrics.sortino_ratio
                    },
                    "lastUpdated": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "VirtualPortfolio",
                "props": {
                    "error": str(e),
                    "positions": [],
                    "performanceMetrics": {},
                    "riskMetrics": {}
                }
            }
    
    async def get_demo_trading_interface(self) -> Dict[str, Any]:
        """Get demo trading interface component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get available symbols and market data
            available_symbols = [
                {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
                {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE Arca"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"}
            ]
            
            # Get recent trades
            recent_trades = await self.portfolio.get_trade_history(10)
            
            # Format recent trades for UI
            formatted_trades = []
            for trade in recent_trades[-10:]:
                formatted_trades.append({
                    "id": trade.get("trade_id", ""),
                    "symbol": trade.get("symbol", ""),
                    "side": trade.get("side", ""),
                    "quantity": trade.get("quantity", 0),
                    "price": trade.get("price", 0),
                    "commission": trade.get("commission", 0),
                    "timestamp": trade.get("timestamp", datetime.now()).isoformat() if trade.get("timestamp") else datetime.now().isoformat(),
                    "status": "Filled"  # Demo trades are always filled
                })
            
            return {
                "component": "DemoTradingInterface",
                "props": {
                    "isDemoActive": self.demo_manager.is_demo_mode_active(),
                    "availableSymbols": available_symbols,
                    "orderTypes": ["Market", "Limit", "Stop", "Stop Limit"],
                    "orderSides": ["Buy", "Sell"],
                    "timeInForce": ["Day", "GTC", "IOC", "FOK"],
                    "recentTrades": formatted_trades,
                    "demoSettings": {
                        "commissionRate": self.demo_manager.config.commission_rate,
                        "slippageRate": self.demo_manager.config.slippage_rate,
                        "maxPositionSize": self.demo_manager.config.max_risk_per_trade
                    },
                    "cashBalance": await self.portfolio.get_portfolio_value() - sum(p.market_value for p in await self.portfolio.get_positions_summary()),
                    "lastUpdated": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoTradingInterface",
                "props": {
                    "error": str(e),
                    "availableSymbols": [],
                    "recentTrades": []
                }
            }
    
    async def get_demo_configuration_panel(self) -> Dict[str, Any]:
        """Get demo configuration panel component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            config = self.demo_manager.config
            
            return {
                "component": "DemoConfigurationPanel",
                "props": {
                    "demoModeEnabled": config.enabled,
                    "environment": config.environment.value,
                    "demoBalance": config.demo_account_balance,
                    "maxRiskPerTrade": config.max_risk_per_trade,
                    "realisticSlippage": config.realistic_slippage,
                    "commissionRate": config.commission_rate,
                    "slippageRate": config.slippage_rate,
                    "marketImpactEnabled": config.market_impact_enabled,
                    "dataQuality": config.data_quality,
                    "volatilitySimulation": config.volatility_simulation,
                    "maxDrawdownLimit": config.max_drawdown_limit,
                    "dailyLossLimit": config.daily_loss_limit,
                    "positionSizeLimit": config.position_size_limit,
                    "detailedLogging": config.detailed_logging,
                    "performanceTracking": config.performance_tracking,
                    "alertsEnabled": config.alerts_enabled
                }
            }
        except Exception as e:
            return {
                "component": "DemoConfigurationPanel",
                "props": {
                    "error": str(e)
                }
            }
    
    async def get_demo_dashboard_widget(self) -> Dict[str, Any]:
        """Get demo dashboard widget component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            dashboard_data = await self.dashboard.get_dashboard_data()
            
            return {
                "component": "DemoDashboardWidget",
                "props": {
                    "portfolioValue": dashboard_data.portfolio.get("total_value", 0),
                    "dailyReturn": dashboard_data.portfolio.get("daily_change_pct", 0),
                    "positionsCount": dashboard_data.portfolio.get("positions_count", 0),
                    "unrealizedPnl": dashboard_data.portfolio.get("unrealized_pnl", 0),
                    "realizedPnl": dashboard_data.portfolio.get("realized_pnl", 0),
                    "totalReturn": dashboard_data.performance.get("total_return_pct", 0),
                    "sharpeRatio": dashboard_data.performance.get("sharpe_ratio", 0),
                    "maxDrawdown": dashboard_data.performance.get("max_drawdown_pct", 0),
                    "activeAlerts": len(dashboard_data.alerts),
                    "demoStatus": {
                        "isActive": dashboard_data.demo_mode_status.get("is_active", False),
                        "sessionId": dashboard_data.demo_mode_status.get("session", {}).get("session_id"),
                        "environment": dashboard_data.demo_mode_status.get("environment", "development")
                    },
                    "timestamp": dashboard_data.timestamp.isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoDashboardWidget",
                "props": {
                    "error": str(e)
                }
            }
    
    async def get_demo_alerts_component(self) -> Dict[str, Any]:
        """Get demo alerts component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get recent alerts from dashboard
            dashboard_data = await self.dashboard.get_dashboard_data()
            alerts = dashboard_data.alerts
            
            # Format alerts for UI
            formatted_alerts = []
            for alert in alerts[-10:]:  # Recent 10 alerts
                formatted_alerts.append({
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "metric": alert.metric,
                    "currentValue": alert.current_value,
                    "threshold": alert.threshold
                })
            
            return {
                "component": "DemoAlerts",
                "props": {
                    "alerts": formatted_alerts,
                    "totalAlerts": len(formatted_alerts),
                    "unacknowledgedCount": len([a for a in formatted_alerts if not a["acknowledged"]]),
                    "criticalAlerts": len([a for a in formatted_alerts if a["level"] == "critical"]),
                    "lastUpdated": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoAlerts",
                "props": {
                    "error": str(e),
                    "alerts": []
                }
            }
    
    async def get_demo_performance_chart(self) -> Dict[str, Any]:
        """Get demo performance chart component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get historical performance data
            historical_data = await self.dashboard.get_historical_data(30)  # 30 days
            
            # Format for chart
            chart_data = []
            for data in historical_data[-30:]:  # Last 30 data points
                chart_data.append({
                    "timestamp": data.timestamp.isoformat(),
                    "portfolioValue": data.portfolio.get("total_value", 0),
                    "dailyReturn": data.portfolio.get("daily_change_pct", 0),
                    "cumulativeReturn": data.performance.get("total_return_pct", 0),
                    "drawdown": data.performance.get("max_drawdown_pct", 0)
                })
            
            # Calculate chart statistics
            values = [point["portfolioValue"] for point in chart_data]
            returns = [point["cumulativeReturn"] for point in chart_data]
            
            chart_stats = {
                "maxValue": max(values) if values else 0,
                "minValue": min(values) if values else 0,
                "avgReturn": sum(returns) / len(returns) if returns else 0,
                "volatility": 0,  # Would calculate properly
                "totalReturn": returns[-1] if returns else 0
            }
            
            return {
                "component": "DemoPerformanceChart",
                "props": {
                    "chartData": chart_data,
                    "chartStats": chart_stats,
                    "timeRange": "30d",
                    "chartTypes": ["portfolio_value", "cumulative_return", "drawdown"],
                    "lastUpdated": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoPerformanceChart",
                "props": {
                    "error": str(e),
                    "chartData": []
                }
            }
    
    async def get_demo_risk_monitor(self) -> Dict[str, Any]:
        """Get demo risk monitor component data"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get risk data
            risk_metrics = await self.portfolio.get_risk_metrics()
            risk_validation = await self.dashboard.get_risk_summary()
            
            # Risk thresholds
            risk_thresholds = {
                "var95": {"warning": 3.0, "critical": 5.0},
                "maxDrawdown": {"warning": 10.0, "critical": 15.0},
                "volatility": {"warning": 25.0, "critical": 35.0},
                "concentration": {"warning": 25.0, "critical": 35.0}
            }
            
            # Risk status calculation
            var_status = "normal"
            if risk_metrics.var_95 * 100 > risk_thresholds["var95"]["critical"]:
                var_status = "critical"
            elif risk_metrics.var_95 * 100 > risk_thresholds["var95"]["warning"]:
                var_status = "warning"
            
            dd_status = "normal"
            max_dd_value = risk_validation.get("max_drawdown", 0)
            if max_dd_value > risk_thresholds["maxDrawdown"]["critical"]:
                dd_status = "critical"
            elif max_dd_value > risk_thresholds["maxDrawdown"]["warning"]:
                dd_status = "warning"
            
            return {
                "component": "DemoRiskMonitor",
                "props": {
                    "riskMetrics": {
                        "var95": {
                            "value": risk_metrics.var_95 * 100,
                            "status": var_status,
                            "threshold": risk_thresholds["var95"]
                        },
                        "maxDrawdown": {
                            "value": max_dd_value,
                            "status": dd_status,
                            "threshold": risk_thresholds["maxDrawdown"]
                        },
                        "volatility": {
                            "value": risk_metrics.volatility * 100,
                            "status": "normal",
                            "threshold": risk_thresholds["volatility"]
                        },
                        "sharpeRatio": {
                            "value": risk_metrics.sharpe_ratio,
                            "status": "normal" if risk_metrics.sharpe_ratio >= 0.5 else "warning"
                        }
                    },
                    "riskLimits": risk_thresholds,
                    "currentAlerts": risk_validation.get("current_alerts", 0),
                    "lastValidation": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "component": "DemoRiskMonitor",
                "props": {
                    "error": str(e),
                    "riskMetrics": {},
                    "riskLimits": {}
                }
            }


# Global UI instance
demo_ui = None


async def get_demo_ui() -> DemoModeUI:
    """Get global demo UI instance"""
    global demo_ui
    if demo_ui is None:
        demo_ui = DemoModeUI()
    return demo_ui


# React Component Templates
REACT_COMPONENTS = {
    "DemoModeIndicator": '''
import React, { useState, useEffect } from 'react';
import { Badge, Alert } from '@/components/ui';

const DemoModeIndicator = ({ isDemoActive, demoState, sessionId, environment, demoBalance }) => {
  const [timeSinceStart, setTimeSinceStart] = useState('0s');

  useEffect(() => {
    if (isDemoActive) {
      const interval = setInterval(() => {
        // Update time since start
        setTimeSinceStart('1m'); // Simplified
      }, 60000);
      return () => clearInterval(interval);
    }
  }, [isDemoActive]);

  if (!isDemoActive) {
    return (
      <Alert variant="warning">
        <Alert.Title>Demo Mode Inactive</Alert.Title>
        <Alert.Description>
          Trading simulation is not currently active. Enable demo mode to start virtual trading.
        </Alert.Description>
      </Alert>
    );
  }

  return (
    <div className="demo-mode-indicator">
      <Badge variant="success">
        <span className="demo-status-dot"></span>
        Demo Mode Active
      </Badge>
      <div className="demo-info">
        <span>Session: {sessionId}</span>
        <span>Environment: {environment}</span>
        <span>Balance: ${demoBalance?.toLocaleString()}</span>
        <span>Active: {timeSinceStart}</span>
      </div>
    </div>
  );
};

export default DemoModeIndicator;
''',
    
    "VirtualPortfolio": '''
import React, { useState, useEffect } from 'react';
import { Card, Table, Metric, Progress, Badge } from '@/components/ui';

const VirtualPortfolio = ({ totalValue, cashBalance, investedValue, totalReturn, dailyChange, positions, performanceMetrics, riskMetrics }) => {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    // Trigger portfolio refresh
    setTimeout(() => setRefreshing(false), 1000);
  };

  return (
    <div className="virtual-portfolio">
      <Card>
        <div className="portfolio-summary">
          <div className="metric-group">
            <Metric label="Total Value" value={`$${totalValue?.toLocaleString()}`} />
            <Metric label="Cash Balance" value={`$${cashBalance?.toLocaleString()}`} />
            <Metric label="Invested Value" value={`$${investedValue?.toLocaleString()}`} />
          </div>
          <div className="performance-group">
            <Metric 
              label="Total Return" 
              value={`${totalReturn?.toFixed(2)}%`}
              trend={totalReturn >= 0 ? 'up' : 'down'}
            />
            <Metric 
              label="Daily Change" 
              value={`${dailyChange?.toFixed(2)}%`}
              trend={dailyChange >= 0 ? 'up' : 'down'}
            />
          </div>
        </div>

        <div className="positions-section">
          <h3>Positions</h3>
          {positions?.length > 0 ? (
            <Table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Quantity</th>
                  <th>Avg Price</th>
                  <th>Current Price</th>
                  <th>Market Value</th>
                  <th>Unrealized P&L</th>
                  <th>P&L %</th>
                  <th>Weight</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position, index) => (
                  <tr key={index}>
                    <td>{position.symbol}</td>
                    <td>
                      <Badge variant={position.side === 'Long' ? 'success' : 'destructive'}>
                        {position.side}
                      </Badge>
                    </td>
                    <td>{position.quantity}</td>
                    <td>${position.avgPrice?.toFixed(2)}</td>
                    <td>${position.currentPrice?.toFixed(2)}</td>
                    <td>${position.marketValue?.toLocaleString()}</td>
                    <td className={position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                      ${position.unrealizedPnl?.toFixed(2)}
                    </td>
                    <td className={position.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {position.pnlPercentage?.toFixed(2)}%
                    </td>
                    <td>{position.weight?.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <p>No open positions</p>
          )}
        </div>

        <div className="metrics-section">
          <div className="performance-metrics">
            <h4>Performance Metrics</h4>
            <div className="metrics-grid">
              <Metric label="Sharpe Ratio" value={performanceMetrics?.sharpeRatio?.toFixed(2)} />
              <Metric label="Max Drawdown" value={`${performanceMetrics?.maxDrawdown?.toFixed(2)}%`} />
              <Metric label="Win Rate" value={`${performanceMetrics?.winRate?.toFixed(1)}%`} />
              <Metric label="Total Trades" value={performanceMetrics?.totalTrades} />
            </div>
          </div>

          <div className="risk-metrics">
            <h4>Risk Metrics</h4>
            <div className="metrics-grid">
              <Metric label="95% VaR" value={`${riskMetrics?.var95?.toFixed(2)}%`} />
              <Metric label="Volatility" value={`${riskMetrics?.volatility?.toFixed(2)}%`} />
              <Metric label="Sortino Ratio" value={riskMetrics?.sortinoRatio?.toFixed(2)} />
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default VirtualPortfolio;
'''
}


async def get_react_component(component_name: str) -> str:
    """Get React component template"""
    return REACT_COMPONENTS.get(component_name, f"// Component {component_name} not found")


async def get_all_demo_ui_components() -> Dict[str, Any]:
    """Get all demo UI components"""
    ui = await get_demo_ui()
    
    components = {}
    
    # Get all component data
    components["DemoModeIndicator"] = await ui.get_demo_indicator_component()
    components["VirtualPortfolio"] = await ui.get_virtual_portfolio_component()
    components["DemoTradingInterface"] = await ui.get_demo_trading_interface()
    components["DemoConfigurationPanel"] = await ui.get_demo_configuration_panel()
    components["DemoDashboardWidget"] = await ui.get_demo_dashboard_widget()
    components["DemoAlerts"] = await ui.get_demo_alerts_component()
    components["DemoPerformanceChart"] = await ui.get_demo_performance_chart()
    components["DemoRiskMonitor"] = await ui.get_demo_risk_monitor()
    
    return components


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize UI
        ui = await get_demo_ui()
        
        # Get component data
        indicator = await ui.get_demo_indicator_component()
        portfolio = await ui.get_virtual_portfolio_component()
        
        print("Demo UI Components:")
        print(f"Demo Mode Indicator: {indicator}")
        print(f"Virtual Portfolio: {portfolio}")
    
    asyncio.run(main())
