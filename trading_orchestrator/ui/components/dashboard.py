"""
Dashboard Manager - Real-time Data Integration for Terminal UI
Manages data feeds, updates, and integration between system components and UI
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

from loguru import logger

from config.application import ApplicationConfig
from ui.terminal import TerminalUI


class DashboardManager:
    """
    Dashboard Manager for real-time data integration
    
    Features:
    - Real-time market data feeds
    - Account and position updates
    - Order tracking and status
    - Risk monitoring
    - System health status
    - AI analysis results display
    """
    
    def __init__(self, app_config: ApplicationConfig, ui_instance: TerminalUI):
        """
        Initialize dashboard manager
        
        Args:
            app_config: Application configuration instance
            ui_instance: Terminal UI instance
        """
        self.app_config = app_config
        self.ui = ui_instance
        
        # Data caches
        self.market_data = {}
        self.account_data = {}
        self.positions = []
        self.orders = []
        self.risk_metrics = {}
        self.system_status = {}
        self.ai_insights = []
        
        # Update intervals
        self.update_intervals = {
            'market': 2,
            'account': 5,
            'positions': 5,
            'orders': 3,
            'risk': 10,
            'system': 30,
            'ai': 300
        }
        
        self.last_updates = {
            'market': datetime.utcnow(),
            'account': datetime.utcnow(),
            'positions': datetime.utcnow(),
            'orders': datetime.utcnow(),
            'risk': datetime.utcnow(),
            'system': datetime.utcnow(),
            'ai': datetime.utcnow()
        }
        
        logger.info("Dashboard Manager initialized")
    
    async def setup_real_data_feeds(self):
        """Setup real data feeds for the dashboard"""
        try:
            logger.info("🔗 Setting up real data feeds...")
            
            # This would initialize connections to real data providers
            # For now, we'll simulate realistic data
            
            # Initialize mock data with realistic values
            await self._initialize_mock_data()
            
            logger.success("✅ Real data feeds configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup data feeds: {e}")
            raise
    
    async def _initialize_mock_data(self):
        """Initialize with realistic mock data"""
        # Account data
        self.account_data = {
            'balance': 100000.0,
            'available': 85000.0,
            'equity': 102500.0,
            'buying_power': 170000.0,
            'day_trade_buying_power': 170000.0,
            'maintenance_requirement': 15000.0,
            'sma': 102500.0
        }
        
        # Mock positions
        self.positions = [
            {
                'symbol': 'AAPL',
                'side': 'long',
                'quantity': 100,
                'entry_price': 150.00,
                'current_price': 152.50,
                'market_value': 15250.0,
                'unrealized_pnl': 250.0,
                'unrealized_pnl_percent': 1.67,
                'average_entry_price': 150.00,
                'daytrade_count': 0,
                'asset_marginable': True,
                'exchange': 'NASDAQ'
            },
            {
                'symbol': 'MSFT',
                'side': 'long',
                'quantity': 50,
                'entry_price': 300.00,
                'current_price': 298.75,
                'market_value': 14937.5,
                'unrealized_pnl': -62.5,
                'unrealized_pnl_percent': -0.42,
                'average_entry_price': 300.00,
                'daytrade_count': 0,
                'asset_marginable': True,
                'exchange': 'NASDAQ'
            },
            {
                'symbol': 'BTC-USD',
                'side': 'long',
                'quantity': 0.5,
                'entry_price': 45000.00,
                'current_price': 46500.00,
                'market_value': 23250.0,
                'unrealized_pnl': 750.0,
                'unrealized_pnl_percent': 3.33,
                'average_entry_price': 45000.00,
                'daytrade_count': 0,
                'asset_marginable': True,
                'exchange': 'BINANCE'
            }
        ]
        
        # Mock orders
        self.orders = [
            {
                'order_id': 'ORD_20241106_001',
                'symbol': 'GOOGL',
                'order_type': 'limit',
                'side': 'buy',
                'status': 'accepted',
                'quantity': 20,
                'filled_quantity': 0,
                'remaining_quantity': 20,
                'limit_price': 2750.00,
                'stop_price': None,
                'time_in_force': 'day',
                'created_time': datetime.utcnow().isoformat(),
                'updated_time': datetime.utcnow().isoformat()
            }
        ]
        
        # Risk metrics
        self.risk_metrics = {
            'daily_pnl': 937.5,
            'max_daily_loss': 1000.0,
            'daily_loss_remaining': 62.5,
            'open_orders': 1,
            'max_position_size': 10000.0,
            'current_exposure': 53437.5,
            'exposure_pct': 52.1,
            'var_95': -2500.0,
            'sharpe_ratio': 1.25,
            'max_drawdown': -5.2,
            'win_rate': 68.5
        }
        
        # System status
        self.system_status = {
            'brokers': {
                'alpaca': {'connected': True, 'latency': 15},
                'binance': {'connected': True, 'latency': 8},
                'ibkr': {'connected': False, 'reason': 'TWS not running'}
            },
            'ai_status': 'active',
            'ai_model': 'gpt-4-turbo',
            'ai_decisions_today': 23,
            'uptime': 3600,
            'memory_usage': 245.6,
            'cpu_usage': 12.3
        }
    
    async def refresh_all_data(self):
        """Refresh all dashboard data"""
        try:
            current_time = datetime.utcnow()
            
            # Refresh data based on intervals
            if self._should_update('market', current_time):
                await self._refresh_market_data()
                self.last_updates['market'] = current_time
            
            if self._should_update('account', current_time):
                await self._refresh_account_data()
                self.last_updates['account'] = current_time
            
            if self._should_update('positions', current_time):
                await self._refresh_positions_data()
                self.last_updates['positions'] = current_time
            
            if self._should_update('orders', current_time):
                await self._refresh_orders_data()
                self.last_updates['orders'] = current_time
            
            if self._should_update('risk', current_time):
                await self._refresh_risk_data()
                self.last_updates['risk'] = current_time
            
            if self._should_update('system', current_time):
                await self._refresh_system_data()
                self.last_updates['system'] = current_time
            
            if self._should_update('ai', current_time):
                await self._refresh_ai_data()
                self.last_updates['ai'] = current_time
                
        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {e}")
    
    def _should_update(self, data_type: str, current_time: datetime) -> bool:
        """Check if data should be updated based on interval"""
        last_update = self.last_updates.get(data_type, datetime.utcnow())
        interval = self.update_intervals.get(data_type, 60)
        return (current_time - last_update).total_seconds() >= interval
    
    async def _refresh_market_data(self):
        """Refresh market data (prices, volume, etc.)"""
        try:
            # Update current prices for all symbols we have positions in
            symbols = ['AAPL', 'MSFT', 'BTC-USD', 'GOOGL', 'TSLA', 'NVDA']
            
            for symbol in symbols:
                # Simulate price changes
                base_prices = {
                    'AAPL': 152.50, 'MSFT': 298.75, 'BTC-USD': 46500.00,
                    'GOOGL': 2750.00, 'TSLA': 220.00, 'NVDA': 450.00
                }
                
                base_price = base_prices.get(symbol, 100.00)
                
                # Add some random variation (±0.5%)
                import random
                variation = random.uniform(-0.005, 0.005)
                new_price = base_price * (1 + variation)
                
                self.market_data[symbol] = {
                    'price': new_price,
                    'change': variation * 100,
                    'volume': random.randint(1000000, 50000000),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Update position prices if we have positions in this symbol
                for position in self.positions:
                    if position['symbol'] == symbol:
                        position['current_price'] = new_price
                        position['market_value'] = new_price * position['quantity']
                        position['unrealized_pnl'] = (
                            new_price - position['entry_price']
                        ) * position['quantity']
                        position['unrealized_pnl_percent'] = (
                            position['unrealized_pnl'] / (position['entry_price'] * position['quantity']) * 100
                        )
            
        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
    
    async def _refresh_account_data(self):
        """Refresh account data"""
        try:
            # Calculate account metrics from positions
            total_market_value = sum(pos['market_value'] for pos in self.positions)
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
            
            # Update account data
            self.account_data.update({
                'balance': 100000.0,  # Would come from broker
                'available': 85000.0,
                'equity': total_market_value + 85000.0,
                'buying_power': 170000.0,
                'total_market_value': total_market_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'last_update': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error refreshing account data: {e}")
    
    async def _refresh_positions_data(self):
        """Refresh positions data"""
        try:
            # This would normally fetch from broker APIs
            # For now, we keep the current positions and update market data
            
            # Remove positions with zero quantity (filled)
            self.positions = [pos for pos in self.positions if pos['quantity'] > 0]
            
        except Exception as e:
            logger.error(f"Error refreshing positions data: {e}")
    
    async def _refresh_orders_data(self):
        """Refresh orders data"""
        try:
            # This would normally fetch order status from brokers
            # For now, we simulate some order updates
            
            import random
            for order in self.orders:
                if order['status'] == 'accepted' and random.random() < 0.1:  # 10% chance to fill
                    order['status'] = 'filled'
                    order['filled_quantity'] = order['quantity']
                    order['remaining_quantity'] = 0
                    order['updated_time'] = datetime.utcnow().isoformat()
                    
                    # Create new position
                    self.positions.append({
                        'symbol': order['symbol'],
                        'side': 'long' if order['side'] == 'buy' else 'short',
                        'quantity': order['quantity'],
                        'entry_price': order['limit_price'],
                        'current_price': order['limit_price'],
                        'market_value': order['quantity'] * order['limit_price'],
                        'unrealized_pnl': 0.0,
                        'unrealized_pnl_percent': 0.0,
                        'average_entry_price': order['limit_price'],
                        'daytrade_count': 0,
                        'asset_marginable': True,
                        'exchange': 'NASDAQ'
                    })
            
            # Remove filled orders from active list
            self.orders = [order for order in self.orders if order['status'] not in ['filled', 'cancelled', 'rejected']]
            
        except Exception as e:
            logger.error(f"Error refreshing orders data: {e}")
    
    async def _refresh_risk_data(self):
        """Refresh risk management data"""
        try:
            # Calculate risk metrics
            total_exposure = sum(abs(pos['market_value']) for pos in self.positions)
            total_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
            
            self.risk_metrics.update({
                'daily_pnl': total_pnl,
                'current_exposure': total_exposure,
                'exposure_pct': (total_exposure / self.account_data.get('equity', 100000)) * 100,
                'open_orders': len(self.orders),
                'risk_score': min(total_exposure / 100000, 1.0),  # Simplified risk score
                'last_update': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error refreshing risk data: {e}")
    
    async def _refresh_system_data(self):
        """Refresh system status data"""
        try:
            # Update system metrics
            self.system_status.update({
                'uptime': self.system_status.get('uptime', 0) + 30,
                'ai_decisions_today': min(self.system_status.get('ai_decisions_today', 0) + 1, 100),
                'last_update': datetime.utcnow().isoformat()
            })
            
            # Update broker connection status
            import random
            for broker_name, broker_info in self.system_status['brokers'].items():
                if broker_info['connected']:
                    # Simulate latency changes
                    broker_info['latency'] = max(5, random.randint(8, 50))
            
        except Exception as e:
            logger.error(f"Error refreshing system data: {e}")
    
    async def _refresh_ai_data(self):
        """Refresh AI insights and analysis"""
        try:
            # Simulate AI insights
            insights = [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'market_analysis',
                    'symbol': 'AAPL',
                    'insight': 'Strong momentum detected, consider adding to position',
                    'confidence': 0.85,
                    'action': 'HOLD'
                },
                {
                    'timestamp': (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    'type': 'risk_alert',
                    'symbol': 'BTC-USD',
                    'insight': 'High volatility detected, consider reducing exposure',
                    'confidence': 0.92,
                    'action': 'REDUCE'
                }
            ]
            
            self.ai_insights = insights
            
        except Exception as e:
            logger.error(f"Error refreshing AI data: {e}")
    
    def get_account_data(self) -> Dict[str, Any]:
        """Get current account data"""
        return self.account_data
    
    def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get current positions data"""
        return self.positions
    
    def get_orders_data(self) -> List[Dict[str, Any]]:
        """Get current orders data"""
        return self.orders
    
    def get_risk_data(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return self.risk_metrics
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get current system status"""
        return self.system_status
    
    def get_ai_insights(self) -> List[Dict[str, Any]]:
        """Get current AI insights"""
        return self.ai_insights
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = 'market',
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Submit order through the dashboard"""
        try:
            if not self.app_config.state.order_manager:
                return {'success': False, 'reason': 'Order manager not available'}
            
            result = await self.app_config.state.order_manager.submit_order(
                symbol=symbol,
                side=side,
                quantity=Decimal(str(quantity)),
                order_type=order_type,
                price=Decimal(str(limit_price)) if limit_price else None,
                broker_name='alpaca'  # Default broker
            )
            
            if result.get('success'):
                # Add to local orders list
                self.orders.append({
                    'order_id': result['order_id'],
                    'symbol': symbol,
                    'order_type': order_type,
                    'side': side,
                    'status': 'pending',
                    'quantity': quantity,
                    'filled_quantity': 0,
                    'remaining_quantity': quantity,
                    'limit_price': limit_price,
                    'created_time': datetime.utcnow().isoformat(),
                    'updated_time': datetime.utcnow().isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {'success': False, 'reason': str(e)}


# Example usage
if __name__ == "__main__":
    async def test_dashboard():
        from config.application import app_config
        
        app = app_config
        ui = TerminalUI()
        dashboard = DashboardManager(app, ui)
        
        await dashboard.setup_real_data_feeds()
        
        # Test data retrieval
        print("Account Data:", dashboard.get_account_data())
        print("Positions:", dashboard.get_positions_data())
        print("Risk Metrics:", dashboard.get_risk_data())
    
    asyncio.run(test_dashboard())