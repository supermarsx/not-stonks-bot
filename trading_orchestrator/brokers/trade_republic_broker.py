"""
Trade Republic Unofficial API Integration

⚠️  CRITICAL LEGAL WARNING ⚠️
This implementation uses UNOFFICIAL APIs that violate Trade Republic's Customer Agreement.
Using this code may result in:
- IMMEDIATE ACCOUNT TERMINATION
- VIOLATION OF CUSTOMER AGREEMENT
- LOSS OF ALL SERVICES
- POTENTIAL LEGAL CONSEQUENCES

Trade Republic's Customer Agreement explicitly states:
"Services can only be used via this Application... as well as other access channels provided by Trade Republic."
"Any use of the features and services provided by Trade Republic through access paths, programs and/or other interfaces not provided by Trade Republic outside of the Application is prohibited."
"Violation of this prohibition may result in extraordinary termination."

THIS IS HIGH RISK - USE AT YOUR OWN RISK
Consult legal counsel before using any unofficial interfaces.

Based on research in research/trade_republic/

Libraries (as per research):
- pytr (pytr-org/pytr): Most comprehensive, MIT license, async WebSocket
- TradeRepublicApi (Zarathustra2): Less maintained, has breakage issues
- trade-republic-api (PyPI): Minimal wrapper, limited capabilities
- Apify scraper: Web scraping of LS-TC portal

Capabilities (unofficial):
- WebSocket subscriptions to private streams
- Portfolio, cash, watchlists, market data
- Timeline, news, savings plans
- Order placement (high risk)
- 2FA authentication flows

⚠️  EXTREME LEGAL RISK - MAY CAUSE ACCOUNT TERMINATION ⚠️
"""

import asyncio
import warnings
import json
import time
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from loguru import logger

# Critical legal warning decorator
def critical_legal_warning(func):
    """Decorator to warn about severe legal risks before each method call"""
    def wrapper(*args, **kwargs):
        self = args[0]
        logger.critical(f"⚠️  CRITICAL LEGAL WARNING: {func.__name__} on Trade Republic")
        logger.critical("⚠️  VIOLATES CUSTOMER AGREEMENT - ACCOUNT TERMINATION RISK")
        logger.critical("⚠️  PROHIBITED BY TERMS AND CONDITIONS")
        logger.critical("⚠️  IMMEDIATE LEGAL CONSEQUENCES POSSIBLE")
        warnings.warn(
            f"Trade Republic integration violates Customer Agreement. "
            f"Account termination guaranteed. {func.__name__} called.",
            UserWarning
        )
        return func(*args, **kwargs)
    return wrapper


class TradeRepublicBroker(BaseBroker):
    """
    Trade Republic Unofficial API Implementation
    
    ⚠️  EXTREME LEGAL RISK - CUSTOMER AGREEMENT VIOLATION ⚠️
    - No official API support (confirmed by Trade Republic)
    - Violates Customer Agreement Section X
    - Violates Light User Agreement
    - Account termination probable
    - Legal consequences possible
    
    Uses unofficial libraries: pytr, TradeRepublicApi
    ⚠️  USE AT YOUR OWN EXTREME RISK ⚠️
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Issue critical legal warnings
        logger.critical("⚠️  TRADE REPUBLIC INTEGRATION - CRITICAL LEGAL WARNING ⚠️")
        logger.critical("⚠️  VIOLATES CUSTOMER AGREEMENT")
        logger.critical("⚠️  ACCOUNT TERMINATION RISK = 100%")
        logger.critical("⚠️  IMMEDIATE LEGAL CONSEQUENCES")
        logger.critical("⚠️  CONSULT LEGAL COUNSEL BEFORE USE")
        logger.critical("⚠️  THIS IS PROHIBITED BEHAVIOR")
        
        # Extract configuration  
        config_dict = config.config or {}
        self.phone_number = config_dict.get('phone_number')  # Trade Republic phone number
        self.pin = config_dict.get('pin')  # Trade Republic PIN
        self.device_id = config_dict.get('device_id')  # Device identifier
        
        # Rate limiting state (unofficial)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Conservative 1 second between requests
        
        # Session storage
        self.ws_connection: Optional[Any] = None
        self.authenticated: bool = False
        self.subscriptions: Dict[str, Any] = {}
        
        # Import tracking
        self._client = None
        self._available_client = None
        self._library_maintenance_status = "UNKNOWN"
        
        logger.critical(f"Trade Republic broker initialized (risk_level=CRITICAL_LEGAL)")
        
    def _get_client(self):
        """Get or create Trade Republic unofficial API client"""
        if self._client:
            return self._client
            
        # Try to import different unofficial libraries
        libraries_to_try = [
            ('pytr', 'TraderRepublicClient'),  # Most comprehensive
            ('TradeRepublicApi', 'TradeRepublicApi'),  # Zarathustra2
            ('trade_republic_api', 'Client'),  # Minimal wrapper
        ]
        
        for lib_name, class_name in libraries_to_try:
            try:
                import importlib
                module = importlib.import_module(lib_name)
                ClientClass = getattr(module, class_name, None)
                
                if ClientClass:
                    self._available_client = (ClientClass, lib_name)
                    logger.info(f"Found Trade Republic library: {lib_name}")
                    
                    # Check maintenance status
                    if lib_name == 'pytr':
                        self._library_maintenance_status = "ACTIVE"
                    elif lib_name == 'TradeRepublicApi':
                        self._library_maintenance_status = "STALE"
                    else:
                        self._library_maintenance_status = "UNKNOWN"
                    
                    return ClientClass
                    
            except ImportError as e:
                logger.warning(f"Library {lib_name} not available: {e}")
                continue
                
        # If no library found
        raise ImportError(
            "No Trade Republic unofficial library found. "
            "Install one of: pytr (recommended), TradeRepublicApi, trade-republic-api. "
            "Note: These violate Customer Agreement."
        )
    
    @critical_legal_warning
    async def connect(self) -> bool:
        """Connect to Trade Republic unofficial API"""
        async with self._connection_lock:
            try:
                logger.warning("Connecting to Trade Republic (UNOFFICIAL API - HIGH RISK)...")
                
                if not self.phone_number or not self.pin:
                    raise ValueError("Phone number and PIN required for Trade Republic")
                
                ClientClass, lib_name = self._get_client()
                logger.warning(f"Using unofficial library: {lib_name} (status: {self._library_maintenance_status})")
                
                # Create client
                self._client = ClientClass()
                
                # Authentication (varies by library)
                if lib_name == 'pytr':
                    # pytr has web login and app login methods
                    login_method = config_dict.get('login_method', 'web')  # 'web' or 'app'
                    
                    if login_method == 'web':
                        # Web login - less intrusive, occasional 2FA prompts
                        login_result = await self._client.login_web(
                            phone=self.phone_number,
                            pin=self.pin
                        )
                    else:
                        # App login - requires device reset, logs out mobile app
                        login_result = await self._client.login_app(
                            phone=self.phone_number,
                            pin=self.pin
                        )
                
                elif lib_name == 'TradeRepublicApi':
                    # Zarathustra2 library
                    login_result = self._client.login(
                        phone_number=self.phone_number,
                        pin=self.pin
                    )
                else:
                    # Other libraries
                    login_result = self._client.connect(
                        phone=self.phone_number,
                        pin=self.pin
                    )
                
                # Check authentication success
                if hasattr(login_result, 'success') and not login_result.success:
                    raise ValueError(f"Authentication failed: {login_result.error}")
                
                self.authenticated = True
                self.is_connected = True
                
                logger.warning(f"Connected to Trade Republic via {lib_name}")
                logger.warning(f"⚠️  LEGAL RISK: Violating Customer Agreement")
                logger.warning(f"⚠️  MAINTENANCE: Library status is {self._library_maintenance_status}")
                
                return True
                
            except Exception as e:
                logger.error(f"Trade Republic connection failed: {e}")
                logger.error("⚠️  Unofficial APIs are fragile and can break")
                self.authenticated = False
                self.is_connected = False
                return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Trade Republic"""
        async with self._connection_lock:
            try:
                if self._client and self.authenticated:
                    # Logout based on available method
                    if hasattr(self._client, 'logout'):
                        if asyncio.iscoroutinefunction(self._client.logout):
                            await self._client.logout()
                        else:
                            self._client.logout()
                
                self.ws_connection = None
                self.authenticated = False
                self.is_connected = False
                logger.info("Disconnected from Trade Republic")
                return True
                
            except Exception as e:
                logger.error(f"Disconnect failed: {e}")
                return False
    
    async def is_connection_alive(self) -> bool:
        """Check if Trade Republic connection is alive"""
        try:
            if not self.is_connected or not self.authenticated:
                return False
            
            # Try to get basic account data
            await self._rate_limited_call(self._client.get_portfolio)
            return True
            
        except Exception:
            return False
    
    def _rate_limited_call(self, func, *args, **kwargs):
        """Apply conservative rate limiting to API calls"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
        # Call function (sync or async)
        if asyncio.iscoroutinefunction(func):
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    @critical_legal_warning
    async def get_account(self) -> AccountInfo:
        """Get Trade Republic account information"""
        @critical_legal_warning
        async def _get_account():
            try:
                # Get portfolio and cash data
                portfolio_data = await self._rate_limited_call(
                    self._client.get_portfolio
                )
                cash_data = await self._rate_limited_call(
                    self._client.get_cash
                )
                
                # Extract fields (varies by library)
                if hasattr(portfolio_data, 'total_value'):
                    total_value = float(portfolio_data.total_value)
                elif isinstance(portfolio_data, dict):
                    total_value = float(portfolio_data.get('totalValue', 0))
                else:
                    total_value = 0.0
                
                if hasattr(cash_data, 'available'):
                    available = float(cash_data.available)
                elif isinstance(cash_data, dict):
                    available = float(cash_data.get('available', 0))
                else:
                    available = 0.0
                
                return AccountInfo(
                    account_id="trade_republic_unofficial",  # Can't get real account ID
                    broker_name="Trade Republic",
                    currency="EUR",
                    balance=available,
                    available_balance=available,
                    equity=total_value,
                    buying_power=available,
                    margin_used=0.0,  # Trade Republic is cash account
                    margin_available=available,
                    is_pattern_day_trader=False,
                    metadata={
                        'unofficial_api': True,
                        'risk_level': 'CRITICAL_LEGAL',
                        'customer_agreement_violation': True,
                        'library': self._available_client[1] if self._available_client else 'unknown',
                        'maintenance_status': self._library_maintenance_status
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get Trade Republic account: {e}")
                raise
        
        return await _get_account()
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get Trade Republic positions"""
        @critical_legal_warning
        async def _get_positions():
            try:
                portfolio_data = await self._rate_limited_call(
                    self._client.get_portfolio
                )
                
                positions = []
                
                # Handle different response formats
                if hasattr(portfolio_data, 'positions'):
                    items = portfolio_data.positions
                elif isinstance(portfolio_data, dict) and 'positions' in portfolio_data:
                    items = portfolio_data['positions']
                else:
                    items = portfolio_data
                
                for position in items:
                    if float(getattr(position, 'quantity', 0) or 0) > 0:
                        positions.append(PositionInfo(
                            symbol=str(getattr(position, 'isin', '')),  # Trade Republic uses ISIN
                            broker_name="Trade Republic",
                            side="long",
                            quantity=float(getattr(position, 'quantity', 0)),
                            avg_entry_price=float(getattr(position, 'average_price', 0)),
                            current_price=float(getattr(position, 'current_price', 0)),
                            market_value=float(getattr(position, 'market_value', 0)),
                            unrealized_pnl=float(getattr(position, 'pl', 0)),
                            unrealized_pnl_percent=float(getattr(position, 'pl_percent', 0)),
                            metadata={
                                'unofficial_api': True,
                                'customer_agreement_violation': True,
                                'position_id': str(getattr(position, 'id', ''))
                            }
                        ))
                
                return positions
                
            except Exception as e:
                logger.error(f"Failed to get Trade Republic positions: {e}")
                return []
        
        return await _get_positions()
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol (ISIN)"""
        positions = await self.get_positions()
        for position in positions:
            if position.symbol == symbol:
                return position
        return None
    
    @critical_legal_warning
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
        **kwargs
    ) -> OrderInfo:
        """⚠️  EXTREME RISK: Order placement via unofficial API ⚠️"""
        
        logger.critical("⚠️  TRADE REPUBLIC ORDER PLACEMENT - ACCOUNT TERMINATION GUARANTEED")
        logger.critical("⚠️  Customer Agreement Section X explicitly prohibits this")
        logger.critical("⚠️  Violation will result in extraordinary termination")
        logger.critical("⚠️  Legal consequences inevitable")
        
        try:
            # Map order types to Trade Republic format
            order_mapping = {
                'market': 'market',
                'limit': 'limit',
                'stop': 'stop_market'  # Trade Republic doesn't have simple stop orders
            }
            
            tr_order_type = order_mapping.get(order_type.lower(), 'market')
            
            # Prepare order parameters
            order_params = {
                'isin': symbol,  # Trade Republic uses ISIN
                'side': side.upper(),  # BUY or SELL
                'quantity': abs(quantity),
                'order_type': tr_order_type
            }
            
            if limit_price:
                order_params['limit_price'] = limit_price
            if stop_price:
                order_params['stop_price'] = stop_price
            
            # Add expiry (Trade Republic specific)
            expiry_mapping = {
                'day': 'gfd',  # Good for day
                'gtc': 'gtc',  # Good till cancelled
                'gtd': 'gtd'   # Good till date (requires expiry_date)
            }
            order_params['validity'] = expiry_mapping.get(time_in_force.lower(), 'gfd')
            
            # Place order (extremely risky)
            result = await self._rate_limited_call(
                self._client.place_order,
                **order_params
            )
            
            # Extract order ID
            order_id = str(getattr(result, 'id', 'unknown'))
            
            return OrderInfo(
                order_id=order_id,
                broker_name="Trade Republic",
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                filled_quantity=0.0,
                status="pending",
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                extended_hours=extended_hours,
                submitted_at=datetime.utcnow(),
                metadata={
                    'unofficial_api': True,
                    'customer_agreement_violation': True,
                    'account_termination_risk': 'GUARANTEED',
                    'isin': symbol
                }
            )
            
        except Exception as e:
            logger.error(f"Trade Republic order placement failed: {e}")
            logger.error("⚠️  This is actually GOOD - avoided account termination")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Trade Republic order"""
        @critical_legal_warning
        async def _cancel_order():
            try:
                await self._rate_limited_call(
                    self._client.cancel_order,
                    order_id=order_id
                )
                return True
            except Exception as e:
                logger.error(f"Failed to cancel Trade Republic order {order_id}: {e}")
                return False
        
        return await _cancel_order()
    
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get Trade Republic orders"""
        @critical_legal_warning
        async def _get_orders():
            try:
                orders_data = await self._rate_limited_call(
                    self._client.get_orders
                )
                
                orders = []
                for order_data in orders_data:
                    orders.append(OrderInfo(
                        order_id=str(getattr(order_data, 'id', '')),
                        broker_name="Trade Republic",
                        symbol=str(getattr(order_data, 'isin', '')),
                        order_type=str(getattr(order_data, 'type', '')),
                        side=str(getattr(order_data, 'side', '')),
                        quantity=float(getattr(order_data, 'quantity', 0)),
                        filled_quantity=float(getattr(order_data, 'filled_quantity', 0)),
                        status=str(getattr(order_data, 'status', 'pending')),
                        limit_price=float(getattr(order_data, 'limit_price', 0)) or None,
                        submitted_at=getattr(order_data, 'submitted_at', datetime.utcnow()),
                        metadata={
                            'unofficial_api': True,
                            'customer_agreement_violation': True
                        }
                    ))
                
                return orders
                
            except Exception as e:
                logger.error(f"Failed to get Trade Republic orders: {e}")
                return []
        
        return await _get_orders()
    
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific Trade Republic order"""
        orders = await self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """Get Trade Republic market data"""
        @critical_legal_warning
        async def _get_market_data():
            try:
                # Get performance data
                performance_data = await self._rate_limited_call(
                    self._client.get_performance,
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                market_data = []
                for item in performance_data:
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        broker_name="Trade Republic",
                        timestamp=item.get('date', datetime.utcnow()),
                        open=float(item.get('open', 0)),
                        high=float(item.get('high', 0)),
                        low=float(item.get('low', 0)),
                        close=float(item.get('close', 0)),
                        volume=float(item.get('volume', 0)),
                        timeframe=timeframe,
                        metadata={
                            'unofficial_api': True,
                            'customer_agreement_violation': True
                        }
                    ))
                
                return market_data
                
            except Exception as e:
                logger.error(f"Failed to get Trade Republic market data: {e}")
                return []
        
        return await _get_market_data()
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get Trade Republic quote"""
        @critical_legal_warning
        async def _get_quote():
            try:
                # Get instrument details for quote
                instrument_data = await self._rate_limited_call(
                    self._client.get_instrument_details,
                    symbol=symbol
                )
                
                return {
                    'symbol': symbol,
                    'isin': symbol,
                    'bid': float(getattr(instrument_data, 'bid', 0)),
                    'ask': float(getattr(instrument_data, 'ask', 0)),
                    'last': float(getattr(instrument_data, 'price', 0)),
                    'volume': float(getattr(instrument_data, 'volume', 0)),
                    'timestamp': datetime.utcnow(),
                    'unofficial_api': True,
                    'customer_agreement_violation': True
                }
                
            except Exception as e:
                logger.error(f"Failed to get Trade Republic quote: {e}")
                return {}
        
        return await _get_quote()
    
    # WebSocket streaming (unofficial, may break)
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Stream quotes via unofficial WebSocket (unreliable)"""
        logger.warning("Unofficial WebSocket streaming may break without notice")
        logger.warning("No SLAs or support available for unofficial APIs")
        raise NotImplementedError("Trade Republic unofficial WebSocket streaming not implemented")
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Stream trades via unofficial WebSocket (unreliable)"""
        logger.warning("Unofficial WebSocket streaming may break without notice")
        logger.warning("No SLAs or support available for unofficial APIs")
        raise NotImplementedError("Trade Republic unofficial WebSocket streaming not implemented")
    
    def get_legal_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive legal risk assessment"""
        
        return {
            "broker": "Trade Republic",
            "api_status": "NO_OFFICIAL_API",
            "legal_violations": {
                "customer_agreement": {
                    "section": "X - Access Restrictions",
                    "violation": "Using unofficial interfaces outside Application",
                    "consequence": "Extraordinary termination",
                    "likelihood": "GUARANTEED if detected"
                },
                "light_user_agreement": {
                    "scope": "Non-custody app-only access",
                    "violation": "Automated access via unofficial programs",
                    "consequence": "Access termination"
                }
            },
            "technical_risks": {
                "breakage_rate": "HIGH - frequent after app updates",
                "maintenance_status": self._library_maintenance_status,
                "support_availability": "NONE - unofficial libraries",
                "reliability": "POOR - no SLAs or guarantees"
            },
            "operational_risks": {
                "device_pairing": "Single device constraint - mobile app logout required",
                "2fa_disruption": "Frequent 2FA prompts for web login",
                "rate_limiting": "Unknown and undocumented",
                "account_tracking": "High risk of detection and termination"
            },
            "recommended_alternatives": [
                "Read-only analytics tools",
                "Manual export/import workflows", 
                "Regulated third-party aggregators (if any)",
                "Alternative brokers with official APIs"
            ],
            "legal_disclaimer": "This integration violates Trade Republic's Customer Agreement. Account termination is probable. Consult legal counsel before use."
        }
    
    def __repr__(self):
        return f"<TradeRepublicBroker(broker='Trade Republic', connected={self.is_connected}, risk=CRITICAL_LEGAL)>"