"""
DEGIRO Unofficial API Integration

⚠️  LEGAL WARNING ⚠️
This implementation uses UNOFFICIAL APIs that violate DEGIRO's Terms of Service.
Using this code may result in:
- Account termination
- Legal consequences
- Loss of access to DEGIRO services
- Violation of Client Agreement and Terms and Conditions

USE AT YOUR OWN RISK. This is provided for educational/research purposes only.
You MUST consult legal counsel before using this integration.

Key Risks:
1. DEGIRO explicitly states NO public API exists
2. Unofficial clients can break without notice
3. Rate limiting is unpredictable
4. 2FA authentication required
5. Session management is fragile
6. May violate securities regulations

Based on research files in research/degiro/

Libraries (as per research):
- DegiroAPI (Python, MIT license, last commit 2020)
- degiroasync (Python, async, 2FA support)  
- degiro-connector (PyPI, BSD-3-Clause)
- icastillejogomez/degiro-api (TypeScript, most comprehensive)

⚠️  THIS IS HIGH RISK AND UNSTABLE ⚠️
"""

import asyncio
import warnings
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
import json
import hashlib
import time

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from loguru import logger

# Legal warning decorator
def legal_warning(func):
    """Decorator to warn about legal risks before each method call"""
    def wrapper(*args, **kwargs):
        self = args[0]
        logger.error(f"⚠️  LEGAL WARNING: {func.__name__} on DEGIRO is UNOFFICIAL and violates ToS!")
        logger.error(f"⚠️  Account termination risk - use at your own risk!")
        warnings.warn(
            f"DEGIRO integration violates Terms of Service. "
            f"Account termination possible. {func.__name__} called.",
            UserWarning
        )
        return func(*args, **kwargs)
    return wrapper


class DegiroBroker(BaseBroker):
    """
    DEGIRO Unofficial API Implementation
    
    ⚠️  HIGH RISK - UNOFFICIAL API ⚠️
    - No official API support
    - May break without notice
    - Violates DEGIRO ToS
    - Account termination possible
    
    Uses pyDegiro or degiroapi libraries (community-maintained)
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Issue prominent legal warning
        logger.critical("⚠️  DEGIRO INTEGRATION - LEGAL WARNING ⚠️")
        logger.critical("⚠️  This uses UNOFFICIAL APIs violating ToS")
        logger.critical("⚠️  Account termination is possible")
        logger.critical("⚠️  Consult legal counsel before use")
        logger.critical("⚠️  USE AT YOUR OWN RISK ⚠️")
        
        # Extract configuration
        config_dict = config.config or {}
        self.username = config_dict.get('username')  # DEGIRO username
        self.password = config_dict.get('password')  # DEGIRO password
        self.twofa_secret = config_dict.get('twofa_secret')  # 2FA secret if enabled
        
        # Rate limiting state
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum 2 seconds between requests (conservative)
        
        # Session storage
        self.session_token: Optional[str] = None
        self.session_id: Optional[str] = None
        
        # Import tracking
        self._client = None
        self._available_client = None
        
        logger.warning(f"DEGIRO broker initialized (risk_level=CRITICAL)")
        
    def _get_client(self):
        """Get or create DEGIRO API client (try multiple libraries)"""
        if self._client:
            return self._client
            
        # Try to import different DEGIRO libraries
        libraries_to_try = [
            ('degiroasync', 'DegiroAsyncClient'),
            ('degiro_connector', 'Connector'),
            ('DegiroAPI', 'Client'),
        ]
        
        for lib_name, class_name in libraries_to_try:
            try:
                import importlib
                module = importlib.import_module(lib_name)
                ClientClass = getattr(module, class_name, None)
                
                if ClientClass:
                    self._available_client = (ClientClass, lib_name)
                    logger.info(f"Found DEGIRO library: {lib_name}")
                    return ClientClass
                    
            except ImportError:
                continue
                
        # If no library found
        raise ImportError(
            "No DEGIRO library found. Install one of: "
            "degiroasync, degiro-connector, DegiroAPI"
        )
    
    @legal_warning
    async def connect(self) -> bool:
        """Connect to DEGIRO unofficial API"""
        async with self._connection_lock:
            try:
                logger.warning("Connecting to DEGIRO (UNOFFICIAL API)...")
                
                if not self.username or not self.password:
                    raise ValueError("Username and password required for DEGIRO")
                
                ClientClass, lib_name = self._get_client()
                
                # Create client with authentication
                self._client = ClientClass()
                
                # Authentication (varies by library)
                if lib_name == 'degiroasync':
                    # degiroasync with 2FA support
                    login_result = await self._client.login(
                        username=self.username,
                        password=self.password,
                        secret=self.twofa_secret
                    )
                elif lib_name == 'degiro_connector':
                    # degiro-connector
                    login_result = self._client.connect(
                        username=self.username,
                        password=self.password,
                        two_fa_code=self.twofa_secret if self.twofa_secret else None
                    )
                else:
                    # DegiroAPI (basic)
                    login_result = self._client.login(
                        username=self.username,
                        password=self.password
                    )
                
                # Extract session info (varies by library)
                if hasattr(login_result, 'session_id'):
                    self.session_id = login_result.session_id
                elif isinstance(login_result, dict) and 'sessionId' in login_result:
                    self.session_id = login_result['sessionId']
                
                self.is_connected = True
                logger.warning(f"Connected to DEGIRO via {lib_name} (session_id={self.session_id[:10]}...)")
                return True
                
            except Exception as e:
                logger.error(f"DEGIRO connection failed: {e}")
                logger.error("⚠️  This is expected - unofficial APIs are fragile")
                self.is_connected = False
                return False
    
    async def disconnect(self) -> bool:
        """Disconnect from DEGIRO"""
        async with self._connection_lock:
            try:
                if self._client:
                    # Logout based on available method
                    if hasattr(self._client, 'logout'):
                        if asyncio.iscoroutinefunction(self._client.logout):
                            await self._client.logout()
                        else:
                            self._client.logout()
                    elif hasattr(self._client, 'disconnect'):
                        self._client.disconnect()
                
                self.is_connected = False
                self._client = None
                logger.info("Disconnected from DEGIRO")
                return True
                
            except Exception as e:
                logger.error(f"Disconnect failed: {e}")
                return False
    
    async def is_connection_alive(self) -> bool:
        """Check if DEGIRO connection is alive"""
        try:
            if not self.is_connected or not self._client:
                return False
            
            # Try to get account info to test connection
            await self._rate_limited_call(self._client.get_account_data)
            return True
            
        except Exception:
            return False
    
    def _rate_limited_call(self, func, *args, **kwargs):
        """Apply rate limiting to API calls"""
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
    
    async def get_account(self) -> AccountInfo:
        """Get DEGIRO account information"""
        @legal_warning
        async def _get_account():
            try:
                # Get account data from client
                account_data = await self._rate_limited_call(
                    self._client.get_account_data
                )
                
                # Handle different response formats
                if hasattr(account_data, 'data'):
                    data = account_data.data
                else:
                    data = account_data
                
                # Extract fields (varies by library)
                balance = float(getattr(data, 'balance', 0) or 0)
                equity = float(getattr(data, 'equity', balance) or balance)
                
                return AccountInfo(
                    account_id=str(getattr(data, 'id', 'unknown')),
                    broker_name="DEGIRO",
                    currency=str(getattr(data, 'currency', 'EUR')),
                    balance=balance,
                    available_balance=equity - balance,
                    equity=equity,
                    buying_power=equity * 0.8,  # Approximation
                    margin_used=0.0,  # DEGIRO typically doesn't show margin
                    margin_available=equity * 0.8,
                    is_pattern_day_trader=False,
                    metadata={
                        'unofficial_api': True,
                        'risk_level': 'CRITICAL',
                        'tos_violation': True,
                        'client_library': self._available_client[1] if self._available_client else 'unknown'
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get DEGIRO account: {e}")
                raise
        
        return await _get_account()
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get DEGIRO positions"""
        @legal_warning
        async def _get_positions():
            try:
                # Get portfolio data
                portfolio = await self._rate_limited_call(
                    self._client.get_portfolio
                )
                
                positions = []
                
                # Handle different response formats
                if hasattr(portfolio, 'data'):
                    items = portfolio.data
                else:
                    items = portfolio
                
                for item in items:
                    if float(getattr(item, 'quantity', 0) or 0) > 0:
                        positions.append(PositionInfo(
                            symbol=str(getattr(item, 'symbol', '')),
                            broker_name="DEGIRO",
                            side="long",
                            quantity=float(getattr(item, 'quantity', 0)),
                            avg_entry_price=float(getattr(item, 'averagePrice', 0)),
                            current_price=float(getattr(item, 'price', 0)),
                            market_value=float(getattr(item, 'marketValue', 0)),
                            unrealized_pnl=float(getattr(item, 'pl', 0)),
                            unrealized_pnl_percent=float(getattr(item, 'plPercent', 0)),
                            metadata={
                                'unofficial_api': True,
                                'position_id': str(getattr(item, 'id', ''))
                            }
                        ))
                
                return positions
                
            except Exception as e:
                logger.error(f"Failed to get DEGIRO positions: {e}")
                return []
        
        return await _get_positions()
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        positions = await self.get_positions()
        for position in positions:
            if position.symbol == symbol:
                return position
        return None
    
    @legal_warning
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
        """⚠️  RISK: Order placement via unofficial API ⚠️"""
        
        logger.critical("⚠️  DEGIRO ORDER PLACEMENT - ACCOUNT TERMINATION RISK ⚠️")
        logger.critical("⚠️  This violates ToS and may trigger account closure")
        logger.critical("⚠️  Use at your own risk - legal consequences possible")
        
        try:
            # First get product ID for symbol (DEGIRO uses internal IDs)
            product_id = await self._get_product_id(symbol)
            if not product_id:
                raise ValueError(f"Product not found: {symbol}")
            
            # Map order types to DEGIRO format
            order_mapping = {
                'market': 'MARKET',
                'limit': 'LIMIT', 
                'stop': 'STOPLOSS',
                'stop_limit': 'STOPLIMIT'
            }
            
            degiro_order_type = order_mapping.get(order_type.lower(), 'MARKET')
            
            # Prepare order parameters
            order_params = {
                'productId': product_id,
                'buySell': side.upper(),
                'orderType': degiro_order_type,
                'quantity': abs(quantity),  # DEGIRO uses positive quantities
                'timeValidity': time_in_force.upper()  # 'DAY' or 'GTC'
            }
            
            if limit_price:
                order_params['price'] = limit_price
            if stop_price:
                order_params['stopPrice'] = stop_price
            
            # Place order (highly risky operation)
            result = await self._rate_limited_call(
                self._client.place_order,
                **order_params
            )
            
            # Extract order ID from response
            order_id = None
            if hasattr(result, 'orderId'):
                order_id = str(result.orderId)
            elif isinstance(result, dict):
                order_id = str(result.get('orderId', ''))
            else:
                order_id = str(hashlib.md5(str(result).encode()).hexdigest())
            
            return OrderInfo(
                order_id=order_id,
                broker_name="DEGIRO",
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
                    'high_risk': True,
                    'tos_violation': True,
                    'product_id': product_id
                }
            )
            
        except Exception as e:
            logger.error(f"DEGIRO order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel DEGIRO order"""
        @legal_warning
        async def _cancel_order():
            try:
                await self._rate_limited_call(
                    self._client.cancel_order,
                    order_id=order_id
                )
                return True
            except Exception as e:
                logger.error(f"Failed to cancel DEGIRO order {order_id}: {e}")
                return False
        
        return await _cancel_order()
    
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get DEGIRO orders"""
        @legal_warning
        async def _get_orders():
            try:
                orders_data = await self._rate_limited_call(
                    self._client.get_orders
                )
                
                orders = []
                for order_data in orders_data:
                    orders.append(OrderInfo(
                        order_id=str(getattr(order_data, 'id', '')),
                        broker_name="DEGIRO",
                        symbol=str(getattr(order_data, 'symbol', '')),
                        order_type=str(getattr(order_data, 'type', '')),
                        side=str(getattr(order_data, 'side', '')),
                        quantity=float(getattr(order_data, 'quantity', 0)),
                        filled_quantity=float(getattr(order_data, 'filledQuantity', 0)),
                        status=str(getattr(order_data, 'status', 'pending')),
                        limit_price=float(getattr(order_data, 'price', 0)) or None,
                        submitted_at=getattr(order_data, 'submittedAt', datetime.utcnow()),
                        metadata={'unofficial_api': True}
                    ))
                
                return orders
                
            except Exception as e:
                logger.error(f"Failed to get DEGIRO orders: {e}")
                return []
        
        return await _get_orders()
    
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific DEGIRO order"""
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
        """Get DEGIRO market data"""
        @legal_warning
        async def _get_market_data():
            try:
                # DEGIRO API for historical data
                product_id = await self._get_product_id(symbol)
                if not product_id:
                    return []
                
                # Get historical data
                data = await self._rate_limited_call(
                    self._client.get_historical_data,
                    product_id=product_id,
                    period=timeframe,
                    start_date=start,
                    end_date=end
                )
                
                market_data = []
                for item in data:
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        broker_name="DEGIRO",
                        timestamp=item.get('date', datetime.utcnow()),
                        open=float(item.get('open', 0)),
                        high=float(item.get('high', 0)),
                        low=float(item.get('low', 0)),
                        close=float(item.get('close', 0)),
                        volume=float(item.get('volume', 0)),
                        timeframe=timeframe,
                        metadata={'unofficial_api': True}
                    ))
                
                return market_data
                
            except Exception as e:
                logger.error(f"Failed to get DEGIRO market data: {e}")
                return []
        
        return await _get_market_data()
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get DEGIRO quote"""
        @legal_warning
        async def _get_quote():
            try:
                product_id = await self._get_product_id(symbol)
                if not product_id:
                    return {}
                
                quote_data = await self._rate_limited_call(
                    self._client.get_quote,
                    product_id=product_id
                )
                
                return {
                    'symbol': symbol,
                    'bid': float(getattr(quote_data, 'bid', 0)),
                    'ask': float(getattr(quote_data, 'ask', 0)),
                    'last': float(getattr(quote_data, 'last', 0)),
                    'volume': float(getattr(quote_data, 'volume', 0)),
                    'timestamp': datetime.utcnow(),
                    'unofficial_api': True
                }
                
            except Exception as e:
                logger.error(f"Failed to get DEGIRO quote: {e}")
                return {}
        
        return await _get_quote()
    
    async def _get_product_id(self, symbol: str) -> Optional[str]:
        """Get DEGIRO product ID for symbol"""
        try:
            # Search for product
            search_results = await self._rate_limited_call(
                self._client.search_products,
                query=symbol
            )
            
            # Find exact match
            for product in search_results:
                if hasattr(product, 'symbol') and product.symbol == symbol:
                    return str(product.id)
                elif isinstance(product, dict) and product.get('symbol') == symbol:
                    return str(product['id'])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get product ID for {symbol}: {e}")
            return None
    
    # Streaming methods (not recommended for unofficial API)
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """⚠️  Not recommended for DEGIRO unofficial API ⚠️"""
        logger.warning("Streaming not recommended for unofficial DEGIRO API")
        raise NotImplementedError("DEGIRO unofficial API does not support reliable streaming")
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """⚠️  Not recommended for DEGIRO unofficial API ⚠️"""
        logger.warning("Streaming not recommended for unofficial DEGIRO API")
        raise NotImplementedError("DEGIRO unofficial API does not support reliable streaming")
    
    def __repr__(self):
        return f"<DegiroBroker(broker='DEGIRO', connected={self.is_connected}, risk=CRITICAL)>"