"""
Trading 212 API Integration

Official API with beta limitations and rate limiting requirements.
Based on research in research/trading212/

Key Features:
- Official Public API (v0) - in beta phase
- Practice (Paper) and Live environments
- Rate limiting per endpoint with explicit headers
- Limited order types in Live (Market only)
- No official streaming market data
- HTTP Basic Authentication
- Scoped API keys

Limitations:
- Practice: Full order types (Limit, Market, Stop, Stop-Limit)
- Live (Beta): Only Market orders
- No WebSocket streaming for market data
- Rate limits per endpoint (often restrictive)

Reference: docs.trading212.com
"""

import asyncio
import base64
import time
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
import aiohttp
import json

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from loguru import logger


class Trading212Broker(BaseBroker):
    """
    Trading 212 Public API Implementation
    
    Official API in beta phase with environment restrictions.
    Practice mode supports all order types, Live only Market orders.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Extract configuration
        self.username = config.api_key  # API key as username
        self.password = config.api_secret  # API secret as password
        
        # Determine base URL based on environment
        if config.is_paper:
            self.base_url = "https://demo.trading212.com/api/v0"
            self.environment = "Practice"
        else:
            self.base_url = "https://live.trading212.com/api/v0"
            self.environment = "Live"
        
        # Rate limiting state
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Set up auth header
        credentials = f"{self.username}:{self.password}"
        auth_bytes = base64.b64encode(credentials.encode()).decode()
        self.auth_header = f"Basic {auth_bytes}"
        
        logger.info(f"Trading212 broker initialized ({self.environment})")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'Authorization': self.auth_header}
            )
        return self.session
    
    def _parse_rate_limit_headers(self, headers) -> Dict[str, Any]:
        """Parse rate limit headers from response"""
        def get_header(name, default=None):
            return headers.get(name.lower().replace('_', '-'), default)
        
        return {
            'limit': int(get_header('x-ratelimit-limit', 1000)),
            'period': int(get_header('x-ratelimit-period', 60)),
            'remaining': int(get_header('x-ratelimit-remaining', 1000)),
            'reset': int(get_header('x-ratelimit-reset', time.time() + 60)),
            'used': int(get_header('x-ratelimit-used', 0))
        }
    
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if we can make a request to this endpoint"""
        if endpoint not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[endpoint]
        current_time = time.time()
        
        # Reset period if needed
        if current_time >= limit_info['reset']:
            limit_info['remaining'] = limit_info['limit']
        
        return limit_info['remaining'] > 0
    
    def _update_rate_limit(self, endpoint: str, headers) -> None:
        """Update rate limit info after API call"""
        limit_info = self._parse_rate_limit_headers(headers)
        self.rate_limits[endpoint] = limit_info
        
        # Log rate limit usage
        if limit_info['remaining'] < limit_info['limit'] * 0.1:
            logger.warning(f"Trading212 rate limit low: {limit_info['remaining']}/{limit_info['limit']}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Trading212 API"""
        
        # Check rate limit
        if not self._check_rate_limit(endpoint):
            # Find reset time and wait
            limit_info = self.rate_limits[endpoint]
            wait_time = max(0, limit_info['reset'] - time.time() + 1)
            logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.request(
                method=method,
                url=url,
                json=data if data else None,
                params=params
            ) as response:
                
                # Update rate limit info
                self._update_rate_limit(endpoint, response.headers)
                
                # Handle errors
                if response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, data, params)
                
                if response.status == 401:
                    raise ValueError("Invalid API credentials")
                
                if response.status == 403:
                    raise ValueError("Insufficient API scope permissions")
                
                response.raise_for_status()
                
                # Parse JSON response
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    text = await response.text()
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"text": text}
                
        except Exception as e:
            logger.error(f"Trading212 API error ({method} {endpoint}): {e}")
            raise
    
    async def connect(self) -> bool:
        """Connect to Trading212 API"""
        async with self._connection_lock:
            try:
                logger.info(f"Connecting to Trading212 ({self.environment})...")
                
                # Test connection with account info request
                account_info = await self._make_request("GET", "/equity/account/info")
                
                self.is_connected = True
                logger.success(f"Connected to Trading212 ({self.environment})")
                return True
                
            except Exception as e:
                logger.error(f"Trading212 connection failed: {e}")
                self.is_connected = False
                return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Trading212"""
        async with self._connection_lock:
            try:
                if self.session:
                    await self.session.close()
                    self.session = None
                
                self.is_connected = False
                logger.info("Disconnected from Trading212")
                return True
                
            except Exception as e:
                logger.error(f"Disconnect failed: {e}")
                return False
    
    async def is_connection_alive(self) -> bool:
        """Check if Trading212 connection is alive"""
        try:
            if not self.is_connected:
                return False
            
            # Try to get account info
            await self._make_request("GET", "/equity/account/info")
            return True
            
        except Exception:
            return False
    
    async def get_account(self) -> AccountInfo:
        """Get Trading212 account information"""
        try:
            # Get cash and info in parallel
            cash_data = await self._make_request("GET", "/equity/account/cash")
            info_data = await self._make_request("GET", "/equity/account/info")
            
            return AccountInfo(
                account_id=str(info_data.get('accountId', 'unknown')),
                broker_name="Trading212",
                currency=str(info_data.get('currency', 'GBP')),
                balance=float(cash_data.get('totalCash', 0)),
                available_balance=float(cash_data.get('freeCash', 0)),
                equity=float(cash_data.get('totalCash', 0)),  # Simplified
                buying_power=float(cash_data.get('freeCash', 0)),
                margin_used=0.0,  # Trading212 is typically cash account
                margin_available=float(cash_data.get('freeCash', 0)),
                is_pattern_day_trader=False,
                metadata={
                    'environment': self.environment,
                    'beta_limitations': True if not self.config.is_paper else False
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get Trading212 account: {e}")
            raise
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get Trading212 positions"""
        try:
            portfolio_data = await self._make_request("GET", "/equity/portfolio")
            
            positions = []
            for position in portfolio_data:
                # Skip zero positions
                if float(position.get('quantity', 0)) <= 0:
                    continue
                
                positions.append(PositionInfo(
                    symbol=str(position.get('ticker', '')),
                    broker_name="Trading212",
                    side="long",  # Trading212 doesn't support short positions
                    quantity=float(position.get('quantity', 0)),
                    avg_entry_price=float(position.get('averagePrice', 0)),
                    current_price=float(position.get('price', 0)),
                    market_value=float(position.get('quantity', 0)) * float(position.get('price', 0)),
                    unrealized_pnl=float(position.get('pl', 0)),
                    unrealized_pnl_percent=float(position.get('plPct', 0)),
                    cost_basis=float(position.get('quantity', 0)) * float(position.get('averagePrice', 0)),
                    metadata={
                        'environment': self.environment,
                        'equity_type': 'Invest'  # Assuming Invest account
                    }
                ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get Trading212 positions: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        try:
            # Trading212 uses ticker format like "AAPL_US_EQ"
            # Try different ticker formats
            ticker_formats = [
                f"{symbol}_US_EQ",  # US Equities
                f"{symbol}_UK_EQ",  # UK Equities  
                f"{symbol}_EU_EQ",  # EU Equities
                symbol
            ]
            
            for ticker in ticker_formats:
                try:
                    response = await self._make_request(
                        "GET", f"/equity/portfolio/{ticker}"
                    )
                    
                    if response:
                        position_data = response
                        return PositionInfo(
                            symbol=ticker,
                            broker_name="Trading212",
                            side="long",
                            quantity=float(position_data.get('quantity', 0)),
                            avg_entry_price=float(position_data.get('averagePrice', 0)),
                            current_price=float(position_data.get('price', 0)),
                            market_value=float(position_data.get('quantity', 0)) * float(position_data.get('price', 0)),
                            unrealized_pnl=float(position_data.get('pl', 0)),
                            metadata={'ticker_format': ticker}
                        )
                        
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Trading212 position for {symbol}: {e}")
            return None
    
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
        """Place Trading212 order (limited in Live mode)"""
        
        # Check if Live mode has beta limitations
        if not self.config.is_paper:
            if order_type.lower() != 'market':
                logger.warning(f"Trading212 Live mode (beta) only supports market orders, got {order_type}")
                logger.warning("Falling back to market order")
                order_type = 'market'
        
        try:
            # Prepare order data
            order_data = {
                "tickerSymbol": symbol,
                "quantity": abs(quantity),  # Trading212 uses positive quantities
                "side": side.upper(),  # BUY or SELL
            }
            
            # Add order type specific parameters
            if order_type.lower() == 'market':
                # Market orders support extended hours
                if extended_hours:
                    order_data["extendedHours"] = True
            
            elif order_type.lower() == 'limit':
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                order_data["limitPrice"] = limit_price
                order_data["timeValidity"] = time_in_force.upper()  # DAY or GOOD_TILL_CANCEL
            
            elif order_type.lower() in ['stop', 'stop_limit']:
                if not stop_price:
                    raise ValueError("Stop price required for stop orders")
                order_data["stopPrice"] = stop_price
                
                if order_type.lower() == 'stop_limit' and limit_price:
                    order_data["limitPrice"] = limit_price
                
                order_data["timeValidity"] = time_in_force.upper()
            
            # Place order via appropriate endpoint
            endpoint_map = {
                'market': '/equity/orders/market',
                'limit': '/equity/orders/limit',
                'stop': '/equity/orders/stop',
                'stop_limit': '/equity/orders/stop_limit'
            }
            
            endpoint = endpoint_map.get(order_type.lower())
            if not endpoint:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            response = await self._make_request("POST", endpoint, order_data)
            
            # Extract order ID
            order_id = str(response.get('orderId', ''))
            
            return OrderInfo(
                order_id=order_id,
                broker_name="Trading212",
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
                    'environment': self.environment,
                    'beta_limitation': not self.config.is_paper and order_type.lower() == 'market'
                }
            )
            
        except Exception as e:
            logger.error(f"Trading212 order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Trading212 order"""
        try:
            await self._make_request("DELETE", f"/equity/orders/{order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel Trading212 order {order_id}: {e}")
            return False
    
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get Trading212 orders"""
        try:
            orders_data = await self._make_request("GET", "/equity/orders")
            
            orders = []
            for order_data in orders_data:
                # Filter by status if specified
                if status and order_data.get('status') != status:
                    continue
                
                orders.append(OrderInfo(
                    order_id=str(order_data.get('id', '')),
                    broker_name="Trading212",
                    symbol=str(order_data.get('tickerSymbol', '')),
                    order_type=str(order_data.get('type', '').lower()),
                    side=str(order_data.get('side', '').lower()),
                    quantity=float(order_data.get('quantity', 0)),
                    filled_quantity=float(order_data.get('quantity', 0)) - float(order_data.get('remainingQuantity', 0)),
                    status=str(order_data.get('status', 'pending')),
                    limit_price=float(order_data.get('limitPrice', 0)) or None,
                    submitted_at=datetime.fromisoformat(order_data.get('dateCreated', '').replace('Z', '+00:00')),
                    metadata={'environment': self.environment}
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get Trading212 orders: {e}")
            return []
    
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific Trading212 order"""
        try:
            order_data = await self._make_request("GET", f"/equity/orders/{order_id}")
            
            return OrderInfo(
                order_id=str(order_data.get('id', '')),
                broker_name="Trading212",
                symbol=str(order_data.get('tickerSymbol', '')),
                order_type=str(order_data.get('type', '').lower()),
                side=str(order_data.get('side', '').lower()),
                quantity=float(order_data.get('quantity', 0)),
                filled_quantity=float(order_data.get('quantity', 0)) - float(order_data.get('remainingQuantity', 0)),
                status=str(order_data.get('status', 'pending')),
                limit_price=float(order_data.get('limitPrice', 0)) or None,
                submitted_at=datetime.fromisoformat(order_data.get('dateCreated', '').replace('Z', '+00:00')),
                metadata={'environment': self.environment}
            )
            
        except Exception as e:
            logger.error(f"Failed to get Trading212 order {order_id}: {e}")
            return None
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """Get Trading212 market data (limited)"""
        
        # Note: Trading212 has limited historical data via API
        # This is a placeholder implementation
        
        logger.warning("Trading212 historical data via API is limited")
        return []
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get Trading212 quote (limited)"""
        
        # Note: Trading212 doesn't provide real-time quotes via public API
        # This is a placeholder implementation
        
        logger.warning("Trading212 real-time quotes via API not available")
        return {}
    
    # Note: Trading212 does not provide official streaming
    # Community solutions exist but violate terms
    
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Trading212 does not support official quote streaming"""
        logger.warning("Trading212 official API does not support streaming")
        raise NotImplementedError("Trading212 does not provide official streaming")
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Trading212 does not support official trade streaming"""
        logger.warning("Trading212 official API does not support streaming")
        raise NotImplementedError("Trading212 does not provide official streaming")
    
    def __repr__(self):
        return f"<Trading212Broker(broker='Trading212', connected={self.is_connected}, env={self.environment})>"