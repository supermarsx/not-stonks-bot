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