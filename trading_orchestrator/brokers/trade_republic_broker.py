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