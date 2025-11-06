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