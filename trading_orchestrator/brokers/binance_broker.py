"""
@file binance_broker.py
@brief Binance Exchange Broker Implementation

@details
This module provides a comprehensive implementation of the BaseBroker interface
for Binance cryptocurrency exchange. It integrates both REST API and WebSocket
real-time data streaming to provide a unified trading interface.

Key Features:
- REST API integration for account, orders, and market data
- WebSocket streaming for real-time quotes and trade updates
- Built-in rate limit compliance and error handling
- Testnet support for paper trading simulation
- Comprehensive position and account management

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note Based on research/binance/binance_api_analysis.md for API specifications

@warning 
- Requires valid Binance API credentials (API_KEY and API_SECRET)
- Testnet mode is enabled by default for safety
- Rate limits: 1200 requests/minute for REST, 5 connections for WebSocket

@see brokers.base.BaseBroker for interface specifications
@see config.binance.example.json for configuration examples
"""

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from binance import AsyncClient, BinanceSocketManager
from binance.enums import *
import asyncio
from loguru import logger


class BinanceBroker(BaseBroker):
    """
    @class BinanceBroker
    @brief Binance Exchange API Integration
    
    @details
    Implements comprehensive trading operations for Binance cryptocurrency exchange
    including spot trading, real-time data streaming, and account management.
    Supports both mainnet and testnet environments with automatic configuration.
    
    @par Key Features:
    - <b>REST API Operations</b>: Account info, order management, market data retrieval
    - <b>WebSocket Streaming</b>: Real-time quotes, trade updates, order book data
    - <b>Rate Limit Compliance</b>: Built-in handling of API rate limits
    - <b>Testnet Support</b>: Paper trading with testnet for safe experimentation
    - <b>Multi-Asset Support</b>: Trading across all Binance spot trading pairs
    
    @par Supported Operations:
    - Account balance and equity queries
    - Market and limit order placement
    - Position tracking and management
    - Historical and real-time market data
    - Real-time quote streaming
    - Order cancellation and modification
    
    @par Configuration:
    The broker requires a BrokerConfig with the following parameters:
    - api_key: Binance API key
    - api_secret: Binance API secret
    - testnet: Boolean flag for testnet mode (default: True)
    
    @note
    This implementation follows the unified broker interface defined in
    brokers.base.BaseBroker for consistent integration across the trading system.
    
    @warning
    Always use testnet mode for testing and development. Real API keys
    with mainnet access can execute actual trades with real funds.
    
    @par Usage Example:
    @code
    from brokers.binance_broker import BinanceBroker
    from brokers.base import BrokerConfig
    
    # Initialize with testnet credentials
    config = BrokerConfig(
        broker_name="binance",
        api_key="your_testnet_api_key",
        api_secret="your_testnet_api_secret",
        is_paper=True,
        config={"testnet": True}
    )
    
    broker = BinanceBroker(config)
    await broker.connect()
    
    # Get account information
    account = await broker.get_account()
    print(f"Balance: {account.balance} USD")
    
    # Place a market order
    order = await broker.place_order(
        symbol="BTCUSDT",
        side="buy",
        order_type="market",
        quantity=0.001
    )
    
    # Stream real-time quotes
    async for quote in broker.stream_quotes(["BTCUSDT"]):
        print(f"BTC/USDT: ${quote['last']}")
    
    await broker.disconnect()
    @endcode
    
    @see BaseBroker for interface definition
    @see BrokerConfig for configuration details
    @see research/binance/binance_api_analysis.md for API documentation
    """
    
    def __init__(self, config: BrokerConfig):
        """
        @brief Initialize Binance broker with configuration
        
        @param config BrokerConfig object containing API credentials and settings
        
        @details
        Sets up the Binance broker instance with provided configuration.
        Automatically determines whether to use testnet based on config.testnet flag.
        
        @throws ValueError if required configuration parameters are missing
        
        @par Example:
        @code
        config = BrokerConfig(
            broker_name="binance",
            api_key="your_api_key",
            api_secret="your_api_secret",
            is_paper=True,
            config={"testnet": True}
        )
        broker = BinanceBroker(config)
        @endcode
        
        @note The broker is not connected after initialization. Call connect() method.
        """
        super().__init__(config)
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None
        self.testnet = config.config.get("testnet", True) if config.config else True
        
        # Validate required configuration
        if not self.config.api_key or not self.config.api_secret:
            raise ValueError("Binance API key and secret are required")
    
    async def connect(self) -> bool:
        """
        @brief Establish connection to Binance API
        
        @return bool True if connection successful, False otherwise
        
        @throws ConnectionError if API credentials are invalid
        @throws asyncio.TimeoutError if connection timeout occurs
        
        @details
        Establishes a connection to the Binance API using provided credentials.
        Creates both REST client for API operations and WebSocket manager for
        real-time data streaming.
        
        @par Connection Process:
        1. Validate API credentials are present
        2. Create Binance AsyncClient (REST API)
        3. Test connection with ping() call
        4. Initialize BinanceSocketManager for WebSocket operations
        5. Set connection state to active
        
        @par Testnet vs Mainnet:
        - Testnet: Safe testing environment with test funds (default)
        - Mainnet: Real trading with actual funds
        
        @par Rate Limits:
        - REST API: 1200 requests per minute
        - WebSocket: 5 connections maximum
        
        @par Example:
        @code
        broker = BinanceBroker(config)
        if await broker.connect():
            print("Connected to Binance successfully")
            account = await broker.get_account()
        else:
            print("Failed to connect to Binance")
        @endcode
        
        @warning
        Always verify connection before attempting trading operations.
        Invalid credentials will cause connection failures.
        
        @note
        This method is thread-safe using asyncio.Lock to prevent
        concurrent connection attempts.
        """
        async with self._connection_lock:
            try:
                logger.info(f"Connecting to Binance ({'Testnet' if self.testnet else 'Mainnet'})...")
                
                # Validate credentials before connection
                if not self.config.api_key or not self.config.api_secret:
                    raise ValueError("API credentials not provided")
                
                # Create client with credentials
                self.client = await AsyncClient.create(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    testnet=self.testnet
                )
                
                # Test connection with ping
                await self.client.ping()
                
                # Initialize WebSocket manager
                self.bm = BinanceSocketManager(self.client)
                
                # Set connection state
                self.is_connected = True
                
                logger.success(f"Connected to Binance ({'Testnet' if self.testnet else 'Mainnet'})")
                logger.info(f"Rate limits: REST=1200/min, WS={self.bm.MAX_CONNECTS if hasattr(self.bm, 'MAX_CONNECTS') else 5}")
                
                return True
                
            except ValueError as e:
                logger.error(f"Invalid Binance credentials: {e}")
                self.is_connected = False
                return False
            except Exception as e:
                logger.error(f"Failed to connect to Binance: {e}")
                self.is_connected = False
                return False