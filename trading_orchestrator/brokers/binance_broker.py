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