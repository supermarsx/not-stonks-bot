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