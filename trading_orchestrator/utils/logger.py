"""
@file logger.py
@brief Matrix-Themed Logging Utility for Day Trading Orchestrator

@details
This module provides structured logging with Matrix-style formatting and
comprehensive trading event tracking. It enhances standard Python logging
with trading-specific features, visual styling, and structured data capture.

Key Features:
- Matrix-themed log formatting with emoji indicators
- Structured logging for trading events
- Multiple output destinations (file, console, network)
- Log rotation and archival management
- Trading-specific event categorization
- Performance and timing tracking
- Security and audit logging

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Logging configuration affects system performance. Excessive logging
can impact trading execution speed and system resources.

@note
This module provides enhanced logging capabilities:

@see LogLevel for logging level definitions
@see TradingEventType for trading event categorization
@see TradingLogRecord for structured log data

@par Log Categories:
- Trading Events: Orders, trades, positions, P&L
- System Events: Service start/stop, errors, warnings
- Risk Events: Limit violations, circuit breakers, compliance
- Market Events: Data updates, price changes, volatility
- Security Events: Authentication, authorization, access logs
- Performance Events: Latency, throughput, resource usage

@par Log Destinations:
- Console: Real-time monitoring with color coding
- Files: Persistent storage with rotation
- Network: Centralized logging servers
- Database: Structured query capability
"""

import logging
import logging.handlers