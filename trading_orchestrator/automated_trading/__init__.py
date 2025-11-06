"""
Automated Trading System

This module provides a comprehensive automated trading system that operates
perpetually during market hours with autonomous decision-making capabilities.

Components:
- Market hours detection and management
- Automated trading engine with perpetual operation
- Autonomous decision making with AI integration
- Continuous monitoring and risk management
- Comprehensive logging and reporting
"""

from .market_hours import MarketHoursManager, MarketSession, MarketType
from .automated_engine import AutomatedTradingEngine
from .autonomous_decisions import AutonomousDecisionEngine
from .continuous_monitor import ContinuousMonitoringSystem
from .risk_management import AutomatedRiskManager
from .config import AutomatedTradingConfig
from .logging_system import TradingLogger

__all__ = [
    "MarketHoursManager",
    "MarketSession", 
    "MarketType",
    "AutomatedTradingEngine",
    "AutonomousDecisionEngine",
    "ContinuousMonitoringSystem", 
    "AutomatedRiskManager",
    "AutomatedTradingConfig",
    "TradingLogger"
]