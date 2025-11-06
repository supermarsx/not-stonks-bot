"""
Order Management System (OMS) - Main Package

Provides comprehensive order management including:
- Order routing across multiple brokers
- Order validation and compliance checks
- Execution monitoring and tracking
- Slippage tracking and analysis
- Position management and reconciliation
- Trade settlement system
- Performance analytics
"""

from .engine import OMSEngine
from .manager import OrderManager
from .router import OrderRouter
from .validator import OrderValidator
from .monitor import ExecutionMonitor
from .tracker import SlippageTracker, PerformanceAnalytics
from .manager import OrderManager
from .position_manager import (
    PositionManager,
    BrokerPosition,
    PositionDiscrepancy,
    PositionAlert,
    PositionReconciliationStatus,
    PositionAlert as PositionAlertType
)
from .settlement import (
    SettlementProcessor,
    TradeSettlement,
    SettlementBatch,
    SettlementStatus,
    SettlementType
)

__all__ = [
    # Core OMS
    'OMSEngine',
    'OrderRouter', 
    'OrderValidator',
    'ExecutionMonitor',
    'SlippageTracker',
    'PerformanceAnalytics',
    'OrderManager',
    
    # Position Management
    'PositionManager',
    'BrokerPosition',
    'PositionDiscrepancy',
    'PositionAlert',
    'PositionReconciliationStatus',
    'PositionAlertType',
    
    # Settlement Processing
    'SettlementProcessor',
    'TradeSettlement',
    'SettlementBatch',
    'SettlementStatus',
    'SettlementType'
]