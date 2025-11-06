# Order Management System

This module implements a comprehensive Order Management System (OMS) for the trading orchestrator.

Key Features:
- Order lifecycle management
- Order routing and execution
- Real-time order tracking
- Order validation and compliance
- Partial fill handling
- Order modification and cancellation
- Execution quality analysis
- Order audit trail

Architecture:
- Asynchronous order processing
- Modular order routing
- Support for multiple broker connections
- Real-time order status updates
- Comprehensive error handling

Order States:
1. PENDING: Order created but not submitted
2. SUBMITTED: Order sent to broker
3. ACKNOWLEDGED: Broker acknowledged receipt
4. PARTIAL_FILLED: Order partially executed
5. FILLED: Order fully executed
6. CANCELLED: Order cancelled by user or system
7. REJECTED: Order rejected by broker
8. EXPIRED: Order expired (if time-in-force specified)

Order Types Supported:
- Market Orders
- Limit Orders
- Stop Orders
- Stop-Limit Orders
- Trailing Stop Orders
- Iceberg Orders

Time-in-Force:
- Day
- Good Till Cancel (GTC)
- Immediate or Cancel (IOC)
- Fill or Kill (FOK)

Integration Points:
- Risk Management System for order validation
- Broker APIs for order execution
- Database for persistent storage
- Alert system for status updates
- Portfolio tracking for position updates
