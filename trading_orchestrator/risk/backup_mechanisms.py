"""
Backup Trading Mechanisms Module

This module implements comprehensive backup trading mechanisms for disaster recovery,
including failover broker routing, backup execution venues, and disaster recovery procedures.

Features:
- Failover broker routing
- Backup execution venues
- Disaster recovery procedures
- Connection health monitoring
- Automatic failover detection
- Trading session continuity
- Data synchronization
- Recovery testing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import json
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

from trading_orchestrator.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class BrokerStatus(Enum):
    """Broker connection status."""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class ExecutionVenueType(Enum):
    """Types of execution venues."""
    PRIMARY_BROKER = "primary_broker"
    SECONDARY_BROKER = "secondary_broker"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    ATS = "ats"
    INTERNAL_VENUE = "internal_venue"


class FailoverTrigger(Enum):
    """Types of failover triggers."""
    CONNECTION_LOST = "connection_lost"
    RESPONSE_TIMEOUT = "response_timeout"
    HIGH_ERROR_RATE = "high_error_rate"
    LATENCY_THRESHOLD = "latency_threshold"
    ORDER_REJECT_RATE = "order_reject_rate"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED = "scheduled"


@dataclass
class BrokerConnection:
    """Broker connection configuration."""
    broker_id: str
    name: str
    connection_string: str
    status: BrokerStatus
    priority: int  # 1 = highest priority
    venue_type: ExecutionVenueType
    capabilities: Set[str] = field(default_factory=set)
    max_orders_per_second: Optional[int] = None
    supported_instruments: Set[str] = field(default_factory=set)
    connection_health: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: Optional[datetime] = None
    failover_candidate: bool = True
    active: bool = True


@dataclass
class FailoverRule:
    """Failover rule configuration."""
    rule_id: str
    name: str
    trigger: FailoverTrigger
    conditions: Dict[str, Any]
    target_broker: str
    backup_broker: str
    enabled: bool = True
    max_failover_time_seconds: int = 30
    recovery_timeout_minutes: int = 15


@dataclass
class TradingSession:
    """Trading session for continuity tracking."""
    session_id: str
    primary_broker: str
    current_broker: str
    started_at: datetime
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    orders_in_flight: Dict[str, str] = field(default_factory=dict)  # order_id -> broker
    connection_failures: int = 0
    failover_count: int = 0
    active: bool = True


@dataclass
class DisasterRecoveryEvent:
    """Disaster recovery event tracking."""
    event_id: str
    event_type: str
    trigger: FailoverTrigger
    primary_broker: str
    backup_broker: str
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    orders_recovered: int = 0
    orders_lost: int = 0
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class BrokerConnectionModel(Base):
    """Database model for broker connections."""
    __tablename__ = "broker_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    broker_id = Column(String(100), nullable=False, unique=True)
    name = Column(String(200), nullable=False)
    connection_string = Column(Text, nullable=False)
    status = Column(String(20), nullable=False)
    priority = Column(Integer, nullable=False)
    venue_type = Column(String(30), nullable=False)
    capabilities = Column(Text, nullable=True)  # JSON data
    max_orders_per_second = Column(Integer, nullable=True)
    supported_instruments = Column(Text, nullable=True)  # JSON data
    connection_health = Column(Text, nullable=True)  # JSON data
    last_heartbeat = Column(DateTime, nullable=True)
    failover_candidate = Column(Boolean, default=True, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class FailoverEvent(Base):
    """Database model for failover events."""
    __tablename__ = "failover_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(100), nullable=False, unique=True)
    rule_id = Column(String(100), nullable=False)
    trigger = Column(String(30), nullable=False)
    from_broker = Column(String(100), nullable=False)
    to_broker = Column(String(100), nullable=False)
    orders_affected = Column(Integer, default=0)
    failover_duration_seconds = Column(Integer, nullable=True)
    success = Column(Boolean, default=False, nullable=False)
    error_message = Column(Text, nullable=True)
    triggered_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)


class TradingSessionModel(Base):
    """Database model for trading sessions."""
    __tablename__ = "trading_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, unique=True)
    primary_broker = Column(String(100), nullable=False)
    current_broker = Column(String(100), nullable=False)
    started_at = Column(DateTime, nullable=False)
    last_heartbeat = Column(DateTime, default=func.now())
    orders_in_flight = Column(Text, nullable=True)  # JSON data
    connection_failures = Column(Integer, default=0)
    failover_count = Column(Integer, default=0)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class BackupTradingManager:
    """Manager for backup trading mechanisms."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._brokers: Dict[str, BrokerConnection] = {}
        self._failover_rules: Dict[str, FailoverRule] = {}
        self._trading_sessions: Dict[str, TradingSession] = {}
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._recovery_monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the backup trading manager."""
        logger.info("Initializing backup trading manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Load broker configurations
            await self._load_broker_configurations()
            
            # Load failover rules
            await self._load_failover_rules()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("Backup trading manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backup trading manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for backup trading."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _load_broker_configurations(self) -> None:
        """Load broker configurations."""
        # This would load from configuration/database
        await self._initialize_default_brokers()
    
    async def _initialize_default_brokers(self) -> None:
        """Initialize default broker configurations."""
        brokers = [
            BrokerConnection(
                broker_id="primary_broker",
                name="Primary Broker - Goldman Sachs",
                connection_string="gs://production.primary",
                status=BrokerStatus.ACTIVE,
                priority=1,
                venue_type=ExecutionVenueType.PRIMARY_BROKER,
                capabilities={"equity", "options", "futures", "forex"},
                max_orders_per_second=1000,
                supported_instruments={"stocks", "options", "futures", "fx"},
                failover_candidate=False
            ),
            BrokerConnection(
                broker_id="backup_broker_1",
                name="Backup Broker 1 - JPMorgan",
                connection_string="jpm://backup.primary",
                status=BrokerStatus.STANDBY,
                priority=2,
                venue_type=ExecutionVenueType.SECONDARY_BROKER,
                capabilities={"equity", "options", "futures"},
                max_orders_per_second=500,
                supported_instruments={"stocks", "options", "futures"},
                failover_candidate=True
            ),
            BrokerConnection(
                broker_id="backup_broker_2",
                name="Backup Broker 2 - Morgan Stanley",
                connection_string="ms://backup.secondary",
                status=BrokerStatus.STANDBY,
                priority=3,
                venue_type=ExecutionVenueType.SECONDARY_BROKER,
                capabilities={"equity", "options"},
                max_orders_per_second=300,
                supported_instruments={"stocks", "options"},
                failover_candidate=True
            ),
            BrokerConnection(
                broker_id="dark_pool",
                name="Internal Dark Pool",
                connection_string="internal://dark_pool",
                status=BrokerStatus.STANDBY,
                priority=4,
                venue_type=ExecutionVenueType.DARK_POOL,
                capabilities={"equity"},
                max_orders_per_second=200,
                supported_instruments={"stocks"},
                failover_candidate=True
            )
        ]
        
        for broker in brokers:
            self._brokers[broker.broker_id] = broker
        
        logger.info("Loaded default broker configurations")
    
    async def _load_failover_rules(self) -> None:
        """Load failover rules."""
        await self._initialize_default_failover_rules()
    
    async def _initialize_default_failover_rules(self) -> None:
        """Initialize default failover rules."""
        rules = [
            FailoverRule(
                rule_id="CONNECTION_TIMEOUT",
                name="Connection Timeout Failover",
                trigger=FailoverTrigger.CONNECTION_LOST,
                conditions={"timeout_seconds": 30},
                target_broker="primary_broker",
                backup_broker="backup_broker_1"
            ),
            FailoverRule(
                rule_id="HIGH_LATENCY",
                name="High Latency Failover",
                trigger=FailoverTrigger.LATENCY_THRESHOLD,
                conditions={"max_latency_ms": 5000},
                target_broker="primary_broker",
                backup_broker="backup_broker_1"
            ),
            FailoverRule(
                rule_id="ERROR_RATE",
                name="High Error Rate Failover",
                trigger=FailoverTrigger.HIGH_ERROR_RATE,
                conditions={"max_error_rate": 0.05},  # 5%
                target_broker="primary_broker",
                backup_broker="backup_broker_1"
            ),
            FailoverRule(
                rule_id="ORDER_REJECT_RATE",
                name="Order Reject Rate Failover",
                trigger=FailoverTrigger.ORDER_REJECT_RATE,
                conditions={"max_reject_rate": 0.10},  # 10%
                target_broker="primary_broker",
                backup_broker="backup_broker_1"
            )
        ]
        
        for rule in rules:
            self._failover_rules[rule.rule_id] = rule
        
        logger.info("Loaded default failover rules")
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Start broker health monitoring
        self._health_monitor_task = asyncio.create_task(self._broker_health_monitor())
        
        # Start recovery monitoring
        self._recovery_monitor_task = asyncio.create_task(self._recovery_monitor())
        
        logger.info("Started backup trading monitoring tasks")
    
    async def _broker_health_monitor(self) -> None:
        """Monitor broker connection health."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for broker_id, broker in self._brokers.items():
                    if broker.active:
                        health_status = await self._check_broker_health(broker)
                        broker.connection_health = health_status
                        broker.last_heartbeat = datetime.utcnow()
                        
                        # Check if failover should be triggered
                        if await self._should_trigger_failover(broker, health_status):
                            await self._trigger_failover(broker_id, FailoverTrigger.CONNECTION_LOST)
                        
            except Exception as e:
                logger.error(f"Error in broker health monitor: {e}")
    
    async def _check_broker_health(self, broker: BrokerConnection) -> Dict[str, Any]:
        """Check the health of a broker connection."""
        try:
            # This would perform actual health checks
            # For now, we'll simulate health metrics
            import random
            
            return {
                "connected": random.choice([True, True, True, False]),  # 75% uptime simulation
                "latency_ms": random.randint(50, 200),
                "error_rate": random.uniform(0.0, 0.02),  # 0-2% error rate
                "throughput_orders_per_second": random.randint(50, 200),
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking health for broker {broker.broker_id}: {e}")
            return {"connected": False, "error": str(e)}
    
    async def _should_trigger_failover(self, broker: BrokerConnection, 
                                     health_status: Dict[str, Any]) -> bool:
        """Determine if failover should be triggered for a broker."""
        try:
            # Check connection status
            if not health_status.get("connected", True):
                return True
            
            # Check latency
            latency = health_status.get("latency_ms", 0)
            if latency > 5000:  # 5 second latency threshold
                return True
            
            # Check error rate
            error_rate = health_status.get("error_rate", 0)
            if error_rate > 0.05:  # 5% error rate threshold
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating failover criteria for {broker.broker_id}: {e}")
            return False
    
    async def _trigger_failover(self, failed_broker_id: str, trigger: FailoverTrigger,
                              initiated_by: str = "system") -> bool:
        """Trigger failover to backup broker."""
        try:
            failed_broker = self._brokers.get(failed_broker_id)
            if not failed_broker:
                logger.warning(f"Failed broker not found: {failed_broker_id}")
                return False
            
            # Find backup broker
            backup_broker = await self._select_backup_broker(failed_broker_id)
            if not backup_broker:
                logger.error(f"No backup broker available for {failed_broker_id}")
                return False
            
            logger.warning(f"Initiating failover from {failed_broker_id} to {backup_broker.broker_id}")
            
            # Generate event ID
            event_id = f"failover_{failed_broker_id}_{int(datetime.utcnow().timestamp())}"
            
            # Create disaster recovery event
            recovery_event = DisasterRecoveryEvent(
                event_id=event_id,
                event_type="failover",
                trigger=trigger,
                primary_broker=failed_broker_id,
                backup_broker=backup_broker.broker_id,
                triggered_at=datetime.utcnow()
            )
            
            # Execute failover
            success = await self._execute_failover(recovery_event)
            
            if success:
                # Update broker statuses
                failed_broker.status = BrokerStatus.FAILED
                backup_broker.status = BrokerStatus.ACTIVE
                backup_broker.priority = 1  # Promote to primary
                
                # Record failover event
                await self._record_failover_event(event_id, failed_broker_id, 
                                                backup_broker.broker_id, success)
                
                logger.info(f"Failover completed: {failed_broker_id} -> {backup_broker.broker_id}")
            else:
                logger.error(f"Failover failed: {failed_broker_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error triggering failover for {failed_broker_id}: {e}")
            return False
    
    async def _select_backup_broker(self, failed_broker_id: str) -> Optional[BrokerConnection]:
        """Select the best backup broker."""
        try:
            # Get candidates with failover capability
            candidates = [
                broker for broker in self._brokers.values()
                if (broker.active and 
                    broker.broker_id != failed_broker_id and 
                    broker.failover_candidate and
                    broker.status in [BrokerStatus.STANDBY, BrokerStatus.DEGRADED])
            ]
            
            if not candidates:
                return None
            
            # Sort by priority (lower number = higher priority)
            candidates.sort(key=lambda b: b.priority)
            
            # Check health of candidates
            for candidate in candidates:
                health = await self._check_broker_health(candidate)
                if health.get("connected", False):
                    return candidate
            
            return candidates[0] if candidates else None
            
        except Exception as e:
            logger.error(f"Error selecting backup broker for {failed_broker_id}: {e}")
            return None
    
    async def _execute_failover(self, event: DisasterRecoveryEvent) -> bool:
        """Execute the failover process."""
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Cancel orders on failed broker
            orders_to_recover = await self._recover_pending_orders(event.primary_broker)
            
            # Step 2: Switch to backup broker
            switch_success = await self._switch_to_broker(event.backup_broker)
            if not switch_success:
                return False
            
            # Step 3: Re-submit recovered orders
            recovered_count = await self._resubmit_orders(orders_to_recover, event.backup_broker)
            
            # Step 4: Update trading sessions
            await self._update_trading_sessions(event.primary_broker, event.backup_broker)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Update event
            event.completed_at = end_time
            event.orders_recovered = recovered_count
            event.success = True
            event.details = {
                "orders_to_recover": len(orders_to_recover),
                "recovered_count": recovered_count,
                "duration_seconds": duration
            }
            
            logger.info(f"Failover executed successfully in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error executing failover: {e}")
            return False
    
    async def _recover_pending_orders(self, failed_broker_id: str) -> List[Dict[str, Any]]:
        """Recover orders from failed broker."""
        try:
            # This would query the order management system
            # For now, we'll return empty list (simulation)
            logger.info(f"Recovering pending orders from {failed_broker_id}")
            
            # In production, this would:
            # 1. Query all active orders on failed broker
            # 2. Extract order details (symbol, side, quantity, price, etc.)
            # 3. Return list of orders to recreate
            
            return []
            
        except Exception as e:
            logger.error(f"Error recovering orders from {failed_broker_id}: {e}")
            return []
    
    async def _switch_to_broker(self, broker_id: str) -> bool:
        """Switch trading operations to backup broker."""
        try:
            broker = self._brokers.get(broker_id)
            if not broker:
                return False
            
            # This would update internal routing configuration
            # to send new orders to the backup broker
            
            logger.info(f"Switched to backup broker: {broker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to broker {broker_id}: {e}")
            return False
    
    async def _resubmit_orders(self, orders: List[Dict[str, Any]], broker_id: str) -> int:
        """Re-submit recovered orders to backup broker."""
        try:
            recovered_count = 0
            
            for order in orders:
                # This would submit order to backup broker
                logger.info(f"Would resubmit order {order.get('order_id', 'unknown')} to {broker_id}")
                recovered_count += 1
            
            return recovered_count
            
        except Exception as e:
            logger.error(f"Error resubmitting orders to {broker_id}: {e}")
            return 0
    
    async def _update_trading_sessions(self, old_broker: str, new_broker: str) -> None:
        """Update trading sessions after failover."""
        try:
            # Update all active trading sessions
            for session_id, session in self._trading_sessions.items():
                if session.current_broker == old_broker:
                    session.current_broker = new_broker
                    session.failover_count += 1
                    session.last_heartbeat = datetime.utcnow()
                    
                    # Update database
                    await self._save_trading_session(session)
            
            logger.info(f"Updated {len(self._trading_sessions)} trading sessions")
            
        except Exception as e:
            logger.error(f"Error updating trading sessions: {e}")
    
    async def _save_trading_session(self, session: TradingSession) -> None:
        """Save trading session to database."""
        async with self.db_manager.get_session() as session_db:
            # Check if session exists
            existing = await session_db.execute(
                TradingSessionModel.__table__.select()
                .where(TradingSessionModel.session_id == session.session_id)
            )
            
            if existing.fetchone():
                # Update existing session
                await session_db.execute(
                    TradingSessionModel.__table__.update()
                    .where(TradingSessionModel.session_id == session.session_id)
                    .values(
                        current_broker=session.current_broker,
                        last_heartbeat=session.last_heartbeat,
                        orders_in_flight=json.dumps(session.orders_in_flight),
                        connection_failures=session.connection_failures,
                        failover_count=session.failover_count
                    )
                )
            else:
                # Create new session
                session_record = TradingSessionModel(
                    session_id=session.session_id,
                    primary_broker=session.primary_broker,
                    current_broker=session.current_broker,
                    started_at=session.started_at,
                    last_heartbeat=session.last_heartbeat,
                    orders_in_flight=json.dumps(session.orders_in_flight),
                    connection_failures=session.connection_failures,
                    failover_count=session.failover_count
                )
                session_db.add(session_record)
            
            await session_db.commit()
    
    async def _record_failover_event(self, event_id: str, from_broker: str, 
                                   to_broker: str, success: bool) -> None:
        """Record failover event in database."""
        async with self.db_manager.get_session() as session:
            event_record = FailoverEvent(
                event_id=event_id,
                rule_id="SYSTEM_TRIGGER",
                trigger="system_monitor",
                from_broker=from_broker,
                to_broker=to_broker,
                success=success,
                triggered_at=datetime.utcnow()
            )
            session.add(event_record)
            await session.commit()
    
    async def _recovery_monitor(self) -> None:
        """Monitor for broker recovery."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for broker_id, broker in self._brokers.items():
                    if broker.status == BrokerStatus.FAILED:
                        # Check if broker has recovered
                        health = await self._check_broker_health(broker)
                        if health.get("connected", False):
                            logger.info(f"Broker {broker_id} has recovered")
                            # Consider promoting back to higher priority
                            
            except Exception as e:
                logger.error(f"Error in recovery monitor: {e}")
    
    async def create_trading_session(self, session_id: str, primary_broker: str) -> bool:
        """Create a new trading session."""
        try:
            session = TradingSession(
                session_id=session_id,
                primary_broker=primary_broker,
                current_broker=primary_broker,
                started_at=datetime.utcnow()
            )
            
            self._trading_sessions[session_id] = session
            await self._save_trading_session(session)
            
            logger.info(f"Created trading session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating trading session {session_id}: {e}")
            return False
    
    async def manual_failover(self, target_broker: str, reason: str) -> bool:
        """Manually trigger failover to specific broker."""
        try:
            logger.critical(f"Manual failover initiated: {reason}")
            
            # Find currently active broker
            active_brokers = [b for b in self._brokers.values() if b.status == BrokerStatus.ACTIVE]
            if not active_brokers:
                logger.error("No active broker found for manual failover")
                return False
            
            current_broker = active_brokers[0]
            
            return await self._trigger_failover(current_broker.broker_id, 
                                              FailoverTrigger.MANUAL_TRIGGER, reason)
            
        except Exception as e:
            logger.error(f"Error in manual failover: {e}")
            return False
    
    async def test_failover(self) -> Dict[str, Any]:
        """Test failover mechanism (non-disruptive)."""
        try:
            logger.info("Starting failover test")
            
            test_results = {
                "test_started": datetime.utcnow().isoformat(),
                "brokers_tested": [],
                "success": True,
                "errors": []
            }
            
            for broker_id, broker in self._brokers.items():
                if broker.active:
                    # Test broker connectivity
                    health = await self._check_broker_health(broker)
                    test_results["brokers_tested"].append({
                        "broker_id": broker_id,
                        "status": broker.status.value,
                        "health": health
                    })
                    
                    if not health.get("connected", False):
                        test_results["success"] = False
                        test_results["errors"].append(f"Broker {broker_id} connectivity failed")
            
            logger.info("Failover test completed")
            return test_results
            
        except Exception as e:
            logger.error(f"Error in failover test: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """Get backup trading system status."""
        try:
            active_brokers = [b for b in self._brokers.values() if b.status == BrokerStatus.ACTIVE]
            standby_brokers = [b for b in self._brokers.values() if b.status == BrokerStatus.STANDBY]
            failed_brokers = [b for b in self._brokers.values() if b.status == BrokerStatus.FAILED]
            
            return {
                "total_brokers": len(self._brokers),
                "active_brokers": len(active_brokers),
                "standby_brokers": len(standby_brokers),
                "failed_brokers": len(failed_brokers),
                "active_trading_sessions": len(self._trading_sessions),
                "failover_rules": len(self._failover_rules),
                "broker_details": {
                    broker_id: {
                        "status": broker.status.value,
                        "priority": broker.priority,
                        "venue_type": broker.venue_type.value,
                        "last_heartbeat": broker.last_heartbeat.isoformat() if broker.last_heartbeat else None,
                        "health": broker.connection_health
                    }
                    for broker_id, broker in self._brokers.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return {}


# Factory function for creating backup trading manager
async def create_backup_trading_manager(db_manager: DatabaseManager) -> BackupTradingManager:
    """Create a configured backup trading manager."""
    manager = BackupTradingManager(db_manager)
    await manager.initialize()
    
    return manager


# Predefined broker configurations for different scenarios
PRODUCTION_BROKER_CONFIG = {
    "primary": BrokerConnection(
        broker_id="production_primary",
        name="Production Primary Broker",
        connection_string="primary://prod.broker1.com",
        status=BrokerStatus.ACTIVE,
        priority=1,
        venue_type=ExecutionVenueType.PRIMARY_BROKER,
        capabilities={"equity", "options", "futures"},
        max_orders_per_second=1000
    ),
    "backup1": BrokerConnection(
        broker_id="production_backup1",
        name="Production Backup Broker 1",
        connection_string="backup://prod.broker2.com",
        status=BrokerStatus.STANDBY,
        priority=2,
        venue_type=ExecutionVenueType.SECONDARY_BROKER,
        capabilities={"equity", "options"},
        max_orders_per_second=500
    )
}

TEST_BROKER_CONFIG = {
    "primary": BrokerConnection(
        broker_id="test_primary",
        name="Test Primary Broker",
        connection_string="test://sim.broker1.com",
        status=BrokerStatus.ACTIVE,
        priority=1,
        venue_type=ExecutionVenueType.PRIMARY_BROKER,
        capabilities={"equity"},
        max_orders_per_second=100
    )
}