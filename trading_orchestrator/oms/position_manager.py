"""
Position Management System
Handles position reconciliation, monitoring, and synchronization across multiple brokers
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from collections import defaultdict

from loguru import logger

from trading.models import Position, PositionSide
from trading.database import get_db_session
from broker.base import BrokerClient


class PositionReconciliationStatus(Enum):
    """Position reconciliation status"""
    SYNCED = "synced"
    DISCREPANCY = "discrepancy"
    MISSING_BROKER = "missing_broker"
    MISSING_INTERNAL = "missing_internal"
    ERROR = "error"


class PositionAlert(Enum):
    """Position alert types"""
    HIGH_EXPOSURE = "high_exposure"
    CONCENTRATION_RISK = "concentration_risk"
    RECONCILIATION_FAILED = "reconciliation_failed"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    PNL_SPIKE = "pnl_spike"
    SYMBOL_CONCENTRATION = "symbol_concentration"


@dataclass
class BrokerPosition:
    """Position from a specific broker"""
    broker_name: str
    symbol: str
    quantity: Decimal
    average_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    market_value: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionDiscrepancy:
    """Represents a position discrepancy"""
    symbol: str
    broker_positions: Dict[str, BrokerPosition]
    internal_position: Optional[Position]
    total_broker_quantity: Decimal
    internal_quantity: Decimal
    quantity_difference: Decimal
    total_broker_value: Decimal
    internal_value: Decimal
    value_difference: Decimal
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    reconciliation_status: PositionReconciliationStatus
    notes: str = ""
    created_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PositionAlert:
    """Position-based alert"""
    alert_id: str
    alert_type: PositionAlert
    symbol: str
    severity: str
    message: str
    current_value: Any
    threshold_value: Any
    broker_name: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False


class PositionManager:
    """
    Position Management System
    
    Features:
    - Multi-broker position reconciliation
    - Position monitoring and alerts
    - Unrealized P&L tracking
    - Position limit enforcement
    - Symbol concentration analysis
    - Historical position tracking
    """
    
    def __init__(self):
        """Initialize position manager"""
        self.brokers: Dict[str, BrokerClient] = {}
        self.position_alerts: Dict[str, PositionAlert] = {}
        self.reconciliation_history: List[PositionDiscrepancy] = []
        
        # Position limits and thresholds
        self.max_position_size = Decimal('1000000')  # $1M max position
        self.max_symbol_concentration = Decimal('0.20')  # 20% max concentration
        self.max_total_exposure = Decimal('10000000')  # $10M max total exposure
        self.position_reconciliation_interval = 60  # seconds
        self.exposure_check_interval = 300  # seconds
        
        # Performance tracking
        self.total_reconciliations = 0
        self.successful_reconciliations = 0
        self.discrepancies_detected = 0
        self.alerts_generated = 0
        
        logger.info("Position Manager initialized")
    
    def register_broker(self, broker_name: str, broker_client: BrokerClient):
        """Register a broker for position management"""
        self.brokers[broker_name] = broker_client
        logger.info(f"Registered broker for position management: {broker_name}")
    
    async def get_all_positions(self, broker_name: Optional[str] = None) -> Dict[str, BrokerPosition]:
        """
        Get all positions from brokers
        
        Args:
            broker_name: Specific broker to query (None for all)
            
        Returns:
            Dictionary mapping broker_name -> position data
        """
        try:
            positions = {}
            
            if broker_name:
                # Get positions from specific broker
                if broker_name in self.brokers:
                    broker_positions = await self._get_broker_positions(broker_name)
                    positions[broker_name] = broker_positions
                else:
                    logger.warning(f"Broker {broker_name} not registered")
            else:
                # Get positions from all brokers
                for name, broker in self.brokers.items():
                    broker_positions = await self._get_broker_positions(name)
                    positions[name] = broker_positions
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    async def _get_broker_positions(self, broker_name: str) -> Dict[str, BrokerPosition]:
        """Get positions from a specific broker"""
        try:
            broker = self.brokers[broker_name]
            raw_positions = await broker.get_positions()
            
            positions = {}
            for raw_pos in raw_positions:
                position = BrokerPosition(
                    broker_name=broker_name,
                    symbol=raw_pos.get('symbol', ''),
                    quantity=Decimal(str(raw_pos.get('quantity', 0))),
                    average_price=Decimal(str(raw_pos.get('average_price', 0))),
                    unrealized_pnl=Decimal(str(raw_pos.get('unrealized_pnl', 0))),
                    realized_pnl=Decimal(str(raw_pos.get('realized_pnl', 0))),
                    market_value=Decimal(str(raw_pos.get('market_value', 0))),
                    timestamp=datetime.utcnow(),
                    metadata=raw_pos.get('metadata', {})
                )
                positions[position.symbol] = position
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions from {broker_name}: {e}")
            return {}
    
    async def reconcile_positions(self, force: bool = False) -> List[PositionDiscrepancy]:
        """
        Reconcile positions across all brokers with internal records
        
        Args:
            force: Force reconciliation even if recently done
            
        Returns:
            List of position discrepancies
        """
        try:
            logger.info("Starting position reconciliation")
            
            # Get broker positions
            broker_positions = await self.get_all_positions()
            
            # Get internal positions from database
            internal_positions = await self._get_internal_positions()
            
            # Find discrepancies
            discrepancies = await self._find_position_discrepancies(
                broker_positions, internal_positions
            )
            
            # Store reconciliation history
            self.reconciliation_history.extend(discrepancies)
            self.total_reconciliations += 1
            
            if not discrepancies:
                self.successful_reconciliations += 1
                logger.info("Position reconciliation completed - all positions synced")
            else:
                self.discrepancies_detected += len(discrepancies)
                logger.warning(f"Position reconciliation completed - {len(discrepancies)} discrepancies found")
                
                # Log detailed discrepancies
                for disc in discrepancies:
                    logger.warning(
                        f"Discrepancy for {disc.symbol}: "
                        f"Total broker Qty: {disc.total_broker_quantity}, "
                        f"Internal Qty: {disc.internal_quantity}, "
                        f"Difference: {disc.quantity_difference}"
                    )
            
            # Run post-reconciliation tasks
            await self._post_reconciliation_tasks(discrepancies)
            
            return discrepancies
            
        except Exception as e:
            logger.error(f"Error during position reconciliation: {e}")
            return []
    
    async def _get_internal_positions(self) -> Dict[str, Position]:
        """Get positions from internal database"""
        try:
            async with get_db_session() as session:
                positions = await Position.get_all_positions(session)
                
                internal_positions = {}
                for pos in positions:
                    internal_positions[pos.symbol] = pos
                
                return internal_positions
                
        except Exception as e:
            logger.error(f"Error getting internal positions: {e}")
            return {}
    
    async def _find_position_discrepancies(
        self,
        broker_positions: Dict[str, Dict[str, BrokerPosition]],
        internal_positions: Dict[str, Position]
    ) -> List[PositionDiscrepancy]:
        """Find discrepancies between broker and internal positions"""
        discrepancies = []
        
        # Get all symbols across brokers and internal records
        all_symbols = set()
        for broker_name, positions in broker_positions.items():
            all_symbols.update(positions.keys())
        all_symbols.update(internal_positions.keys())
        
        for symbol in all_symbols:
            try:
                # Get broker positions for this symbol
                symbol_broker_positions = {}
                total_broker_quantity = Decimal('0')
                total_broker_value = Decimal('0')
                
                for broker_name, positions in broker_positions.items():
                    if symbol in positions:
                        pos = positions[symbol]
                        symbol_broker_positions[broker_name] = pos
                        total_broker_quantity += pos.quantity
                        total_broker_value += pos.market_value
                
                # Get internal position
                internal_position = internal_positions.get(symbol)
                internal_quantity = internal_position.quantity if internal_position else Decimal('0')
                internal_value = (internal_position.quantity * internal_position.average_price) if internal_position else Decimal('0')
                
                # Calculate differences
                quantity_difference = total_broker_quantity - internal_quantity
                value_difference = total_broker_value - internal_value
                
                # Determine if there's a significant discrepancy
                quantity_threshold = Decimal('0.01')  # $0.01 worth
                value_threshold = Decimal('0.01')     # $0.01
                
                if abs(quantity_difference) > quantity_threshold or abs(value_difference) > value_threshold:
                    # Determine reconciliation status
                    if not symbol_broker_positions and internal_position:
                        status = PositionReconciliationStatus.MISSING_BROKER
                        severity = "HIGH"
                        notes = "Internal position not found in any broker"
                    elif symbol_broker_positions and not internal_position:
                        status = PositionReconciliationStatus.MISSING_INTERNAL
                        severity = "HIGH"
                        notes = "Broker positions not reflected in internal records"
                    else:
                        status = PositionReconciliationStatus.DISCREPANCY
                        
                        # Determine severity based on magnitude
                        if abs(quantity_difference) > Decimal('100') or abs(value_difference) > Decimal('10000'):
                            severity = "CRITICAL"
                        elif abs(quantity_difference) > Decimal('10') or abs(value_difference) > Decimal('1000'):
                            severity = "HIGH"
                        elif abs(quantity_difference) > Decimal('1') or abs(value_difference) > Decimal('100'):
                            severity = "MEDIUM"
                        else:
                            severity = "LOW"
                        notes = f"Quantity diff: {quantity_difference}, Value diff: {value_difference}"
                    
                    discrepancy = PositionDiscrepancy(
                        symbol=symbol,
                        broker_positions=symbol_broker_positions,
                        internal_position=internal_position,
                        total_broker_quantity=total_broker_quantity,
                        internal_quantity=internal_quantity,
                        quantity_difference=quantity_difference,
                        total_broker_value=total_broker_value,
                        internal_value=internal_value,
                        value_difference=value_difference,
                        severity=severity,
                        reconciliation_status=status,
                        notes=notes
                    )
                    
                    discrepancies.append(discrepancy)
            
            except Exception as e:
                logger.error(f"Error analyzing discrepancy for {symbol}: {e}")
        
        return discrepancies
    
    async def _post_reconciliation_tasks(self, discrepancies: List[PositionDiscrepancy]):
        """Run tasks after reconciliation"""
        try:
            # Generate alerts for discrepancies
            await self._generate_discrepancy_alerts(discrepancies)
            
            # Update position exposure metrics
            await self._update_exposure_metrics()
            
            # Check concentration limits
            await self._check_concentration_limits()
            
        except Exception as e:
            logger.error(f"Error in post-reconciliation tasks: {e}")
    
    async def _generate_discrepancy_alerts(self, discrepancies: List[PositionDiscrepancy]):
        """Generate alerts for position discrepancies"""
        for discrepancy in discrepancies:
            if discrepancy.severity in ["CRITICAL", "HIGH"]:
                alert = PositionAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=PositionAlert.RECONCILIATION_FAILED,
                    symbol=discrepancy.symbol,
                    severity=discrepancy.severity,
                    message=f"Position reconciliation failed for {discrepancy.symbol}: {discrepancy.notes}",
                    current_value=discrepancy.quantity_difference,
                    threshold_value=Decimal('0'),
                    created_time=datetime.utcnow()
                )
                
                self.position_alerts[alert.alert_id] = alert
                self.alerts_generated += 1
                
                logger.warning(f"Position alert generated: {alert.message}")
    
    async def _update_exposure_metrics(self):
        """Update portfolio exposure metrics"""
        try:
            # Get all positions
            broker_positions = await self.get_all_positions()
            
            total_exposure = Decimal('0')
            symbol_exposures = defaultdict(Decimal)
            
            for broker_name, positions in broker_positions.items():
                for position in positions.values():
                    total_exposure += position.market_value
                    symbol_exposures[position.symbol] += position.market_value
            
            # Check total exposure limit
            if total_exposure > self.max_total_exposure:
                alert = PositionAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=PositionAlert.HIGH_EXPOSURE,
                    symbol="PORTFOLIO",
                    severity="HIGH",
                    message=f"Total portfolio exposure ({total_exposure}) exceeds limit ({self.max_total_exposure})",
                    current_value=total_exposure,
                    threshold_value=self.max_total_exposure,
                    created_time=datetime.utcnow()
                )
                
                self.position_alerts[alert.alert_id] = alert
                self.alerts_generated += 1
                
                logger.warning(f"Exposure alert generated: {alert.message}")
            
            # Store exposure metrics (could be written to database for historical tracking)
            
        except Exception as e:
            logger.error(f"Error updating exposure metrics: {e}")
    
    async def _check_concentration_limits(self):
        """Check symbol concentration limits"""
        try:
            # Get all positions and calculate concentration
            broker_positions = await self.get_all_positions()
            
            total_portfolio_value = Decimal('0')
            symbol_values = defaultdict(Decimal)
            
            # Calculate total and per-symbol values
            for broker_name, positions in broker_positions.items():
                for position in positions.values():
                    total_portfolio_value += position.market_value
                    symbol_values[position.symbol] += position.market_value
            
            # Check concentration limits
            for symbol, symbol_value in symbol_values.items():
                if total_portfolio_value > 0:
                    concentration = symbol_value / total_portfolio_value
                    
                    if concentration > self.max_symbol_concentration:
                        alert = PositionAlert(
                            alert_id=str(uuid.uuid4()),
                            alert_type=PositionAlert.CONCENTRATION_RISK,
                            symbol=symbol,
                            severity="MEDIUM",
                            message=f"Symbol concentration for {symbol} ({concentration:.1%}) exceeds limit ({self.max_symbol_concentration:.1%})",
                            current_value=concentration,
                            threshold_value=self.max_symbol_concentration,
                            created_time=datetime.utcnow()
                        )
                        
                        self.position_alerts[alert.alert_id] = alert
                        self.alerts_generated += 1
                        
                        logger.warning(f"Concentration alert generated: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error checking concentration limits: {e}")
    
    async def synchronize_position(self, symbol: str, broker_name: str) -> bool:
        """
        Synchronize a specific position between broker and internal records
        
        Args:
            symbol: Symbol to synchronize
            broker_name: Broker holding the position
            
        Returns:
            True if synchronization successful
        """
        try:
            logger.info(f"Synchronizing position for {symbol} from {broker_name}")
            
            # Get broker position
            broker_positions = await self.get_all_positions(broker_name)
            broker_position = broker_positions.get(broker_name, {}).get(symbol)
            
            if not broker_position:
                logger.warning(f"No position found for {symbol} at {broker_name}")
                return False
            
            # Update internal position record
            async with get_db_session() as session:
                # Find existing position
                position = await Position.get_position(session, symbol)
                
                if position:
                    # Update existing position
                    position.quantity = broker_position.quantity
                    position.average_price = broker_position.average_price
                    position.unrealized_pnl = broker_position.unrealized_pnl
                    position.realized_pnl = broker_position.realized_pnl
                    position.market_value = broker_position.market_value
                    position.updated_at = datetime.utcnow()
                else:
                    # Create new position
                    position = Position(
                        symbol=symbol,
                        broker=broker_name,
                        quantity=broker_position.quantity,
                        average_price=broker_position.average_price,
                        side=PositionSide.LONG if broker_position.quantity > 0 else PositionSide.SHORT,
                        unrealized_pnl=broker_position.unrealized_pnl,
                        realized_pnl=broker_position.realized_pnl,
                        market_value=broker_position.market_value,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                
                session.add(position)
                await session.commit()
            
            logger.info(f"Position synchronized for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error synchronizing position {symbol}: {e}")
            return False
    
    async def get_position_alerts(
        self,
        severity_filter: Optional[List[str]] = None,
        acknowledged: bool = False,
        unresolved_only: bool = True
    ) -> List[PositionAlert]:
        """
        Get position alerts with optional filtering
        
        Args:
            severity_filter: Filter by severity levels
            acknowledged: Include acknowledged alerts
            unresolved_only: Only include unresolved alerts
            
        Returns:
            List of matching alerts
        """
        alerts = []
        
        for alert in self.position_alerts.values():
            # Apply filters
            if severity_filter and alert.severity not in severity_filter:
                continue
            
            if not acknowledged and alert.acknowledged:
                continue
            
            if unresolved_only and alert.resolved:
                continue
            
            alerts.append(alert)
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_time, reverse=True)
        
        return alerts
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a position alert"""
        if alert_id in self.position_alerts:
            self.position_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a position alert"""
        if alert_id in self.position_alerts:
            self.position_alerts[alert_id].resolved = True
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    async def get_reconciliation_history(
        self,
        symbol: Optional[str] = None,
        hours: int = 24
    ) -> List[PositionDiscrepancy]:
        """
        Get reconciliation history
        
        Args:
            symbol: Filter by symbol (None for all)
            hours: Hours of history to return
            
        Returns:
            List of historical discrepancies
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_history = []
        for discrepancy in self.reconciliation_history:
            if discrepancy.created_time >= cutoff_time:
                if symbol is None or discrepancy.symbol == symbol:
                    filtered_history.append(discrepancy)
        
        # Sort by creation time (newest first)
        filtered_history.sort(key=lambda x: x.created_time, reverse=True)
        
        return filtered_history
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary"""
        try:
            # Get current positions
            broker_positions = await self.get_all_positions()
            
            # Calculate totals
            total_market_value = Decimal('0')
            total_unrealized_pnl = Decimal('0')
            total_realized_pnl = Decimal('0')
            position_count = 0
            symbol_count = set()
            
            for broker_name, positions in broker_positions.items():
                for position in positions.values():
                    total_market_value += position.market_value
                    total_unrealized_pnl += position.unrealized_pnl
                    total_realized_pnl += position.realized_pnl
                    symbol_count.add(position.symbol)
                    if position.quantity != 0:
                        position_count += 1
            
            # Get recent alerts
            recent_alerts = await self.get_position_alerts(
                hours=24, unresolved_only=True
            )
            
            return {
                'portfolio': {
                    'total_market_value': float(total_market_value),
                    'total_unrealized_pnl': float(total_unrealized_pnl),
                    'total_realized_pnl': float(total_realized_pnl),
                    'position_count': position_count,
                    'symbol_count': len(symbol_count),
                    'symbols': list(symbol_count)
                },
                'reconciliation': {
                    'total_reconciliations': self.total_reconciliations,
                    'successful_reconciliations': self.successful_reconciliations,
                    'success_rate': self.successful_reconciliations / max(self.total_reconciliations, 1),
                    'discrepancies_detected': self.discrepancies_detected
                },
                'alerts': {
                    'active_alerts': len([a for a in recent_alerts if not a.resolved]),
                    'total_alerts_24h': len(recent_alerts),
                    'alerts_generated': self.alerts_generated
                },
                'brokers': {
                    'registered': list(self.brokers.keys()),
                    'active': len([b for b in self.brokers.values() if b])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating position summary: {e}")
            return {}
    
    async def start_monitoring(self):
        """Start continuous position monitoring"""
        logger.info("Starting position monitoring")
        
        # Initial reconciliation
        await self.reconcile_positions()
        
        # Set up periodic tasks
        async def periodic_reconciliation():
            while True:
                try:
                    await asyncio.sleep(self.position_reconciliation_interval)
                    await self.reconcile_positions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic reconciliation: {e}")
        
        async def periodic_exposure_check():
            while True:
                try:
                    await asyncio.sleep(self.exposure_check_interval)
                    await self._update_exposure_metrics()
                    await self._check_concentration_limits()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic exposure check: {e}")
        
        # Start background tasks
        reconciliation_task = asyncio.create_task(periodic_reconciliation())
        exposure_task = asyncio.create_task(periodic_exposure_check())
        
        return [reconciliation_task, exposure_task]
    
    async def stop_monitoring(self, tasks: List[asyncio.Task]):
        """Stop position monitoring"""
        logger.info("Stopping position monitoring")
        
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)


# Example usage and testing
if __name__ == "__main__":
    async def test_position_manager():
        from broker.alpaca import AlpacaClient
        from broker.binance import BinanceClient
        
        # Create position manager
        position_manager = PositionManager()
        
        # Register brokers (these would be real broker clients)
        # position_manager.register_broker('alpaca', AlpacaClient(...))
        # position_manager.register_broker('binance', BinanceClient(...))
        
        # Run reconciliation
        discrepancies = await position_manager.reconcile_positions()
        
        print(f"Reconciliation completed. Discrepancies: {len(discrepancies)}")
        
        for discrepancy in discrepancies:
            print(f"  {discrepancy.symbol}: {discrepancy.reconciliation_status.value} - {discrepancy.notes}")
        
        # Get position summary
        summary = await position_manager.get_position_summary()
        print("\nPosition Summary:")
        print(f"  Total Value: ${summary['portfolio']['total_market_value']:,.2f}")
        print(f"  Unrealized P&L: ${summary['portfolio']['total_unrealized_pnl']:,.2f}")
        print(f"  Position Count: {summary['portfolio']['position_count']}")
        print(f"  Symbol Count: {summary['portfolio']['symbol_count']}")
    
    asyncio.run(test_position_manager())