"""
Trade Settlement System
Handles trade settlement, position updates, and P&L tracking
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from collections import defaultdict

from loguru import logger

from trading.models import Position, PositionSide, Order, OrderStatus, Trade
from trading.database import get_db_session
from broker.base import BrokerClient


class SettlementStatus(Enum):
    """Settlement status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_SETTLED = "partially_settled"
    RECONCILED = "reconciled"


class SettlementType(Enum):
    """Settlement type"""
    TRADE_EXECUTION = "trade_execution"
    POSITION_ADJUSTMENT = "position_adjustment"
    DIVIDEND = "dividend"
    CORPORATE_ACTION = "corporate_action"
    MANUAL_ADJUSTMENT = "manual_adjustment"


@dataclass
class TradeSettlement:
    """Trade settlement record"""
    settlement_id: str
    trade_id: str
    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: Decimal
    execution_price: Decimal
    commission: Decimal
    settlement_type: SettlementType
    status: SettlementStatus
    broker_name: str
    broker_trade_id: Optional[str] = None
    settlement_date: Optional[datetime] = None
    actual_settlement_date: Optional[datetime] = None
    currency: str = "USD"
    fees: Decimal = Decimal('0')
    taxes: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


@dataclass
class SettlementBatch:
    """Batch of settlements for processing"""
    batch_id: str
    settlements: List[TradeSettlement]
    total_trades: int = 0
    total_volume: Decimal = Decimal('0')
    total_commission: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    status: SettlementStatus = SettlementStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    error_count: int = 0


class SettlementProcessor:
    """
    Trade Settlement Processing System
    
    Features:
    - Automated trade settlement after execution
    - Position updates and reconciliation
    - P&L calculation and tracking
    - Settlement failure handling and retry logic
    - Historical settlement tracking and reporting
    - Multi-currency support
    - Commission and fee calculation
    - Tax implications tracking
    """
    
    def __init__(self):
        """Initialize settlement processor"""
        self.brokers: Dict[str, BrokerClient] = {}
        self.settlement_queue: List[TradeSettlement] = []
        self.active_batches: Dict[str, SettlementBatch] = {}
        self.settlement_history: List[TradeSettlement] = []
        
        # Settlement configuration
        self.settlement_delay_seconds = 30  # Delay before settlement processing
        self.max_retry_attempts = 3
        self.retry_delay_seconds = 60
        self.batch_size = 100
        self.settlement_timeout_seconds = 300
        
        # Performance tracking
        self.total_settlements = 0
        self.successful_settlements = 0
        self.failed_settlements = 0
        self.total_volume_settled = Decimal('0')
        self.total_commission_collected = Decimal('0')
        self.total_fees_collected = Decimal('0')
        
        # P&L tracking
        self.realized_pnl_by_symbol: Dict[str, Decimal] = defaultdict(Decimal)
        self.realized_pnl_by_day: Dict[str, Decimal] = defaultdict(Decimal)
        
        logger.info("Settlement Processor initialized")
    
    def register_broker(self, broker_name: str, broker_client: BrokerClient):
        """Register a broker for settlement processing"""
        self.brokers[broker_name] = broker_client
        logger.info(f"Registered broker for settlement: {broker_name}")
    
    async def process_trade_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        execution_price: Decimal,
        broker_name: str,
        broker_trade_id: Optional[str] = None,
        commission: Decimal = Decimal('0'),
        fees: Decimal = Decimal('0'),
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a trade execution and create settlement record
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Executed quantity
            execution_price: Execution price
            broker_name: Broker name
            broker_trade_id: Broker trade ID
            commission: Commission amount
            fees: Additional fees
            metadata: Additional metadata
            
        Returns:
            Settlement ID
        """
        try:
            # Create settlement record
            settlement_id = f"SET_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            trade_id = f"TRADE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            settlement = TradeSettlement(
                settlement_id=settlement_id,
                trade_id=trade_id,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                execution_price=execution_price,
                commission=commission,
                settlement_type=SettlementType.TRADE_EXECUTION,
                status=SettlementStatus.PENDING,
                broker_name=broker_name,
                broker_trade_id=broker_trade_id,
                fees=fees,
                currency=metadata.get('currency', 'USD') if metadata else 'USD',
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            # Add to settlement queue
            self.settlement_queue.append(settlement)
            
            logger.info(
                f"Trade execution queued for settlement: {settlement_id} "
                f"{side} {quantity} {symbol} @ {execution_price}"
            )
            
            # Trigger settlement processing if batch size reached
            if len(self.settlement_queue) >= self.batch_size:
                await self._process_settlement_batch()
            
            return settlement_id
            
        except Exception as e:
            logger.error(f"Error processing trade execution: {e}")
            raise
    
    async def _process_settlement_batch(self):
        """Process a batch of settlements"""
        if not self.settlement_queue:
            return
        
        try:
            # Create batch
            batch_id = f"BATCH_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Take batch from queue
            batch_settlements = self.settlement_queue[:self.batch_size]
            self.settlement_queue = self.settlement_queue[self.batch_size:]
            
            batch = SettlementBatch(
                batch_id=batch_id,
                settlements=batch_settlements,
                total_trades=len(batch_settlements),
                status=SettlementStatus.PENDING,
                created_at=datetime.utcnow()
            )
            
            # Calculate batch totals
            for settlement in batch_settlements:
                batch.total_volume += settlement.quantity * settlement.execution_price
                batch.total_commission += settlement.commission
                batch.total_fees += settlement.fees
            
            self.active_batches[batch_id] = batch
            
            logger.info(f"Processing settlement batch {batch_id} with {batch.total_trades} trades")
            
            # Process the batch
            await self._settle_batch(batch)
            
        except Exception as e:
            logger.error(f"Error processing settlement batch: {e}")
    
    async def _settle_batch(self, batch: SettlementBatch):
        """Process settlement for a batch"""
        try:
            batch.status = SettlementStatus.IN_PROGRESS
            batch.processed_at = datetime.utcnow()
            
            successful_settlements = 0
            failed_settlements = 0
            
            # Group settlements by symbol for efficient processing
            symbol_batches = defaultdict(list)
            for settlement in batch.settlements:
                symbol_batches[settlement.symbol].append(settlement)
            
            # Process each symbol batch
            for symbol, settlements in symbol_batches.items():
                try:
                    symbol_success = await self._settle_symbol_batch(symbol, settlements)
                    
                    if symbol_success:
                        successful_settlements += len(settlements)
                    else:
                        failed_settlements += len(settlements)
                        
                except Exception as e:
                    logger.error(f"Error settling symbol batch {symbol}: {e}")
                    failed_settlements += len(settlements)
            
            # Update batch status
            if failed_settlements == 0:
                batch.status = SettlementStatus.COMPLETED
                self.successful_settlements += successful_settlements
            elif successful_settlements == 0:
                batch.status = SettlementStatus.FAILED
                self.failed_settlements += failed_settlements
            else:
                batch.status = SettlementStatus.PARTIALLY_SETTLED
                self.successful_settlements += successful_settlements
                self.failed_settlements += failed_settlements
            
            batch.error_count = failed_settlements
            
            logger.info(
                f"Settlement batch {batch.batch_id} completed: "
                f"{successful_settlements} successful, {failed_settlements} failed"
            )
            
        except Exception as e:
            logger.error(f"Error settling batch {batch.batch_id}: {e}")
            batch.status = SettlementStatus.FAILED
            batch.error_count = len(batch.settlements)
    
    async def _settle_symbol_batch(self, symbol: str, settlements: List[TradeSettlement]) -> bool:
        """Process settlement for a specific symbol"""
        try:
            # Calculate net position change
            net_quantity = Decimal('0')
            total_cost = Decimal('0')
            total_commission = Decimal('0')
            total_fees = Decimal('0')
            
            for settlement in settlements:
                if settlement.side.lower() == 'buy':
                    net_quantity += settlement.quantity
                    total_cost += settlement.quantity * settlement.execution_price
                else:  # sell
                    net_quantity -= settlement.quantity
                    total_cost -= settlement.quantity * settlement.execution_price
                
                total_commission += settlement.commission
                total_fees += settlement.fees
            
            # Update positions for each settlement
            for settlement in settlements:
                success = await self._settle_individual_trade(settlement)
                
                if not success:
                    settlement.status = SettlementStatus.FAILED
                    settlement.error_message = "Failed to settle individual trade"
                else:
                    settlement.status = SettlementStatus.COMPLETED
                    settlement.actual_settlement_date = datetime.utcnow()
            
            # Update portfolio-level metrics
            self.total_volume_settled += abs(total_cost)
            self.total_commission_collected += total_commission
            self.total_fees_collected += total_fees
            
            return True
            
        except Exception as e:
            logger.error(f"Error settling symbol batch for {symbol}: {e}")
            return False
    
    async def _settle_individual_trade(self, settlement: TradeSettlement) -> bool:
        """Settle an individual trade"""
        try:
            async with get_db_session() as session:
                # Get or create position
                position = await Position.get_position(session, settlement.symbol)
                
                if not position:
                    # Create new position
                    position = Position(
                        symbol=settlement.symbol,
                        broker=settlement.broker_name,
                        quantity=Decimal('0'),
                        average_price=Decimal('0'),
                        side=PositionSide.LONG,
                        unrealized_pnl=Decimal('0'),
                        realized_pnl=Decimal('0'),
                        market_value=Decimal('0'),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                
                # Calculate P&L impact
                pnl_change = await self._calculate_trade_pnl(position, settlement)
                settlement.realized_pnl = pnl_change
                
                # Update position
                await self._update_position_from_trade(position, settlement)
                
                # Update realized P&L tracking
                day_key = settlement.created_at.strftime('%Y-%m-%d')
                self.realized_pnl_by_symbol[settlement.symbol] += pnl_change
                self.realized_pnl_by_day[day_key] += pnl_change
                
                # Save position
                session.add(position)
                
                # Create trade record
                trade = Trade(
                    trade_id=settlement.trade_id,
                    order_id=settlement.order_id,
                    symbol=settlement.symbol,
                    side=settlement.side,
                    quantity=settlement.quantity,
                    price=settlement.execution_price,
                    commission=settlement.commission,
                    fees=settlement.fees,
                    realized_pnl=pnl_change,
                    broker=settlement.broker_name,
                    broker_trade_id=settlement.broker_trade_id,
                    trade_time=settlement.created_at,
                    created_at=datetime.utcnow()
                )
                
                session.add(trade)
                await session.commit()
                
                logger.debug(f"Trade settled: {settlement.settlement_id} - P&L: {pnl_change}")
                return True
                
        except Exception as e:
            logger.error(f"Error settling individual trade {settlement.settlement_id}: {e}")
            return False
    
    async def _calculate_trade_pnl(self, position: Position, settlement: TradeSettlement) -> Decimal:
        """Calculate realized P&L from a trade"""
        try:
            # For simple FIFO calculation
            trade_value = settlement.quantity * settlement.execution_price
            
            if settlement.side.lower() == 'buy':
                # For buys, we just add to position cost basis
                if position.quantity > 0:
                    # Calculate new average price
                    total_cost_before = position.quantity * position.average_price
                    total_cost_after = total_cost_before + trade_value + settlement.commission
                    new_quantity = position.quantity + settlement.quantity
                    new_average_price = total_cost_after / new_quantity if new_quantity > 0 else Decimal('0')
                    
                    position.average_price = new_average_price
                
                return Decimal('0')  # No realized P&L on buy
                
            else:  # sell
                # For sells, calculate realized P&L
                if position.quantity >= settlement.quantity:
                    # Calculate cost basis
                    cost_basis = settlement.quantity * position.average_price
                    proceeds = trade_value - settlement.commission
                    realized_pnl = proceeds - cost_basis
                    
                    # Update position quantity
                    new_quantity = position.quantity - settlement.quantity
                    
                    if new_quantity > 0:
                        position.quantity = new_quantity
                        # Average price remains the same
                    else:
                        # Position closed, reset
                        position.quantity = Decimal('0')
                        position.average_price = Decimal('0')
                    
                    return realized_pnl
                else:
                    # Short position or insufficient shares
                    logger.warning(f"Selling more shares than held for {settlement.symbol}")
                    return Decimal('0')
                    
        except Exception as e:
            logger.error(f"Error calculating trade P&L: {e}")
            return Decimal('0')
    
    async def _update_position_from_trade(self, position: Position, settlement: TradeSettlement):
        """Update position based on trade"""
        try:
            if settlement.side.lower() == 'buy':
                # Add to position
                old_quantity = position.quantity
                old_value = old_quantity * position.average_price
                
                new_quantity = old_quantity + settlement.quantity
                new_cost = old_value + (settlement.quantity * settlement.execution_price) + settlement.commission
                
                if new_quantity > 0:
                    position.quantity = new_quantity
                    position.average_price = new_cost / new_quantity
                else:
                    position.quantity = Decimal('0')
                    position.average_price = Decimal('0')
                
                position.side = PositionSide.LONG if position.quantity > 0 else PositionSide.SHORT
                
            else:  # sell
                # Subtract from position
                position.quantity -= settlement.quantity
                
                if position.quantity == 0:
                    position.average_price = Decimal('0')
                    position.side = PositionSide.LONG if position.quantity > 0 else PositionSide.SHORT
                elif position.quantity < 0:
                    position.side = PositionSide.SHORT
            
            position.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating position from trade: {e}")
    
    async def retry_failed_settlements(self, max_age_hours: int = 24) -> int:
        """
        Retry failed settlements
        
        Args:
            max_age_hours: Maximum age of settlements to retry
            
        Returns:
            Number of settlements retried
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            # Find failed settlements in history
            failed_settlements = [
                s for s in self.settlement_history
                if s.status == SettlementStatus.FAILED and s.created_at >= cutoff_time
            ]
            
            retry_count = 0
            for settlement in failed_settlements:
                if self._should_retry_settlement(settlement):
                    # Reset settlement for retry
                    settlement.status = SettlementStatus.PENDING
                    settlement.error_message = None
                    settlement.updated_at = datetime.utcnow()
                    
                    # Add back to queue
                    self.settlement_queue.append(settlement)
                    retry_count += 1
            
            if retry_count > 0:
                logger.info(f"Queued {retry_count} failed settlements for retry")
                
                # Process retry queue
                if self.settlement_queue:
                    await self._process_settlement_batch()
            
            return retry_count
            
        except Exception as e:
            logger.error(f"Error retrying failed settlements: {e}")
            return 0
    
    def _should_retry_settlement(self, settlement: TradeSettlement) -> bool:
        """Check if a settlement should be retried"""
        # Count retry attempts in metadata
        retry_count = settlement.metadata.get('retry_count', 0)
        
        # Only retry if under max attempts
        if retry_count < self.max_retry_attempts:
            # Increment retry count
            settlement.metadata['retry_count'] = retry_count + 1
            settlement.metadata['last_retry'] = datetime.utcnow().isoformat()
            return True
        
        return False
    
    async def get_settlement_status(self, settlement_id: str) -> Optional[Dict[str, Any]]:
        """Get settlement status"""
        # Search in active batches
        for batch in self.active_batches.values():
            for settlement in batch.settlements:
                if settlement.settlement_id == settlement_id:
                    return self._settlement_to_dict(settlement)
        
        # Search in settlement queue
        for settlement in self.settlement_queue:
            if settlement.settlement_id == settlement_id:
                return self._settlement_to_dict(settlement)
        
        # Search in history
        for settlement in self.settlement_history:
            if settlement.settlement_id == settlement_id:
                return self._settlement_to_dict(settlement)
        
        return None
    
    def _settlement_to_dict(self, settlement: TradeSettlement) -> Dict[str, Any]:
        """Convert settlement to dictionary"""
        return {
            'settlement_id': settlement.settlement_id,
            'trade_id': settlement.trade_id,
            'order_id': settlement.order_id,
            'symbol': settlement.symbol,
            'side': settlement.side,
            'quantity': float(settlement.quantity),
            'execution_price': float(settlement.execution_price),
            'commission': float(settlement.commission),
            'fees': float(settlement.fees),
            'settlement_type': settlement.settlement_type.value,
            'status': settlement.status.value,
            'broker_name': settlement.broker_name,
            'broker_trade_id': settlement.broker_trade_id,
            'currency': settlement.currency,
            'realized_pnl': float(settlement.realized_pnl),
            'created_at': settlement.created_at.isoformat(),
            'updated_at': settlement.updated_at.isoformat(),
            'actual_settlement_date': settlement.actual_settlement_date.isoformat() if settlement.actual_settlement_date else None,
            'error_message': settlement.error_message
        }
    
    async def get_pending_settlements(
        self,
        symbol: Optional[str] = None,
        broker: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get pending settlements"""
        settlements = []
        
        # Check active batches
        for batch in self.active_batches.values():
            for settlement in batch.settlements:
                if settlement.status == SettlementStatus.PENDING:
                    if self._settlement_matches_filters(settlement, symbol, broker):
                        settlements.append(self._settlement_to_dict(settlement))
        
        # Check settlement queue
        for settlement in self.settlement_queue:
            if self._settlement_matches_filters(settlement, symbol, broker):
                settlements.append(self._settlement_to_dict(settlement))
        
        # Sort by creation time
        settlements.sort(key=lambda x: x['created_at'], reverse=True)
        
        return settlements[:limit]
    
    def _settlement_matches_filters(
        self,
        settlement: TradeSettlement,
        symbol: Optional[str],
        broker: Optional[str]
    ) -> bool:
        """Check if settlement matches filters"""
        if symbol and settlement.symbol != symbol:
            return False
        
        if broker and settlement.broker_name != broker:
            return False
        
        return True
    
    async def get_settlement_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get settlement summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter settlements by time
        recent_settlements = [
            s for s in self.settlement_history
            if s.created_at >= cutoff_time
        ]
        
        # Calculate metrics
        total_volume = sum(
            float(s.quantity * s.execution_price) for s in recent_settlements
        )
        total_commission = sum(float(s.commission) for s in recent_settlements)
        total_fees = sum(float(s.fees) for s in recent_settlements)
        total_realized_pnl = sum(float(s.realized_pnl) for s in recent_settlements)
        
        # Status breakdown
        status_counts = defaultdict(int)
        for settlement in recent_settlements:
            status_counts[settlement.status.value] += 1
        
        # Symbol breakdown
        symbol_volume = defaultdict(float)
        for settlement in recent_settlements:
            symbol_volume[settlement.symbol] += float(settlement.quantity * settlement.execution_price)
        
        return {
            'period_hours': hours,
            'metrics': {
                'total_settlements': len(recent_settlements),
                'total_volume': total_volume,
                'total_commission': total_commission,
                'total_fees': total_fees,
                'total_realized_pnl': total_realized_pnl,
                'average_trade_size': total_volume / len(recent_settlements) if recent_settlements else 0
            },
            'status_breakdown': dict(status_counts),
            'top_symbols_by_volume': dict(
                sorted(symbol_volume.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            'lifetime_metrics': {
                'total_settlements': self.total_settlements,
                'successful_settlements': self.successful_settlements,
                'failed_settlements': self.failed_settlements,
                'success_rate': self.successful_settlements / max(self.total_settlements, 1),
                'total_volume_settled': float(self.total_volume_settled),
                'total_commission_collected': float(self.total_commission_collected),
                'total_fees_collected': float(self.total_fees_collected)
            },
            'realized_pnl': {
                'by_symbol': {k: float(v) for k, v in self.realized_pnl_by_symbol.items()},
                'by_day': {k: float(v) for k, v in self.realized_pnl_by_day.items()}
            }
        }
    
    async def cleanup_completed_settlements(self, keep_days: int = 30) -> int:
        """Clean up old completed settlements"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=keep_days)
            
            # Remove from settlement queue
            initial_queue_size = len(self.settlement_queue)
            self.settlement_queue = [
                s for s in self.settlement_queue
                if s.created_at >= cutoff_time or s.status == SettlementStatus.PENDING
            ]
            removed_from_queue = initial_queue_size - len(self.settlement_queue)
            
            # Remove from active batches if completed
            removed_from_batches = 0
            completed_batches = [
                bid for bid, batch in self.active_batches.items()
                if batch.status in [SettlementStatus.COMPLETED, SettlementStatus.FAILED, SettlementStatus.PARTIALLY_SETTLED]
                and batch.created_at < cutoff_time
            ]
            
            for batch_id in completed_batches:
                del self.active_batches[batch_id]
                removed_from_batches += 1
            
            logger.info(
                f"Cleanup completed: {removed_from_queue} from queue, "
                f"{removed_from_batches} batches removed"
            )
            
            return removed_from_queue + removed_from_batches
            
        except Exception as e:
            logger.error(f"Error cleaning up completed settlements: {e}")
            return 0
    
    async def force_process_queue(self):
        """Force process all pending settlements"""
        while self.settlement_queue:
            await self._process_settlement_batch()
        
        logger.info("All settlement queues processed")
    
    async def start_settlement_monitoring(self):
        """Start settlement processing monitoring"""
        logger.info("Starting settlement monitoring")
        
        async def process_queue_periodically():
            while True:
                try:
                    if self.settlement_queue:
                        await self._process_settlement_batch()
                    
                    # Retry failed settlements periodically
                    if len(self.settlement_history) > 0:
                        await self.retry_failed_settlements()
                    
                    # Wait before next check
                    await asyncio.sleep(self.settlement_delay_seconds)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in settlement monitoring: {e}")
                    await asyncio.sleep(self.retry_delay_seconds)
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(process_queue_periodically())
        return [monitoring_task]
    
    async def stop_settlement_monitoring(self, tasks: List[asyncio.Task]):
        """Stop settlement monitoring"""
        logger.info("Stopping settlement monitoring")
        
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process remaining queue before shutdown
        if self.settlement_queue:
            logger.info("Processing remaining settlements before shutdown")
            await self.force_process_queue()


# Example usage and testing
if __name__ == "__main__":
    async def test_settlement_processor():
        from broker.alpaca import AlpacaClient
        
        # Create settlement processor
        processor = SettlementProcessor()
        
        # Register broker
        # processor.register_broker('alpaca', AlpacaClient(...))
        
        # Process some trade executions
        await processor.process_trade_execution(
            order_id="ORD_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            execution_price=Decimal("150.00"),
            broker_name="alpaca",
            commission=Decimal("1.00")
        )
        
        await processor.process_trade_execution(
            order_id="ORD_124",
            symbol="AAPL",
            side="sell",
            quantity=Decimal("50"),
            execution_price=Decimal("155.00"),
            broker_name="alpaca",
            commission=Decimal("0.50")
        )
        
        # Force process queue
        await processor.force_process_queue()
        
        # Get summary
        summary = await processor.get_settlement_summary()
        print("Settlement Summary:")
        print(f"  Total Volume: ${summary['metrics']['total_volume']:,.2f}")
        print(f"  Total Commission: ${summary['metrics']['total_commission']:,.2f}")
        print(f"  Realized P&L: ${summary['metrics']['total_realized_pnl']:,.2f}")
        print(f"  Success Rate: {summary['lifetime_metrics']['success_rate']:.1%}")
    
    asyncio.run(test_settlement_processor())