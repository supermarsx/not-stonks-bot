"""
Circuit Breaker Management System

Manages emergency trading halts and kill switches:
- Automatic circuit breaker triggers
- Manual kill switches
- Recovery mechanisms
- Emergency actions (cancel orders, close positions)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from config.database import get_db
from database.models.risk import CircuitBreaker

logger = logging.getLogger(__name__)


class CircuitBreakerManager:
    """
    Manages circuit breakers and emergency trading controls.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Circuit breaker types and their default behaviors
        self.breaker_configs = {
            "daily_loss_limit": {
                "default_actions": ["halt_new_orders"],
                "auto_recovery": False,
                "expiry_hours": 24
            },
            "position_limit_breach": {
                "default_actions": ["cancel_existing_orders"],
                "auto_recovery": True,
                "expiry_hours": 1
            },
            "volatility_spike": {
                "default_actions": ["halt_new_orders"],
                "auto_recovery": True,
                "expiry_hours": 2
            },
            "manual_halt": {
                "default_actions": ["halt_new_orders", "cancel_existing_orders", "close_positions"],
                "auto_recovery": False,
                "expiry_hours": None
            },
            "system_emergency": {
                "default_actions": ["halt_new_orders", "cancel_existing_orders", "close_positions"],
                "auto_recovery": False,
                "expiry_hours": None
            }
        }
        
        logger.info(f"CircuitBreakerManager initialized for user {self.user_id}")
    
    async def check_circuit_status(self, symbol: Optional[str] = None, 
                                 strategy_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Check current circuit breaker status for trading operations.
        
        Args:
            symbol: Symbol to check (optional)
            strategy_id: Strategy to check (optional)
            
        Returns:
            Current circuit breaker status
        """
        try:
            status = {
                "trading_halted": False,
                "active_breakers": [],
                "halt_reasons": [],
                "symbol_specific": None,
                "strategy_specific": None
            }
            
            # Check global breakers
            global_breakers = self.db.query(CircuitBreaker).filter(
                and_(
                    CircuitBreaker.is_active == True,
                    CircuitBreaker.user_id.is_(None)  # Global breakers
                )
            ).all()
            
            for breaker in global_breakers:
                if not self._is_breaker_expired(breaker):
                    status["active_breakers"].append({
                        "name": breaker.breaker_name,
                        "type": breaker.breaker_type,
                        "triggered_at": breaker.triggered_at,
                        "reason": breaker.trigger_condition
                    })
                    status["halt_reasons"].append(breaker.trigger_condition)
                    
                    if breaker.halt_new_orders:
                        status["trading_halted"] = True
            
            # Check user-specific breakers
            user_breakers = self.db.query(CircuitBreaker).filter(
                and_(
                    CircuitBreaker.is_active == True,
                    CircuitBreaker.user_id == self.user_id
                )
            ).all()
            
            for breaker in user_breakers:
                if not self._is_breaker_expired(breaker):
                    status["active_breakers"].append({
                        "name": breaker.breaker_name,
                        "type": breaker.breaker_type,
                        "triggered_at": breaker.triggered_at,
                        "reason": breaker.trigger_condition
                    })
                    
                    if breaker.halt_new_orders:
                        status["trading_halted"] = True
            
            # Check symbol-specific breakers
            if symbol:
                symbol_breakers = self.db.query(CircuitBreaker).filter(
                    and_(
                        CircuitBreaker.is_active == True,
                        CircuitBreaker.breaker_type == "symbol",
                        CircuitBreaker.scope_target == symbol
                    )
                ).all()
                
                symbol_halted = False
                for breaker in symbol_breakers:
                    if not self._is_breaker_expired(breaker):
                        symbol_halted = True
                        break
                
                status["symbol_specific"] = {
                    "symbol": symbol,
                    "halted": symbol_halted,
                    "halt_reason": "Symbol-specific trading halt" if symbol_halted else None
                }
                
                if symbol_halted:
                    status["trading_halted"] = True
                    status["halt_reasons"].append(f"Trading halted for symbol: {symbol}")
            
            # Check strategy-specific breakers
            if strategy_id:
                strategy_breakers = self.db.query(CircuitBreaker).filter(
                    and_(
                        CircuitBreaker.is_active == True,
                        CircuitBreaker.breaker_type == "strategy",
                        CircuitBreaker.scope_target == str(strategy_id)
                    )
                ).all()
                
                strategy_halted = False
                for breaker in strategy_breakers:
                    if not self._is_breaker_expired(breaker):
                        strategy_halted = True
                        break
                
                status["strategy_specific"] = {
                    "strategy_id": strategy_id,
                    "halted": strategy_halted,
                    "halt_reason": "Strategy-specific trading halt" if strategy_halted else None
                }
                
                if strategy_halted:
                    status["trading_halted"] = True
                    status["halt_reasons"].append(f"Trading halted for strategy: {strategy_id}")
            
            # Add summary
            status["halt_count"] = len(status["active_breakers"])
            status["global_halt"] = any(breaker.get("type") == "global" for breaker in status["active_breakers"])
            
            return status
            
        except Exception as e:
            logger.error(f"Circuit status check error: {str(e)}")
            return {
                "trading_halted": True,  # Default to halt on error
                "error": str(e),
                "reason": "System error - trading halted for safety"
            }
    
    async def trigger_circuit_breaker(self, breaker_type: str, reason: str,
                                    triggered_by: str = "system",
                                    symbol: Optional[str] = None,
                                    strategy_id: Optional[int] = None,
                                    halt_new_orders: bool = True,
                                    cancel_existing_orders: bool = False,
                                    close_positions: bool = False,
                                    expiry_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Manually or automatically trigger a circuit breaker.
        
        Args:
            breaker_type: Type of circuit breaker
            reason: Reason for triggering
            triggered_by: Who/what triggered it
            symbol: Symbol to halt (optional)
            strategy_id: Strategy to halt (optional)
            halt_new_orders: Whether to halt new orders
            cancel_existing_orders: Whether to cancel existing orders
            close_positions: Whether to close positions
            expiry_hours: How long to keep breaker active
            
        Returns:
            Circuit breaker creation result
        """
        try:
            # Get configuration for this breaker type
            config = self.breaker_configs.get(breaker_type, {})
            
            # Use default actions if not specified
            if halt_new_orders is None and cancel_existing_orders is None and close_positions is None:
                default_actions = config.get("default_actions", [])
                halt_new_orders = "halt_new_orders" in default_actions
                cancel_existing_orders = "cancel_existing_orders" in default_actions
                close_positions = "close_positions" in default_actions
            
            # Determine breaker scope
            if symbol:
                breaker_scope = "symbol"
                scope_target = symbol
            elif strategy_id is not None:
                breaker_scope = "strategy"
                scope_target = str(strategy_id)
            else:
                breaker_scope = "account"
                scope_target = None
            
            # Determine expiry
            if expiry_hours is None:
                expiry_hours = config.get("expiry_hours")
            
            # Create circuit breaker
            breaker = CircuitBreaker(
                user_id=self.user_id if triggered_by != "system" else None,
                breaker_name=f"{breaker_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                breaker_type=breaker_scope,
                scope_target=scope_target,
                is_active=True,
                triggered_by=triggered_by,
                trigger_condition=reason,
                halt_new_orders=halt_new_orders,
                cancel_existing_orders=cancel_existing_orders,
                close_positions=close_positions,
                triggered_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=expiry_hours) if expiry_hours else None,
                metadata={
                    "breaker_type": breaker_type,
                    "trigger_method": "manual" if triggered_by == "user" else "automatic",
                    "config": config
                }
            )
            
            self.db.add(breaker)
            self.db.commit()
            
            logger.critical(f"Circuit breaker triggered for user {self.user_id}: {breaker_type} - {reason}")
            
            # Execute emergency actions if needed
            actions_taken = []
            if halt_new_orders:
                actions_taken.append("halt_new_orders")
            if cancel_existing_orders:
                actions_taken.append(await self._cancel_all_orders_for_scope(scope_target))
            if close_positions:
                actions_taken.append(await self._close_positions_for_scope(scope_target))
            
            return {
                "success": True,
                "breaker_id": breaker.id,
                "breaker_name": breaker.breaker_name,
                "type": breaker_type,
                "scope": breaker_scope,
                "scope_target": scope_target,
                "actions_taken": actions_taken,
                "triggered_at": breaker.triggered_at,
                "expires_at": breaker.expires_at
            }
            
        except Exception as e:
            logger.error(f"Circuit breaker trigger error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def clear_circuit_breaker(self, breaker_id: int, cleared_by: str = "system") -> Dict[str, Any]:
        """
        Manually clear a circuit breaker.
        
        Args:
            breaker_id: ID of circuit breaker to clear
            cleared_by: Who cleared it
            
        Returns:
            Circuit breaker clearing result
        """
        try:
            breaker = self.db.query(CircuitBreaker).filter(
                CircuitBreaker.id == breaker_id
            ).first()
            
            if not breaker:
                return {
                    "success": False,
                    "error": "Circuit breaker not found"
                }
            
            # Update breaker status
            breaker.is_active = False
            breaker.cleared_at = datetime.now()
            
            # Add metadata about clearing
            if breaker.metadata is None:
                breaker.metadata = {}
            breaker.metadata["cleared_by"] = cleared_by
            breaker.metadata["cleared_at"] = datetime.now().isoformat()
            
            self.db.commit()
            
            logger.info(f"Circuit breaker cleared for user {self.user_id}: {breaker.breaker_name}")
            
            return {
                "success": True,
                "breaker_id": breaker.id,
                "breaker_name": breaker.breaker_name,
                "cleared_at": breaker.cleared_at,
                "cleared_by": cleared_by
            }
            
        except Exception as e:
            logger.error(f"Circuit breaker clear error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_active_breakers(self) -> List[Dict[str, Any]]:
        """
        Get all currently active circuit breakers.
        
        Returns:
            List of active circuit breakers
        """
        try:
            breakers = self.db.query(CircuitBreaker).filter(
                CircuitBreaker.is_active == True
            ).all()
            
            active_breakers = []
            for breaker in breakers:
                if not self._is_breaker_expired(breaker):
                    active_breakers.append({
                        "id": breaker.id,
                        "name": breaker.breaker_name,
                        "type": breaker.breaker_type,
                        "scope_target": breaker.scope_target,
                        "triggered_by": breaker.triggered_by,
                        "trigger_condition": breaker.trigger_condition,
                        "triggered_at": breaker.triggered_at,
                        "expires_at": breaker.expires_at,
                        "actions": {
                            "halt_new_orders": breaker.halt_new_orders,
                            "cancel_existing_orders": breaker.cancel_existing_orders,
                            "close_positions": breaker.close_positions
                        },
                        "time_active": (datetime.now() - breaker.triggered_at).total_seconds() / 3600  # hours
                    })
            
            return active_breakers
            
        except Exception as e:
            logger.error(f"Active breakers retrieval error: {str(e)}")
            return []
    
    async def auto_recovery_check(self) -> List[Dict[str, Any]]:
        """
        Check for breakers that should auto-recover based on their configuration.
        
        Returns:
            List of recovered breakers
        """
        recovered = []
        
        try:
            # Get all active breakers with auto-recovery enabled
            active_breakers = self.db.query(CircuitBreaker).filter(
                CircuitBreaker.is_active == True
            ).all()
            
            for breaker in active_breakers:
                breaker_type = breaker.metadata.get("breaker_type", "") if breaker.metadata else ""
                config = self.breaker_configs.get(breaker_type, {})
                
                # Check if this breaker should auto-recover
                if (config.get("auto_recovery", False) and 
                    self._is_breaker_expired(breaker)):
                    
                    # Clear the breaker
                    await self.clear_circuit_breaker(breaker.id, "auto_recovery")
                    
                    recovered.append({
                        "breaker_id": breaker.id,
                        "breaker_name": breaker.breaker_name,
                        "type": breaker_type,
                        "reason": "Auto-recovery after expiry"
                    })
                    
                    logger.info(f"Circuit breaker auto-recovered: {breaker.breaker_name}")
            
        except Exception as e:
            logger.error(f"Auto recovery check error: {str(e)}")
        
        return recovered
    
    async def emergency_kill_switch(self, reason: str = "Emergency manual halt",
                                  close_all_positions: bool = True) -> Dict[str, Any]:
        """
        Execute emergency kill switch - immediately halt all trading.
        
        Args:
            reason: Reason for kill switch
            close_all_positions: Whether to close all positions
            
        Returns:
            Kill switch execution result
        """
        try:
            logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED for user {self.user_id}: {reason}")
            
            # Trigger global circuit breaker
            kill_result = await self.trigger_circuit_breaker(
                breaker_type="manual_halt",
                reason=reason,
                triggered_by="user",
                halt_new_orders=True,
                cancel_existing_orders=True,
                close_positions=close_all_positions
            )
            
            if kill_result["success"]:
                # Execute immediate actions
                actions = []
                
                # Cancel all pending orders
                cancel_result = await self._cancel_all_orders_for_scope(None)
                actions.append(("cancel_orders", cancel_result))
                
                if close_all_positions:
                    # Close all positions
                    close_result = await self._close_positions_for_scope(None)
                    actions.append(("close_positions", close_result))
                
                return {
                    "success": True,
                    "kill_switch_activated": True,
                    "circuit_breaker": kill_result,
                    "actions_executed": actions,
                    "timestamp": datetime.now(),
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to activate kill switch circuit breaker",
                    "partial_actions": actions if 'actions' in locals() else []
                }
            
        except Exception as e:
            logger.error(f"Emergency kill switch error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _is_breaker_expired(self, breaker: CircuitBreaker) -> bool:
        """Check if a circuit breaker has expired."""
        try:
            if not breaker.expires_at:
                return False  # No expiry, keep active
            
            return datetime.now() > breaker.expires_at
            
        except Exception:
            return False
    
    async def _cancel_all_orders_for_scope(self, scope_target: Optional[str]) -> Dict[str, Any]:
        """Cancel orders based on scope."""
        try:
            # This would integrate with the OMS to cancel orders
            # For now, just return a placeholder result
            
            query = and_(
                Order.user_id == self.user_id,
                Order.status.in_(["pending", "new", "partially_filled"])
            )
            
            if scope_target:
                # Symbol-specific or strategy-specific cancellation
                # This would need more specific logic
                pass
            
            # In practice, this would call the OMS cancel_orders method
            return {
                "cancelled_count": 0,
                "scope": "all" if scope_target is None else scope_target,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Order cancellation error: {str(e)}")
            return {
                "error": str(e),
                "cancelled_count": 0
            }
    
    async def _close_positions_for_scope(self, scope_target: Optional[str]) -> Dict[str, Any]:
        """Close positions based on scope."""
        try:
            # This would integrate with the OMS to close positions
            # For now, just return a placeholder result
            
            positions_query = Position.user_id == self.user_id
            
            if scope_target:
                # Symbol-specific or strategy-specific position closing
                # This would need more specific logic
                pass
            
            # In practice, this would call the OMS close_positions method
            return {
                "positions_closed": 0,
                "scope": "all" if scope_target is None else scope_target,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Position closing error: {str(e)}")
            return {
                "error": str(e),
                "positions_closed": 0
            }
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
