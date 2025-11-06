"""
Order Router - Multi-Broker Order Routing System

Routes orders to appropriate brokers based on:
- Broker availability and status
- Asset class availability
- Cost considerations (commissions, spreads)
- Execution quality
- Regulatory requirements
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from config.database import get_db
from database.models.trading import Order, OrderStatus, OrderType
from database.models.broker import BrokerConnection

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Order routing strategies."""
    COST_OPTIMIZATION = "cost_optimization"
    EXECUTION_QUALITY = "execution_quality"
    BROKER_PREFERENCE = "broker_preference"
    ROUND_ROBIN = "round_robin"
    FASTEST_EXECUTION = "fastest_execution"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class OrderRouter:
    """
    Routes orders across multiple broker connections.
    
    Implements intelligent order routing based on:
    - Broker availability
    - Asset class support
    - Cost optimization
    - Execution quality
    - Regulatory requirements
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Broker routing configuration
        self.broker_capabilities = {
            "binance": {
                "asset_classes": ["crypto", "forex"],
                "order_types": ["market", "limit", "stop", "stop_limit"],
                "commissions": {"maker": 0.001, "taker": 0.001},
                "avg_execution_time": 0.5,  # seconds
                "reliability": 0.95
            },
            "alpaca": {
                "asset_classes": ["equities", "crypto"],
                "order_types": ["market", "limit", "stop", "stop_limit"],
                "commissions": {"equities": 0.0, "crypto": 0.0025},
                "avg_execution_time": 0.3,
                "reliability": 0.98
            },
            "ibkr": {
                "asset_classes": ["equities", "options", "futures", "forex", "bonds"],
                "order_types": ["market", "limit", "stop", "stop_limit", "trailing_stop", "bracket"],
                "commissions": {"equities": 0.005, "options": 0.65, "futures": 2.25},
                "avg_execution_time": 0.8,
                "reliability": 0.92
            },
            "trading212": {
                "asset_classes": ["equities", "etfs", "crypto"],
                "order_types": ["market", "limit"],
                "commissions": {"equities": 0.0, "etfs": 0.0, "crypto": 0.002},
                "avg_execution_time": 1.2,
                "reliability": 0.88
            }
        }
        
        # Routing preferences
        self.routing_strategy = RoutingStrategy.COST_OPTIMIZATION
        self.preferred_brokers = ["alpaca", "binance", "ibkr", "trading212"]
        self.avoid_brokers = []
        
        # Round-robin tracking
        self.round_robin_index = 0
        
        logger.info(f"OrderRouter initialized for user {self.user_id}")
    
    async def route_order(self, order: Order) -> Dict[str, Any]:
        """
        Route an order to the most appropriate broker.
        
        Args:
            order: Order to route
            
        Returns:
            Routing result with broker selection and submission status
        """
        result = {
            "success": False,
            "broker_name": None,
            "broker_order_id": None,
            "routing_path": [],
            "errors": [],
            "warnings": [],
            "estimated_cost": 0.0,
            "estimated_execution_time": 0.0
        }
        
        try:
            # Get available brokers for this order
            eligible_brokers = await self._get_eligible_brokers(order)
            
            if not eligible_brokers:
                result["errors"] = ["No eligible brokers found for this order"]
                return result
            
            # Select best broker based on routing strategy
            selected_broker = await self._select_broker(order, eligible_brokers)
            
            if not selected_broker:
                result["errors"] = ["No suitable broker selected"]
                return result
            
            result["broker_name"] = selected_broker
            
            # Submit order to selected broker
            submission_result = await self._submit_to_broker(order, selected_broker)
            
            if submission_result["success"]:
                result.update({
                    "success": True,
                    "broker_order_id": submission_result["broker_order_id"],
                    "estimated_cost": submission_result.get("estimated_cost", 0.0),
                    "estimated_execution_time": submission_result.get("estimated_execution_time", 0.0),
                    "routing_path": [selected_broker]
                })
                
                logger.info(f"Order routed successfully to {selected_broker}: {order.id}")
            else:
                result["errors"] = submission_result.get("errors", ["Order submission failed"])
                
                # Try fallback broker if primary fails
                fallback_result = await self._try_fallback_brokers(order, eligible_brokers, selected_broker)
                if fallback_result["success"]:
                    result.update(fallback_result)
                    logger.info(f"Order routed to fallback broker {fallback_result['broker_name']}: {order.id}")
            
        except Exception as e:
            logger.error(f"Order routing error: {str(e)}")
            result["errors"] = [f"Routing error: {str(e)}"]
        
        return result
    
    async def cancel_order(self, order: Order) -> Dict[str, Any]:
        """
        Cancel an order through its broker.
        
        Args:
            order: Order to cancel
            
        Returns:
            Cancellation result
        """
        try:
            broker_name = order.broker_name
            
            # Cancel through broker API
            if broker_name == "binance":
                return await self._cancel_binance_order(order)
            elif broker_name == "alpaca":
                return await self._cancel_alpaca_order(order)
            elif broker_name == "ibkr":
                return await self._cancel_ibkr_order(order)
            elif broker_name == "trading212":
                return await self._cancel_trading212_order(order)
            else:
                return {"success": False, "errors": [f"Unknown broker: {broker_name}"]}
                
        except Exception as e:
            logger.error(f"Order cancellation error: {str(e)}")
            return {"success": False, "errors": [str(e)]}
    
    async def modify_order(self, order: Order, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an order through its broker.
        
        Args:
            order: Order to modify
            modifications: Modifications to apply
            
        Returns:
            Modification result
        """
        try:
            broker_name = order.broker_name
            
            # Modify through broker API
            if broker_name == "binance":
                return await self._modify_binance_order(order, modifications)
            elif broker_name == "alpaca":
                return await self._modify_alpaca_order(order, modifications)
            elif broker_name == "ibkr":
                return await self._modify_ibkr_order(order, modifications)
            elif broker_name == "trading212":
                return await self._modify_trading212_order(order, modifications)
            else:
                return {"success": False, "errors": [f"Unknown broker: {broker_name}"]}
                
        except Exception as e:
            logger.error(f"Order modification error: {str(e)}")
            return {"success": False, "errors": [str(e)]}
    
    async def get_broker_status(self) -> Dict[str, Any]:
        """
        Get status of all configured brokers.
        
        Returns:
            Status of all brokers
        """
        status = {
            "user_id": self.user_id,
            "brokers": {},
            "routing_strategy": self.routing_strategy.value,
            "preferred_brokers": self.preferred_brokers
        }
        
        try:
            # Get broker connections from database
            broker_connections = self.db.query(BrokerConnection).filter(
                BrokerConnection.user_id == self.user_id,
                BrokerConnection.is_active == True
            ).all()
            
            for broker_conn in broker_connections:
                broker_name = broker_conn.broker_name
                
                # Check broker capabilities and status
                capabilities = self.broker_capabilities.get(broker_name, {})
                
                status["brokers"][broker_name] = {
                    "is_configured": True,
                    "is_active": broker_conn.is_active,
                    "last_connected": broker_conn.last_connected,
                    "capabilities": capabilities,
                    "routing_enabled": broker_name not in self.avoid_brokers,
                    "priority": self._get_broker_priority(broker_name)
                }
            
            # Add brokers not in database but in capabilities
            for broker_name in self.broker_capabilities.keys():
                if broker_name not in status["brokers"]:
                    status["brokers"][broker_name] = {
                        "is_configured": False,
                        "is_active": False,
                        "capabilities": self.broker_capabilities[broker_name],
                        "routing_enabled": broker_name not in self.avoid_brokers,
                        "priority": self._get_broker_priority(broker_name)
                    }
            
        except Exception as e:
            logger.error(f"Broker status retrieval error: {str(e)}")
            status["error"] = str(e)
        
        return status
    
    async def set_routing_preferences(self, strategy: RoutingStrategy, 
                                    preferred_brokers: List[str] = None,
                                    avoid_brokers: List[str] = None):
        """
        Set routing preferences and strategy.
        
        Args:
            strategy: Routing strategy to use
            preferred_brokers: Preferred broker list (optional)
            avoid_brokers: Brokers to avoid (optional)
        """
        try:
            self.routing_strategy = strategy
            
            if preferred_brokers:
                self.preferred_brokers = preferred_brokers
            
            if avoid_brokers:
                self.avoid_brokers = avoid_brokers
            
            logger.info(f"Routing preferences updated: strategy={strategy.value}")
            
        except Exception as e:
            logger.error(f"Routing preferences update error: {str(e)}")
    
    async def _get_eligible_brokers(self, order: Order) -> List[str]:
        """Get list of brokers eligible for this order."""
        eligible_brokers = []
        
        try:
            asset_class = order.asset_class
            order_type = order.order_type.value
            
            for broker_name, capabilities in self.broker_capabilities.items():
                # Skip if broker is in avoid list
                if broker_name in self.avoid_brokers:
                    continue
                
                # Check if broker supports the asset class
                if asset_class not in capabilities["asset_classes"]:
                    continue
                
                # Check if broker supports the order type
                if order_type not in capabilities["order_types"]:
                    continue
                
                # Check if broker connection exists and is active
                broker_conn = self.db.query(BrokerConnection).filter(
                    and_(
                        BrokerConnection.user_id == self.user_id,
                        BrokerConnection.broker_name == broker_name,
                        BrokerConnection.is_active == True
                    )
                ).first()
                
                if broker_conn:
                    eligible_brokers.append(broker_name)
            
            logger.debug(f"Eligible brokers for {order.symbol}: {eligible_brokers}")
            
        except Exception as e:
            logger.error(f"Eligible brokers calculation error: {str(e)}")
        
        return eligible_brokers
    
    async def _select_broker(self, order: Order, eligible_brokers: List[str]) -> Optional[str]:
        """Select best broker from eligible list based on routing strategy."""
        if not eligible_brokers:
            return None
        
        try:
            if self.routing_strategy == RoutingStrategy.BROKER_PREFERENCE:
                # Use preferred broker list order
                for broker_name in self.preferred_brokers:
                    if broker_name in eligible_brokers:
                        return broker_name
                
                # Fall back to first eligible broker
                return eligible_brokers[0]
            
            elif self.routing_strategy == RoutingStrategy.COST_OPTIMIZATION:
                # Select broker with lowest estimated cost
                best_broker = None
                best_cost = float('inf')
                
                for broker_name in eligible_brokers:
                    cost = await self._estimate_order_cost(order, broker_name)
                    if cost < best_cost:
                        best_cost = cost
                        best_broker = broker_name
                
                return best_broker
            
            elif self.routing_strategy == RoutingStrategy.EXECUTION_QUALITY:
                # Select broker with best execution quality metrics
                best_broker = None
                best_score = 0
                
                for broker_name in eligible_brokers:
                    score = await self._calculate_execution_score(order, broker_name)
                    if score > best_score:
                        best_score = score
                        best_broker = broker_name
                
                return best_broker
            
            elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                # Round-robin selection
                broker = eligible_brokers[self.round_robin_index % len(eligible_brokers)]
                self.round_robin_index += 1
                return broker
            
            elif self.routing_strategy == RoutingStrategy.FASTEST_EXECUTION:
                # Select broker with fastest average execution
                best_broker = None
                fastest_time = float('inf')
                
                for broker_name in eligible_brokers:
                    exec_time = self.broker_capabilities[broker_name]["avg_execution_time"]
                    if exec_time < fastest_time:
                        fastest_time = exec_time
                        best_broker = broker_name
                
                return best_broker
            
            else:
                # Default to first eligible broker
                return eligible_brokers[0]
                
        except Exception as e:
            logger.error(f"Broker selection error: {str(e)}")
            return eligible_brokers[0] if eligible_brokers else None
    
    async def _estimate_order_cost(self, order: Order, broker_name: str) -> float:
        """Estimate total cost (commissions + spread) for order."""
        try:
            capabilities = self.broker_capabilities[broker_name]
            
            # Commission calculation
            order_value = order.quantity * (order.limit_price or 100)  # Assume $100 if no limit price
            commission_rate = capabilities["commissions"].get(order.asset_class, 0.001)
            commission = order_value * commission_rate
            
            # Estimated spread cost (simplified)
            spread_bps = 1.0  # 1 basis point as estimate
            spread_cost = order_value * (spread_bps / 10000)
            
            return commission + spread_cost
            
        except Exception as e:
            logger.error(f"Cost estimation error: {str(e)}")
            return 0.0
    
    async def _calculate_execution_score(self, order: Order, broker_name: str) -> float:
        """Calculate execution quality score for broker."""
        try:
            capabilities = self.broker_capabilities[broker_name]
            
            # Score components
            reliability = capabilities["reliability"]
            speed_score = 1.0 / (capabilities["avg_execution_time"] + 0.1)  # Favor faster execution
            
            # Weight the components
            total_score = (reliability * 0.7) + (speed_score * 0.3)
            
            return total_score
            
        except Exception as e:
            logger.error(f"Execution score calculation error: {str(e)}")
            return 0.5
    
    def _get_broker_priority(self, broker_name: str) -> int:
        """Get routing priority for broker."""
        try:
            priority_map = {
                "binance": 1,
                "alpaca": 2,
                "ibkr": 3,
                "trading212": 4
            }
            return priority_map.get(broker_name, 999)
        except Exception:
            return 999
    
    async def _submit_to_broker(self, order: Order, broker_name: str) -> Dict[str, Any]:
        """Submit order to specific broker."""
        try:
            if broker_name == "binance":
                return await self._submit_binance_order(order)
            elif broker_name == "alpaca":
                return await self._submit_alpaca_order(order)
            elif broker_name == "ibkr":
                return await self._submit_ibkr_order(order)
            elif broker_name == "trading212":
                return await self._submit_trading212_order(order)
            else:
                return {"success": False, "errors": [f"Unknown broker: {broker_name}"]}
                
        except Exception as e:
            logger.error(f"Broker submission error: {str(e)}")
            return {"success": False, "errors": [str(e)]}
    
    async def _try_fallback_brokers(self, order: Order, eligible_brokers: List[str], 
                                  failed_broker: str) -> Dict[str, Any]:
        """Try fallback brokers if primary broker fails."""
        try:
            fallback_brokers = [b for b in eligible_brokers if b != failed_broker]
            
            for fallback_broker in fallback_brokers:
                result = await self._submit_to_broker(order, fallback_broker)
                if result["success"]:
                    return {
                        "success": True,
                        "broker_name": fallback_broker,
                        "broker_order_id": result["broker_order_id"],
                        "routing_path": [fallback_broker],
                        "is_fallback": True
                    }
            
            return {"success": False, "errors": ["All brokers failed"]}
            
        except Exception as e:
            logger.error(f"Fallback broker attempt error: {str(e)}")
            return {"success": False, "errors": [str(e)]}
    
    # Broker-specific implementations (simplified)
    
    async def _submit_binance_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to Binance."""
        # This would integrate with actual Binance API
        try:
            # Placeholder implementation
            broker_order_id = f"BN_{order.client_order_id}"
            
            return {
                "success": True,
                "broker_order_id": broker_order_id,
                "estimated_cost": 0.001 * order.quantity * (order.limit_price or 100),
                "estimated_execution_time": 0.5
            }
            
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    async def _submit_alpaca_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to Alpaca."""
        try:
            broker_order_id = f"ALP_{order.client_order_id}"
            
            return {
                "success": True,
                "broker_order_id": broker_order_id,
                "estimated_cost": 0.0,  # Commission-free
                "estimated_execution_time": 0.3
            }
            
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    async def _submit_ibkr_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to Interactive Brokers."""
        try:
            broker_order_id = f"IBKR_{order.client_order_id}"
            
            return {
                "success": True,
                "broker_order_id": broker_order_id,
                "estimated_cost": 0.005 * order.quantity,
                "estimated_execution_time": 0.8
            }
            
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    async def _submit_trading212_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to Trading 212."""
        try:
            broker_order_id = f"T212_{order.client_order_id}"
            
            return {
                "success": True,
                "broker_order_id": broker_order_id,
                "estimated_cost": 0.0,
                "estimated_execution_time": 1.2
            }
            
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    # Cancellation methods
    
    async def _cancel_binance_order(self, order: Order) -> Dict[str, Any]:
        """Cancel Binance order."""
        return {"success": True, "message": "Order cancelled successfully"}
    
    async def _cancel_alpaca_order(self, order: Order) -> Dict[str, Any]:
        """Cancel Alpaca order."""
        return {"success": True, "message": "Order cancelled successfully"}
    
    async def _cancel_ibkr_order(self, order: Order) -> Dict[str, Any]:
        """Cancel IBKR order."""
        return {"success": True, "message": "Order cancelled successfully"}
    
    async def _cancel_trading212_order(self, order: Order) -> Dict[str, Any]:
        """Cancel Trading 212 order."""
        return {"success": True, "message": "Order cancelled successfully"}
    
    # Modification methods
    
    async def _modify_binance_order(self, order: Order, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify Binance order."""
        return {"success": True, "message": "Order modified successfully"}
    
    async def _modify_alpaca_order(self, order: Order, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify Alpaca order."""
        return {"success": True, "message": "Order modified successfully"}
    
    async def _modify_ibkr_order(self, order: Order, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify IBKR order."""
        return {"success": True, "message": "Order modified successfully"}
    
    async def _modify_trading212_order(self, order: Order, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify Trading 212 order."""
        return {"success": True, "message": "Order modified successfully"}
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
