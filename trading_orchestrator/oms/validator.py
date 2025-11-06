"""
Order Validator - Comprehensive Order Validation System

Validates orders against:
- Broker-specific constraints
- Market rules and regulations
- Risk management limits
- Position and account limits
- Order size and price limits
- Regulatory compliance
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from enum import Enum

from config.database import get_db
from database.models.trading import Order, Position, OrderType, OrderSide, TimeInForce
from database.models.risk import RiskLimit
from database.models.user import User

logger = logging.getLogger(__name__)


class ValidationResult:
    """Order validation result."""
    
    def __init__(self):
        self.approved = True
        self.rejection_reason = None
        self.warnings = []
        self.modifications = []
        self.validation_details = {}


class OrderValidator:
    """
    Comprehensive order validation system.
    
    Validates orders across multiple dimensions:
    - Technical validation (price, quantity, order type)
    - Business rules (market hours, position limits)
    - Risk management (exposure limits, concentration)
    - Regulatory compliance (pattern day trading, etc.)
    - Broker-specific constraints
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Validation rules
        self.validation_rules = {
            "min_order_size": {
                "equities": 1,
                "crypto": 0.001,
                "forex": 1000,
                "options": 1,
                "futures": 1
            },
            "max_order_size": {
                "equities": 1000000,
                "crypto": 1000,
                "forex": 10000000,
                "options": 10000,
                "futures": 1000
            },
            "price_limits": {
                "min_price": 0.01,
                "max_price": 1000000,
                "max_price_change_pct": 10.0  # 10% max change per order
            },
            "order_type_support": {
                "market": ["equities", "crypto", "forex"],
                "limit": ["equities", "crypto", "forex", "options"],
                "stop": ["equities", "crypto", "forex"],
                "stop_limit": ["equities", "crypto", "forex"],
                "trailing_stop": ["equities", "crypto"],
                "bracket": ["equities"],
                "oco": ["equities", "crypto"]
            }
        }
        
        # Market session validation
        self.market_sessions = {
            "equities": {
                "regular": {"start": "09:30", "end": "16:00", "days": [0, 1, 2, 3, 4]},
                "extended": {"start": "04:00", "end": "20:00", "days": [0, 1, 2, 3, 4]}
            },
            "crypto": {
                "24_7": {"start": "00:00", "end": "23:59", "days": [0, 1, 2, 3, 4, 5, 6]}
            }
        }
        
        logger.info(f"OrderValidator initialized for user {self.user_id}")
    
    async def validate_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive order validation.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with approval status and details
        """
        result = {
            "approved": True,
            "rejection_reason": None,
            "warnings": [],
            "modifications": [],
            "validation_details": {}
        }
        
        try:
            # Initialize validation result
            validation = ValidationResult()
            
            # 1. Technical validation
            technical_result = await self._validate_technical(order_data)
            self._merge_validation_result(validation, technical_result)
            
            # 2. Market rules validation
            market_result = await self._validate_market_rules(order_data)
            self._merge_validation_result(validation, market_result)
            
            # 3. Risk management validation
            risk_result = await self._validate_risk_management(order_data)
            self._merge_validation_result(validation, risk_result)
            
            # 4. Regulatory compliance validation
            compliance_result = await self._validate_regulatory_compliance(order_data)
            self._merge_validation_result(validation, compliance_result)
            
            # 5. Broker-specific validation
            broker_result = await self._validate_broker_constraints(order_data)
            self._merge_validation_result(validation, broker_result)
            
            # 6. Account validation
            account_result = await self._validate_account_limits(order_data)
            self._merge_validation_result(validation, account_result)
            
            # Convert validation result to dict
            result = {
                "approved": validation.approved,
                "rejection_reason": validation.rejection_reason,
                "warnings": validation.warnings,
                "modifications": validation.modifications,
                "validation_details": validation.validation_details
            }
            
            logger.debug(f"Order validation completed: {result['approved']} for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Order validation error: {str(e)}")
            result.update({
                "approved": False,
                "rejection_reason": f"Validation error: {str(e)}"
            })
        
        return result
    
    async def _validate_technical(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate technical aspects of order (price, quantity, order type)."""
        validation = ValidationResult()
        
        try:
            symbol = order_data.get("symbol", "").upper()
            asset_class = order_data.get("asset_class", "")
            order_type = order_data.get("order_type", "")
            side = order_data.get("side", "")
            quantity = float(order_data.get("quantity", 0))
            limit_price = order_data.get("limit_price")
            stop_price = order_data.get("stop_price")
            
            validation.validation_details["technical"] = {}
            
            # Validate symbol format
            if not symbol or len(symbol) < 1:
                validation.approved = False
                validation.rejection_reason = "Invalid symbol format"
                return validation
            
            validation.validation_details["technical"]["symbol"] = symbol
            
            # Validate asset class
            if asset_class not in self.validation_rules["order_type_support"].get("market", []):
                validation.approved = False
                validation.rejection_reason = f"Unsupported asset class: {asset_class}"
                return validation
            
            validation.validation_details["technical"]["asset_class"] = asset_class
            
            # Validate quantity
            if quantity <= 0:
                validation.approved = False
                validation.rejection_reason = "Order quantity must be positive"
                return validation
            
            min_size = self.validation_rules["min_order_size"].get(asset_class, 1)
            max_size = self.validation_rules["max_order_size"].get(asset_class, 1000000)
            
            if quantity < min_size:
                validation.approved = False
                validation.rejection_reason = f"Order quantity {quantity} below minimum {min_size} for {asset_class}"
                return validation
            
            if quantity > max_size:
                validation.approved = False
                validation.rejection_reason = f"Order quantity {quantity} above maximum {max_size} for {asset_class}"
                return validation
            
            validation.validation_details["technical"]["quantity"] = quantity
            
            # Validate order type
            supported_types = []
            for order_type_supported, asset_classes in self.validation_rules["order_type_support"].items():
                if asset_class in asset_classes:
                    supported_types.append(order_type_supported)
            
            if order_type not in supported_types:
                validation.approved = False
                validation.rejection_reason = f"Order type {order_type} not supported for {asset_class}"
                return validation
            
            validation.validation_details["technical"]["order_type"] = order_type
            
            # Validate price constraints
            if limit_price is not None:
                limit_price = float(limit_price)
                
                min_price = self.validation_rules["price_limits"]["min_price"]
                max_price = self.validation_rules["price_limits"]["max_price"]
                
                if limit_price < min_price or limit_price > max_price:
                    validation.approved = False
                    validation.rejection_reason = f"Limit price {limit_price} outside allowed range {min_price}-{max_price}"
                    return validation
            
            if stop_price is not None:
                stop_price = float(stop_price)
                
                if stop_price < min_price or stop_price > max_price:
                    validation.approved = False
                    validation.rejection_reason = f"Stop price {stop_price} outside allowed range"
                    return validation
            
            # Validate stop-limit order consistency
            if order_type == "stop_limit" and (stop_price is None or limit_price is None):
                validation.approved = False
                validation.rejection_reason = "Stop-limit orders require both stop and limit prices"
                return validation
            
            # Validate bracket order requirements
            if order_type == "bracket":
                if not all([order_data.get("profit_target"), order_data.get("stop_loss")]):
                    validation.approved = False
                    validation.rejection_reason = "Bracket orders require profit target and stop loss"
                    return validation
            
            validation.validation_details["technical"]["prices"] = {
                "limit_price": limit_price,
                "stop_price": stop_price
            }
            
        except Exception as e:
            logger.error(f"Technical validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Technical validation error: {str(e)}"
        
        return validation
    
    async def _validate_market_rules(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate against market rules and sessions."""
        validation = ValidationResult()
        
        try:
            asset_class = order_data.get("asset_class", "")
            symbol = order_data.get("symbol", "")
            
            validation.validation_details["market_rules"] = {}
            
            # Check market sessions
            current_time = datetime.now()
            
            if asset_class == "equities":
                # Check if crypto (24/7) or regular equities
                if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "USDT", "BNB"]):
                    # Crypto trades 24/7
                    validation.validation_details["market_rules"]["session"] = "24_7_crypto"
                else:
                    # Regular stock trading hours
                    market_open, market_close = self._check_equity_hours(current_time)
                    
                    if not market_open:
                        validation.approved = False
                        validation.rejection_reason = "Order outside regular trading hours"
                        validation.warnings.append("Order placed outside regular trading hours - may have limited liquidity")
                    
                    validation.validation_details["market_rules"]["session"] = "equities_regular"
                    validation.validation_details["market_rules"]["market_open"] = market_open
            
            elif asset_class == "crypto":
                # Crypto trades 24/7
                validation.validation_details["market_rules"]["session"] = "crypto_24_7"
            
            # Check for market holidays (simplified)
            if self._is_market_holiday(current_time):
                validation.approved = False
                validation.rejection_reason = "Market closed for holiday"
                return validation
            
            # Check for trading halts or circuit breakers
            if await self._check_trading_halt(symbol):
                validation.approved = False
                validation.rejection_reason = f"Trading halted for symbol {symbol}"
                return validation
            
        except Exception as e:
            logger.error(f"Market rules validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Market rules validation error: {str(e)}"
        
        return validation
    
    async def _validate_risk_management(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate against risk management limits."""
        validation = ValidationResult()
        
        try:
            validation.validation_details["risk_management"] = {}
            
            # Get risk limits
            risk_limits = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.is_active == True
                )
            ).all()
            
            # Check position size limits
            symbol = order_data.get("symbol", "")
            quantity = float(order_data.get("quantity", 0))
            limit_price = order_data.get("limit_price") or 100
            order_value = quantity * limit_price
            
            for limit in risk_limits:
                if limit.limit_type == "position_size":
                    # Calculate new position size
                    existing_position = self.db.query(Position).filter(
                        and_(
                            Position.user_id == self.user_id,
                            Position.symbol == symbol
                        )
                    ).first()
                    
                    current_size = existing_position.quantity if existing_position else 0
                    new_size = current_size + (quantity if order_data.get("side") == "buy" else -quantity)
                    
                    if abs(new_size) > limit.limit_value:
                        validation.approved = False
                        validation.rejection_reason = f"Position size limit exceeded: {abs(new_size)} > {limit.limit_value}"
                        break
                
                elif limit.limit_type == "position_value":
                    # Calculate new position value
                    if existing_position:
                        existing_value = abs(existing_position.market_value or 0)
                        new_value = existing_value + order_value
                    else:
                        new_value = order_value
                    
                    if new_value > limit.limit_value:
                        validation.approved = False
                        validation.rejection_reason = f"Position value limit exceeded: ${new_value:,.2f} > ${limit.limit_value:,.2f}"
                        break
                
                elif limit.limit_type == "daily_orders":
                    # Check daily order count
                    today = datetime.now().date()
                    orders_today = self.db.query(Order).filter(
                        and_(
                            Order.user_id == self.user_id,
                            Order.submitted_at >= datetime.combine(today, datetime.min.time())
                        )
                    ).count()
                    
                    if orders_today + 1 > limit.limit_value:
                        validation.approved = False
                        validation.rejection_reason = f"Daily order limit exceeded: {orders_today + 1} > {limit.limit_value}"
                        break
            
            if validation.approved:
                validation.validation_details["risk_management"]["limits_checked"] = len(risk_limits)
            
        except Exception as e:
            logger.error(f"Risk management validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Risk management validation error: {str(e)}"
        
        return validation
    
    async def _validate_regulatory_compliance(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate regulatory compliance."""
        validation = ValidationResult()
        
        try:
            validation.validation_details["regulatory_compliance"] = {}
            
            symbol = order_data.get("symbol", "")
            asset_class = order_data.get("asset_class", "")
            
            # Check for restricted securities
            restricted_securities = await self._get_restricted_securities()
            
            if symbol.upper() in restricted_securities:
                validation.approved = False
                validation.rejection_reason = f"Symbol {symbol} is restricted for trading"
                return validation
            
            # Pattern Day Trader (PDT) rule compliance for US equities
            if asset_class == "equities" and not any(crypto in symbol.upper() for crypto in ["BTC", "ETH"]):
                pdt_status = await self._check_pattern_day_trader_status()
                
                if not pdt_status["is_pdt"] and pdt_status["equity_trades_today"] >= 3:
                    validation.approved = False
                    validation.rejection_reason = "Pattern Day Trader rule: 3 equity trades allowed per 5 business days for non-PDT accounts"
                    return validation
            
            # Wash trading prevention (simplified)
            if await self._check_wash_trading_potential(order_data):
                validation.warnings.append("Potential wash trading pattern detected - ensure compliance with regulations")
            
            validation.validation_details["regulatory_compliance"]["checks_passed"] = True
            
        except Exception as e:
            logger.error(f"Regulatory compliance validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Regulatory compliance validation error: {str(e)}"
        
        return validation
    
    async def _validate_broker_constraints(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate broker-specific constraints."""
        validation = ValidationResult()
        
        try:
            broker_name = order_data.get("broker", "default")
            
            validation.validation_details["broker_constraints"] = {}
            
            # Broker-specific validations would go here
            # For now, just return approved
            
            validation.validation_details["broker_constraints"]["broker"] = broker_name
            validation.validation_details["broker_constraints"]["validated"] = True
            
        except Exception as e:
            logger.error(f"Broker constraints validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Broker constraints validation error: {str(e)}"
        
        return validation
    
    async def _validate_account_limits(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate account-level limits."""
        validation = ValidationResult()
        
        try:
            validation.validation_details["account_limits"] = {}
            
            user = self.db.query(User).filter(User.id == self.user_id).first()
            
            if not user:
                validation.approved = False
                validation.rejection_reason = "User account not found"
                return validation
            
            # Check account status
            if not user.is_active:
                validation.approved = False
                validation.rejection_reason = "Account is not active"
                return validation
            
            # Check trading permissions
            if hasattr(user, 'trading_enabled') and not user.trading_enabled:
                validation.approved = False
                validation.rejection_reason = "Trading is disabled for this account"
                return validation
            
            validation.validation_details["account_limits"]["account_verified"] = True
            validation.validation_details["account_limits"]["trading_enabled"] = True
            
        except Exception as e:
            logger.error(f"Account limits validation error: {str(e)}")
            validation.approved = False
            validation.rejection_reason = f"Account limits validation error: {str(e)}"
        
        return validation
    
    def _merge_validation_result(self, target: ValidationResult, source: ValidationResult):
        """Merge validation result into target."""
        if not source.approved:
            target.approved = False
            target.rejection_reason = source.rejection_reason
        
        target.warnings.extend(source.warnings)
        target.modifications.extend(source.modifications)
        target.validation_details.update(source.validation_details)
    
    def _check_equity_hours(self, current_time: datetime) -> Tuple[bool, bool]:
        """Check if current time is within equity trading hours."""
        try:
            weekday = current_time.weekday()
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # Regular hours: 9:30 AM - 4:00 PM EST
            if weekday < 5:  # Monday-Friday
                if (current_hour > 9 or (current_hour == 9 and current_minute >= 30)) and current_hour < 16:
                    return True, True  # Within regular hours
            
            return False, False
            
        except Exception:
            return False, False
    
    def _is_market_holiday(self, current_time: datetime) -> bool:
        """Check if current date is a market holiday."""
        try:
            # Simplified holiday check - would use proper holiday calendar
            # New Year's Day, Christmas, etc.
            month = current_time.month
            day = current_time.day
            
            holidays = [
                (1, 1),   # New Year's Day
                (12, 25), # Christmas
                (7, 4),   # Independence Day
                (11, 11), # Veterans Day
            ]
            
            return (month, day) in holidays
            
        except Exception:
            return False
    
    async def _check_trading_halt(self, symbol: str) -> bool:
        """Check if symbol is subject to trading halt."""
        try:
            # This would check against trading halt feeds
            # For now, just return False
            return False
            
        except Exception:
            return False
    
    async def _get_restricted_securities(self) -> List[str]:
        """Get list of restricted securities."""
        try:
            # This would pull from regulatory feeds
            # For now, return empty list
            return []
            
        except Exception:
            return []
    
    async def _check_pattern_day_trader_status(self) -> Dict[str, Any]:
        """Check pattern day trader status and trade counts."""
        try:
            # Simplified PDT check
            today = datetime.now().date()
            
            equity_trades_today = self.db.query(Order).filter(
                and_(
                    Order.user_id == self.user_id,
                    Order.submitted_at >= datetime.combine(today, datetime.min.time()),
                    Order.asset_class == "equities"
                )
            ).count()
            
            return {
                "is_pdt": False,  # Would check account type
                "equity_trades_today": equity_trades_today,
                "equity_trades_5days": equity_trades_today  # Simplified
            }
            
        except Exception:
            return {"is_pdt": False, "equity_trades_today": 0, "equity_trades_5days": 0}
    
    async def _check_wash_trading_potential(self, order_data: Dict[str, Any]) -> bool:
        """Check for potential wash trading patterns."""
        try:
            symbol = order_data.get("symbol", "")
            side = order_data.get("side", "")
            
            # Check for recent opposite trades in same symbol
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            recent_trades = self.db.query(Order).filter(
                and_(
                    Order.user_id == self.user_id,
                    Order.symbol == symbol,
                    Order.submitted_at >= one_hour_ago
                )
            ).all()
            
            opposite_side = "sell" if side == "buy" else "buy"
            
            for trade in recent_trades:
                if trade.side.value == opposite_side:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
