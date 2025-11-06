"""
Risk Management System
Monitors and controls trading risk across all positions and orders
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import asyncio

from loguru import logger


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of risk violations"""
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    MAX_OPEN_ORDERS_EXCEEDED = "max_open_orders_exceeded"
    CONCENTRATION_RISK = "concentration_risk"
    MARGIN_CALL_RISK = "margin_call_risk"
    VOLATILITY_RISK = "volatility_risk"


@dataclass
class PositionRisk:
    """Position risk assessment"""
    symbol: str
    quantity: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    risk_score: float
    risk_level: RiskLevel
    concentration: float  # % of portfolio


@dataclass
class RiskViolation:
    """Risk violation record"""
    timestamp: datetime
    violation_type: ViolationType
    description: str
    current_value: float
    limit_value: float
    severity: RiskLevel
    position_details: Optional[Dict] = None


class RiskManager:
    """
    Comprehensive risk management system
    
    Features:
    - Position size limits
    - Daily loss limits
    - Concentration risk monitoring
    - Volatility-based limits
    - Real-time risk assessment
    """
    
    def __init__(
        self,
        max_position_size: float = 10000.0,
        max_daily_loss: float = 1000.0,
        max_open_orders: int = 50,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size in currency
            max_daily_loss: Maximum daily loss in currency
            max_open_orders: Maximum number of open orders
            risk_per_trade: Risk per trade as fraction of capital (2% default)
        """
        self.max_position_size = Decimal(str(max_position_size))
        self.max_daily_loss = Decimal(str(max_daily_loss))
        self.max_open_orders = max_open_orders
        self.risk_per_trade = Decimal(str(risk_per_trade))
        
        # Risk tracking
        self.daily_pnl = Decimal('0')
        self.daily_start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.open_orders = []
        self.positions = {}
        self.violations = []
        
        # Risk metrics
        self.portfolio_value = Decimal('0')
        self.cash_available = Decimal('0')
        
        logger.info(f"Risk Manager initialized - Max position: {max_position_size}, Max loss: {max_daily_loss}")
    
    async def reset_daily_counters(self):
        """Reset daily risk counters"""
        current_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if current_day > self.daily_start_time:
            self.daily_pnl = Decimal('0')
            self.daily_start_time = current_day
            logger.info("Daily risk counters reset")
    
    async def check_trade_risk(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        account_value: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Check if a trade complies with risk limits
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Order price
            account_value: Total account value
            
        Returns:
            Risk assessment result
        """
        try:
            # Reset daily counters if needed
            await self.reset_daily_counters()
            
            # Calculate trade value
            trade_value = quantity * price
            
            # Check position size limit
            if abs(trade_value) > self.max_position_size:
                return self._create_violation_response(
                    ViolationType.POSITION_SIZE_EXCEEDED,
                    f"Trade value {trade_value} exceeds maximum position size {self.max_position_size}",
                    float(abs(trade_value)),
                    float(self.max_position_size),
                    {"symbol": symbol, "side": side, "quantity": float(quantity), "price": float(price)}
                )
            
            # Check daily loss limit
            current_pnl = self.daily_pnl
            if side.upper() == 'BUY' and current_pnl < -self.max_daily_loss:
                return self._create_violation_response(
                    ViolationType.DAILY_LOSS_EXCEEDED,
                    f"Current daily P&L {current_pnl} would exceed loss limit {self.max_daily_loss}",
                    float(abs(current_pnl)),
                    float(self.max_daily_loss),
                    {"symbol": symbol, "side": side, "current_pnl": float(current_pnl)}
                )
            
            # Check concentration risk
            if account_value:
                concentration = abs(trade_value) / account_value
                if concentration > 0.1:  # 10% concentration limit
                    return self._create_violation_response(
                        ViolationType.CONCENTRATION_RISK,
                        f"Position concentration {concentration:.2%} exceeds 10% limit",
                        float(concentration),
                        0.1,
                        {"symbol": symbol, "concentration": float(concentration)}
                    )
            
            # Check risk per trade
            if account_value and account_value > 0:
                risk_amount = account_value * self.risk_per_trade
                if abs(trade_value) > risk_amount:
                    return self._create_violation_response(
                        ViolationType.POSITION_SIZE_EXCEEDED,
                        f"Trade risk {abs(trade_value)} exceeds per-trade limit {risk_amount}",
                        float(abs(trade_value)),
                        float(risk_amount),
                        {"symbol": symbol, "risk_amount": float(risk_amount)}
                    )
            
            # Trade approved
            return {
                'approved': True,
                'risk_level': RiskLevel.LOW,
                'warnings': [],
                'position_size': float(abs(trade_value)),
                'risk_score': float(abs(trade_value) / (account_value or Decimal('1')))
            }
            
        except Exception as e:
            logger.error(f"Error in trade risk check: {e}")
            return {
                'approved': False,
                'error': str(e),
                'risk_level': RiskLevel.HIGH
            }
    
    async def check_order_limit(
        self,
        current_open_orders: List[Dict[str, Any]],
        new_order_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check if adding new orders would exceed limits
        
        Args:
            current_open_orders: Currently open orders
            new_order_symbols: Symbols for new orders (optional)
            
        Returns:
            Order limit check result
        """
        try:
            total_orders = len(current_open_orders)
            if new_order_symbols:
                total_orders += len(new_order_symbols)
            
            if total_orders > self.max_open_orders:
                return self._create_violation_response(
                    ViolationType.MAX_OPEN_ORDERS_EXCEEDED,
                    f"Total orders {total_orders} exceed maximum {self.max_open_orders}",
                    total_orders,
                    self.max_open_orders,
                    {"current_orders": len(current_open_orders), "new_orders": len(new_order_symbols or [])}
                )
            
            return {
                'approved': True,
                'current_orders': total_orders,
                'remaining_capacity': self.max_open_orders - total_orders
            }
            
        except Exception as e:
            logger.error(f"Error in order limit check: {e}")
            return {
                'approved': False,
                'error': str(e)
            }
    
    async def calculate_position_risks(
        self,
        positions: List[Dict[str, Any]]
    ) -> List[PositionRisk]:
        """
        Calculate risk metrics for all positions
        
        Args:
            positions: Current positions list
            
        Returns:
            List of position risk assessments
        """
        try:
            position_risks = []
            total_portfolio_value = sum(
                Decimal(str(pos.get('market_value', 0)))
                for pos in positions
            ) + self.cash_available
            
            self.portfolio_value = total_portfolio_value
            
            for position in positions:
                symbol = position.get('symbol', 'UNKNOWN')
                quantity = Decimal(str(position.get('quantity', 0)))
                market_value = Decimal(str(position.get('market_value', 0)))
                unrealized_pnl = Decimal(str(position.get('unrealized_pnl', 0)))
                
                # Calculate risk metrics
                concentration = abs(market_value) / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Risk score based on multiple factors
                risk_score = self._calculate_risk_score(
                    concentration, 
                    abs(unrealized_pnl) / market_value if market_value != 0 else 0,
                    abs(market_value) / self.max_position_size
                )
                
                risk_level = self._determine_risk_level(risk_score, concentration)
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    concentration=concentration
                )
                
                position_risks.append(position_risk)
            
            return position_risks
            
        except Exception as e:
            logger.error(f"Error calculating position risks: {e}")
            return []
    
    def _calculate_risk_score(
        self,
        concentration: float,
        volatility: float,
        size_ratio: float
    ) -> float:
        """Calculate composite risk score (0.0 to 1.0)"""
        # Weighted risk factors
        concentration_weight = 0.4
        volatility_weight = 0.3
        size_weight = 0.3
        
        risk_score = (
            min(concentration / 0.2, 1.0) * concentration_weight +
            min(volatility, 1.0) * volatility_weight +
            min(size_ratio, 1.0) * size_weight
        )
        
        return min(risk_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float, concentration: float) -> RiskLevel:
        """Determine risk level based on risk score and concentration"""
        if risk_score >= 0.8 or concentration >= 0.3:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6 or concentration >= 0.2:
            return RiskLevel.HIGH
        elif risk_score >= 0.3 or concentration >= 0.1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _create_violation_response(
        self,
        violation_type: ViolationType,
        description: str,
        current_value: float,
        limit_value: float,
        details: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a risk violation response"""
        violation = RiskViolation(
            timestamp=datetime.utcnow(),
            violation_type=violation_type,
            description=description,
            current_value=current_value,
            limit_value=limit_value,
            severity=self._determine_violation_severity(current_value, limit_value),
            position_details=details
        )
        
        self.violations.append(violation)
        
        logger.warning(f"Risk violation: {description}")
        
        return {
            'approved': False,
            'violation': {
                'type': violation_type.value,
                'description': description,
                'current_value': current_value,
                'limit_value': limit_value,
                'severity': violation.severity.value,
                'timestamp': violation.timestamp.isoformat()
            },
            'risk_level': violation.severity
        }
    
    def _determine_violation_severity(self, current: float, limit: float) -> RiskLevel:
        """Determine violation severity"""
        ratio = current / limit if limit > 0 else 2.0
        
        if ratio >= 2.0:
            return RiskLevel.CRITICAL
        elif ratio >= 1.5:
            return RiskLevel.HIGH
        elif ratio >= 1.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def update_daily_pnl(self, pnl_change: Decimal):
        """Update daily P&L tracking"""
        await self.reset_daily_counters()
        self.daily_pnl += pnl_change
        
        # Check if daily loss limit exceeded
        if self.daily_pnl < -self.max_daily_loss:
            logger.error(f"Daily loss limit exceeded: {self.daily_pnl} < -{self.max_daily_loss}")
            # Trigger emergency stop logic here
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        await self.reset_daily_counters()
        
        # Get recent violations
        recent_violations = [
            v for v in self.violations 
            if v.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            'risk_metrics': {
                'daily_pnl': float(self.daily_pnl),
                'max_daily_loss': float(self.max_daily_loss),
                'daily_loss_remaining': float(self.max_daily_loss + self.daily_pnl),
                'portfolio_value': float(self.portfolio_value),
                'cash_available': float(self.cash_available),
                'open_orders_count': len(self.open_orders),
                'max_position_size': float(self.max_position_size),
                'risk_per_trade': float(self.risk_per_trade)
            },
            'risk_level': self._calculate_overall_risk_level(),
            'violations_24h': len(recent_violations),
            'recent_violations': [
                {
                    'timestamp': v.timestamp.isoformat(),
                    'type': v.violation_type.value,
                    'description': v.description,
                    'severity': v.severity.value
                }
                for v in recent_violations[-10:]  # Last 10 violations
            ],
            'compliance_status': 'PASS' if len(recent_violations) == 0 else 'VIOLATIONS'
        }
    
    def _calculate_overall_risk_level(self) -> RiskLevel:
        """Calculate overall portfolio risk level"""
        # Simple implementation - can be enhanced
        recent_violations = [
            v for v in self.violations 
            if v.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        if any(v.severity == RiskLevel.CRITICAL for v in recent_violations):
            return RiskLevel.CRITICAL
        elif any(v.severity == RiskLevel.HIGH for v in recent_violations):
            return RiskLevel.HIGH
        elif any(v.severity == RiskLevel.MEDIUM for v in recent_violations):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def update_portfolio_value(self, total_value: Decimal, cash: Decimal):
        """Update portfolio and cash values"""
        self.portfolio_value = total_value
        self.cash_available = cash


# Example usage and testing
if __name__ == "__main__":
    async def test_risk_manager():
        manager = RiskManager(
            max_position_size=5000.0,
            max_daily_loss=500.0,
            max_open_orders=10,
            risk_per_trade=0.02
        )
        
        # Test trade approval
        result = await manager.check_trade_risk(
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("150.00"),
            account_value=Decimal("100000")
        )
        
        print("Trade Risk Check:", result)
        
        # Test position risk calculation
        positions = [
            {"symbol": "AAPL", "quantity": 10, "market_value": 1500, "unrealized_pnl": 50},
            {"symbol": "GOOGL", "quantity": 5, "market_value": 1250, "unrealized_pnl": -25}
        ]
        
        risks = await manager.calculate_position_risks(positions)
        for risk in risks:
            print(f"Position {risk.symbol}: Risk Level {risk.risk_level.value}")
    
    asyncio.run(test_risk_manager())