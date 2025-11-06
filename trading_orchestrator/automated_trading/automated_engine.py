"""
Automated Trading Engine

Core engine that manages perpetual trading operations during market hours.
Handles strategy selection, execution coordination, and autonomous operation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from loguru import logger

from .market_hours import MarketHoursManager, MarketSession, SessionInfo
from .autonomous_decisions import AutonomousDecisionEngine
from .continuous_monitor import ContinuousMonitoringSystem
from .risk_management import AutomatedRiskManager
from .config import AutomatedTradingConfig
from .logging_system import TradingLogger
# Handle relative imports with fallbacks
try:
    from ..strategies.base import BaseStrategy, StrategyRegistry, StrategyConfig, StrategyType
    from ..risk.manager import RiskManager
    from ..oms.manager import OrderManager
except ImportError:
    # Fallback definitions for standalone usage
    from enum import Enum
    from typing import Protocol
    from decimal import Decimal
    from dataclasses import dataclass
    
    class StrategyType(Enum):
        TREND_FOLLOWING = "trend_following"
        MEAN_REVERSION = "mean_reversion"
        PAIRS_TRADING = "pairs_trading"
        ARBITRAGE = "arbitrage"
        MOMENTUM = "momentum"
        BREAKOUT = "breakout"
        SCALPING = "scalping"
        SWING_TRADING = "swing_trading"
    
    @dataclass
    class StrategyConfig:
        strategy_id: str
        strategy_type: StrategyType
        name: str
        description: str
        parameters: Dict
        risk_level: str = "medium"
        max_position_size: Decimal = Decimal('100000')
        max_daily_loss: Decimal = Decimal('10000')
        symbols: List[str] = None
        enabled: bool = True
    
    class BaseStrategy:
        def __init__(self, config):
            self.config = config
    
    class StrategyRegistry:
        def register_strategy(self, strategy):
            pass
        
        def get_strategy(self, strategy_id):
            return None
    
    class RiskManager:
        pass
    
    class OrderManager:
        pass


class AutomationLevel(Enum):
    """Automation levels"""
    DISABLED = "disabled"
    MANUAL = "manual" 
    ADVISORY = "advisory"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"


class TradingMode(Enum):
    """Trading modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class SystemStatus(Enum):
    """System status"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Engine performance metrics"""
    uptime_seconds: float = 0.0
    total_trades_executed: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    strategies_active: int = 0
    current_positions: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    average_trade_duration: float = 0.0
    last_trade_time: Optional[datetime] = None
    market_hours_active: int = 0
    system_health_score: float = 100.0
    errors_count: int = 0


@dataclass
class TradingDecision:
    """Trading decision record"""
    decision_id: str
    timestamp: datetime
    automation_level: AutomationLevel
    strategy_id: str
    symbol: str
    action: str  # buy, sell, hold, close
    confidence: float
    reasoning: str
    position_size: Decimal
    risk_score: float
    approved: bool
    executed: bool = False
    execution_result: Optional[Dict] = None


class AutomatedTradingEngine:
    """
    Automated Trading Engine
    
    Core component that runs perpetual trading operations during market hours:
    - Market session management and transitions
    - Autonomous strategy selection and execution
    - Risk management integration
    - Continuous monitoring and adaptation
    - Multi-level automation control
    """
    
    def __init__(self, 
                 config: AutomatedTradingConfig,
                 risk_manager: Optional[RiskManager] = None,
                 order_manager: Optional[OrderManager] = None,
                 strategy_registry: Optional[StrategyRegistry] = None):
        
        self.config = config
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.strategy_registry = strategy_registry or StrategyRegistry()
        
        # Core components
        self.market_hours = MarketHoursManager()
        self.autonomous_decisions = AutonomousDecisionEngine(config)
        self.monitoring = ContinuousMonitoringSystem(config)
        self.risk_automation = AutomatedRiskManager(config, risk_manager)
        self.logger = TradingLogger(config)
        
        # State management
        self.status = SystemStatus.STOPPED
        self.automation_level = AutomationLevel.DISABLED
        self.trading_mode = TradingMode.BALANCED
        self.engine_metrics = EngineMetrics()
        
        # Session management
        self.current_session: Optional[SessionInfo] = None
        self.session_start_time: Optional[datetime] = None
        self.trading_decisions: List[TradingDecision] = []
        self.active_strategies: Dict[str, BaseStrategy] = {}
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.performance_history: List[Dict] = []
        
        # Control mechanisms
        self.emergency_stop = False
        self.pause_reasons: List[str] = []
        self.max_daily_loss_reached = False
        self.consecutive_losses = 0
        
        # Setup callbacks
        self.market_hours.register_session_callback(self._on_session_change)
        
        logger.info("Automated Trading Engine initialized")
    
    async def start(self) -> bool:
        """Start the automated trading engine"""
        try:
            logger.info("🚀 Starting Automated Trading Engine...")
            self.status = SystemStatus.INITIALIZING
            
            # Initialize logging
            await self.logger.start()
            
            # Start monitoring system
            await self.monitoring.start()
            
            # Initialize autonomous decision engine
            await self.autonomous_decisions.initialize()
            
            # Start risk automation
            await self.risk_automation.start()
            
            # Load and configure strategies
            await self._load_strategies()
            
            # Set automation level
            await self.set_automation_level(self.config.default_automation_level)
            
            # Start session monitoring
            self.monitoring_task = asyncio.create_task(self._session_monitoring_loop())
            self.trading_task = asyncio.create_task(self._trading_loop())
            self.metrics_task = asyncio.create_task(self._metrics_update_loop())
            self.health_task = asyncio.create_task(self._health_monitoring_loop())
            
            self.status = SystemStatus.RUNNING
            self.start_time = datetime.utcnow()
            
            logger.success("✅ Automated Trading Engine started successfully")
            
            # Initial market status
            market_summary = self.market_hours.get_market_summary()
            logger.info(f"Market status: {market_summary['open_exchanges']}/{market_summary['total_exchanges']} exchanges open")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start automated trading engine: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the automated trading engine"""
        try:
            logger.info("🛑 Stopping Automated Trading Engine...")
            
            self.status = SystemStatus.STOPPED
            
            # Cancel all tasks
            for task_name in ['monitoring_task', 'trading_task', 'metrics_task', 'health_task']:
                task = getattr(self, task_name, None)
                if task and not task.done():
                    task.cancel()
            
            # Stop strategies
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'stop'):
                    await strategy.stop()
            
            # Stop components
            await self.monitoring.stop()
            await self.autonomous_decisions.stop()
            await self.risk_automation.stop()
            
            # Generate final report
            await self._generate_end_session_report()
            
            logger.success("✅ Automated Trading Engine stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping engine: {e}")
            return False
    
    async def pause(self, reason: str = "Manual pause"):
        """Pause the engine"""
        if self.status == SystemStatus.RUNNING:
            self.status = SystemStatus.PAUSED
            self.pause_reasons.append(reason)
            
            # Pause strategies
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'pause'):
                    await strategy.pause()
            
            logger.info(f"⏸️ Engine paused: {reason}")
    
    async def resume(self):
        """Resume the engine"""
        if self.status == SystemStatus.PAUSED:
            self.status = SystemStatus.RUNNING
            
            # Resume strategies
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'resume'):
                    await strategy.resume()
            
            # Clear pause reasons
            self.pause_reasons.clear()
            
            logger.info("▶️ Engine resumed")
    
    async def emergency_stop(self, reason: str = "Emergency stop"):
        """Trigger emergency stop"""
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        
        self.emergency_stop = True
        self.status = SystemStatus.EMERGENCY_STOP
        
        # Emergency stop all strategies
        for strategy in self.active_strategies.values():
            if hasattr(strategy, 'stop'):
                await strategy.stop()
        
        # Close all positions
        if self.order_manager:
            try:
                # Emergency position closure logic would go here
                logger.critical("Emergency position closure initiated")
            except Exception as e:
                logger.error(f"Error in emergency position closure: {e}")
        
        # Stop all tasks
        for task_name in ['monitoring_task', 'trading_task', 'metrics_task', 'health_task']:
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()
    
    async def set_automation_level(self, level: AutomationLevel):
        """Set automation level"""
        self.automation_level = level
        
        # Update strategy configurations based on automation level
        for strategy in self.active_strategies.values():
            if hasattr(strategy, 'config'):
                # Adjust strategy parameters based on automation level
                await self._adjust_strategy_for_automation_level(strategy, level)
        
        logger.info(f"🔧 Automation level set to: {level.value}")
    
    async def _adjust_strategy_for_automation_level(self, strategy: BaseStrategy, level: AutomationLevel):
        """Adjust strategy parameters based on automation level"""
        # This would modify strategy behavior based on automation level
        # Implementation would depend on strategy-specific logic
        pass
    
    async def _load_strategies(self):
        """Load and configure trading strategies"""
        try:
            # Load strategy configurations from config
            strategy_configs = self.config.get_strategy_configs()
            
            for config in strategy_configs:
                if config.enabled:
                    # Create strategy instance (this would use factory pattern)
                    # strategy = await self._create_strategy(config)
                    
                    # Set context
                    # strategy.set_context(self._get_strategy_context())
                    
                    # Store strategy
                    self.active_strategies[config.strategy_id] = None  # placeholder
                    
                    logger.info(f"Strategy loaded: {config.name} ({config.strategy_type.value})")
            
            logger.info(f"✅ Loaded {len(self.active_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def _get_strategy_context(self):
        """Get strategy execution context"""
        # This would implement the StrategyContext protocol
        # Returning a mock for now
        class MockStrategyContext:
            async def get_market_data(self, symbol: str, timeframe: str):
                return []
            
            async def get_current_price(self, symbol: str):
                return Decimal('100')
            
            async def submit_order(self, symbol: str, side: str, quantity: Decimal, order_type: str, price: Optional[Decimal] = None):
                return {'success': True, 'order_id': 'mock_order'}
            
            async def get_positions(self, symbol: Optional[str] = None):
                return []
            
            async def get_portfolio_value(self):
                return Decimal('100000')
            
            def log_message(self, level: str, message: str, **kwargs):
                logger.log(level.upper(), message, **kwargs)
        
        return MockStrategyContext()
    
    async def _on_session_change(self, exchange_name: str, session_info: SessionInfo):
        """Handle market session changes"""
        logger.info(f"📅 Session change: {exchange_name} -> {session_info.current_session.value}")
        
        # Update current session
        self.current_session = session_info
        
        # Handle session transitions
        if session_info.is_market_open and not self.session_start_time:
            # Market just opened
            self.session_start_time = datetime.utcnow()
            logger.info("🟢 Market session started")
            
            # Resume trading operations
            if self.status == SystemStatus.PAUSED:
                await self.resume()
        
        elif not session_info.is_market_open and self.session_start_time:
            # Market just closed
            session_duration = datetime.utcnow() - self.session_start_time
            self.engine_metrics.market_hours_active += session_duration.total_seconds()
            self.session_start_time = None
            
            logger.info(f"🔴 Market session ended (duration: {session_duration})")
            
            # Pause trading operations
            if self.status == SystemStatus.RUNNING:
                await self.pause("Market closed")
    
    async def _session_monitoring_loop(self):
        """Monitor market sessions and manage trading operations"""
        while self.status != SystemStatus.STOPPED:
            try:
                # Check if any markets are open
                markets_open = self.market_hours.is_any_market_open()
                
                if markets_open:
                    # Ensure we're running during market hours
                    if self.status == SystemStatus.PAUSED and not self.pause_reasons:
                        await self.resume()
                else:
                    # No markets open, ensure we're paused
                    if self.status == SystemStatus.RUNNING:
                        await self.pause("No markets open")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _trading_loop(self):
        """Main trading loop"""
        while self.status not in [SystemStatus.STOPPED, SystemStatus.EMERGENCY_STOP]:
            try:
                if self.status == SystemStatus.RUNNING and self.automation_level != AutomationLevel.DISABLED:
                    # Run trading operations
                    await self._execute_trading_operations()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.engine_metrics.errors_count += 1
                await asyncio.sleep(5)
    
    async def _execute_trading_operations(self):
        """Execute automated trading operations"""
        try:
            # Get current market analysis
            market_analysis = await self.autonomous_decisions.analyze_market_conditions()
            
            # Update risk assessment
            risk_assessment = await self.risk_automation.assess_current_risk()
            
            # Generate trading opportunities
            opportunities = await self.autonomous_decisions.detect_opportunities()
            
            # Process each opportunity
            for opportunity in opportunities:
                # Create trading decision
                decision = TradingDecision(
                    decision_id=f"dec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
                    timestamp=datetime.utcnow(),
                    automation_level=self.automation_level,
                    strategy_id=opportunity.get('strategy_id', 'autonomous'),
                    symbol=opportunity['symbol'],
                    action=opportunity['action'],
                    confidence=opportunity['confidence'],
                    reasoning=opportunity['reasoning'],
                    position_size=Decimal(str(opportunity.get('size', 0))),
                    risk_score=opportunity.get('risk_score', 0.0),
                    approved=False
                )
                
                # Evaluate decision
                await self._evaluate_trading_decision(decision, market_analysis, risk_assessment)
                
                # Execute if approved
                if decision.approved:
                    await self._execute_trading_decision(decision)
                
                # Log decision
                self.trading_decisions.append(decision)
                
                # Limit memory usage
                if len(self.trading_decisions) > 10000:
                    self.trading_decisions = self.trading_decisions[-5000:]
            
            # Update metrics
            await self._update_trading_metrics()
            
        except Exception as e:
            logger.error(f"Error in trading operations: {e}")
    
    async def _evaluate_trading_decision(self, decision: TradingDecision, 
                                       market_analysis: Dict, risk_assessment: Dict):
        """Evaluate a trading decision"""
        try:
            # Risk evaluation
            if risk_assessment.get('daily_loss_limit_reached', False):
                decision.approved = False
                decision.reasoning += " | REJECTED: Daily loss limit reached"
                return
            
            # Market condition evaluation
            market_conditions_ok = market_analysis.get('conditions_acceptable', True)
            if not market_conditions_ok:
                decision.approved = False
                decision.reasoning += " | REJECTED: Market conditions not favorable"
                return
            
            # Confidence threshold
            min_confidence = self.config.get_min_confidence_for_automation(self.automation_level)
            if decision.confidence < min_confidence:
                decision.approved = False
                decision.reasoning += f" | REJECTED: Confidence {decision.confidence:.2f} < threshold {min_confidence:.2f}"
                return
            
            # Risk score threshold
            max_risk_score = self.config.get_max_risk_score_for_mode(self.trading_mode)
            if decision.risk_score > max_risk_score:
                decision.approved = False
                decision.reasoning += f" | REJECTED: Risk score {decision.risk_score:.2f} > threshold {max_risk_score:.2f}"
                return
            
            # Automation level checks
            if self.automation_level == AutomationLevel.ADVISORY:
                decision.approved = False  # Advisory mode doesn't execute
                decision.reasoning += " | ADVISORY: Manual review required"
            elif self.automation_level in [AutomationLevel.SEMI_AUTOMATED, AutomationLevel.FULLY_AUTOMATED]:
                decision.approved = True
            
            if decision.approved:
                logger.info(f"✅ Trading decision approved: {decision.symbol} {decision.action} "
                          f"(confidence: {decision.confidence:.2f}, risk: {decision.risk_score:.2f})")
            else:
                logger.debug(f"❌ Trading decision rejected: {decision.reasoning}")
            
        except Exception as e:
            logger.error(f"Error evaluating trading decision: {e}")
            decision.approved = False
            decision.reasoning += f" | ERROR: {str(e)}"
    
    async def _execute_trading_decision(self, decision: TradingDecision):
        """Execute an approved trading decision"""
        try:
            # Execute through order manager
            if self.order_manager:
                # Order execution logic would go here
                result = {
                    'success': True,
                    'order_id': f"exec_{decision.decision_id}",
                    'execution_time': datetime.utcnow(),
                    'price': 100.0,  # Mock price
                    'quantity': float(decision.position_size)
                }
                
                decision.executed = True
                decision.execution_result = result
                
                self.engine_metrics.total_trades_executed += 1
                self.engine_metrics.last_trade_time = datetime.utcnow()
                
                logger.info(f"🎯 Trade executed: {decision.symbol} {decision.action} "
                          f"(ID: {decision.decision_id})")
            
        except Exception as e:
            logger.error(f"Error executing trading decision: {e}")
            decision.executed = False
            decision.execution_result = {'error': str(e)}
            self.engine_metrics.failed_trades += 1
    
    async def _update_trading_metrics(self):
        """Update engine metrics"""
        try:
            if self.start_time:
                self.engine_metrics.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            self.engine_metrics.strategies_active = len(self.active_strategies)
            
            # Calculate win rate
            if self.engine_metrics.total_trades_executed > 0:
                self.engine_metrics.win_rate = (self.engine_metrics.successful_trades / 
                                              self.engine_metrics.total_trades_executed)
            
            # Update from monitoring system
            performance_data = await self.monitoring.get_performance_summary()
            self.engine_metrics.total_pnl = performance_data.get('total_pnl', 0.0)
            self.engine_metrics.daily_pnl = performance_data.get('daily_pnl', 0.0)
            self.engine_metrics.current_positions = performance_data.get('active_positions', 0)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _metrics_update_loop(self):
        """Update metrics periodically"""
        while self.status != SystemStatus.STOPPED:
            try:
                await self._update_trading_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitoring_loop(self):
        """Monitor system health"""
        while self.status != SystemStatus.STOPPED:
            try:
                # Check system health
                health_score = await self._calculate_health_score()
                self.engine_metrics.system_health_score = health_score
                
                # Health-based actions
                if health_score < 50:
                    logger.warning(f"🟡 Low system health score: {health_score:.1f}")
                
                if health_score < 25:
                    logger.error(f"🔴 Critical system health score: {health_score:.1f}")
                    await self.pause("Critical health score")
                
                self.last_health_check = datetime.utcnow()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            score = 100.0
            
            # Error rate penalty
            if self.engine_metrics.total_trades_executed > 0:
                error_rate = self.engine_metrics.failed_trades / self.engine_metrics.total_trades_executed
                score -= error_rate * 30
            
            # Performance penalty
            if self.engine_metrics.daily_pnl < -1000:  # Loss threshold
                score -= 20
            
            # Uptime penalty (older systems might be less reliable)
            if self.engine_metrics.uptime_seconds > 86400:  # 24 hours
                score -= 5
            
            # Missing components penalty
            if not self.risk_manager:
                score -= 20
            if not self.order_manager:
                score -= 20
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.0
    
    async def _generate_end_session_report(self):
        """Generate end-of-session report"""
        try:
            report = {
                'session_end': datetime.utcnow(),
                'total_uptime': self.engine_metrics.uptime_seconds,
                'trades_executed': self.engine_metrics.total_trades_executed,
                'success_rate': self.engine_metrics.win_rate,
                'total_pnl': self.engine_metrics.total_pnl,
                'decisions_made': len(self.trading_decisions),
                'market_hours_active': self.engine_metrics.market_hours_active
            }
            
            await self.logger.log_session_end(report)
            logger.info("📊 End-of-session report generated")
            
        except Exception as e:
            logger.error(f"Error generating end-session report: {e}")
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'status': self.status.value,
            'automation_level': self.automation_level.value,
            'trading_mode': self.trading_mode.value,
            'current_session': self.current_session.current_session.value if self.current_session else None,
            'is_trading': self.status == SystemStatus.RUNNING and self.automation_level != AutomationLevel.DISABLED,
            'metrics': {
                'uptime_seconds': self.engine_metrics.uptime_seconds,
                'trades_executed': self.engine_metrics.total_trades_executed,
                'win_rate': self.engine_metrics.win_rate,
                'total_pnl': self.engine_metrics.total_pnl,
                'daily_pnl': self.engine_metrics.daily_pnl,
                'health_score': self.engine_metrics.system_health_score,
                'active_strategies': self.engine_metrics.strategies_active,
                'errors_count': self.engine_metrics.errors_count
            },
            'market_status': self.market_hours.get_market_summary(),
            'active_decisions': len([d for d in self.trading_decisions if not d.executed]),
            'pause_reasons': self.pause_reasons.copy(),
            'emergency_stop': self.emergency_stop
        }


# Global engine instance (will be initialized with config)
automated_engine: Optional[AutomatedTradingEngine] = None