"""
Demo Mode Manager - Central component for managing simulation trading
"""

import os
import json
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from loguru import logger


class DemoEnvironment(Enum):
    """Demo environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    SANDBOX = "sandbox"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"


class DemoModeState(Enum):
    """Demo mode states"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    INITIALIZING = "initializing"
    SUSPENDED = "suspended"


@dataclass
class DemoModeConfig:
    """Configuration for demo mode"""
    # Basic settings
    enabled: bool = False
    environment: DemoEnvironment = DemoEnvironment.DEVELOPMENT
    demo_account_balance: float = 100000.0  # Starting virtual balance
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    
    # Simulation parameters
    realistic_slippage: bool = True
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% average slippage
    market_impact_enabled: bool = True
    order_fill_delay: float = 0.1  # Seconds
    
    # Data simulation
    use_synthetic_data: bool = True
    data_quality: str = "high"  # low, medium, high
    volatility_simulation: bool = True
    
    # Risk simulation
    max_drawdown_limit: float = 0.10  # 10% max drawdown
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    position_size_limit: float = 0.05  # 5% max position size
    
    # Logging and monitoring
    detailed_logging: bool = True
    performance_tracking: bool = True
    alerts_enabled: bool = True
    
    # Persistence
    save_state_interval: int = 300  # seconds
    state_file_path: str = "demo_state.json"


@dataclass
class DemoSession:
    """Demo trading session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    trades_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    is_active: bool = True


class DemoModeManager:
    """
    Central manager for demo mode operations
    
    Manages demo mode state, configuration, and lifecycle operations.
    Provides unified interface for all demo mode functionality.
    """
    
    def __init__(self, config: Optional[DemoModeConfig] = None):
        self.config = config or DemoModeConfig()
        self.state = DemoModeState.DISABLED
        self.current_session: Optional[DemoSession] = None
        self._state_lock = asyncio.Lock()
        self._session_data: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize demo mode manager"""
        try:
            async with self._state_lock:
                self.state = DemoModeState.INITIALIZING
                
                # Load persisted state if exists
                if os.path.exists(self.config.state_file_path):
                    await self._load_state()
                
                # Validate configuration
                if not await self._validate_config():
                    logger.error("Demo mode configuration validation failed")
                    self.state = DemoModeState.DISABLED
                    return False
                
                # Set up session tracking
                await self._initialize_session()
                
                self.state = DemoModeState.ENABLED if self.config.enabled else DemoModeState.DISABLED
                self._initialized = True
                
                logger.info(f"Demo mode manager initialized: state={self.state.value}, env={self.config.environment.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize demo mode manager: {e}")
            self.state = DemoModeState.DISABLED
            return False
    
    async def enable_demo_mode(self) -> bool:
        """Enable demo mode"""
        try:
            async with self._state_lock:
                if not self._initialized:
                    await self.initialize()
                
                if self.state == DemoModeState.ENABLED:
                    logger.warning("Demo mode already enabled")
                    return True
                
                self.config.enabled = True
                self.state = DemoModeState.ENABLED
                await self._start_new_session()
                await self._save_state()
                
                logger.info("Demo mode enabled successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to enable demo mode: {e}")
            return False
    
    async def disable_demo_mode(self) -> bool:
        """Disable demo mode"""
        try:
            async with self._state_lock:
                if self.state == DemoModeState.DISABLED:
                    logger.warning("Demo mode already disabled")
                    return True
                
                self.config.enabled = False
                self.state = DemoModeState.DISABLED
                
                if self.current_session:
                    self.current_session.end_time = datetime.now()
                    self.current_session.is_active = False
                
                await self._save_state()
                
                logger.info("Demo mode disabled")
                return True
                
        except Exception as e:
            logger.error(f"Failed to disable demo mode: {e}")
            return False
    
    async def suspend_demo_mode(self, reason: str = "Manual suspension") -> bool:
        """Suspend demo mode temporarily"""
        try:
            async with self._state_lock:
                if self.state != DemoModeState.ENABLED:
                    logger.warning(f"Cannot suspend demo mode from state: {self.state}")
                    return False
                
                self.state = DemoModeState.SUSPENDED
                logger.warning(f"Demo mode suspended: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to suspend demo mode: {e}")
            return False
    
    async def resume_demo_mode(self) -> bool:
        """Resume demo mode from suspension"""
        try:
            async with self._state_lock:
                if self.state != DemoModeState.SUSPENDED:
                    logger.warning(f"Cannot resume demo mode from state: {self.state}")
                    return False
                
                self.state = DemoModeState.ENABLED
                logger.info("Demo mode resumed")
                return True
                
        except Exception as e:
            logger.error(f"Failed to resume demo mode: {e}")
            return False
    
    async def get_demo_status(self) -> Dict[str, Any]:
        """Get current demo mode status"""
        status = {
            "state": self.state.value,
            "enabled": self.config.enabled,
            "environment": self.config.environment.value,
            "is_active": self.state == DemoModeState.ENABLED,
            "session": asdict(self.current_session) if self.current_session else None,
            "config_summary": {
                "demo_balance": self.config.demo_account_balance,
                "max_risk_per_trade": self.config.max_risk_per_trade,
                "slippage_enabled": self.config.realistic_slippage,
                "commission_rate": self.config.commission_rate
            }
        }
        
        return status
    
    async def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update demo mode configuration"""
        try:
            # Update config values
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            # Validate updated config
            if not await self._validate_config():
                logger.error("Updated configuration validation failed")
                return False
            
            await self._save_state()
            logger.info("Demo mode configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update demo mode config: {e}")
            return False
    
    def is_demo_mode_active(self) -> bool:
        """Check if demo mode is currently active"""
        return self.state == DemoModeState.ENABLED
    
    def is_demo_enabled(self) -> bool:
        """Check if demo mode is enabled in config"""
        return self.config.enabled
    
    def get_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self.current_session.session_id if self.current_session else None
    
    async def start_performance_monitoring(self):
        """Start demo mode performance monitoring"""
        if not self.config.performance_tracking:
            return
        
        # Schedule periodic performance updates
        asyncio.create_task(self._performance_monitor_loop())
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.current_session:
            return {}
        
        metrics = {
            "session_duration": (
                datetime.now() - self.current_session.start_time
            ).total_seconds(),
            "trades_count": self.current_session.trades_count,
            "total_pnl": self.current_session.total_pnl,
            "win_rate": self.current_session.win_rate,
            "max_drawdown": self.current_session.max_drawdown,
            "sharpe_ratio": self.current_session.sharpe_ratio
        }
        
        return metrics
    
    # Private methods
    
    async def _validate_config(self) -> bool:
        """Validate demo mode configuration"""
        try:
            # Validate numerical ranges
            if not 0 < self.config.max_risk_per_trade <= 1:
                logger.error("max_risk_per_trade must be between 0 and 1")
                return False
            
            if self.config.demo_account_balance <= 0:
                logger.error("demo_account_balance must be positive")
                return False
            
            if not 0 <= self.config.commission_rate <= 1:
                logger.error("commission_rate must be between 0 and 1")
                return False
            
            # Validate environment
            if not isinstance(self.config.environment, DemoEnvironment):
                logger.error("Invalid environment type")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _initialize_session(self):
        """Initialize demo session"""
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = DemoSession(
            session_id=session_id,
            start_time=datetime.now()
        )
    
    async def _start_new_session(self):
        """Start a new demo session"""
        await self._initialize_session()
    
    async def _load_state(self):
        """Load persisted demo mode state"""
        try:
            with open(self.config.state_file_path, 'r') as f:
                data = json.load(f)
            
            # Restore configuration
            config_data = data.get('config', {})
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    if key == 'environment':
                        setattr(self.config, key, DemoEnvironment(value))
                    else:
                        setattr(self.config, key, value)
            
            # Restore session if exists
            session_data = data.get('session')
            if session_data:
                self.current_session = DemoSession(**session_data)
            
            logger.info("Demo mode state loaded from file")
            
        except Exception as e:
            logger.warning(f"Failed to load demo mode state: {e}")
    
    async def _save_state(self):
        """Persist demo mode state"""
        try:
            state_data = {
                'config': asdict(self.config),
                'session': asdict(self.current_session) if self.current_session else None,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config.state_file_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.warning(f"Failed to save demo mode state: {e}")
    
    async def _performance_monitor_loop(self):
        """Background task for performance monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.save_state_interval)
                await self._update_performance_metrics()
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics for current session"""
        if not self.current_session:
            return
        
        # This would be implemented with actual performance calculation
        # based on demo trading data
        pass


# Global demo mode manager instance
demo_manager = DemoModeManager()


async def get_demo_manager() -> DemoModeManager:
    """Get global demo mode manager instance"""
    if not demo_manager._initialized:
        await demo_manager.initialize()
    return demo_manager


# Demo mode context manager
class DemoModeContext:
    """Context manager for demo mode operations"""
    
    def __init__(self, manager: DemoModeManager):
        self.manager = manager
        self.was_enabled = False
    
    async def __aenter__(self):
        """Enter demo mode context"""
        self.was_enabled = self.manager.is_demo_mode_active()
        if not self.was_enabled:
            await self.manager.enable_demo_mode()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit demo mode context"""
        if not self.was_enabled:
            await self.manager.disable_demo_mode()


# Convenience functions
async def enable_demo_mode() -> bool:
    """Convenience function to enable demo mode"""
    manager = await get_demo_manager()
    return await manager.enable_demo_mode()


async def disable_demo_mode() -> bool:
    """Convenience function to disable demo mode"""
    manager = await get_demo_manager()
    return await manager.disable_demo_mode()


async def is_demo_mode() -> bool:
    """Convenience function to check demo mode status"""
    manager = await get_demo_manager()
    return manager.is_demo_mode_active()


async def demo_mode_context():
    """Convenience function for demo mode context"""
    manager = await get_demo_manager()
    return DemoModeContext(manager)


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = await get_demo_manager()
        
        # Enable demo mode
        await manager.enable_demo_mode()
        
        # Check status
        status = await manager.get_demo_status()
        print(f"Demo mode status: {status}")
        
        # Use context manager
        async with demo_mode_context():
            print("Trading in demo mode...")
            # Your trading operations here
        
        # Disable demo mode
        await manager.disable_demo_mode()
    
    # Run example
    asyncio.run(main())
