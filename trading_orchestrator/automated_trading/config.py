"""
Automated Trading Configuration System

Configuration management for automated trading with user-configurable automation levels,
strategy preferences, risk tolerance, market hours, and emergency stop conditions.
"""

import json
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from loguru import logger

# Handle import for StrategyType and RiskLevel
try:
    from ..strategies.base import StrategyType, RiskLevel
except ImportError:
    # Fallback for standalone usage
    from enum import Enum
    class StrategyType(Enum):
        TREND_FOLLOWING = "trend_following"
        MEAN_REVERSION = "mean_reversion"
        MOMENTUM = "momentum"
        BREAKOUT = "breakout"
        PAIRS_TRADING = "pairs_trading"
        ARBITRAGE = "arbitrage"
        SCALPING = "scalping"
        SWING_TRADING = "swing_trading"
    
    class RiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"


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


@dataclass
class MarketHoursConfig:
    """Market hours configuration"""
    # Trading session preferences
    enable_pre_market: bool = True
    enable_after_hours: bool = True
    pre_market_start: time = field(default_factory=lambda: time(4, 0))
    pre_market_end: time = field(default_factory=lambda: time(9, 30))
    after_hours_start: time = field(default_factory=lambda: time(16, 0))
    after_hours_end: time = field(default_factory=lambda: time(20, 0))
    
    # Exchange preferences
    preferred_exchanges: List[str] = field(default_factory=lambda: ["NYSE", "NASDAQ"])
    enabled_exchanges: List[str] = field(default_factory=lambda: ["NYSE", "NASDAQ", "LSE"])
    
    # Trading schedule
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    skip_holidays: bool = True
    timezone: str = "America/New_York"


@dataclass
class StrategyPreferences:
    """Strategy preferences configuration"""
    # Enabled strategies
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "trend_following", "mean_reversion", "momentum"
    ])
    
    # Strategy weights (0.0 to 1.0)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend_following": 0.4,
        "mean_reversion": 0.3,
        "momentum": 0.3
    })
    
    # Symbol preferences
    preferred_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"
    ])
    avoid_symbols: List[str] = field(default_factory=list)
    
    # Sector preferences
    preferred_sectors: List[str] = field(default_factory=lambda: [
        "technology", "healthcare", "financial"
    ])
    
    # Strategy-specific parameters
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "trend_following": {
            "fast_period": 10,
            "slow_period": 30,
            "signal_threshold": 0.02
        },
        "mean_reversion": {
            "lookback_period": 20,
            "std_dev_threshold": 2.0,
            "entry_threshold": 0.05
        },
        "momentum": {
            "momentum_period": 14,
            "volume_threshold": 1.5,
            "price_change_threshold": 0.03
        }
    })


@dataclass
class RiskToleranceConfig:
    """Risk tolerance configuration"""
    # Overall risk level
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Position sizing
    base_position_size: float = 0.02  # 2% of portfolio
    min_position_size: float = 0.005  # 0.5% minimum
    max_position_size: float = 0.10   # 10% maximum
    
    # Risk limits
    max_daily_loss: float = 5000.0    # $5,000 daily loss limit
    max_drawdown: float = 0.15        # 15% maximum drawdown
    max_consecutive_losses: int = 5
    
    # Portfolio limits
    max_sector_exposure: float = 0.30  # 30% per sector
    max_single_position: float = 0.15  # 15% per position
    max_total_positions: int = 20
    
    # Stop loss settings
    default_stop_loss: float = 0.025   # 2.5% default stop loss
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 0.02  # 2% trailing stop
    
    # Take profit settings
    default_take_profit: float = 0.05   # 5% default take profit
    risk_reward_ratio: float = 2.0      # 2:1 risk-reward


@dataclass
class EmergencyStopConfig:
    """Emergency stop configuration"""
    # Automatic emergency stop triggers
    enable_automatic_stops: bool = True
    
    # Loss-based triggers
    emergency_stop_loss_threshold: float = 0.10  # 10% portfolio loss
    daily_loss_emergency_stop: float = 10000.0   # $10,000 daily loss
    
    # Performance-based triggers
    max_drawdown_emergency: float = 0.20         # 20% drawdown
    consecutive_loss_emergency: int = 10
    
    # System-based triggers
    enable_system_health_stops: bool = True
    max_system_error_rate: float = 0.05          # 5% error rate
    
    # Manual override settings
    require_confirmation: bool = False
    auto_restart_after_cooldown: bool = True
    cooldown_period_minutes: int = 60
    
    # Emergency procedures
    close_all_positions_on_emergency: bool = True
    halt_all_trading: bool = True
    notify_emergency_contacts: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Alert settings
    enable_email_alerts: bool = True
    enable_sms_alerts: bool = False
    enable_webhook_alerts: bool = False
    
    # Performance monitoring
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    metrics_retention_days: int = 7
    
    # Alert thresholds
    alert_cpu_threshold: float = 80.0       # 80% CPU usage
    alert_memory_threshold: float = 85.0    # 85% memory usage
    alert_drawdown_threshold: float = 0.05  # 5% drawdown
    alert_daily_loss_threshold: float = 2000.0  # $2,000 daily loss
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_retention_days: int = 30


@dataclass
class AutomationConfig:
    """Automation behavior configuration"""
    # Default automation level
    default_automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATED
    
    # Maximum simultaneous opportunities
    max_simultaneous_opportunities: int = 10
    
    # Decision confidence thresholds
    min_confidence_disabled: float = 0.0
    min_confidence_manual: float = 0.8
    min_confidence_advisory: float = 0.6
    min_confidence_semi_automated: float = 0.7
    min_confidence_fully_automated: float = 0.8
    
    # Risk score thresholds by mode
    max_risk_score_conservative: float = 0.3
    max_risk_score_balanced: float = 0.5
    max_risk_score_aggressive: float = 0.7
    
    # Market condition filters
    min_liquidity_score: float = 0.5
    max_volatility_threshold: float = 0.5
    require_market_hours: bool = True
    
    # Strategy selection
    auto_select_strategies: bool = True
    dynamic_strategy_switching: bool = True
    strategy_rotation_enabled: bool = False


class AutomatedTradingConfig:
    """
    Main configuration class for automated trading system
    
    Manages all configuration aspects:
    - Automation levels and behavior
    - Strategy preferences and parameters
    - Risk tolerance and limits
    - Market hours and scheduling
    - Emergency stop conditions
    - Monitoring and alerting
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "automated_trading_config.json"
        
        # Default configurations
        self.market_hours = MarketHoursConfig()
        self.strategies = StrategyPreferences()
        self.risk = RiskToleranceConfig()
        self.emergency_stops = EmergencyStopConfig()
        self.monitoring = MonitoringConfig()
        self.automation = AutomationConfig()
        
        # Load configuration if file exists
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            # Save default configuration
            self.save_config()
        
        logger.info(f"Automated Trading Config initialized from {self.config_file}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load market hours config
            if 'market_hours' in config_data:
                self._update_dataclass_from_dict(self.market_hours, config_data['market_hours'])
            
            # Load strategy preferences
            if 'strategies' in config_data:
                self._update_dataclass_from_dict(self.strategies, config_data['strategies'])
            
            # Load risk config
            if 'risk' in config_data:
                self._update_dataclass_from_dict(self.risk, config_data['risk'])
            
            # Load emergency stops config
            if 'emergency_stops' in config_data:
                self._update_dataclass_from_dict(self.emergency_stops, config_data['emergency_stops'])
            
            # Load monitoring config
            if 'monitoring' in config_data:
                self._update_dataclass_from_dict(self.monitoring, config_data['monitoring'])
            
            # Load automation config
            if 'automation' in config_data:
                self._update_dataclass_from_dict(self.automation, config_data['automation'])
            
            logger.info(f"✅ Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_data = {
                'market_hours': asdict(self.market_hours),
                'strategies': asdict(self.strategies),
                'risk': asdict(self.risk),
                'emergency_stops': asdict(self.emergency_stops),
                'monitoring': asdict(self.monitoring),
                'automation': asdict(self.automation),
                'version': '1.0.0',
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Convert time objects to strings
            config_data = self._convert_times_to_strings(config_data)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _update_dataclass_from_dict(self, dataclass_obj, data_dict):
        """Update dataclass from dictionary"""
        for key, value in data_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def _convert_times_to_strings(self, obj):
        """Convert time objects and enums to strings for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_times_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_times_to_strings(item) for item in obj]
        elif isinstance(obj, time):
            return obj.strftime('%H:%M:%S')
        elif hasattr(obj, 'value'):  # Handle enums
            return obj.value
        else:
            return obj
    
    # Convenience methods for accessing configuration
    
    def get_min_confidence_for_automation(self, level: AutomationLevel) -> float:
        """Get minimum confidence threshold for automation level"""
        confidence_map = {
            AutomationLevel.DISABLED: self.automation.min_confidence_disabled,
            AutomationLevel.MANUAL: self.automation.min_confidence_manual,
            AutomationLevel.ADVISORY: self.automation.min_confidence_advisory,
            AutomationLevel.SEMI_AUTOMATED: self.automation.min_confidence_semi_automated,
            AutomationLevel.FULLY_AUTOMATED: self.automation.min_confidence_fully_automated
        }
        return confidence_map.get(level, 0.5)
    
    def get_max_risk_score_for_mode(self, mode: TradingMode) -> float:
        """Get maximum risk score for trading mode"""
        risk_map = {
            TradingMode.CONSERVATIVE: self.automation.max_risk_score_conservative,
            TradingMode.BALANCED: self.automation.max_risk_score_balanced,
            TradingMode.AGGRESSIVE: self.automation.max_risk_score_aggressive,
            TradingMode.CUSTOM: 0.6  # Default for custom
        }
        return risk_map.get(mode, 0.5)
    
    def get_strategy_configs(self) -> List[Any]:
        """Get strategy configurations based on preferences"""
        from ..strategies.base import StrategyConfig
        
        configs = []
        
        for strategy_name in self.strategies.enabled_strategies:
            strategy_type = self._get_strategy_type_from_name(strategy_name)
            weight = self.strategies.strategy_weights.get(strategy_name, 1.0)
            
            config = StrategyConfig(
                strategy_id=strategy_name,
                strategy_type=strategy_type,
                name=strategy_name.replace('_', ' ').title(),
                description=f"Automated {strategy_name} strategy",
                parameters=self.strategies.parameters.get(strategy_name, {}),
                risk_level=self.risk.risk_level,
                max_position_size=self.risk.max_single_position,
                max_daily_loss=self.risk.max_daily_loss,
                symbols=self.strategies.preferred_symbols,
                enabled=True
            )
            
            configs.append(config)
        
        return configs
    
    def _get_strategy_type_from_name(self, strategy_name: str) -> StrategyType:
        """Get StrategyType enum from name"""
        name_map = {
            'trend_following': StrategyType.TREND_FOLLOWING,
            'mean_reversion': StrategyType.MEAN_REVERSION,
            'momentum': StrategyType.MOMENTUM,
            'breakout': StrategyType.BREAKOUT,
            'pairs_trading': StrategyType.PAIRS_TRADING,
            'arbitrage': StrategyType.ARBITRAGE,
            'scalping': StrategyType.SCALPING
        }
        
        return name_map.get(strategy_name, StrategyType.MOMENTUM)
    
    # Configuration update methods
    
    def update_automation_level(self, level: AutomationLevel):
        """Update default automation level"""
        self.automation.default_automation_level = level
        self.save_config()
        logger.info(f"Automation level updated to: {level.value}")
    
    def update_risk_level(self, risk_level: RiskLevel):
        """Update risk level"""
        self.risk.risk_level = risk_level
        self.save_config()
        logger.info(f"Risk level updated to: {risk_level.value}")
    
    def update_market_hours(self, **kwargs):
        """Update market hours configuration"""
        for key, value in kwargs.items():
            if hasattr(self.market_hours, key):
                setattr(self.market_hours, key, value)
        
        self.save_config()
        logger.info("Market hours configuration updated")
    
    def update_strategy_preferences(self, **kwargs):
        """Update strategy preferences"""
        for key, value in kwargs.items():
            if hasattr(self.strategies, key):
                setattr(self.strategies, key, value)
        
        self.save_config()
        logger.info("Strategy preferences updated")
    
    def update_risk_limits(self, **kwargs):
        """Update risk limits"""
        for key, value in kwargs.items():
            if hasattr(self.risk, key):
                setattr(self.risk, key, value)
        
        self.save_config()
        logger.info("Risk limits updated")
    
    def update_emergency_stops(self, **kwargs):
        """Update emergency stop configuration"""
        for key, value in kwargs.items():
            if hasattr(self.emergency_stops, key):
                setattr(self.emergency_stops, key, value)
        
        self.save_config()
        logger.info("Emergency stop configuration updated")
    
    def update_monitoring_settings(self, **kwargs):
        """Update monitoring configuration"""
        for key, value in kwargs.items():
            if hasattr(self.monitoring, key):
                setattr(self.monitoring, key, value)
        
        self.save_config()
        logger.info("Monitoring settings updated")
    
    # Validation methods
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate risk limits
        if self.risk.max_position_size > 0.25:  # Max 25%
            errors.append("Max position size cannot exceed 25%")
        
        if self.risk.max_daily_loss <= 0:
            errors.append("Max daily loss must be positive")
        
        if self.risk.base_position_size <= 0:
            errors.append("Base position size must be positive")
        
        if self.risk.base_position_size > self.risk.max_position_size:
            errors.append("Base position size cannot exceed max position size")
        
        # Validate automation settings
        if self.automation.max_simultaneous_opportunities <= 0:
            errors.append("Max simultaneous opportunities must be positive")
        
        # Validate monitoring settings
        if self.monitoring.monitoring_interval_seconds < 5:
            errors.append("Monitoring interval must be at least 5 seconds")
        
        # Validate strategy weights
        total_weight = sum(self.strategies.strategy_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small variance
            errors.append(f"Strategy weights should sum to 1.0, currently {total_weight:.2f}")
        
        # Validate market hours
        if (self.market_hours.pre_market_start >= self.market_hours.pre_market_end or
            self.market_hours.after_hours_start >= self.market_hours.after_hours_end):
            errors.append("Invalid market hours configuration")
        
        return errors
    
    def export_config(self, file_path: str):
        """Export configuration to specified file"""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    'market_hours': asdict(self.market_hours),
                    'strategies': asdict(self.strategies),
                    'risk': asdict(self.risk),
                    'emergency_stops': asdict(self.emergency_stops),
                    'monitoring': asdict(self.monitoring),
                    'automation': asdict(self.automation)
                }, f, indent=2)
            
            logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
    
    def import_config(self, file_path: str):
        """Import configuration from specified file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            # Backup current config
            backup_file = f"{self.config_file}.backup"
            self.save_config()
            os.rename(self.config_file, backup_file)
            
            # Load new config
            original_config = self.config_file
            self.config_file = file_path
            self.load_config()
            
            # Update the original file with new config
            self.config_file = original_config
            self.save_config()
            
            # Clean up backup
            if os.path.exists(backup_file):
                os.remove(backup_file)
            
            logger.info(f"Configuration imported from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            # Restore backup if it exists
            backup_file = f"{self.config_file}.backup"
            if os.path.exists(backup_file):
                os.rename(backup_file, self.config_file)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        # Handle both enum and string values
        automation_level = (self.automation.default_automation_level.value 
                          if hasattr(self.automation.default_automation_level, 'value') 
                          else str(self.automation.default_automation_level))
        
        risk_level = (self.risk.risk_level.value 
                     if hasattr(self.risk.risk_level, 'value') 
                     else str(self.risk.risk_level))
        
        return {
            "automation_level": automation_level,
            "risk_level": risk_level,
            "enabled_strategies": len(self.strategies.enabled_strategies),
            "preferred_symbols": len(self.strategies.preferred_symbols),
            "max_position_size": f"{self.risk.max_position_size:.1%}",
            "max_daily_loss": f"${self.risk.max_daily_loss:,.0f}",
            "emergency_stops_enabled": self.emergency_stops.enable_automatic_stops,
            "monitoring_enabled": self.monitoring.enable_real_time_monitoring,
            "config_file": self.config_file
        }


# Global configuration instance
config = AutomatedTradingConfig()