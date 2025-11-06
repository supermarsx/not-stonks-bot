"""
Configuration Management for Day Trading Orchestrator
Supports environment-based configuration with validation
"""

from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from enum import Enum
import os


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class Settings(BaseSettings):
    """Main configuration class"""
    
    # Application Settings
    app_name: str = "Day Trading Orchestrator"
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Database Settings
    db_type: DatabaseType = Field(default=DatabaseType.SQLITE)
    db_path: str = Field(default="./trading_orchestrator.db")
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)
    
    # Security Settings
    secret_key: str = Field(default="CHANGE_THIS_IN_PRODUCTION")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    encryption_key: Optional[str] = None
    
    # Broker API Keys (Encrypted Storage Recommended)
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: bool = Field(default=True)
    
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=7497)  # 7497=TWS paper, 7496=TWS live
    ibkr_client_id: int = Field(default=1)
    
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_paper: bool = Field(default=True)
    
    trading212_api_key: Optional[str] = None
    trading212_practice: bool = Field(default=True)
    
    # Risk Management Settings
    max_position_size: float = Field(default=10000.0)
    max_daily_loss: float = Field(default=1000.0)
    max_open_orders: int = Field(default=50)
    risk_per_trade: float = Field(default=0.02)  # 2% of capital
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_period: int = Field(default=60)  # seconds
    
    # WebSocket Settings
    ws_heartbeat_interval: int = Field(default=30)
    ws_reconnect_delay: int = Field(default=5)
    ws_max_reconnect_attempts: int = Field(default=10)
    
    # AI/LLM Settings
    ai_provider: str = Field(default="openai")
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ai_model_routing_enabled: bool = Field(default=True)
    ai_max_tokens: int = Field(default=4000)
    
    # Monitoring & Logging
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    log_file_path: str = Field(default="./logs/trading_orchestrator.log")
    log_rotation: str = Field(default="1 day")
    log_retention: str = Field(default="30 days")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @property
    def database_url(self) -> str:
        """Generate database connection URL"""
        if self.db_type == DatabaseType.SQLITE:
            return f"sqlite+aiosqlite:///{self.db_path}"
        elif self.db_type == DatabaseType.POSTGRESQL:
            return (
                f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING
    
    def get_broker_config(self, broker_name: str) -> Dict[str, Any]:
        """Get configuration for specific broker"""
        broker_configs = {
            "binance": {
                "api_key": self.binance_api_key,
                "api_secret": self.binance_api_secret,
                "testnet": self.binance_testnet
            },
            "ibkr": {
                "host": self.ibkr_host,
                "port": self.ibkr_port,
                "client_id": self.ibkr_client_id
            },
            "alpaca": {
                "api_key": self.alpaca_api_key,
                "api_secret": self.alpaca_api_secret,
                "paper": self.alpaca_paper
            },
            "trading212": {
                "api_key": self.trading212_api_key,
                "practice": self.trading212_practice
            }
        }
        return broker_configs.get(broker_name.lower(), {})


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection for FastAPI"""
    return settings
