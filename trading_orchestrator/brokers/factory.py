"""
Broker Factory - Instantiate broker connections
"""

from typing import Dict, Type, Optional, Any
from brokers.base import BaseBroker, BrokerConfig
from loguru import logger


# Broker registry with risk levels - populated lazily
BROKER_REGISTRY: Dict[str, Type[BaseBroker]] = {}


def _populate_broker_registry():
    """Populate broker registry with available broker classes"""
    global BROKER_REGISTRY
    
    # Try to import each broker, add to registry if successful
    brokers_to_try = [
        ("binance", "brokers.binance_broker", "BinanceBroker"),
        ("ibkr", "brokers.ibkr_broker", "IBKRBroker"),
        ("alpaca", "brokers.alpaca_broker", "AlpacaBroker"),
        ("trading212", "brokers.trading212_broker", "Trading212Broker"),
        ("xtb", "brokers.xtb_broker", "XTBBroker"),
        ("degiro", "brokers.degiro_broker", "DegiroBroker"),
        ("trade_republic", "brokers.trade_republic_broker", "TradeRepublicBroker")
    ]
    
    for broker_name, module_name, class_name in brokers_to_try:
        try:
            # Dynamic import
            module = __import__(module_name, fromlist=[class_name])
            broker_class = getattr(module, class_name)
            BROKER_REGISTRY[broker_name] = broker_class
            logger.debug(f"Loaded {broker_name} broker")
        except ImportError as e:
            logger.debug(f"Could not load {broker_name} broker: {e}")
        except Exception as e:
            logger.warning(f"Error loading {broker_name} broker: {e}")
    
    logger.info(f"Broker registry populated with: {list(BROKER_REGISTRY.keys())}")


# Initialize registry on module load
_populate_broker_registry()


class BrokerFactory:
    """
    Factory for creating broker instances
    
    Official APIs (Recommended):
    - Binance (Official API - REST + WebSocket)
    - IBKR (Official TWS API)
    - Alpaca (Official API - REST + WebSocket)
    - Trading 212 (Official API, beta, limited features)
    
    Discontinued APIs (Legacy):
    - XTB (API discontinued March 14, 2025)
    
    Unofficial APIs (HIGH LEGAL RISK):
    - DEGIRO (No official API - ToS violation, account termination risk)
    - Trade Republic (No official API - Customer Agreement violation, guaranteed termination)
    
    ⚠️  WARNING: Unofficial API usage violates broker terms and may result in account termination.
    Use at your own risk and consult legal counsel before implementation.
    """
    
    @staticmethod
    def create_broker(config: BrokerConfig) -> BaseBroker:
        """
        Create broker instance from configuration
        
        Args:
            config: BrokerConfig with broker name and credentials
            
        Returns:
            BaseBroker: Instantiated broker
            
        Raises:
            ValueError: If broker not supported
        """
        broker_name = config.broker_name.lower()
        
        if broker_name not in BROKER_REGISTRY:
            supported = ", ".join(BROKER_REGISTRY.keys())
            raise ValueError(
                f"Broker '{broker_name}' not supported. "
                f"Supported brokers: {supported}"
            )
        
        broker_class = BROKER_REGISTRY[broker_name]
        logger.info(f"Creating {broker_name} broker instance")
        
        return broker_class(config)
    
    @staticmethod
    def get_supported_brokers() -> list:
        """Get list of supported brokers"""
        return list(BROKER_REGISTRY.keys())
    
    @staticmethod
    def is_broker_supported(broker_name: str) -> bool:
        """Check if broker is supported"""
        return broker_name.lower() in BROKER_REGISTRY
    
    @staticmethod
    def get_broker_risk_info(broker_name: str) -> Dict[str, str]:
        """Get risk information for broker"""
        risk_info = {
            "binance": {"risk_level": "LOW", "api_status": "OFFICIAL", "notes": "Official API with WebSocket support"},
            "ibkr": {"risk_level": "LOW", "api_status": "OFFICIAL", "notes": "Official TWS API"},
            "alpaca": {"risk_level": "LOW", "api_status": "OFFICIAL", "notes": "Official API, developer-friendly"},
            "trading212": {"risk_level": "LOW", "api_status": "OFFICIAL", "notes": "Official API in beta, limited order types in Live"},
            "xtb": {"risk_level": "HIGH", "api_status": "DISCONTINUED", "notes": "API disabled March 2025 - migration required"},
            "degiro": {"risk_level": "CRITICAL", "api_status": "UNOFFICIAL", "notes": "No official API - ToS violation, account termination risk"},
            "trade_republic": {"risk_level": "CRITICAL", "api_status": "UNOFFICIAL", "notes": "Customer Agreement violation - guaranteed account termination"}
        }
        return risk_info.get(broker_name.lower(), {"risk_level": "UNKNOWN", "api_status": "UNKNOWN", "notes": "No risk information available"})