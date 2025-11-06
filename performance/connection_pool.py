"""
Connection Pool Manager

Comprehensive connection pooling and management for databases and broker APIs
with performance optimization and monitoring.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import contextlib
import weakref
import json


class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    TESTING = "testing"
    CLOSED = "closed"
    ERROR = "error"


class PoolType(Enum):
    """Connection pool type enumeration"""
    DATABASE = "database"
    BROKER_API = "broker_api"
    CACHE = "cache"
    HTTP = "http"


@dataclass
class PoolConfig:
    """Connection pool configuration"""
    pool_type: PoolType
    max_connections: int = 50
    min_connections: int = 5
    max_idle_time: int = 300  # 5 minutes
    max_lifetime: int = 1800  # 30 minutes
    connection_timeout: int = 30
    validation_query: Optional[str] = None
    test_on_borrow: bool = True
    test_on_return: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60
    connection_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_connections < self.min_connections:
            raise ValueError("max_connections must be >= min_connections")


@dataclass
class ConnectionInfo:
    """Connection information"""
    connection_id: str
    created_at: datetime
    last_used_at: datetime
    last_validated_at: Optional[datetime] = None
    usage_count: int = 0
    total_time_active: float = 0.0
    state: ConnectionState = ConnectionState.IDLE
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseConnection:
    """Base connection class for all connection types"""
    
    def __init__(self, connection_id: str, config: PoolConfig):
        self.connection_id = connection_id
        self.config = config
        self.created_at = datetime.now()
        self.last_used_at = datetime.now()
        self.last_validated_at = None
        self.state = ConnectionState.IDLE
        self.error_count = 0
        self.total_time_active = 0.0
        self._lock = threading.Lock()
        self._raw_connection = None
        
    def connect(self) -> bool:
        """Establish connection"""
        raise NotImplementedError
    
    def disconnect(self):
        """Close connection"""
        raise NotImplementedError
    
    def is_alive(self) -> bool:
        """Check if connection is alive"""
        raise NotImplementedError
    
    def execute(self, query: str, params: Any = None) -> Any:
        """Execute query or operation"""
        raise NotImplementedError
    
    def validate(self) -> bool:
        """Validate connection health"""
        raise NotImplementedError
    
    def get_raw_connection(self):
        """Get raw connection object"""
        return self._raw_connection
    
    def mark_as_used(self):
        """Mark connection as used"""
        with self._lock:
            self.last_used_at = datetime.now()
            self.state = ConnectionState.ACTIVE
    
    def mark_as_idle(self):
        """Mark connection as idle"""
        with self._lock:
            self.state = ConnectionState.IDLE
    
    def increment_error(self, error: Exception):
        """Increment error count"""
        with self._lock:
            self.error_count += 1
            self.last_error = str(error)
            if self.error_count >= self.config.retry_attempts:
                self.state = ConnectionState.ERROR
    
    def reset_errors(self):
        """Reset error count"""
        with self._lock:
            self.error_count = 0
            self.last_error = None
    
    def get_age(self) -> timedelta:
        """Get connection age"""
        return datetime.now() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """Get idle time since last use"""
        return datetime.now() - self.last_used_at


class DatabaseConnection(BaseConnection):
    """Database connection wrapper"""
    
    def __init__(self, connection_id: str, config: PoolConfig, connection_string: str):
        super().__init__(connection_id, config)
        self.connection_string = connection_string
        self._connection_kwargs = config.connection_kwargs
        self._connection_class = self._detect_connection_class()
    
    def _detect_connection_class(self):
        """Detect database connection class"""
        # Try different database connection classes
        connection_classes = []
        
        try:
            import sqlite3
            connection_classes.append(('sqlite3', sqlite3.Connection))
        except ImportError:
            pass
        
        try:
            import psycopg2
            connection_classes.append(('postgresql', psycopg2.extensions.connection))
        except ImportError:
            pass
        
        try:
            import pymysql
            connection_classes.append(('mysql', pymysql.Connection))
        except ImportError:
            pass
        
        try:
            import cx_Oracle
            connection_classes.append(('oracle', cx_Oracle.Connection))
        except ImportError:
            pass
        
        # Detect connection type from connection string
        if self.connection_string.startswith('sqlite'):
            return 'sqlite3'
        elif self.connection_string.startswith('postgresql') or self.connection_string.startswith('postgres'):
            return 'postgresql'
        elif self.connection_string.startswith('mysql'):
            return 'mysql'
        elif self.connection_string.startswith('oracle'):
            return 'oracle'
        else:
            # Default to sqlite3
            return 'sqlite3'
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            if self._connection_class == 'sqlite3':
                import sqlite3
                self._raw_connection = sqlite3.connect(
                    self.connection_string.replace('sqlite3://', ''),
                    **self._connection_kwargs
                )
            elif self._connection_class == 'postgresql':
                import psycopg2
                self._raw_connection = psycopg2.connect(
                    self.connection_string,
                    **self._connection_kwargs
                )
            elif self._connection_class == 'mysql':
                import pymysql
                self._raw_connection = pymysql.connect(
                    self.connection_string,
                    **self._connection_kwargs
                )
            else:
                raise ValueError(f"Unsupported connection class: {self._connection_class}")
            
            self.state = ConnectionState.IDLE
            return True
            
        except Exception as e:
            self.increment_error(e)
            logging.error(f"Database connection failed for {self.connection_id}: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self._raw_connection:
                self._raw_connection.close()
            self.state = ConnectionState.CLOSED
        except Exception as e:
            logging.error(f"Error closing database connection {self.connection_id}: {e}")
    
    def is_alive(self) -> bool:
        """Check if database connection is alive"""
        try:
            if not self._raw_connection:
                return False
            
            # Try a simple query
            cursor = self._raw_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False
    
    def execute(self, query: str, params: Any = None) -> Any:
        """Execute database query"""
        start_time = time.time()
        try:
            self.mark_as_used()
            
            cursor = self._raw_connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Check if it's a select query
            if query.strip().lower().startswith('select'):
                result = cursor.fetchall()
            else:
                self._raw_connection.commit()
                result = cursor.rowcount
            
            cursor.close()
            self.reset_errors()
            
            execution_time = time.time() - start_time
            with self._lock:
                self.total_time_active += execution_time
            
            return result
            
        except Exception as e:
            self.increment_error(e)
            self._raw_connection.rollback() if self._raw_connection else None
            logging.error(f"Database query failed for {self.connection_id}: {e}")
            raise
        finally:
            self.mark_as_idle()
    
    def validate(self) -> bool:
        """Validate database connection"""
        try:
            if not self._raw_connection:
                return False
            
            if self.config.validation_query:
                cursor = self._raw_connection.cursor()
                cursor.execute(self.config.validation_query)
                cursor.fetchone()
                cursor.close()
            else:
                # Simple validation
                cursor = self._raw_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            
            self.last_validated_at = datetime.now()
            return True
            
        except Exception as e:
            logging.error(f"Database validation failed for {self.connection_id}: {e}")
            return False


class BrokerAPConnection(BaseConnection):
    """Broker API connection wrapper"""
    
    def __init__(self, connection_id: str, config: PoolConfig, broker_config: Dict[str, Any]):
        super().__init__(connection_id, config)
        self.broker_config = broker_config
        self._broker_type = broker_config.get('type', 'alpaca')
        self._raw_connection = None
    
    def connect(self) -> bool:
        """Establish broker API connection"""
        try:
            if self._broker_type == 'alpaca':
                import alpaca_trade_api
                self._raw_connection = alpaca_trade_api.REST(
                    key_id=self.broker_config.get('api_key'),
                    secret_key=self.broker_config.get('secret_key'),
                    base_url=self.broker_config.get('base_url', 'https://paper-api.alpaca.markets')
                )
            elif self._broker_type == 'binance':
                import binance
                self._raw_connection = binance.Client(
                    api_key=self.broker_config.get('api_key'),
                    api_secret=self.broker_config.get('secret_key')
                )
            elif self._broker_type == 'ibkr':
                # Interactive Brokers connection would be implemented here
                # This is a placeholder for the actual implementation
                self._raw_connection = {"type": "ibkr", "status": "connected"}
            else:
                raise ValueError(f"Unsupported broker type: {self._broker_type}")
            
            self.state = ConnectionState.IDLE
            return True
            
        except Exception as e:
            self.increment_error(e)
            logging.error(f"Broker API connection failed for {self.connection_id}: {e}")
            return False
    
    def disconnect(self):
        """Close broker API connection"""
        try:
            if self._raw_connection and hasattr(self._raw_connection, 'close'):
                self._raw_connection.close()
            self.state = ConnectionState.CLOSED
        except Exception as e:
            logging.error(f"Error closing broker API connection {self.connection_id}: {e}")
    
    def is_alive(self) -> bool:
        """Check if broker API connection is alive"""
        try:
            if not self._raw_connection:
                return False
            
            # Try a simple API call
            if self._broker_type == 'alpaca':
                account = self._raw_connection.get_account()
                return account is not None
            elif self._broker_type == 'binance':
                status = self._raw_connection.get_system_status()
                return status.get('status') == 'normal'
            else:
                return True  # Assume alive for unknown types
                
        except Exception:
            return False
    
    def execute(self, method: str, endpoint: str, params: Any = None) -> Any:
        """Execute broker API call"""
        start_time = time.time()
        try:
            self.mark_as_used()
            
            if self._broker_type == 'alpaca':
                # Alpaca API calls
                if method == 'get_account':
                    result = self._raw_connection.get_account()
                elif method == 'get_positions':
                    result = self._raw_connection.list_positions()
                elif method == 'place_order':
                    result = self._raw_connection.submit_order(**params)
                else:
                    raise ValueError(f"Unsupported Alpaca method: {method}")
            elif self._broker_type == 'binance':
                # Binance API calls
                if method == 'get_account':
                    result = self._raw_connection.get_account()
                elif method == 'get_positions':
                    result = self._raw_connection.futures_position_information()
                elif method == 'place_order':
                    result = self._raw_connection.futures_create_order(**params)
                else:
                    raise ValueError(f"Unsupported Binance method: {method}")
            else:
                result = {"status": "ok"}
            
            self.reset_errors()
            
            execution_time = time.time() - start_time
            with self._lock:
                self.total_time_active += execution_time
            
            return result
            
        except Exception as e:
            self.increment_error(e)
            logging.error(f"Broker API call failed for {self.connection_id}: {e}")
            raise
        finally:
            self.mark_as_idle()
    
    def validate(self) -> bool:
        """Validate broker API connection"""
        try:
            return self.is_alive()
        except Exception as e:
            logging.error(f"Broker API validation failed for {self.connection_id}: {e}")
            return False


class ConnectionPool:
    """Generic connection pool"""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self._connections = {}  # connection_id -> BaseConnection
        self._idle_connections = queue.Queue(maxsize=config.max_connections)
        self._active_connections = set()
        self._connection_counter = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._total_connections_created = 0
        self._total_connections_destroyed = 0
        self._total_requests = 0
        self._total_timeouts = 0
        
        # Health monitoring
        self._health_check_thread = None
        self._stop_health_check = False
        
        # Initialize with minimum connections
        self._initialize_min_connections()
        
        # Start health check if enabled
        if config.health_check_interval > 0:
            self._start_health_check()
    
    def _initialize_min_connections(self):
        """Initialize minimum number of connections"""
        for _ in range(self.config.min_connections):
            connection = self._create_connection()
            if connection:
                self._add_to_pool(connection)
    
    def _create_connection(self) -> Optional[BaseConnection]:
        """Create new connection"""
        try:
            self._connection_counter += 1
            connection_id = f"{self.config.pool_type.value}_{self._connection_counter}"
            
            # Create appropriate connection type
            if self.config.pool_type == PoolType.DATABASE:
                connection_string = self.config.connection_kwargs.get('connection_string', 'sqlite3://:memory:')
                connection = DatabaseConnection(connection_id, self.config, connection_string)
            elif self.config.pool_type == PoolType.BROKER_API:
                broker_config = self.config.connection_kwargs.get('broker_config', {})
                connection = BrokerAPConnection(connection_id, self.config, broker_config)
            else:
                raise ValueError(f"Unsupported pool type: {self.config.pool_type}")
            
            # Establish connection
            if connection.connect():
                self._total_connections_created += 1
                self.logger.debug(f"Created new connection: {connection_id}")
                return connection
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def _add_to_pool(self, connection: BaseConnection):
        """Add connection to pool"""
        with self._lock:
            self._connections[connection.connection_id] = connection
            self._idle_connections.put(connection)
    
    def get_connection(self, timeout: Optional[float] = None) -> Optional[BaseConnection]:
        """Get connection from pool"""
        timeout = timeout or self.config.connection_timeout
        start_time = time.time()
        
        self._total_requests += 1
        
        while time.time() - start_time < timeout:
            try:
                # Try to get idle connection
                connection = self._idle_connections.get_nowait()
                
                # Validate connection
                if self.config.test_on_borrow and not connection.validate():
                    self._destroy_connection(connection)
                    continue
                
                # Mark as active
                with self._lock:
                    self._active_connections.add(connection.connection_id)
                
                connection.mark_as_used()
                self.logger.debug(f"Provided connection: {connection.connection_id}")
                return connection
                
            except queue.Empty:
                # No idle connections, try to create new one
                with self._lock:
                    if len(self._connections) < self.config.max_connections:
                        connection = self._create_connection()
                        if connection:
                            self._add_to_pool(connection)
                            continue
                
                # Pool exhausted, wait and retry
                time.sleep(0.1)
        
        self._total_timeouts += 1
        self.logger.warning(f"Connection pool timeout after {timeout}s")
        return None
    
    def return_connection(self, connection: BaseConnection):
        """Return connection to pool"""
        try:
            connection.mark_as_idle()
            
            with self._lock:
                self._active_connections.discard(connection.connection_id)
                
                # Validate before returning if required
                if self.config.test_on_return and not connection.validate():
                    self._destroy_connection(connection)
                    return
                
                # Check if connection should be destroyed
                if self._should_destroy_connection(connection):
                    self._destroy_connection(connection)
                    return
                
                # Return to idle pool
                self._idle_connections.put_nowait(connection)
                self.logger.debug(f"Returned connection to pool: {connection.connection_id}")
                
        except Exception as e:
            self.logger.error(f"Error returning connection {connection.connection_id}: {e}")
            self._destroy_connection(connection)
    
    def _should_destroy_connection(self, connection: BaseConnection) -> bool:
        """Check if connection should be destroyed"""
        # Check connection age
        age = connection.get_age()
        if age.total_seconds() > self.config.max_lifetime:
            self.logger.info(f"Destroying connection {connection.connection_id}: exceeded max lifetime")
            return True
        
        # Check idle time
        idle_time = connection.get_idle_time()
        if idle_time.total_seconds() > self.config.max_idle_time:
            self.logger.info(f"Destroying connection {connection.connection_id}: exceeded max idle time")
            return True
        
        # Check error count
        if connection.error_count >= self.config.retry_attempts:
            self.logger.warning(f"Destroying connection {connection.connection_id}: too many errors")
            return True
        
        # Check if pool has too many connections
        if len(self._connections) > self.config.max_connections:
            # Keep at least min_connections
            if len(self._connections) - len(self._active_connections) > self.config.min_connections:
                self.logger.debug(f"Destroying connection {connection.connection_id}: pool size optimization")
                return True
        
        return False
    
    def _destroy_connection(self, connection: BaseConnection):
        """Destroy connection"""
        try:
            connection.disconnect()
            with self._lock:
                if connection.connection_id in self._connections:
                    del self._connections[connection.connection_id]
                self._active_connections.discard(connection.connection_id)
            self._total_connections_destroyed += 1
            self.logger.debug(f"Destroyed connection: {connection.connection_id}")
        except Exception as e:
            self.logger.error(f"Error destroying connection {connection.connection_id}: {e}")
    
    def _start_health_check(self):
        """Start health check background thread"""
        if self._health_check_thread and self._health_check_thread.is_alive():
            return
        
        self._stop_health_check = False
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        self.logger.info("Started connection pool health check")
    
    def _health_check_loop(self):
        """Health check loop"""
        while not self._stop_health_check:
            try:
                time.sleep(self.config.health_check_interval)
                
                # Check all connections
                connections_to_check = []
                with self._lock:
                    connections_to_check = list(self._connections.values())
                
                for connection in connections_to_check:
                    if connection.connection_id not in self._active_connections:
                        # Test idle connections
                        connection.state = ConnectionState.TESTING
                        if not connection.validate():
                            self.logger.warning(f"Health check failed for {connection.connection_id}")
                            self._destroy_connection(connection)
                        else:
                            connection.last_validated_at = datetime.now()
                            connection.state = ConnectionState.IDLE
                
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
    
    def _cleanup_connections(self):
        """Cleanup idle connections"""
        with self._lock:
            current_idle_connections = list(self._connections.values())
        
        for connection in current_idle_connections:
            if connection.connection_id not in self._active_connections:
                if self._should_destroy_connection(connection):
                    self._destroy_connection(connection)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_type': self.config.pool_type.value,
                'total_connections': len(self._connections),
                'active_connections': len(self._active_connections),
                'idle_connections': len(self._connections) - len(self._active_connections),
                'total_created': self._total_connections_created,
                'total_destroyed': self._total_connections_destroyed,
                'total_requests': self._total_requests,
                'total_timeouts': self._total_timeouts,
                'timeout_rate': self._total_timeouts / max(self._total_requests, 1),
                'utilization_rate': len(self._active_connections) / max(self.config.max_connections, 1)
            }
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get detailed connection information"""
        with self._lock:
            connections = list(self._connections.values())
        
        return [connection.to_dict() for connection in connections]
    
    def resize_pool(self, new_max_connections: int):
        """Resize connection pool"""
        if new_max_connections < self.config.min_connections:
            raise ValueError("new_max_connections must be >= min_connections")
        
        self.config.max_connections = new_max_connections
        self.logger.info(f"Resized connection pool to {new_max_connections} max connections")
        
        # Cleanup excess connections
        self._cleanup_connections()
    
    def close_all_connections(self):
        """Close all connections"""
        self._stop_health_check = True
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        
        with self._lock:
            connections_to_close = list(self._connections.values())
            self._connections.clear()
            self._active_connections.clear()
        
        for connection in connections_to_close:
            self._destroy_connection(connection)
        
        self.logger.info("Closed all connections in pool")


class ConnectionPoolManager:
    """Manager for multiple connection pools"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._pools = {}  # pool_name -> ConnectionPool
        self._pool_configs = {}
        self._lock = threading.RLock()
    
    def create_pool(self, pool_name: str, config: PoolConfig) -> ConnectionPool:
        """Create new connection pool"""
        with self._lock:
            if pool_name in self._pools:
                raise ValueError(f"Pool '{pool_name}' already exists")
            
            pool = ConnectionPool(config)
            self._pools[pool_name] = pool
            self._pool_configs[pool_name] = config
            
            self.logger.info(f"Created connection pool: {pool_name}")
            return pool
    
    def get_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """Get existing connection pool"""
        return self._pools.get(pool_name)
    
    def remove_pool(self, pool_name: str):
        """Remove and close connection pool"""
        with self._lock:
            if pool_name not in self._pools:
                raise ValueError(f"Pool '{pool_name}' does not exist")
            
            pool = self._pools[pool_name]
            pool.close_all_connections()
            del self._pools[pool_name]
            del self._pool_configs[pool_name]
            
            self.logger.info(f"Removed connection pool: {pool_name}")
    
    def get_pool_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools"""
        return {
            name: pool.get_statistics()
            for name, pool in self._pools.items()
        }
    
    def get_pool_connection_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get connection information for all pools"""
        return {
            name: pool.get_connection_info()
            for name, pool in self._pools.items()
        }
    
    def health_check_all_pools(self) -> Dict[str, bool]:
        """Perform health check on all pools"""
        results = {}
        for name, pool in self._pools.items():
            try:
                stats = pool.get_statistics()
                results[name] = stats['active_connections'] > 0 or stats['idle_connections'] > 0
            except Exception as e:
                self.logger.error(f"Health check failed for pool {name}: {e}")
                results[name] = False
        return results
    
    def close_all_pools(self):
        """Close all connection pools"""
        for pool_name in list(self._pools.keys()):
            self.remove_pool(pool_name)
        
        self.logger.info("Closed all connection pools")


# Context managers for easy usage
@contextlib.asynccontextmanager
async def get_db_connection(pool_name: str, timeout: Optional[float] = None):
    """Get database connection from pool"""
    pool_manager = get_pool_manager()
    pool = pool_manager.get_pool(pool_name)
    
    if not pool:
        raise ValueError(f"Pool '{pool_name}' not found")
    
    connection = pool.get_connection(timeout)
    if not connection:
        raise Exception("Failed to get connection from pool")
    
    try:
        yield connection
    finally:
        pool.return_connection(connection)


@contextlib.contextmanager
def get_connection_context(pool_name: str, timeout: Optional[float] = None):
    """Get connection context manager"""
    pool_manager = get_pool_manager()
    pool = pool_manager.get_pool(pool_name)
    
    if not pool:
        raise ValueError(f"Pool '{pool_name}' not found")
    
    connection = pool.get_connection(timeout)
    if not connection:
        raise Exception("Failed to get connection from pool")
    
    try:
        yield connection
    finally:
        pool.return_connection(connection)


# Global connection pool manager instance
_pool_manager = None

def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager instance"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


def create_database_pool(pool_name: str, 
                        connection_string: str,
                        max_connections: int = 50,
                        min_connections: int = 5,
                        **kwargs) -> ConnectionPool:
    """Create database connection pool"""
    config = PoolConfig(
        pool_type=PoolType.DATABASE,
        max_connections=max_connections,
        min_connections=min_connections,
        connection_kwargs={
            'connection_string': connection_string,
            **kwargs
        }
    )
    
    manager = get_pool_manager()
    return manager.create_pool(pool_name, config)


def create_broker_pool(pool_name: str,
                      broker_config: Dict[str, Any],
                      max_connections: int = 10,
                      min_connections: int = 2,
                      **kwargs) -> ConnectionPool:
    """Create broker API connection pool"""
    config = PoolConfig(
        pool_type=PoolType.BROKER_API,
        max_connections=max_connections,
        min_connections=min_connections,
        connection_kwargs={
            'broker_config': broker_config,
            **kwargs
        }
    )
    
    manager = get_pool_manager()
    return manager.create_pool(pool_name, config)