"""
Base Crawler Framework
Provides common functionality for all crawlers
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import aiofiles
from pathlib import Path


class CrawlerStatus(Enum):
    """Crawler execution status"""
    STOPPED = "stopped"
    RUNNING = "running" 
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class DataType(Enum):
    """Supported data types"""
    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC = "economic"
    TECHNICAL = "technical"


@dataclass
class CrawlerConfig:
    """Crawler configuration"""
    name: str
    data_type: DataType
    interval: int  # seconds between runs
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 100  # requests per minute
    enable_cache: bool = True
    cache_duration: int = 300  # seconds
    enable_storage: bool = True
    storage_path: str = "./data"
    enable_monitoring: bool = True
    error_threshold: int = 5
    recovery_timeout: int = 60


@dataclass
class CrawlResult:
    """Result of a crawl operation"""
    success: bool
    data: Any
    timestamp: datetime
    crawl_duration: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        with self.lock:
            # Remove old requests outside time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check if we've exceeded the limit
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            # Add current request
            self.requests.append(now)


class Cache:
    """Simple file-based cache for crawler data"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key filename"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        with self.lock:
            cache_key = self._get_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check if cache is still valid
                    if time.time() - data['timestamp'] < data['ttl']:
                        return data['content']
                    else:
                        cache_file.unlink()  # Remove expired cache
                except Exception:
                    pass
            
            return None
    
    def set(self, key: str, content: Any, ttl: int = 300):
        """Set cached data"""
        with self.lock:
            cache_key = self._get_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            data = {
                'content': content,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logging.warning(f"Failed to cache data: {e}")
    
    def clear(self):
        """Clear all cached data"""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass


class BaseCrawler(ABC):
    """Base crawler class providing common functionality"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.status = CrawlerStatus.STOPPED
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.cache = Cache() if config.enable_cache else None
        self.last_run_time: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
        self.success_count = 0
        self.last_error: Optional[str] = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Monitoring
        self.performance_metrics = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_duration': 0.0,
            'last_run_duration': 0.0,
            'data_points_collected': 0
        }
    
    async def start(self):
        """Start the crawler"""
        if self.status == CrawlerStatus.RUNNING:
            self.logger.warning("Crawler is already running")
            return
        
        self.logger.info(f"Starting crawler {self.config.name}")
        self.status = CrawlerStatus.RUNNING
        self._running = True
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': f'MarketCrawler/1.0'}
        )
        
        # Start crawler loop
        self._task = asyncio.create_task(self._crawl_loop())
    
    async def stop(self):
        """Stop the crawler"""
        self.logger.info(f"Stopping crawler {self.config.name}")
        self._running = False
        self.status = CrawlerStatus.STOPPED
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            self.session = None
    
    async def pause(self):
        """Pause the crawler"""
        if self.status == CrawlerStatus.RUNNING:
            self.status = CrawlerStatus.PAUSED
            self.logger.info(f"Crawler {self.config.name} paused")
    
    async def resume(self):
        """Resume the crawler"""
        if self.status == CrawlerStatus.PAUSED:
            self.status = CrawlerStatus.RUNNING
            self.logger.info(f"Crawler {self.config.name} resumed")
    
    async def _crawl_loop(self):
        """Main crawler loop"""
        try:
            while self._running:
                if self.status == CrawlerStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    await self._single_crawl()
                    self.success_count += 1
                except Exception as e:
                    self.error_count += 1
                    self.last_error = str(e)
                    self.logger.error(f"Error in crawl: {e}")
                    
                    if self.error_count >= self.config.error_threshold:
                        self.status = CrawlerStatus.ERROR
                        self.logger.error(f"Crawler {self.config.name} entering error state")
                
                self.last_run_time = datetime.now()
                
                # Wait for next interval
                await asyncio.sleep(self.config.interval)
        
        except asyncio.CancelledError:
            self.logger.info("Crawler loop cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error in crawler loop: {e}")
            self.status = CrawlerStatus.ERROR
    
    async def _single_crawl(self):
        """Execute a single crawl operation"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{self.config.name}_{int(time.time() // self.config.interval)}"
            cached_data = None
            
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.logger.debug("Using cached data")
                    await self._process_data(cached_data, source="cache")
                    return
            
            # Fetch new data
            await self.rate_limiter.acquire()
            data = await self._fetch_data()
            
            if data:
                # Cache the data
                if self.cache:
                    self.cache.set(cache_key, data, self.config.cache_duration)
                
                # Store data
                if self.config.enable_storage:
                    await self._store_data(data)
                
                # Process data
                await self._process_data(data)
                
                crawl_duration = time.time() - start_time
                self.performance_metrics['last_run_duration'] = crawl_duration
                self.performance_metrics['data_points_collected'] += 1
                
                self.logger.info(f"Crawl completed in {crawl_duration:.2f}s")
            else:
                self.logger.warning("No data retrieved")
        
        except Exception as e:
            crawl_duration = time.time() - start_time
            self.logger.error(f"Crawl failed after {crawl_duration:.2f}s: {e}")
            raise
    
    async def _fetch_data(self) -> Any:
        """Fetch data from source - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _fetch_data")
    
    async def _process_data(self, data: Any, source: str = "live"):
        """Process fetched data - can be overridden by subclasses"""
        self.logger.debug(f"Processing {len(data) if isinstance(data, (list, dict)) else 1} data points")
    
    async def _store_data(self, data: Any):
        """Store data to file/database - can be overridden by subclasses"""
        if not self.config.enable_storage:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        filepath = Path(self.config.storage_path) / filename
        
        try:
            await aiofiles.makedirs(filepath.parent, exist_ok=True)
            
            if isinstance(data, (list, dict)):
                async with aiofiles.open(filepath, 'w') as f:
                    await f.write(json.dumps(data, indent=2, default=str))
            else:
                async with aiofiles.open(filepath, 'w') as f:
                    await f.write(str(data))
            
            self.logger.debug(f"Data stored to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
    
    async def fetch_once(self) -> CrawlResult:
        """Execute a single crawl and return result"""
        start_time = time.time()
        
        try:
            if not self.session:
                connector = aiohttp.TCPConnector()
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
            
            await self.rate_limiter.acquire()
            data = await self._fetch_data()
            
            crawl_duration = time.time() - start_time
            
            return CrawlResult(
                success=True,
                data=data,
                timestamp=datetime.now(),
                crawl_duration=crawl_duration,
                source=self.config.name
            )
        
        except Exception as e:
            crawl_duration = time.time() - start_time
            self.logger.error(f"Crawl failed: {e}")
            
            return CrawlResult(
                success=False,
                data=None,
                timestamp=datetime.now(),
                crawl_duration=crawl_duration,
                error_message=str(e),
                source=self.config.name
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        return {
            'name': self.config.name,
            'status': self.status.value,
            'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
            'run_count': self.run_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'performance_metrics': self.performance_metrics.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'healthy': self.status not in [CrawlerStatus.ERROR],
            'status': self.status.value,
            'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'uptime': (datetime.now() - self.last_run_time).total_seconds() if self.last_run_time else 0
        }
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols/tickers"""
        pass
    
    @abstractmethod
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        pass