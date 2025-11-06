"""
Request Management System

Handles request queuing, prioritization, batching, and retry logic
for API rate limiting compliance across all broker integrations.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Type, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import threading
import json
from datetime import datetime, timedelta
import heapq
from contextlib import asynccontextmanager

from .rate_limiter import RateLimiterManager, RequestType, RateLimitStatus
from .exceptions import (
    RateLimitExceededException, RequestTimeoutException, 
    InvalidRequestException, BrokerAPIException
)


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Critical for trading operations
    HIGH = 2        # Important but not critical
    NORMAL = 3      # Standard priority
    LOW = 4         # Background operations
    BATCH = 5       # Batch operations


class RequestStatus(Enum):
    """Request processing status"""
    PENDING = auto()
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class RequestCategory(Enum):
    """Request categories for batching"""
    ACCOUNT = "account"
    ORDERS = "orders"
    POSITIONS = "positions"
    MARKET_DATA = "market_data"
    HISTORICAL = "historical"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    timeout: float = 30.0


@dataclass
class Request:
    """Individual API request"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: RequestType = RequestType.ACCOUNT_INFO
    priority: RequestPriority = RequestPriority.NORMAL
    category: RequestCategory = RequestCategory.ACCOUNT
    callback: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Retry configuration
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    
    # Status tracking
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Retry tracking
    attempts: int = 0
    last_error: Optional[Exception] = None
    next_retry_at: Optional[datetime] = None
    
    # Request metadata
    symbol: Optional[str] = None
    estimated_cost: float = 0.0
    requires_order_confirmation: bool = False
    metadata: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue (lower number = higher priority)"""
        if not isinstance(other, Request):
            return NotImplemented
        
        # Compare by priority first
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # Then by creation time (FIFO for same priority)
        return self.created_at < other.created_at


@dataclass
class BatchOperation:
    """Batch operation combining multiple similar requests"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: RequestCategory = RequestCategory.ACCOUNT
    requests: List[Request] = field(default_factory=list)
    max_batch_size: int = 10
    batch_timeout: float = 5.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: RequestStatus = RequestStatus.PENDING
    
    def can_add_request(self, request: Request) -> bool:
        """Check if request can be added to this batch"""
        return (
            len(self.requests) < self.max_batch_size and
            request.category == self.category and
            (datetime.utcnow() - self.created_at).total_seconds() < self.batch_timeout
        )
    
    def add_request(self, request: Request):
        """Add request to batch"""
        if self.can_add_request(request):
            self.requests.append(request)


class DeadLetterQueue:
    """Dead letter queue for failed requests"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._failed_callbacks: List[Callable] = []
    
    def add_failed_request(self, request: Request, error: Exception):
        """Add failed request to dead letter queue"""
        with self._lock:
            failed_entry = {
                "request": request,
                "error": error,
                "failed_at": datetime.utcnow(),
                "error_type": type(error).__name__,
                "retry_count": request.attempts
            }
            
            self._queue.append(failed_entry)
            
            # Notify callbacks
            for callback in self._failed_callbacks:
                try:
                    callback(failed_entry)
                except Exception as e:
                    print(f"Dead letter queue callback error: {e}")
    
    def get_failed_requests(self) -> List[Dict]:
        """Get all failed requests"""
        with self._lock:
            return list(self._queue)
    
    def add_callback(self, callback: Callable):
        """Add callback for failed request notifications"""
        self._failed_callbacks.append(callback)


class RequestManager:
    """
    Central request management system
    
    Manages request queuing, prioritization, batching, and retry logic
    to ensure compliance with broker API rate limits.
    """
    
    def __init__(self, rate_limiter_manager: RateLimiterManager):
        self.rate_limiter = rate_limiter_manager
        self._lock = threading.RLock()
        
        # Request queues
        self._pending_requests: List[Request] = []
        self._processing_requests: Dict[str, Request] = {}
        self._dead_letter_queue = DeadLetterQueue()
        
        # Batch operations
        self._batches: List[BatchOperation] = []
        self._batch_waiting_queue: List[Request] = []
        
        # Processing statistics
        self._stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "average_processing_time": 0.0,
            "requests_by_category": defaultdict(int),
            "requests_by_priority": defaultdict(int),
            "batch_operations": 0,
            "max_queue_depth": 0,
            "average_queue_depth": 0.0
        }
        
        # Control flags
        self._is_running = False
        self._is_processing = False
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self._queue_depth_history = deque(maxlen=100)
        self._processing_times = deque(maxlen=1000)
        
        # Setup batch processing
        self._batch_processors: Dict[RequestCategory, Callable] = {}
        self._setup_default_batch_processors()
    
    def _setup_default_batch_processors(self):
        """Setup default batch processors for different categories"""
        self._batch_processors = {
            RequestCategory.ACCOUNT: self._batch_account_requests,
            RequestCategory.MARKET_DATA: self._batch_market_data_requests,
            RequestCategory.POSITIONS: self._batch_position_requests,
            RequestCategory.ORDERS: self._batch_order_requests,
            RequestCategory.HISTORICAL: self._batch_historical_requests
        }
    
    async def submit_request(
        self,
        request_type: RequestType,
        callback: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        category: RequestCategory = RequestCategory.ACCOUNT,
        symbol: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> str:
        """
        Submit request for processing
        
        Args:
            request_type: Type of API request
            callback: Function to execute the request
            priority: Request priority
            category: Request category for batching
            symbol: Symbol for market data requests
            timeout: Request timeout
            retry_config: Retry configuration
            **kwargs: Additional arguments for callback
            
        Returns:
            str: Request ID for tracking
        """
        request = Request(
            request_type=request_type,
            priority=priority,
            category=category,
            callback=callback,
            args=(callback,),
            kwargs=kwargs,
            retry_config=retry_config or RetryConfig(),
            symbol=symbol
        )
        
        if timeout:
            request.retry_config.timeout = timeout
        
        await self._queue_request(request)
        
        self._stats["total_requests"] += 1
        self._stats["requests_by_category"][category.value] += 1
        self._stats["requests_by_priority"][priority.name] += 1
        
        return request.id
    
    async def _queue_request(self, request: Request):
        """Add request to queue"""
        with self._lock:
            # Try to add to existing batch first
            batch = self._find_or_create_batch(request.category)
            if batch and batch.can_add_request(request):
                batch.add_request(request)
                if len(batch.requests) == 1:
                    # First request in batch, schedule batch processing
                    asyncio.create_task(self._process_batch(batch))
            else:
                # Add to regular queue
                heapq.heappush(self._pending_requests, request)
            
            # Track queue depth
            queue_depth = len(self._pending_requests)
            self._queue_depth_history.append(queue_depth)
            self._stats["max_queue_depth"] = max(self._stats["max_queue_depth"], queue_depth)
            self._stats["average_queue_depth"] = (
                sum(self._queue_depth_history) / len(self._queue_depth_history)
            )
    
    def _find_or_create_batch(self, category: RequestCategory) -> Optional[BatchOperation]:
        """Find existing batch or create new one for category"""
        # Look for existing batch with waiting requests
        for i, request in enumerate(self._batch_waiting_queue):
            if request.category == category:
                batch = BatchOperation(category=category)
                batch.add_request(request)
                del self._batch_waiting_queue[i]
                return batch
        
        # Look for incomplete batch
        for batch in self._batches:
            if (batch.category == category and 
                batch.status != RequestStatus.COMPLETED and
                batch.can_add_request(Request(category=category))):
                return batch
        
        # Create new batch
        new_batch = BatchOperation(category=category)
        self._batches.append(new_batch)
        return new_batch
    
    async def _process_batch(self, batch: BatchOperation):
        """Process batch of similar requests"""
        try:
            batch.status = RequestStatus.PROCESSING
            self._stats["batch_operations"] += 1
            
            # Get appropriate batch processor
            processor = self._batch_processors.get(batch.category)
            if not processor:
                # Fall back to individual processing
                for request in batch.requests:
                    await self._process_single_request(request)
                return
            
            await processor(batch)
            batch.status = RequestStatus.COMPLETED
            
        except Exception as e:
            batch.status = RequestStatus.FAILED
            for request in batch.requests:
                await self._handle_request_error(request, e)
        finally:
            # Clean up completed batches
            self._batches = [b for b in self._batches if b.status != RequestStatus.COMPLETED]
    
    async def _batch_account_requests(self, batch: BatchOperation):
        """Process batched account information requests"""
        # This would typically make a single bulk API call
        # For now, process individually but with rate limiting
        for request in batch.requests:
            await self._process_single_request(request)
    
    async def _batch_market_data_requests(self, batch: BatchOperation):
        """Process batched market data requests"""
        # Group requests by symbol and timeframe
        symbol_groups = defaultdict(list)
        
        for request in batch.requests:
            symbol_groups[request.symbol].append(request)
        
        # Process each symbol group
        for symbol, requests in symbol_groups.items():
            # Combine requests for same symbol
            combined_request = Request(
                request_type=RequestType.MARKET_DATA,
                category=RequestCategory.MARKET_DATA,
                symbol=symbol,
                metadata={
                    "batch_size": len(requests),
                    "original_requests": [r.id for r in requests]
                }
            )
            await self._process_single_request(combined_request)
    
    async def _batch_position_requests(self, batch: BatchOperation):
        """Process batched position requests"""
        for request in batch.requests:
            await self._process_single_request(request)
    
    async def _batch_order_requests(self, batch: BatchOperation):
        """Process batched order requests - high priority, no batching for actual orders"""
        for request in batch.requests:
            await self._process_single_request(request)
    
    async def _batch_historical_requests(self, batch: BatchOperation):
        """Process batched historical data requests"""
        for request in batch.requests:
            await self._process_single_request(request)
    
    async def _process_single_request(self, request: Request):
        """Process individual request with rate limiting and retry logic"""
        start_time = time.time()
        
        try:
            request.status = RequestStatus.PROCESSING
            request.started_at = datetime.utcnow()
            
            with self._lock:
                self._processing_requests[request.id] = request
            
            # Apply rate limiting
            async with self.rate_limiter.rate_limited_request(request.request_type):
                # Execute request
                result = await self._execute_request(request)
                
                # Mark as completed
                request.status = RequestStatus.COMPLETED
                request.completed_at = datetime.utcnow()
                
                # Record processing time
                processing_time = time.time() - start_time
                self._processing_times.append(processing_time)
                self._stats["average_processing_time"] = (
                    sum(self._processing_times) / len(self._processing_times)
                )
                
                self._stats["completed_requests"] += 1
                
                # Store result if callback provided
                if request.callback:
                    try:
                        if asyncio.iscoroutinefunction(request.callback):
                            await request.callback(result, *request.args, **request.kwargs)
                        else:
                            request.callback(result, *request.args, **request.kwargs)
                    except Exception as e:
                        # Callback error doesn't affect request processing
                        print(f"Request callback error: {e}")
        
        except Exception as e:
            await self._handle_request_error(request, e)
        finally:
            with self._lock:
                self._processing_requests.pop(request.id, None)
    
    async def _execute_request(self, request: Request) -> Any:
        """Execute the actual request"""
        if not request.callback:
            raise InvalidRequestException("Request callback is required")
        
        # Execute callback with timeout
        try:
            if asyncio.iscoroutinefunction(request.callback):
                result = await asyncio.wait_for(
                    request.callback(*request.args, **request.kwargs),
                    timeout=request.retry_config.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: request.callback(*request.args, **request.kwargs)
                    ),
                    timeout=request.retry_config.timeout
                )
            return result
        except asyncio.TimeoutError:
            raise RequestTimeoutException(f"Request timed out after {request.retry_config.timeout}s")
        except Exception as e:
            # Check if error should trigger retry
            if self._should_retry(e, request):
                raise
            else:
                raise
    
    def _should_retry(self, error: Exception, request: Request) -> bool:
        """Determine if request should be retried"""
        # Check retry count
        if request.attempts >= request.retry_config.max_retries:
            return False
        
        # Check error type
        if isinstance(error, (RateLimitExceededException, BrokerAPIException)):
            # Check status code if available
            if hasattr(error, 'status_code'):
                if error.status_code in request.retry_config.retry_on_status_codes:
                    return True
            return True
        elif isinstance(error, RequestTimeoutException):
            return True
        
        return False
    
    async def _handle_request_error(self, request: Request, error: Exception):
        """Handle request error with retry logic"""
        request.last_error = error
        request.attempts += 1
        
        # Determine if we should retry
        if self._should_retry(error, request):
            # Schedule retry
            delay = self._calculate_retry_delay(request)
            request.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
            request.status = RequestStatus.RETRYING
            
            self._stats["retried_requests"] += 1
            
            # Re-queue for retry
            asyncio.create_task(self._schedule_retry(request, delay))
        else:
            # Mark as failed
            request.status = RequestStatus.FAILED
            request.completed_at = datetime.utcnow()
            
            self._stats["failed_requests"] += 1
            
            # Add to dead letter queue
            self._dead_letter_queue.add_failed_request(request, error)
    
    def _calculate_retry_delay(self, request: Request) -> float:
        """Calculate delay for next retry attempt"""
        base_delay = request.retry_config.initial_delay
        
        # Exponential backoff
        if request.attempts > 1:
            delay = base_delay * (request.retry_config.exponential_base ** (request.attempts - 1))
        else:
            delay = base_delay
        
        # Apply maximum delay
        delay = min(delay, request.retry_config.max_delay)
        
        # Add jitter if enabled
        if request.retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.9)
            delay *= jitter
        
        return delay
    
    async def _schedule_retry(self, request: Request, delay: float):
        """Schedule request retry after delay"""
        await asyncio.sleep(delay)
        
        # Reset status and re-queue
        request.status = RequestStatus.PENDING
        request.next_retry_at = None
        
        await self._queue_request(request)
    
    async def start_processing(self):
        """Start request processing loop"""
        if self._is_running:
            return
        
        self._is_running = True
        self._shutdown_event.clear()
        
        # Start processing tasks
        processing_task = asyncio.create_task(self._processing_loop())
        batch_task = asyncio.create_task(self._batch_processing_loop())
        
        # Store tasks for cleanup
        self._processing_tasks = [processing_task, batch_task]
    
    async def stop_processing(self):
        """Stop request processing"""
        if not self._is_running:
            return
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel processing tasks
        for task in getattr(self, '_processing_tasks', []):
            task.cancel()
        
        # Wait for tasks to complete
        for task in getattr(self, '_processing_tasks', []):
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self._is_running:
            try:
                # Get next request from queue
                with self._lock:
                    if self._pending_requests:
                        request = heapq.heappop(self._pending_requests)
                    else:
                        request = None
                
                if request:
                    # Process request
                    asyncio.create_task(self._process_single_request(request))
                else:
                    # No requests to process, wait
                    await asyncio.sleep(0.1)
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                    
            except Exception as e:
                print(f"Processing loop error: {e}")
                await asyncio.sleep(1)
    
    async def _batch_processing_loop(self):
        """Batch processing loop"""
        while self._is_running:
            try:
                # Process waiting batches
                current_time = datetime.utcnow()
                for batch in list(self._batches):
                    if (batch.status == RequestStatus.PENDING and
                        (current_time - batch.created_at).total_seconds() >= batch.batch_timeout):
                        await self._process_batch(batch)
                
                await asyncio.sleep(0.5)
                
                if self._shutdown_event.is_set():
                    break
                    
            except Exception as e:
                print(f"Batch processing loop error: {e}")
                await asyncio.sleep(1)
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific request"""
        with self._lock:
            # Check pending queue
            for request in self._pending_requests:
                if request.id == request_id:
                    return self._serialize_request(request)
            
            # Check processing
            if request_id in self._processing_requests:
                return self._serialize_request(self._processing_requests[request_id])
            
            # Check batches
            for batch in self._batches:
                for request in batch.requests:
                    if request.id == request_id:
                        return self._serialize_request(request)
            
            # Check dead letter queue
            for failed_entry in self._dead_letter_queue.get_failed_requests():
                if failed_entry["request"].id == request_id:
                    entry = failed_entry.copy()
                    entry["status"] = "FAILED"
                    return entry
            
            return None
    
    def _serialize_request(self, request: Request) -> Dict[str, Any]:
        """Serialize request for status reporting"""
        return {
            "id": request.id,
            "request_type": request.request_type.value,
            "category": request.category.value,
            "priority": request.priority.name,
            "status": request.status.name,
            "attempts": request.attempts,
            "created_at": request.created_at.isoformat(),
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "next_retry_at": request.next_retry_at.isoformat() if request.next_retry_at else None,
            "symbol": request.symbol,
            "estimated_cost": request.estimated_cost,
            "last_error": str(request.last_error) if request.last_error else None
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request manager statistics"""
        with self._lock:
            return {
                "total_requests": self._stats["total_requests"],
                "completed_requests": self._stats["completed_requests"],
                "failed_requests": self._stats["failed_requests"],
                "retried_requests": self._stats["retried_requests"],
                "processing_requests": len(self._processing_requests),
                "pending_requests": len(self._pending_requests),
                "active_batches": len(self._batches),
                "batch_operations": self._stats["batch_operations"],
                "average_processing_time": self._stats["average_processing_time"],
                "average_queue_depth": self._stats["average_queue_depth"],
                "max_queue_depth": self._stats["max_queue_depth"],
                "success_rate": (
                    self._stats["completed_requests"] / 
                    max(1, self._stats["total_requests"])
                ),
                "retry_rate": (
                    self._stats["retried_requests"] / 
                    max(1, self._stats["total_requests"])
                ),
                "requests_by_category": dict(self._stats["requests_by_category"]),
                "requests_by_priority": dict(self._stats["requests_by_priority"]),
                "dead_letter_queue_size": len(self._dead_letter_queue.get_failed_requests())
            }