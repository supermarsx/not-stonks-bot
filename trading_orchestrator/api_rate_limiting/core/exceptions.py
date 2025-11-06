"""
Rate Limiting Exception Classes
"""

class RateLimitException(Exception):
    """Base exception for rate limiting"""
    pass

class RateLimitExceededException(RateLimitException):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, reset_time: int = None, limit: int = None, remaining: int = None):
        super().__init__(message)
        self.reset_time = reset_time
        self.limit = limit
        self.remaining = remaining

class CircuitBreakerOpenException(RateLimitException):
    """Exception raised when circuit breaker is open"""
    pass

class RequestTimeoutException(RateLimitException):
    """Exception raised when request times out"""
    pass

class InvalidRequestException(RateLimitException):
    """Exception raised for invalid requests"""
    pass

class AuthenticationException(RateLimitException):
    """Exception raised for authentication failures"""
    pass

class BrokerAPIException(RateLimitException):
    """Exception raised for broker-specific API errors"""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response