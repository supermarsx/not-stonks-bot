"""
Error handlers for API

Provides centralized error handling for:
- HTTP exceptions
- Validation errors
- Database errors
- External API errors
- System errors
"""

import traceback
from typing import Any, Dict, Optional, List
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import ValidationError as PydanticValidationError
from loguru import logger

from .response_formatter import ResponseFormatter


def setup_error_handlers(app):
    """Setup error handlers for the FastAPI application"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ResponseFormatter.error_response(
                message=_get_http_message(exc.status_code),
                detail=exc.detail,
                status_code=exc.status_code
            )
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc}")
        
        errors = []
        for error in exc.errors():
            field = ".".join(str(x) for x in error["loc"] if isinstance(x, str))
            errors.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content=ResponseFormatter.validation_error_response(errors)
        )
    
    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle database exceptions"""
        logger.error(f"Database error: {exc}")
        
        # Log traceback for debugging
        logger.debug(f"Database error traceback: {traceback.format_exc()}")
        
        if isinstance(exc, IntegrityError):
            return JSONResponse(
                status_code=400,
                content=ResponseFormatter.error_response(
                    message="Database integrity error",
                    detail="Data integrity constraint violation",
                    status_code=400
                )
            )
        elif isinstance(exc, OperationalError):
            return JSONResponse(
                status_code=503,
                content=ResponseFormatter.error_response(
                    message="Database service unavailable",
                    detail="Database connection or operation failed",
                    status_code=503
                )
            )
        else:
            return JSONResponse(
                status_code=500,
                content=ResponseFormatter.error_response(
                    message="Database error",
                    detail="An error occurred while accessing the database",
                    status_code=500
                )
            )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: PydanticValidationError):
        """Handle Pydantic validation errors"""
        logger.warning(f"Pydantic validation error: {exc}")
        
        errors = []
        for error in exc.errors():
            field = ".".join(str(x) for x in error["loc"] if isinstance(x, str))
            errors.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content=ResponseFormatter.validation_error_response(errors)
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors"""
        logger.warning(f"Value error: {exc}")
        
        return JSONResponse(
            status_code=400,
            content=ResponseFormatter.error_response(
                message="Invalid value",
                detail=str(exc),
                status_code=400
            )
        )
    
    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        """Handle key errors"""
        logger.warning(f"Key error: {exc}")
        
        return JSONResponse(
            status_code=400,
            content=ResponseFormatter.error_response(
                message="Missing required key",
                detail=f"Missing key: {str(exc)}",
                status_code=400
            )
        )
    
    @app.exception_handler(TypeError)
    async def type_error_handler(request: Request, exc: TypeError):
        """Handle type errors"""
        logger.warning(f"Type error: {exc}")
        
        return JSONResponse(
            status_code=400,
            content=ResponseFormatter.error_response(
                message="Type error",
                detail=str(exc),
                status_code=400
            )
        )
    
    @app.exception_handler(AttributeError)
    async def attribute_error_handler(request: Request, exc: AttributeError):
        """Handle attribute errors"""
        logger.warning(f"Attribute error: {exc}")
        
        return JSONResponse(
            status_code=500,
            content=ResponseFormatter.error_response(
                message="Attribute error",
                detail="An internal attribute access error occurred",
                status_code=500
            )
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found errors"""
        logger.warning(f"File not found: {exc}")
        
        return JSONResponse(
            status_code=404,
            content=ResponseFormatter.error_response(
                message="File not found",
                detail=str(exc),
                status_code=404
            )
        )
    
    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError):
        """Handle connection errors"""
        logger.error(f"Connection error: {exc}")
        
        return JSONResponse(
            status_code=503,
            content=ResponseFormatter.error_response(
                message="Connection error",
                detail="Failed to connect to external service",
                status_code=503
            )
        )
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        """Handle timeout errors"""
        logger.warning(f"Timeout error: {exc}")
        
        return JSONResponse(
            status_code=408,
            content=ResponseFormatter.error_response(
                message="Request timeout",
                detail="The request took too long to complete",
                status_code=408
            )
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        logger.exception(f"Unhandled exception: {exc}")
        
        # Log traceback for debugging
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content=ResponseFormatter.error_response(
                message="Internal server error",
                detail="An unexpected error occurred",
                status_code=500
            )
        )


def _get_http_message(status_code: int) -> str:
    """Get HTTP status code message"""
    messages = {
        400: "Bad request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not found",
        405: "Method not allowed",
        406: "Not acceptable",
        408: "Request timeout",
        409: "Conflict",
        410: "Gone",
        411: "Length required",
        412: "Precondition failed",
        413: "Payload too large",
        414: "URI too long",
        415: "Unsupported media type",
        416: "Range not satisfiable",
        417: "Expectation failed",
        418: "I'm a teapot",
        422: "Unprocessable entity",
        423: "Locked",
        424: "Failed dependency",
        425: "Too early",
        426: "Upgrade required",
        428: "Precondition required",
        429: "Too many requests",
        431: "Request header fields too large",
        451: "Unavailable for legal reasons",
        500: "Internal server error",
        501: "Not implemented",
        502: "Bad gateway",
        503: "Service unavailable",
        504: "Gateway timeout",
        505: "HTTP version not supported",
        506: "Variant also negotiates",
        507: "Insufficient storage",
        508: "Loop detected",
        510: "Not extended",
        511: "Network authentication required"
    }
    
    return messages.get(status_code, "Unknown error")


class APIException(Exception):
    """Custom API exception"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        detail: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.message)


class ValidationException(APIException):
    """Validation error exception"""
    
    def __init__(self, message: str = "Validation failed", errors: Optional[List[str]] = None):
        super().__init__(
            message=message,
            status_code=422,
            detail="Request validation failed",
            error_code="VALIDATION_ERROR"
        )
        self.errors = errors or []


class AuthenticationException(APIException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=401,
            detail="Invalid or missing authentication credentials",
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationException(APIException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            status_code=403,
            detail="You don't have permission to access this resource",
            error_code="AUTHORIZATION_ERROR"
        )


class NotFoundException(APIException):
    """Resource not found exception"""
    
    def __init__(self, resource: str = "Resource"):
        super().__init__(
            message=f"{resource} not found",
            status_code=404,
            detail=f"The requested {resource.lower()} could not be found",
            error_code="NOT_FOUND"
        )


class ConflictException(APIException):
    """Resource conflict exception"""
    
    def __init__(self, message: str = "Resource conflict"):
        super().__init__(
            message=message,
            status_code=409,
            detail="A conflict occurred with the current state of the resource",
            error_code="CONFLICT"
        )


class DatabaseException(APIException):
    """Database error exception"""
    
    def __init__(self, message: str = "Database error"):
        super().__init__(
            message=message,
            status_code=500,
            detail="A database error occurred",
            error_code="DATABASE_ERROR"
        )


class ExternalServiceException(APIException):
    """External service error exception"""
    
    def __init__(self, service: str = "external service", message: str = None):
        super().__init__(
            message=message or f"{service} error",
            status_code=502,
            detail=f"Error communicating with {service}",
            error_code="EXTERNAL_SERVICE_ERROR"
        )


class RateLimitException(APIException):
    """Rate limit exceeded exception"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            detail="Too many requests, please try again later",
            error_code="RATE_LIMIT_EXCEEDED"
        )