"""
Utility functions for API response formatting

Provides standardized response formatting for:
- Success responses
- Error responses
- Pagination formatting
- Data validation
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from fastapi import status
from fastapi.responses import JSONResponse


class ResponseFormatter:
    """
    Standardized response formatting utilities
    """
    
    @staticmethod
    def success_response(
        data: Any = None,
        message: str = "Success",
        status_code: int = status.HTTP_200_OK,
        pagination: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format successful API response"""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        if pagination:
            response["pagination"] = pagination
        
        return response
    
    @staticmethod
    def error_response(
        message: str,
        detail: Optional[str] = None,
        errors: Optional[List[str]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> Dict[str, Any]:
        """Format error API response"""
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if detail:
            response["detail"] = detail
        
        if errors:
            response["errors"] = errors
        
        return response
    
    @staticmethod
    def validation_error_response(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format validation error response"""
        return ResponseFormatter.error_response(
            message="Validation error",
            detail="Request validation failed",
            errors=errors,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    
    @staticmethod
    def not_found_response(resource: str = "Resource") -> Dict[str, Any]:
        """Format not found response"""
        return ResponseFormatter.error_response(
            message=f"{resource} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )
    
    @staticmethod
    def unauthorized_response(message: str = "Authentication required") -> Dict[str, Any]:
        """Format unauthorized response"""
        return ResponseFormatter.error_response(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    @staticmethod
    def forbidden_response(message: str = "Access denied") -> Dict[str, Any]:
        """Format forbidden response"""
        return ResponseFormatter.error_response(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN
        )
    
    @staticmethod
    def paginated_response(
        items: List[Any],
        total: int,
        page: int,
        size: int
    ) -> Dict[str, Any]:
        """Format paginated response"""
        pages = (total + size - 1) // size
        
        return {
            "success": True,
            "message": "Data retrieved successfully",
            "data": items,
            "pagination": {
                "total": total,
                "page": page,
                "size": size,
                "pages": pages,
                "has_next": page < pages,
                "has_prev": page > 1
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def created_response(
        data: Any,
        message: str = "Resource created successfully"
    ) -> Dict[str, Any]:
        """Format created response"""
        return ResponseFormatter.success_response(
            data=data,
            message=message,
            status_code=status.HTTP_201_CREATED
        )
    
    @staticmethod
    def updated_response(
        data: Optional[Any] = None,
        message: str = "Resource updated successfully"
    ) -> Dict[str, Any]:
        """Format updated response"""
        return ResponseFormatter.success_response(
            data=data,
            message=message,
            status_code=status.HTTP_200_OK
        )
    
    @staticmethod
    def deleted_response(message: str = "Resource deleted successfully") -> Dict[str, Any]:
        """Format deleted response"""
        return ResponseFormatter.success_response(
            message=message,
            status_code=status.HTTP_204_NO_CONTENT
        )
    
    @staticmethod
    def backtest_started_response(backtest_id: str) -> Dict[str, Any]:
        """Format backtest started response"""
        return ResponseFormatter.success_response(
            data={
                "backtest_id": backtest_id,
                "status": "started",
                "message": "Backtest has been queued for execution"
            },
            message="Backtest started successfully"
        )
    
    @staticmethod
    def websocket_connected_response(connection_id: str) -> Dict[str, Any]:
        """Format WebSocket connected response"""
        return ResponseFormatter.success_response(
            data={
                "connection_id": connection_id,
                "status": "connected",
                "message": "WebSocket connection established"
            },
            message="WebSocket connected successfully"
        )
    
    @staticmethod
    def strategy_performance_response(performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format strategy performance response"""
        return ResponseFormatter.success_response(
            data=performance_data,
            message="Performance data retrieved successfully"
        )
    
    @staticmethod
    def strategy_signals_response(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format strategy signals response"""
        return ResponseFormatter.success_response(
            data=signals,
            message="Strategy signals retrieved successfully"
        )
    
    @staticmethod
    def strategy_comparison_response(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format strategy comparison response"""
        return ResponseFormatter.success_response(
            data=comparison_data,
            message="Strategy comparison completed successfully"
        )
    
    @staticmethod
    def health_check_response(status: str, components: Dict[str, str]) -> Dict[str, Any]:
        """Format health check response"""
        return ResponseFormatter.success_response(
            data={
                "status": status,
                "components": components
            },
            message="Health check completed"
        )
    
    @staticmethod
    def system_overview_response(overview_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format system overview response"""
        return ResponseFormatter.success_response(
            data=overview_data,
            message="System overview retrieved successfully"
        )
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate required fields in data"""
        errors = []
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Required field '{field}' is missing")
        
        return errors
    
    @staticmethod
    def format_validation_errors(field_errors: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Format field validation errors"""
        formatted_errors = []
        
        for field, errors in field_errors.items():
            for error in errors:
                formatted_errors.append({
                    "field": field,
                    "message": error,
                    "type": "validation_error"
                })
        
        return formatted_errors
    
    @staticmethod
    def standardize_error_code(error_code: str) -> str:
        """Standardize error codes"""
        error_code_mapping = {
            "validation_error": "VALIDATION_ERROR",
            "authentication_error": "AUTHENTICATION_ERROR",
            "authorization_error": "AUTHORIZATION_ERROR",
            "not_found": "NOT_FOUND",
            "duplicate": "DUPLICATE_RESOURCE",
            "invalid_parameters": "INVALID_PARAMETERS",
            "external_api_error": "EXTERNAL_API_ERROR",
            "database_error": "DATABASE_ERROR",
            "system_error": "INTERNAL_ERROR"
        }
        
        return error_code_mapping.get(error_code.lower(), "UNKNOWN_ERROR")
    
    @staticmethod
    def get_response_headers(
        cache_control: Optional[str] = None,
        content_type: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Get standardized response headers"""
        headers = {}
        
        if cache_control:
            headers["Cache-Control"] = cache_control
        
        if content_type:
            headers["Content-Type"] = content_type
        
        if custom_headers:
            headers.update(custom_headers)
        
        return headers
