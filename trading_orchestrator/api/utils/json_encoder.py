"""
JSON encoder for API responses

Provides custom JSON encoding for:
- datetime objects
- Enum values
- Decimal objects
- Complex data structures
"""

import json
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for API responses
    
    Handles datetime, Decimal, Enum, UUID and other complex types
    """
    
    def default(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle date objects
        elif isinstance(obj, date):
            return obj.isoformat()
        
        # Handle time objects
        elif isinstance(obj, time):
            return obj.isoformat()
        
        # Handle Decimal objects
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Handle Enum objects
        elif isinstance(obj, Enum):
            return obj.value
        
        # Handle UUID objects
        elif isinstance(obj, UUID):
            return str(obj)
        
        # Handle bytes objects
        elif isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return obj.hex()
        
        # Handle set objects
        elif isinstance(obj, set):
            return list(obj)
        
        # Handle complex numbers
        elif isinstance(obj, complex):
            return {
                "real": obj.real,
                "imag": obj.imag,
                "__complex__": True
            }
        
        # Handle custom objects with to_dict method
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        # Handle custom objects with dict method
        elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            return obj.dict()
        
        # Handle numpy arrays (if available)
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        
        # Default: convert to string
        return str(obj)
    
    @classmethod
    def encode(cls, obj: Any) -> str:
        """Encode object to JSON string"""
        return json.dumps(obj, cls=cls, separators=(',', ':'))
    
    @classmethod
    def dumps(cls, obj: Any, **kwargs) -> str:
        """Encode object to JSON string with additional options"""
        kwargs.setdefault('cls', cls)
        kwargs.setdefault('separators', (',', ':'))
        return json.dumps(obj, **kwargs)
    
    @classmethod
    def pretty_dumps(cls, obj: Any, **kwargs) -> str:
        """Encode object to pretty-printed JSON string"""
        kwargs.setdefault('cls', cls)
        kwargs.setdefault('separators', (',', ': '))
        kwargs.setdefault('indent', 2)
        return json.dumps(obj, **kwargs)


def encode_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, list):
        return [encode_datetime(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: encode_datetime(value) for key, value in obj.items()}
    else:
        return obj


def encode_for_api_response(data: Any) -> Any:
    """Encode data for API response"""
    if isinstance(data, dict):
        return {key: encode_for_api_response(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [encode_for_api_response(item) for item in data]
    elif isinstance(data, tuple):
        return [encode_for_api_response(item) for item in data]
    elif isinstance(data, set):
        return [encode_for_api_response(item) for item in data]
    else:
        return JSONEncoder().default(data)


def safe_json_dumps(obj: Any, default: Any = None) -> str:
    """
    Safely encode object to JSON string
    
    Args:
        obj: Object to encode
        default: Default value if encoding fails
    
    Returns:
        JSON string or default value
    """
    try:
        return JSONEncoder.encode(obj)
    except Exception as e:
        logger.warning(f"JSON encoding failed: {e}")
        return JSONEncoder.encode(default or str(obj))


def validate_json_string(json_str: str) -> bool:
    """Validate if string is valid JSON"""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def format_json_error(error_msg: str, json_str: str = None) -> Dict[str, Any]:
    """Format JSON validation error"""
    result = {
        "error": "Invalid JSON",
        "message": error_msg
    }
    
    if json_str:
        result["provided"] = json_str[:100] + "..." if len(json_str) > 100 else json_str
    
    return result


def create_json_response(
    data: Any,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None
):
    """Create JSON response with proper encoding"""
    from fastapi.responses import JSONResponse
    
    json_data = JSONEncoder.encode(data)
    
    response_headers = headers or {}
    response_headers["Content-Type"] = "application/json"
    
    return JSONResponse(
        content=data,  # FastAPI will handle JSON encoding
        status_code=status_code,
        headers=response_headers
    )


def pretty_format_json(obj: Any) -> str:
    """Pretty-format object as JSON string"""
    return JSONEncoder.pretty_dumps(obj)


def minify_json(obj: Any) -> str:
    """Minify object as JSON string"""
    return JSONEncoder.dumps(obj, separators=(',', ':'))


def encode_pagination_data(
    items: List[Any],
    total: int,
    page: int,
    size: int
) -> Dict[str, Any]:
    """Encode pagination data for API response"""
    pages = (total + size - 1) // size
    
    return {
        "items": [encode_for_api_response(item) for item in items],
        "pagination": {
            "total": total,
            "page": page,
            "size": size,
            "pages": pages,
            "has_next": page < pages,
            "has_prev": page > 1
        }
    }


def encode_strategy_data(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Encode strategy data for API response"""
    encoded_data = {}
    
    for key, value in strategy_data.items():
        if key in ["parameters", "tags", "symbols", "metadata"]:
            # Handle complex data structures
            encoded_data[key] = encode_for_api_response(value)
        elif key in ["created_at", "updated_at", "last_executed", "signal_time", "expires_at"]:
            # Handle datetime fields
            if value:
                encoded_data[key] = encode_datetime(value)
        else:
            # Handle other fields
            encoded_data[key] = JSONEncoder().default(value)
    
    return encoded_data


def encode_performance_data(performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """Encode performance data for API response"""
    encoded_data = {}
    
    for key, value in performance_data.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Handle numeric values
            if value is not None:
                encoded_data[key] = float(value) if isinstance(value, Decimal) else value
        else:
            # Handle other values
            encoded_data[key] = JSONEncoder().default(value)
    
    return encoded_data


# Logging setup
import logging
logger = logging.getLogger(__name__)
