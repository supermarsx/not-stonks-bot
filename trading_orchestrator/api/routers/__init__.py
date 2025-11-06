"""
Routers module for API

Exports all API route handlers
"""

from . import strategies, dependencies
from .strategies import router as strategies_router

__all__ = [
    "strategies_router",
    "dependencies",
]
