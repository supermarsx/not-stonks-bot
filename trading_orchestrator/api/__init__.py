"""
API module initialization

Exports main application instance and core components
"""

from .main import app
from .routers import strategies
from .schemas import *

__all__ = [
    "app",
    "strategies",
]
