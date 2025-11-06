"""
Analytics Configuration

Central configuration definitions for the analytics system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class AnalyticsConfig:
    """Configuration for analytics engine"""
    real_time_update_interval: int = 30  # seconds
    batch_processing_interval: int = 300  # 5 minutes
    report_generation_interval: int = 3600  # 1 hour
    enable_real_time: bool = True
    enable_batch_processing: bool = True
    enable_automatic_reports: bool = True
    max_concurrent_calculations: int = 10
    data_retention_days: int = 90
    benchmark_indices: List[str] = None
    risk_free_rate: float = 0.02