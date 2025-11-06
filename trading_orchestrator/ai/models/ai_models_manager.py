"""
AI Models Manager - Multi-tier LLM orchestration for trading decisions

Supports multiple LLM providers and implements a tiered approach:
- Tier 1 (Reasoning): Claude 3.5 Sonnet, GPT-4 for complex analysis
- Tier 2 (Fast): GPT-3.5 Turbo, Claude Haiku for quick decisions  
- Tier 3 (Local): Local SLMs for high-frequency operations

Each tier is optimized for different latency/cost/quality tradeoffs.
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json
import asyncio

from loguru import logger

# Optional imports for LLM providers
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


class ModelTier(Enum):
    """LLM tier classification"""
    REASONING = "reasoning"  # High-quality, slower, expensive
    FAST = "fast"            # Balanced quality/speed/cost
    LOCAL = "local"          # Local models, fastest, cheapest


class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass