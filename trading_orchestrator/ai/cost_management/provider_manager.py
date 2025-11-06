"""
Provider Manager - LLM provider health monitoring and cost optimization

Monitors provider performance, costs, and enables intelligent provider switching
based on cost, availability, and performance metrics.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
import sqlite3
from pathlib import Path

from loguru import logger


class ProviderHealth(Enum):
    """Provider health status levels"""
    EXCELLENT = "excellent"     # All metrics optimal
    GOOD = "good"              # Minor issues, fully operational
    DEGRADED = "degraded"      # Performance issues, partially operational
    POOR = "poor"              # Significant issues, limited functionality
    DOWN = "down"              # Unavailable or critical issues


class ProviderMetric(Enum):
    """Provider metrics to monitor"""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    COST_PER_1K_TOKENS = "cost_per_1k_tokens"
    TOKEN_THROUGHPUT = "token_throughput"
    SUCCESS_RATE = "success_rate"


@dataclass
class ProviderStats:
    """Statistics for a provider"""
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    avg_cost_per_1k_tokens: float = 0.0
    last_request: Optional[datetime] = None
    health_score: float = 1.0  # 0.0 to 1.0
    health_status: ProviderHealth = ProviderHealth.GOOD
    
    def update_from_request(self, success: bool, response_time: float, cost: float, tokens: int):
        """Update stats from a new request"""
        self.total_requests += 1
        self.last_request = datetime.now()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update totals
        self.total_cost += cost
        self.total_tokens += tokens
        
        # Update averages
        if self.successful_requests > 0:
            self.avg_response_time = (
                (self.avg_response_time * (self.successful_requests - 1) + response_time) /
                self.successful_requests
            )
        
        # Calculate health metrics
        self._update_health_metrics()
    
    def _update_health_metrics(self):
        """Update health metrics and status"""
        if self.total_requests == 0:
            self.health_status = ProviderHealth.DOWN
            self.health_score = 0.0
            return
        
        # Calculate success rate
        success_rate = self.successful_requests / self.total_requests
        
        # Calculate health score (weighted average)
        response_score = min(1.0, 2.0 / max(self.avg_response_time, 0.1))  # 2s = full score
        error_score = 1.0 - (self.failed_requests / self.total_requests)
        cost_score = 1.0  # Would be calculated based on provider cost ranking
        
        self.health_score = (success_rate * 0.5 + response_score * 0.3 + error_score * 0.2)
        
        # Update health status
        if self.health_score >= 0.9:
            self.health_status = ProviderHealth.EXCELLENT
        elif self.health_score >= 0.8:
            self.health_status = ProviderHealth.GOOD
        elif self.health_score >= 0.6:
            self.health_status = ProviderHealth.DEGRADED
        elif self.health_score >= 0.3:
            self.health_status = ProviderHealth.POOR
        else:
            self.health_status = ProviderHealth.DOWN


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    name: str
    base_url: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens_per_request: int
    rate_limit_per_minute: int
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    health_check_enabled: bool = True
    backup_provider: Optional[str] = None


class ProviderManager:
    """
    Advanced LLM provider management system
    
    Features:
    - Real-time provider health monitoring
    - Automatic failover and load balancing
    - Cost-based provider selection
    - Performance optimization
    - Provider capacity management
    - Intelligent routing based on request characteristics
    """
    
    def __init__(
        self,
        database_path: str = "data/provider_stats.db",
        health_check_interval: int = 300  # 5 minutes
    ):
        """
        Initialize provider manager
        
        Args:
            database_path: Path to SQLite database for provider stats
            health_check_interval: Health check interval in seconds
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Provider configurations
        self.providers: Dict[str, ProviderConfig] = {}
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.provider_health_history: List[Dict[str, Any]] = []
        
        # Health monitoring
        self.health_check_interval = health_check_interval
        self.last_health_check = datetime.now()
        self.health_check_enabled = True
        
        # Cost-based selection
        self.cost_optimization_enabled = True
        self.load_balancing_enabled = True
        self.failover_enabled = True
        
        # Initialize default providers
        self._setup_default_providers()
        
        # Initialize database
        self._init_database()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor_loop())
        
        logger.info("Provider Manager initialized")
    
    def _setup_default_providers(self):
        """Setup default provider configurations"""
        
        # OpenAI providers
        self.add_provider(ProviderConfig(
            name="openai-gpt4",
            base_url="https://api.openai.com",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            max_tokens_per_request=4096,
            rate_limit_per_minute=1000
        ))
        
        self.add_provider(ProviderConfig(
            name="openai-gpt35",
            base_url="https://api.openai.com",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.0015,
            max_tokens_per_request=4096,
            rate_limit_per_minute=5000
        ))
        
        # Anthropic providers
        self.add_provider(ProviderConfig(
            name="anthropic-sonnet",
            base_url="https://api.anthropic.com",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            max_tokens_per_request=8192,
            rate_limit_per_minute=500
        ))
        
        self.add_provider(ProviderConfig(
            name="anthropic-haiku",
            base_url="https://api.anthropic.com",
            cost_per_1k_input_tokens=0.0001,
            cost_per_1k_output_tokens=0.0005,
            max_tokens_per_request=4096,
            rate_limit_per_minute=1000
        ))
    
    def _init_database(self):
        """Initialize SQLite database for provider statistics"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Provider performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                request_duration REAL,
                success BOOLEAN,
                tokens_used INTEGER,
                cost REAL,
                error_message TEXT
            )
        """)
        
        # Provider health table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                health_status TEXT NOT NULL,
                health_score REAL,
                avg_response_time REAL,
                error_rate REAL,
                availability REAL,
                cost_per_1k_tokens REAL
            )
        """)
        
        # Provider cost tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_cost REAL,
                cost_per_1k_tokens REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_provider(self, config: ProviderConfig):
        """Add a provider configuration"""
        self.providers[config.name] = config
        
        if config.name not in self.provider_stats:
            self.provider_stats[config.name] = ProviderStats(provider=config.name)
        
        logger.info(f"Added provider: {config.name}")
    
    def remove_provider(self, provider_name: str):
        """Remove a provider"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            
        if provider_name in self.provider_stats:
            del self.provider_stats[provider_name]
        
        logger.info(f"Removed provider: {provider_name}")
    
    async def select_optimal_provider(
        self,
        task_type: str = "general",
        token_count: int = 1000,
        budget_constraint: Optional[float] = None,
        max_response_time: Optional[float] = None,
        require_backup: bool = False
    ) -> str:
        """
        Select optimal provider based on cost, performance, and availability
        
        Args:
            task_type: Type of task (affects provider selection)
            token_count: Estimated token count for the request
            budget_constraint: Maximum cost allowed
            max_response_time: Maximum acceptable response time
            require_backup: Whether to ensure backup provider is available
            
        Returns:
            Name of selected provider
        """
        available_providers = await self._get_available_providers()
        
        if not available_providers:
            raise RuntimeError("No available providers")
        
        # Filter by budget constraint
        if budget_constraint:
            available_providers = [
                p for p in available_providers
                if self._estimate_request_cost(p, token_count) <= budget_constraint
            ]
        
        # Filter by response time
        if max_response_time:
            available_providers = [
                p for p in available_providers
                if self.provider_stats[p].avg_response_time <= max_response_time
            ]
        
        # Score providers
        provider_scores = {}
        for provider in available_providers:
            score = await self._calculate_provider_score(
                provider, task_type, token_count
            )
            provider_scores[provider] = score
        
        # Select best provider
        if not provider_scores:
            # Fallback to most available provider
            best_provider = max(
                self.provider_stats.keys(),
                key=lambda p: self.provider_stats[p].health_score
            )
        else:
            best_provider = max(provider_scores, key=provider_scores.get)
        
        # Check if backup needed
        if require_backup and len(available_providers) < 2:
            logger.warning("Only one provider available, backup not guaranteed")
        
        logger.info(f"Selected provider: {best_provider} (score: {provider_scores.get(best_provider, 0):.2f})")
        return best_provider
    
    async def _get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        
        for name, stats in self.provider_stats.items():
            config = self.providers[name]
            
            # Check if provider is operational
            if (stats.health_status != ProviderHealth.DOWN and
                stats.health_score >= 0.3):  # Minimum health threshold
                
                # Check rate limits
                current_requests = await self._get_current_request_rate(name)
                if current_requests < config.rate_limit_per_minute:
                    available.append(name)
        
        return available
    
    async def _get_current_request_rate(self, provider: str) -> int:
        """Get current request rate for provider (simplified)"""
        # In a real implementation, this would track actual request rates
        return 0
    
    async def _calculate_provider_score(
        self,
        provider: str,
        task_type: str,
        token_count: int
    ) -> float:
        """Calculate score for provider selection"""
        stats = self.provider_stats[provider]
        config = self.providers[provider]
        
        # Cost score (lower cost = higher score)
        estimated_cost = self._estimate_request_cost(provider, token_count)
        max_cost = max(self._estimate_request_cost(p, token_count) for p in self.provider_stats.keys())
        cost_score = 1.0 - (estimated_cost / max_cost) if max_cost > 0 else 0.5
        
        # Performance score
        performance_score = stats.health_score
        
        # Availability score
        availability_score = 1.0 if stats.health_status != ProviderHealth.DOWN else 0.0
        
        # Task-specific adjustments
        task_score = 1.0
        if task_type == "reasoning":
            # Prefer higher-quality models for reasoning
            if "gpt-4" in provider or "sonnet" in provider:
                task_score *= 1.2
        elif task_type == "fast":
            # Prefer faster models for quick tasks
            if stats.avg_response_time < 2.0:
                task_score *= 1.3
        
        # Calculate weighted score
        final_score = (
            cost_score * 0.4 +
            performance_score * 0.4 +
            availability_score * 0.2
        ) * task_score
        
        return final_score
    
    def _estimate_request_cost(self, provider: str, token_count: int) -> float:
        """Estimate cost for a request"""
        config = self.providers[provider]
        
        # Assume 50% input, 50% output tokens (simplified)
        input_tokens = token_count // 2
        output_tokens = token_count // 2
        
        return (
            (input_tokens / 1000) * config.cost_per_1k_input_tokens +
            (output_tokens / 1000) * config.cost_per_1k_output_tokens
        )
    
    async def record_request(
        self,
        provider: str,
        success: bool,
        response_time: float,
        tokens_used: int,
        cost: float,
        error_message: Optional[str] = None
    ):
        """Record a request and update provider statistics"""
        
        # Update real-time stats
        if provider in self.provider_stats:
            self.provider_stats[provider].update_from_request(
                success, response_time, cost, tokens_used
            )
        
        # Store in database
        await self._store_request_record(
            provider, success, response_time, tokens_used, cost, error_message
        )
        
        # Log if failed
        if not success:
            logger.warning(f"Provider request failed: {provider} - {error_message}")
    
    async def _store_request_record(
        self,
        provider: str,
        success: bool,
        response_time: float,
        tokens_used: int,
        cost: float,
        error_message: Optional[str]
    ):
        """Store request record in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO provider_performance
            (provider, request_duration, success, tokens_used, cost, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (provider, response_time, success, tokens_used, cost, error_message))
        
        conn.commit()
        conn.close()
    
    async def _health_monitor_loop(self):
        """Background task for provider health monitoring"""
        while True:
            try:
                if self.health_check_enabled:
                    await self._perform_health_checks()
                
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        current_time = datetime.now()
        
        for provider_name, config in self.providers.items():
            if not config.health_check_enabled:
                continue
            
            try:
                # Simple health check (ping or lightweight API call)
                health_status = await self._ping_provider(provider_name)
                
                # Update stats
                if provider_name in self.provider_stats:
                    # Calculate availability based on recent requests
                    stats = self.provider_stats[provider_name]
                    if stats.total_requests > 0:
                        availability = stats.successful_requests / stats.total_requests
                    else:
                        availability = 1.0
                    
                    # Update health status
                    if stats.health_score >= 0.9:
                        stats.health_status = ProviderHealth.EXCELLENT
                    elif stats.health_score >= 0.7:
                        stats.health_status = ProviderHealth.GOOD
                    elif stats.health_score >= 0.5:
                        stats.health_status = ProviderHealth.DEGRADED
                    else:
                        stats.health_status = ProviderHealth.POOR
                
                # Store health record
                await self._store_health_record(provider_name, health_status)
                
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
    
    async def _ping_provider(self, provider_name: str) -> Dict[str, Any]:
        """Perform a simple health check on provider"""
        # In a real implementation, this would make actual API calls
        # For now, return mock data based on provider stats
        
        stats = self.provider_stats.get(provider_name)
        if not stats:
            return {"status": ProviderHealth.DOWN, "response_time": float('inf')}
        
        return {
            "status": stats.health_status,
            "response_time": stats.avg_response_time,
            "availability": stats.health_score
        }
    
    async def _store_health_record(self, provider: str, health_data: Dict[str, Any]):
        """Store health record in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        stats = self.provider_stats.get(provider)
        
        cursor.execute("""
            INSERT INTO provider_health
            (provider, health_status, health_score, avg_response_time, error_rate, availability)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            provider,
            health_data["status"].value,
            stats.health_score if stats else 0.0,
            stats.avg_response_time if stats else 0.0,
            (stats.failed_requests / stats.total_requests) if stats and stats.total_requests > 0 else 0.0,
            health_data["availability"]
        ))
        
        conn.commit()
        conn.close()
    
    async def get_provider_health_report(self) -> Dict[str, Any]:
        """Get comprehensive provider health report"""
        report = {
            'providers': {},
            'summary': {
                'total_providers': len(self.providers),
                'available_providers': 0,
                'healthy_providers': 0,
                'total_requests': 0,
                'total_cost': 0.0
            }
        }
        
        for provider_name, stats in self.provider_stats.items():
            if stats.health_status != ProviderHealth.DOWN:
                report['summary']['available_providers'] += 1
            
            if stats.health_status in [ProviderHealth.EXCELLENT, ProviderHealth.GOOD]:
                report['summary']['healthy_providers'] += 1
            
            report['summary']['total_requests'] += stats.total_requests
            report['summary']['total_cost'] += stats.total_cost
            
            report['providers'][provider_name] = {
                'status': stats.health_status.value,
                'health_score': f"{stats.health_score:.2f}",
                'total_requests': stats.total_requests,
                'success_rate': f"{(stats.successful_requests / max(stats.total_requests, 1)) * 100:.1f}%",
                'avg_response_time': f"{stats.avg_response_time:.2f}s",
                'total_cost': f"${stats.total_cost:.2f}",
                'avg_cost_per_1k_tokens': f"${stats.avg_cost_per_1k_tokens:.4f}",
                'last_request': stats.last_request.isoformat() if stats.last_request else None
            }
        
        return report
    
    async def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get cost optimization suggestions"""
        suggestions = []
        
        # Analyze provider usage patterns
        provider_costs = {}
        for provider, stats in self.provider_stats.items():
            if stats.total_requests > 0:
                avg_cost = stats.total_cost / stats.total_requests
                provider_costs[provider] = {
                    'avg_cost': avg_cost,
                    'usage': stats.total_requests,
                    'efficiency': stats.total_requests / max(stats.total_cost, 0.01)
                }
        
        # Find most expensive providers
        if provider_costs:
            most_expensive = max(provider_costs, key=lambda p: provider_costs[p]['avg_cost'])
            suggestions.append({
                'type': 'provider_cost',
                'provider': most_expensive,
                'message': f"Consider optimizing usage of {most_expensive} - highest cost per request",
                'potential_savings': provider_costs[most_expensive]['avg_cost'] * 0.2
            })
        
        # Find underutilized providers
        total_requests = sum(stats.total_requests for stats in self.provider_stats.values())
        if total_requests > 0:
            for provider, stats in self.provider_stats.items():
                usage_percentage = (stats.total_requests / total_requests) * 100
                if usage_percentage < 5:  # Less than 5% usage
                    suggestions.append({
                        'type': 'provider_utilization',
                        'provider': provider,
                        'message': f"{provider} is underutilized ({usage_percentage:.1f}% of requests)",
                        'recommendation': 'Consider using more for load balancing'
                    })
        
        return suggestions
    
    async def switch_provider(
        self,
        current_provider: str,
        reason: str = "manual"
    ) -> Optional[str]:
        """Switch from current provider to backup"""
        current_config = self.providers.get(current_provider)
        if not current_config:
            return None
        
        # Find backup provider
        backup_provider = current_config.backup_provider
        if backup_provider and backup_provider in self.providers:
            backup_stats = self.provider_stats.get(backup_provider)
            
            # Check if backup is healthy
            if backup_stats and backup_stats.health_score > 0.5:
                logger.info(f"Switching from {current_provider} to {backup_provider}: {reason}")
                return backup_provider
        
        # Find alternative provider
        for provider_name in self.providers.keys():
            if (provider_name != current_provider and
                self.provider_stats[provider_name].health_score > 0.7):
                logger.info(f"Switching from {current_provider} to {provider_name}: {reason}")
                return provider_name
        
        logger.warning(f"No suitable backup provider found for {current_provider}")
        return None
    
    def get_provider_comparison(self) -> Dict[str, Any]:
        """Get detailed provider comparison"""
        comparison = {}
        
        for provider_name in self.providers.keys():
            stats = self.provider_stats.get(provider_name)
            config = self.providers[provider_name]
            
            if stats and stats.total_requests > 0:
                comparison[provider_name] = {
                    'cost_per_request': f"${stats.total_cost / stats.total_requests:.4f}",
                    'cost_per_1k_tokens': f"${stats.avg_cost_per_1k_tokens:.4f}",
                    'success_rate': f"{(stats.successful_requests / stats.total_requests) * 100:.1f}%",
                    'avg_response_time': f"{stats.avg_response_time:.2f}s",
                    'health_score': f"{stats.health_score:.2f}",
                    'total_requests': stats.total_requests,
                    'total_cost': f"${stats.total_cost:.2f}",
                    'provider_type': 'openai' if 'openai' in provider_name else 'anthropic'
                }
        
        return comparison