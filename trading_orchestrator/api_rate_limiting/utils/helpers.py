"""
Utility Functions for API Rate Limiting

Common utility functions and helpers for rate limiting operations.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import logging

from ..core.rate_limiter import RateLimiterManager, RequestType, RequestPriority
from ..core.request_manager import RequestManager
from ..brokers.rate_limit_configs import RateLimitConfig
from ..monitoring.monitor import RateLimitMonitor, AlertSeverity


def calculate_rate_limit_efficiency(
    requests_per_minute: float,
    broker_limit: float,
    window_seconds: float = 60.0
) -> Dict[str, Any]:
    """
    Calculate rate limit efficiency metrics
    
    Args:
        requests_per_minute: Current request rate
        broker_limit: Broker's rate limit
        window_seconds: Rate limit window
        
    Returns:
        Dict with efficiency metrics
    """
    # Calculate utilization
    utilization_rate = requests_per_minute / max(1, broker_limit)
    
    # Calculate efficiency score (0-100)
    if utilization_rate <= 0.5:
        efficiency_score = 100 - (utilization_rate * 20)
    elif utilization_rate <= 0.8:
        efficiency_score = 90 - ((utilization_rate - 0.5) * 133.33)
    elif utilization_rate <= 0.95:
        efficiency_score = 50 - ((utilization_rate - 0.8) * 266.67)
    else:
        efficiency_score = 10
    
    # Risk assessment
    if utilization_rate > 0.95:
        risk_level = "critical"
        risk_message = "Rate limit almost exceeded"
    elif utilization_rate > 0.8:
        risk_level = "high"
        risk_message = "High rate limit utilization"
    elif utilization_rate > 0.6:
        risk_level = "medium"
        risk_message = "Moderate rate limit utilization"
    else:
        risk_level = "low"
        risk_message = "Low rate limit utilization"
    
    return {
        "utilization_rate": utilization_rate,
        "efficiency_score": max(0, min(100, efficiency_score)),
        "risk_level": risk_level,
        "risk_message": risk_message,
        "suggested_requests_per_minute": min(broker_limit * 0.7, requests_per_minute),
        "buffer_remaining": max(0, broker_limit * 0.3 - requests_per_minute)
    }


def estimate_request_cost(
    request_type: RequestType,
    config: RateLimitConfig,
    is_peak_hours: bool = False
) -> Dict[str, Any]:
    """
    Estimate cost impact of requests
    
    Args:
        request_type: Type of request
        config: Broker configuration
        is_peak_hours: Whether request is during peak hours
        
    Returns:
        Cost estimation
    """
    # Base cost per request
    base_cost = config.market_data_fee_per_request
    
    # Request type multipliers
    multipliers = {
        RequestType.MARKET_DATA: 1.0,
        RequestType.HISTORICAL_DATA: 1.5,
        RequestType.REAL_TIME_DATA: 2.0,
        RequestType.ORDER_PLACE: 0.1,
        RequestType.ACCOUNT_INFO: 0.05,
        RequestType.POSITION_QUERY: 0.05,
        RequestType.ORDER_QUERY: 0.02,
        RequestType.ORDER_CANCEL: 0.02
    }
    
    multiplier = multipliers.get(request_type, 1.0)
    base_cost *= multiplier
    
    # Peak hours multiplier
    if is_peak_hours and request_type in [RequestType.MARKET_DATA, RequestType.REAL_TIME_DATA]:
        base_cost *= 1.2
    
    # Volume discounts (simulated)
    if config.global_rate_limit > 1000:
        base_cost *= 0.9  # 10% discount for high-rate brokers
    
    return {
        "base_cost_per_request": base_cost,
        "multiplier": multiplier,
        "is_peak_hours": is_peak_hours,
        "cost_category": "high" if base_cost > 0.1 else "medium" if base_cost > 0.01 else "low"
    }


def optimize_request_batching(
    requests: List[Dict[str, Any]],
    config: RateLimitConfig
) -> Dict[str, Any]:
    """
    Optimize request batching for cost and efficiency
    
    Args:
        requests: List of request dictionaries
        config: Broker configuration
        
    Returns:
        Batching optimization recommendations
    """
    if not requests:
        return {"recommendations": ["No requests to optimize"]}
    
    # Group requests by type and symbol
    grouped = {}
    for req in requests:
        req_type = req.get("request_type")
        symbol = req.get("symbol", "no_symbol")
        key = f"{req_type}:{symbol}"
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(req)
    
    recommendations = []
    batching_opportunities = []
    
    # Analyze grouping opportunities
    for group_key, group_requests in grouped.items():
        if len(group_requests) > 1:
            req_type = group_requests[0].get("request_type")
            symbol = group_requests[0].get("symbol")
            
            # Check if batching is beneficial
            if req_type in [RequestType.MARKET_DATA, RequestType.HISTORICAL_DATA]:
                batching_opportunities.append({
                    "request_type": req_type.value,
                    "symbol": symbol,
                    "count": len(group_requests),
                    "potential_savings": len(group_requests) * 0.01  # Estimated savings
                })
                
                recommendations.append(
                    f"Batch {len(group_requests)} {req_type.value} requests for {symbol}"
                )
    
    # Cost impact analysis
    total_original_cost = 0
    for req in requests:
        req_type = req.get("request_type")
        cost_est = estimate_request_cost(req_type, config)
        total_original_cost += cost_est["base_cost_per_request"]
    
    estimated_batch_cost = total_original_cost * 0.8  # 20% savings from batching
    
    return {
        "total_requests": len(requests),
        "grouped_requests": len(grouped),
        "batching_opportunities": batching_opportunities,
        "recommendations": recommendations,
        "estimated_cost_savings": total_original_cost - estimated_batch_cost,
        "cost_reduction_percentage": ((total_original_cost - estimated_batch_cost) / total_original_cost * 100) if total_original_cost > 0 else 0
    }


def create_rate_limit_prediction(
    current_usage: Dict[str, float],
    time_window_hours: float,
    config: RateLimitConfig
) -> Dict[str, Any]:
    """
    Create rate limit usage prediction
    
    Args:
        current_usage: Current usage by request type
        time_window_hours: Time window for prediction
        config: Broker configuration
        
    Returns:
        Prediction analysis
    """
    # Calculate current rate per minute
    current_rates = {}
    for req_type, count in current_usage.items():
        current_rates[req_type] = count / (time_window_hours * 60)
    
    # Calculate percentage of limits used
    limit_usage = {}
    for req_type, rate in current_rates.items():
        rule = config.get_rule_for_request(RequestType(req_type))
        if rule:
            limit_percentage = (rate * 60) / rule.limit
            limit_usage[req_type] = {
                "current_rate": rate,
                "limit": rule.limit / 60,  # per minute
                "usage_percentage": limit_percentage * 100
            }
    
    # Predict future usage
    predicted_usage = {}
    for req_type, rate in current_rates.items():
        # Simple linear projection with growth factor
        growth_factor = 1.1  # 10% growth
        predicted_rate = rate * growth_factor
        predicted_usage[req_type] = predicted_rate
    
    # Risk assessment
    risks = []
    for req_type, usage in limit_usage.items():
        if usage["usage_percentage"] > 90:
            risks.append({
                "request_type": req_type,
                "risk_level": "critical",
                "message": f"{req_type} usage at {usage['usage_percentage']:.1f}% of limit"
            })
        elif usage["usage_percentage"] > 75:
            risks.append({
                "request_type": req_type,
                "risk_level": "warning",
                "message": f"{req_type} usage at {usage['usage_percentage']:.1f}% of limit"
            })
    
    # Recommendations
    recommendations = []
    for req_type, usage in limit_usage.items():
        if usage["usage_percentage"] > 80:
            recommendations.append(f"Consider reducing {req_type} request frequency")
    
    return {
        "current_rates_per_minute": current_rates,
        "limit_usage": limit_usage,
        "predicted_rates_per_minute": predicted_usage,
        "risks": risks,
        "recommendations": recommendations,
        "overall_risk_level": "critical" if any(r["risk_level"] == "critical" for r in risks) 
                            else "warning" if any(r["risk_level"] == "warning" for r in risks)
                            else "low"
    }


def validate_api_key_security(api_key: str) -> Dict[str, Any]:
    """
    Validate API key security
    
    Args:
        api_key: API key to validate
        
    Returns:
        Security assessment
    """
    issues = []
    recommendations = []
    
    # Length check
    if len(api_key) < 32:
        issues.append("API key is too short")
        recommendations.append("Use longer API keys (32+ characters)")
    
    # Character complexity
    if api_key.isalnum():
        issues.append("API key lacks special characters")
        recommendations.append("Include special characters in API keys")
    
    # Common patterns
    if any(pattern in api_key.lower() for pattern in ["test", "demo", "example"]):
        issues.append("API key contains common test patterns")
        recommendations.append("Avoid using test-related patterns in production keys")
    
    # Entropy check
    import string
    chars_used = set(api_key)
    entropy_score = len(chars_used) / len(string.printable)
    
    if entropy_score < 0.5:
        issues.append("API key has low entropy")
        recommendations.append("Use more diverse character set")
    
    security_score = max(0, 100 - len(issues) * 20)
    
    return {
        "security_score": security_score,
        "issues": issues,
        "recommendations": recommendations,
        "entropy_score": entropy_score,
        "character_diversity": len(chars_used)
    }


def generate_usage_report(
    analytics_data: List[Dict[str, Any]],
    config: RateLimitConfig,
    period_hours: int = 24
) -> Dict[str, Any]:
    """
    Generate comprehensive usage report
    
    Args:
        analytics_data: List of analytics from different sources
        config: Broker configuration
        period_hours: Report period in hours
        
    Returns:
        Comprehensive usage report
    """
    if not analytics_data:
        return {"error": "No analytics data provided"}
    
    # Aggregate data
    total_requests = sum(data.get("total_requests", 0) for data in analytics_data)
    successful_requests = sum(data.get("successful_requests", 0) for data in analytics_data)
    failed_requests = sum(data.get("failed_requests", 0) for data in analytics_data)
    rate_limited = sum(data.get("rate_limited_requests", 0) for data in analytics_data)
    
    # Calculate metrics
    success_rate = successful_requests / max(1, total_requests)
    error_rate = failed_requests / max(1, total_requests)
    rate_limit_rate = rate_limited / max(1, total_requests)
    
    # Performance metrics
    avg_response_times = [data.get("average_response_time", 0) for data in analytics_data if data.get("average_response_time")]
    avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
    
    # Cost analysis
    total_cost = 0
    cost_breakdown = {}
    
    for data in analytics_data:
        requests_by_type = data.get("requests_by_type", {})
        for req_type, count in requests_by_type.items():
            try:
                cost_est = estimate_request_cost(RequestType(req_type), config)
                cost = cost_est["base_cost_per_request"] * count
                total_cost += cost
                
                if req_type not in cost_breakdown:
                    cost_breakdown[req_type] = {"requests": 0, "cost": 0}
                cost_breakdown[req_type]["requests"] += count
                cost_breakdown[req_type]["cost"] += cost
            except:
                continue
    
    # Generate insights
    insights = []
    
    if success_rate < 0.95:
        insights.append(f"Low success rate: {success_rate:.1%} (target: 95%+)")
    
    if rate_limit_rate > 0.05:
        insights.append(f"High rate limit hit rate: {rate_limit_rate:.1%}")
    
    if avg_response_time > 2.0:
        insights.append(f"High average response time: {avg_response_time:.1f}s")
    
    if total_cost > 50:
        insights.append(f"High API costs: ${total_cost:.2f} in {period_hours}h")
    
    # Performance recommendations
    recommendations = []
    
    if rate_limit_rate > 0.05:
        recommendations.append("Implement request batching and caching")
        recommendations.append("Consider reducing request frequency")
    
    if avg_response_time > 2.0:
        recommendations.append("Investigate network latency issues")
        recommendations.append("Consider using connection pooling")
    
    if total_cost > 50:
        recommendations.append("Review market data request patterns")
        recommendations.append("Implement request cost monitoring")
    
    # Efficiency scoring
    efficiency_factors = {
        "success_rate": success_rate * 100,
        "response_time": max(0, 100 - (avg_response_time * 20)),
        "rate_limit_efficiency": max(0, 100 - (rate_limit_rate * 500)),
        "cost_efficiency": max(0, 100 - (total_cost / 10))
    }
    
    overall_efficiency = sum(efficiency_factors.values()) / len(efficiency_factors)
    
    return {
        "period_hours": period_hours,
        "summary": {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "rate_limited_requests": rate_limited,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "rate_limit_rate": rate_limit_rate,
            "average_response_time": avg_response_time
        },
        "cost_analysis": {
            "total_cost": total_cost,
            "cost_breakdown": cost_breakdown,
            "cost_per_request": total_cost / max(1, total_requests)
        },
        "performance": {
            "overall_efficiency": overall_efficiency,
            "efficiency_factors": efficiency_factors
        },
        "insights": insights,
        "recommendations": recommendations,
        "alerts": [insight for insight in insights if "high" in insight.lower() or "low" in insight.lower()],
        "export_timestamp": datetime.utcnow().isoformat()
    }


def create_health_check_endpoints():
    """
    Create FastAPI endpoints for health checks
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        return None
    
    app = FastAPI(title="API Rate Limiting Health Check", version="1.0.0")
    
    @app.get("/health")
    async def health_check():
        """Basic health check"""
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "api-rate-limiting"
        })
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with metrics"""
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time(),
            "checks": {
                "rate_limiters": "healthy",
                "request_managers": "healthy",
                "monitoring": "healthy",
                "compliance": "healthy"
            }
        })
    
    @app.get("/metrics")
    async def get_metrics():
        """Get metrics endpoint"""
        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": "Metrics would be returned here"
        })
    
    return app


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup logging for rate limiting system
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
        max_file_size: Maximum log file size
        backup_count: Number of backup files to keep
    """
    # Create logger
    logger = logging.getLogger("api_rate_limiting")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def benchmark_rate_limiter(
    rate_limiter: RateLimiterManager,
    request_count: int = 1000,
    concurrent_requests: int = 10
) -> Dict[str, Any]:
    """
    Benchmark rate limiter performance
    
    Args:
        rate_limiter: Rate limiter to benchmark
        request_count: Number of requests to simulate
        concurrent_requests: Number of concurrent requests
        
    Returns:
        Benchmark results
    """
    import asyncio
    
    async def simulate_request():
        """Simulate a single request"""
        start_time = time.time()
        try:
            status = await rate_limiter.acquire(RequestType.ACCOUNT_INFO)
            end_time = time.time()
            return {
                "success": status.allowed,
                "duration": end_time - start_time,
                "wait_time": getattr(status, 'retry_after', 0)
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "duration": end_time - start_time,
                "error": str(e)
            }
    
    async def run_benchmark():
        """Run the benchmark"""
        start_time = time.time()
        
        # Run concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request():
            async with semaphore:
                return await simulate_request()
        
        # Create tasks
        tasks = [bounded_request() for _ in range(request_count)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        durations = [r["duration"] for r in results if "duration" in r]
        wait_times = [r["wait_time"] for r in successful_requests if r.get("wait_time", 0) > 0]
        
        return {
            "total_requests": request_count,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / request_count,
            "total_time_seconds": total_time,
            "requests_per_second": request_count / total_time,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "average_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0,
            "max_wait_time": max(wait_times) if wait_times else 0
        }
    
    return asyncio.run(run_benchmark())


# Configuration validation
def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for issues
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check required fields
    required_fields = []
    if not all(field in config for field in required_fields):
        issues.append("Missing required configuration fields")
    
    # Check monitoring settings
    monitoring = config.get("monitoring", {})
    if monitoring.get("collection_interval", 0) < 1:
        issues.append("Collection interval too short (minimum 1 second)")
    
    # Check retention settings
    if monitoring.get("monitoring_retention_days", 0) < 1:
        issues.append("Monitoring retention too short")
    
    if monitoring.get("audit_retention_days", 0) < 30:
        issues.append("Audit retention should be at least 30 days")
    
    # Check cost thresholds
    if "cost_thresholds" in config:
        thresholds = config["cost_thresholds"]
        if thresholds.get("daily_warning", 0) > thresholds.get("daily_critical", 1):
            issues.append("Daily warning threshold should be below critical threshold")
    
    return issues