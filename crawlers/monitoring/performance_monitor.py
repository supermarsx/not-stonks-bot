"""
Crawler Performance Monitoring System
Tracks performance metrics, analyzes bottlenecks, and provides optimization insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics
import json
from pathlib import Path

from ..base.base_crawler import CrawlerStatus
from ..scheduling.crawler_manager import CrawlerManager


class MetricType(Enum):
    """Types of performance metrics"""
    TIMING = "timing"          # Execution times, delays
    THROUGHPUT = "throughput"  # Data points processed per unit time
    RESOURCE = "resource"      # Memory, CPU usage
    QUALITY = "quality"        # Success rates, data quality
    ERROR = "error"            # Error rates and types
    RELIABILITY = "reliability"  # Uptime, availability


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    crawler_name: str
    metric_type: MetricType
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance at a point in time"""
    timestamp: datetime
    crawler_name: str
    execution_time: float
    data_points_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis result"""
    crawler_name: str
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    description: str
    impact_score: float  # 0-1
    recommendations: List[str]
    detected_at: datetime


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion"""
    crawler_name: str
    category: str  # performance, reliability, cost
    title: str
    description: str
    expected_improvement: str
    effort_level: str  # low, medium, high
    priority: int  # 1-10
    implementation_steps: List[str]


class PerformanceMonitor:
    """Monitors and analyzes crawler performance"""
    
    def __init__(self, manager: CrawlerManager):
        self.manager = manager
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.metrics: List[PerformanceMetric] = []
        self.snapshots: List[PerformanceSnapshot] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            'execution_time_warning': 60,  # seconds
            'execution_time_critical': 300,
            'memory_usage_warning': 100,  # MB
            'memory_usage_critical': 500,
            'cpu_usage_warning': 70,  # percent
            'cpu_usage_critical': 90,
            'error_rate_warning': 0.1,  # 10%
            'error_rate_critical': 0.2,
            'success_rate_warning': 0.8,  # 80%
            'success_rate_critical': 0.6
        }
        
        # Analysis configuration
        self.analysis_config = {
            'analysis_window_hours': 24,
            'min_samples_for_analysis': 10,
            'bottleneck_detection_interval': 300,  # 5 minutes
            'optimization_check_interval': 3600   # 1 hour
        }
        
        # Background analysis tasks
        self._bottleneck_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Performance history
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self._bottleneck_task = asyncio.create_task(self._bottleneck_analysis_loop())
        self._optimization_task = asyncio.create_task(self._optimization_analysis_loop())
        self.logger.info("Started performance monitoring")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        tasks = [self._bottleneck_task, self._optimization_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Stopped performance monitoring")
    
    async def record_metric(self, crawler_name: str, metric_type: MetricType, 
                          metric_name: str, value: float, unit: str = "",
                          tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            crawler_name=crawler_name,
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics (30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    async def record_execution_snapshot(self, crawler_name: str, execution_time: float,
                                      data_points_processed: int, success: bool,
                                      error_message: Optional[str] = None,
                                      metadata: Dict[str, Any] = None):
        """Record execution snapshot"""
        # Simulate resource usage (in real implementation, would get actual metrics)
        import random
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            crawler_name=crawler_name,
            execution_time=execution_time,
            data_points_processed=data_points_processed,
            memory_usage_mb=random.uniform(50, 200),  # Simulated
            cpu_usage_percent=random.uniform(10, 80),  # Simulated
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots (30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        # Record timing metrics
        await self.record_metric(
            crawler_name=crawler_name,
            metric_type=MetricType.TIMING,
            metric_name="execution_time",
            value=execution_time,
            unit="seconds"
        )
        
        # Record throughput metric
        if execution_time > 0:
            throughput = data_points_processed / execution_time
            await self.record_metric(
                crawler_name=crawler_name,
                metric_type=MetricType.THROUGHPUT,
                metric_name="data_points_per_second",
                value=throughput,
                unit="points/second"
            )
        
        # Record quality metric
        success_rate = 1.0 if success else 0.0
        await self.record_metric(
            crawler_name=crawler_name,
            metric_type=MetricType.QUALITY,
            metric_name="success_rate",
            value=success_rate,
            unit="ratio"
        )
        
        # Update performance history
        if crawler_name not in self.performance_history:
            self.performance_history[crawler_name] = []
        
        self.performance_history[crawler_name].append({
            'timestamp': snapshot.timestamp.isoformat(),
            'execution_time': execution_time,
            'success': success,
            'data_points': data_points_processed
        })
        
        # Keep only last 1000 records per crawler
        if len(self.performance_history[crawler_name]) > 1000:
            self.performance_history[crawler_name] = self.performance_history[crawler_name][-1000:]
    
    async def _bottleneck_analysis_loop(self):
        """Continuous bottleneck analysis"""
        while True:
            try:
                await self._analyze_bottlenecks()
                await asyncio.sleep(self.analysis_config['bottleneck_detection_interval'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in bottleneck analysis: {e}")
                await asyncio.sleep(self.analysis_config['bottleneck_detection_interval'])
    
    async def _optimization_analysis_loop(self):
        """Continuous optimization analysis"""
        while True:
            try:
                await self._analyze_optimizations()
                await asyncio.sleep(self.analysis_config['optimization_check_interval'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization analysis: {e}")
                await asyncio.sleep(self.analysis_config['optimization_check_interval'])
    
    async def _analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        try:
            crawler_names = self.manager.crawlers.keys()
            
            for crawler_name in crawler_names:
                crawler_bottlenecks = await self._analyze_crawler_bottlenecks(crawler_name)
                bottlenecks.extend(crawler_bottlenecks)
            
            # Log detected bottlenecks
            if bottlenecks:
                critical_bottlenecks = [b for b in bottlenecks if b.severity == 'critical']
                high_bottlenecks = [b for b in bottlenecks if b.severity == 'high']
                
                if critical_bottlenecks:
                    self.logger.warning(f"Detected {len(critical_bottlenecks)} critical bottlenecks")
                
                if high_bottlenecks:
                    self.logger.info(f"Detected {len(high_bottlenecks)} high-priority bottlenecks")
        
        except Exception as e:
            self.logger.error(f"Error analyzing bottlenecks: {e}")
        
        return bottlenecks
    
    async def _analyze_crawler_bottlenecks(self, crawler_name: str) -> List[BottleneckAnalysis]:
        """Analyze bottlenecks for a specific crawler"""
        bottlenecks = []
        
        try:
            # Get recent metrics for this crawler
            recent_metrics = self._get_recent_metrics(crawler_name, hours=1)
            
            if len(recent_metrics) < self.analysis_config['min_samples_for_analysis']:
                return bottlenecks
            
            # Analyze execution time bottlenecks
            execution_times = [m.value for m in recent_metrics if m.metric_name == "execution_time"]
            if execution_times:
                avg_time = statistics.mean(execution_times)
                max_time = max(execution_times)
                p95_time = self._percentile(execution_times, 95)
                
                if p95_time > self.performance_thresholds['execution_time_critical']:
                    bottlenecks.append(BottleneckAnalysis(
                        crawler_name=crawler_name,
                        bottleneck_type="execution_time",
                        severity="critical",
                        description=f"Critical execution time: 95th percentile is {p95_time:.1f}s (max: {max_time:.1f}s)",
                        impact_score=1.0,
                        recommendations=[
                            "Review data source performance",
                            "Optimize data processing logic",
                            "Consider parallel processing",
                            "Check for resource contention"
                        ],
                        detected_at=datetime.now()
                    ))
                elif p95_time > self.performance_thresholds['execution_time_warning']:
                    bottlenecks.append(BottleneckAnalysis(
                        crawler_name=crawler_name,
                        bottleneck_type="execution_time",
                        severity="high",
                        description=f"High execution time: 95th percentile is {p95_time:.1f}s",
                        impact_score=0.7,
                        recommendations=[
                            "Optimize data queries",
                            "Review processing pipeline",
                            "Consider caching improvements"
                        ],
                        detected_at=datetime.now()
                    ))
            
            # Analyze error rate bottlenecks
            success_rates = [m.value for m in recent_metrics if m.metric_name == "success_rate"]
            if success_rates:
                avg_success_rate = statistics.mean(success_rates)
                
                if avg_success_rate < self.performance_thresholds['success_rate_critical']:
                    bottlenecks.append(BottleneckAnalysis(
                        crawler_name=crawler_name,
                        bottleneck_type="reliability",
                        severity="critical",
                        description=f"Critical reliability: Success rate is {avg_success_rate:.1%}",
                        impact_score=1.0,
                        recommendations=[
                            "Investigate data source issues",
                            "Review error handling logic",
                            "Implement better retry mechanisms",
                            "Check API rate limits"
                        ],
                        detected_at=datetime.now()
                    ))
                elif avg_success_rate < self.performance_thresholds['success_rate_warning']:
                    bottlenecks.append(BottleneckAnalysis(
                        crawler_name=crawler_name,
                        bottleneck_type="reliability",
                        severity="medium",
                        description=f"Low reliability: Success rate is {avg_success_rate:.1%}",
                        impact_score=0.5,
                        recommendations=[
                            "Improve error handling",
                            "Add health checks",
                            "Monitor data source status"
                        ],
                        detected_at=datetime.now()
                    ))
            
            # Analyze throughput bottlenecks
            throughputs = [m.value for m in recent_metrics if m.metric_name == "data_points_per_second"]
            if throughputs:
                avg_throughput = statistics.mean(throughputs)
                
                # Compare with historical average
                historical_avg = self._get_historical_average(crawler_name, "data_points_per_second", hours=24)
                if historical_avg:
                    throughput_decline = (historical_avg - avg_throughput) / historical_avg
                    
                    if throughput_decline > 0.5:  # 50% decline
                        bottlenecks.append(BottleneckAnalysis(
                            crawler_name=crawler_name,
                            bottleneck_type="throughput",
                            severity="high",
                            description=f"Significant throughput decline: {throughput_decline:.1%} drop",
                            impact_score=0.8,
                            recommendations=[
                                "Check data source responsiveness",
                                "Review rate limiting settings",
                                "Optimize data parsing",
                                "Consider batch processing"
                            ],
                            detected_at=datetime.now()
                        ))
            
            # Analyze consecutive failures
            error_count = len([m for m in recent_metrics if m.metric_name == "success_rate" and m.value == 0])
            if error_count >= 5:
                bottlenecks.append(BottleneckAnalysis(
                    crawler_name=crawler_name,
                    bottleneck_type="stability",
                    severity="high",
                    description=f"Multiple consecutive failures: {error_count} in recent period",
                    impact_score=0.9,
                    recommendations=[
                        "Immediate investigation required",
                        "Check data source availability",
                        "Review authentication/permissions",
                        "Implement circuit breaker pattern"
                    ],
                    detected_at=datetime.now()
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing bottlenecks for {crawler_name}: {e}")
        
        return bottlenecks
    
    async def _analyze_optimizations(self) -> List[OptimizationSuggestion]:
        """Analyze optimization opportunities"""
        suggestions = []
        
        try:
            crawler_names = self.manager.crawlers.keys()
            
            for crawler_name in crawler_names:
                crawler_suggestions = await self._analyze_crawler_optimizations(crawler_name)
                suggestions.extend(crawler_suggestions)
            
            # Log optimization suggestions
            high_priority = [s for s in suggestions if s.priority >= 8]
            if high_priority:
                self.logger.info(f"Found {len(high_priority)} high-priority optimization opportunities")
        
        except Exception as e:
            self.logger.error(f"Error analyzing optimizations: {e}")
        
        return suggestions
    
    async def _analyze_crawler_optimizations(self, crawler_name: str) -> List[OptimizationSuggestion]:
        """Analyze optimization opportunities for a crawler"""
        suggestions = []
        
        try:
            # Get recent performance data
            recent_metrics = self._get_recent_metrics(crawler_name, hours=24)
            
            if len(recent_metrics) < self.analysis_config['min_samples_for_analysis']:
                return suggestions
            
            # Performance optimization suggestions
            execution_times = [m.value for m in recent_metrics if m.metric_name == "execution_time"]
            if execution_times:
                avg_time = statistics.mean(execution_times)
                
                if avg_time > 60:  # Average execution time > 1 minute
                    suggestions.append(OptimizationSuggestion(
                        crawler_name=crawler_name,
                        category="performance",
                        title="Optimize Execution Time",
                        description=f"Average execution time is {avg_time:.1f}s, which is above optimal threshold",
                        expected_improvement="30-50% reduction in execution time",
                        effort_level="medium",
                        priority=7,
                        implementation_steps=[
                            "Profile code to identify bottlenecks",
                            "Implement parallel processing where applicable",
                            "Optimize database queries",
                            "Add caching for frequently accessed data"
                        ]
                    ))
            
            # Reliability optimization suggestions
            success_rates = [m.value for m in recent_metrics if m.metric_name == "success_rate"]
            if success_rates:
                avg_success_rate = statistics.mean(success_rates)
                
                if avg_success_rate < 0.95:  # Less than 95% success rate
                    suggestions.append(OptimizationSuggestion(
                        crawler_name=crawler_name,
                        category="reliability",
                        title="Improve Reliability",
                        description=f"Success rate is {avg_success_rate:.1%}, which can be improved",
                        expected_improvement="10-20% improvement in success rate",
                        effort_level="medium",
                        priority=8,
                        implementation_steps=[
                            "Implement comprehensive error handling",
                            "Add retry logic with exponential backoff",
                            "Implement health checks",
                            "Add circuit breaker pattern"
                        ]
                    ))
            
            # Cost optimization suggestions
            memory_usages = [m.value for m in recent_metrics if m.metric_name == "memory_usage_mb"]
            if memory_usages:
                avg_memory = statistics.mean(memory_usages)
                
                if avg_memory > 200:  # Average memory usage > 200MB
                    suggestions.append(OptimizationSuggestion(
                        crawler_name=crawler_name,
                        category="cost",
                        title="Reduce Memory Usage",
                        description=f"Average memory usage is {avg_memory:.0f}MB, which can be optimized",
                        expected_improvement="20-30% reduction in memory usage",
                        effort_level="high",
                        priority=6,
                        implementation_steps=[
                            "Profile memory usage patterns",
                            "Implement streaming for large datasets",
                            "Optimize data structures",
                            "Add garbage collection optimization"
                        ]
                    ))
            
            # Rate limiting optimization
            rate_limit_issues = self._detect_rate_limit_issues(crawler_name, recent_metrics)
            if rate_limit_issues:
                suggestions.append(OptimizationSuggestion(
                    crawler_name=crawler_name,
                    category="performance",
                    title="Optimize Rate Limiting",
                    description="Detected potential rate limiting issues",
                    expected_improvement="Improved data collection efficiency",
                    effort_level="low",
                    priority=5,
                    implementation_steps=[
                        "Review current rate limit settings",
                        "Implement adaptive rate limiting",
                        "Add request batching",
                        "Optimize API call patterns"
                    ]
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing optimizations for {crawler_name}: {e}")
        
        return suggestions
    
    def _detect_rate_limit_issues(self, crawler_name: str, metrics: List[PerformanceMetric]) -> bool:
        """Detect rate limiting issues from metrics"""
        try:
            # Look for patterns that suggest rate limiting
            error_patterns = [m for m in metrics if "error" in m.metric_name.lower()]
            
            # Check for HTTP 429 (Too Many Requests) patterns
            rate_limit_errors = [m for m in error_patterns if "429" in str(m.metadata)]
            
            # Check for increasing execution times (throttling)
            execution_times = [m.value for m in metrics if m.metric_name == "execution_time"]
            if len(execution_times) > 10:
                recent_times = execution_times[-10:]
                older_times = execution_times[-20:-10] if len(execution_times) > 20 else execution_times[:10]
                
                if older_times and statistics.mean(recent_times) > statistics.mean(older_times) * 1.5:
                    return True
            
            return len(rate_limit_errors) > 0
        
        except Exception:
            return False
    
    def _get_recent_metrics(self, crawler_name: str, hours: int = 1) -> List[PerformanceMetric]:
        """Get recent metrics for a crawler"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics 
                if m.crawler_name == crawler_name and m.timestamp >= cutoff_time]
    
    def _get_historical_average(self, crawler_name: str, metric_name: str, hours: int = 24) -> Optional[float]:
        """Get historical average for a metric"""
        try:
            recent_metrics = self._get_recent_metrics(crawler_name, hours=hours)
            metric_values = [m.value for m in recent_metrics if m.metric_name == metric_name]
            
            if metric_values:
                return statistics.mean(metric_values)
            return None
        
        except Exception:
            return None
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for all crawlers"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'crawler_summaries': {},
                'overall_metrics': {}
            }
            
            crawler_names = self.manager.crawlers.keys()
            
            for crawler_name in crawler_names:
                crawler_summary = self._get_crawler_performance_summary(crawler_name, hours)
                summary['crawler_summaries'][crawler_name] = crawler_summary
            
            # Calculate overall metrics
            all_execution_times = []
            all_success_rates = []
            all_throughputs = []
            
            for crawler_name in crawler_names:
                crawler_summary = summary['crawler_summaries'][crawler_name]
                if 'execution_time' in crawler_summary:
                    all_execution_times.extend(crawler_summary['execution_time']['samples'])
                if 'success_rate' in crawler_summary:
                    all_success_rates.extend(crawler_summary['success_rate']['samples'])
                if 'throughput' in crawler_summary:
                    all_throughputs.extend(crawler_summary['throughput']['samples'])
            
            if all_execution_times:
                summary['overall_metrics']['avg_execution_time'] = statistics.mean(all_execution_times)
                summary['overall_metrics']['p95_execution_time'] = self._percentile(all_execution_times, 95)
            
            if all_success_rates:
                summary['overall_metrics']['avg_success_rate'] = statistics.mean(all_success_rates)
            
            if all_throughputs:
                summary['overall_metrics']['avg_throughput'] = statistics.mean(all_throughputs)
                summary['overall_metrics']['total_throughput'] = sum(all_throughputs)
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def _get_crawler_performance_summary(self, crawler_name: str, hours: int) -> Dict[str, Any]:
        """Get performance summary for a specific crawler"""
        try:
            recent_metrics = self._get_recent_metrics(crawler_name, hours)
            
            summary = {
                'total_metrics': len(recent_metrics),
                'metric_breakdown': {}
            }
            
            # Group metrics by type
            metrics_by_name = {}
            for metric in recent_metrics:
                if metric.metric_name not in metrics_by_name:
                    metrics_by_name[metric.metric_name] = []
                metrics_by_name[metric.metric_name].append(metric)
            
            # Analyze each metric type
            for metric_name, metrics in metrics_by_name.items():
                values = [m.value for m in metrics]
                
                if values:
                    summary['metric_breakdown'][metric_name] = {
                        'count': len(values),
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                        'latest': values[-1],
                        'unit': metrics[0].unit
                    }
            
            # Extract key metrics for easy access
            if 'execution_time' in metrics_by_name:
                execution_times = [m.value for m in metrics_by_name['execution_time']]
                summary['execution_time'] = {
                    'average': statistics.mean(execution_times),
                    'median': statistics.median(execution_times),
                    'p95': self._percentile(execution_times, 95),
                    'samples': execution_times
                }
            
            if 'success_rate' in metrics_by_name:
                success_rates = [m.value for m in metrics_by_name['success_rate']]
                summary['success_rate'] = {
                    'average': statistics.mean(success_rates),
                    'samples': success_rates
                }
            
            if 'data_points_per_second' in metrics_by_name:
                throughputs = [m.value for m in metrics_by_name['data_points_per_second']]
                summary['throughput'] = {
                    'average': statistics.mean(throughputs),
                    'samples': throughputs
                }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating crawler summary for {crawler_name}: {e}")
            return {'error': str(e)}
    
    def get_bottleneck_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get bottleneck analysis report"""
        try:
            # Run bottleneck analysis
            bottlenecks = asyncio.run(self._analyze_bottlenecks())
            
            # Group by severity
            bottlenecks_by_severity = {}
            for bottleneck in bottlenecks:
                severity = bottleneck.severity
                if severity not in bottlenecks_by_severity:
                    bottlenecks_by_severity[severity] = []
                bottlenecks_by_severity[severity].append(asdict(bottleneck))
            
            # Group by crawler
            bottlenecks_by_crawler = {}
            for bottleneck in bottlenecks:
                crawler_name = bottleneck.crawler_name
                if crawler_name not in bottlenecks_by_crawler:
                    bottlenecks_by_crawler[crawler_name] = []
                bottlenecks_by_crawler[crawler_name].append(asdict(bottleneck))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'total_bottlenecks': len(bottlenecks),
                'by_severity': bottlenecks_by_severity,
                'by_crawler': bottlenecks_by_crawler,
                'recommendations': self._consolidate_recommendations(bottlenecks)
            }
        
        except Exception as e:
            self.logger.error(f"Error generating bottleneck report: {e}")
            return {'error': str(e)}
    
    def _consolidate_recommendations(self, bottlenecks: List[BottleneckAnalysis]) -> Dict[str, List[str]]:
        """Consolidate recommendations across all bottlenecks"""
        recommendations = {}
        
        for bottleneck in bottlenecks:
            for recommendation in bottleneck.recommendations:
                if recommendation not in recommendations:
                    recommendations[recommendation] = []
                recommendations[recommendation].append({
                    'crawler': bottleneck.crawler_name,
                    'bottleneck_type': bottleneck.bottleneck_type,
                    'severity': bottleneck.severity
                })
        
        return recommendations
    
    def get_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get optimization analysis report"""
        try:
            # Run optimization analysis
            suggestions = asyncio.run(self._analyze_optimizations())
            
            # Group by category and priority
            suggestions_by_category = {}
            suggestions_by_priority = {}
            
            for suggestion in suggestions:
                # By category
                category = suggestion.category
                if category not in suggestions_by_category:
                    suggestions_by_category[category] = []
                suggestions_by_category[category].append(asdict(suggestion))
                
                # By priority
                priority = suggestion.priority
                if priority not in suggestions_by_priority:
                    suggestions_by_priority[priority] = []
                suggestions_by_priority[priority].append(asdict(suggestion))
            
            # Sort by priority
            for priority in suggestions_by_priority:
                suggestions_by_priority[priority].sort(key=lambda x: x['priority'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'total_suggestions': len(suggestions),
                'by_category': suggestions_by_category,
                'by_priority': suggestions_by_priority,
                'high_priority_count': len([s for s in suggestions if s.priority >= 8])
            }
        
        except Exception as e:
            self.logger.error(f"Error generating optimization report: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, export_path: str, hours: int = 24):
        """Export performance data to file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'performance_summary': self.get_performance_summary(hours),
                'bottleneck_report': self.get_bottleneck_report(hours),
                'optimization_report': self.get_optimization_report(hours),
                'raw_metrics': [asdict(metric) for metric in self._get_recent_metrics(
                    "all", hours  # This would need to be implemented
                )]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Performance data exported to {export_path}")
        
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            raise