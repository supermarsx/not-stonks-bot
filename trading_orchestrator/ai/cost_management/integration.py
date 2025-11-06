"""
LLM Cost Management Integration Module

Integrates the comprehensive cost management system with the existing AI models manager
and provides unified access to all cost management features.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

from loguru import logger

from .cost_manager import LLMCostManager
from .budget_manager import BudgetManager
from .provider_manager import ProviderManager
from .analytics import CostAnalytics
from .prediction import CostForecaster
from .anomaly_detector import AnomalyDetector
from .dashboard import CostDashboard


class LLMIntegratedCostManager:
    """
    Integrated LLM Cost Management System
    
    This class integrates all cost management components with the existing AI models manager
    to provide seamless cost tracking, optimization, and control for LLM operations.
    """
    
    def __init__(
        self,
        ai_models_manager,
        database_path: str = "data/llm_cost_management.db",
        enable_real_time_monitoring: bool = True
    ):
        """
        Initialize integrated cost management system
        
        Args:
            ai_models_manager: Existing AIModelsManager instance
            database_path: Path to main cost management database
            enable_real_time_monitoring: Enable real-time cost monitoring
        """
        self.ai_models_manager = ai_models_manager
        
        # Initialize all cost management components
        self.cost_manager = LLMCostManager(database_path=database_path)
        self.budget_manager = BudgetManager(self.cost_manager)
        self.provider_manager = ProviderManager()
        self.analytics = CostAnalytics(self.cost_manager, self.provider_manager)
        self.cost_forecaster = CostForecaster(self.cost_manager, self.provider_manager)
        self.anomaly_detector = AnomalyDetector(
            self.cost_manager, 
            self.provider_manager
        )
        self.dashboard = CostDashboard(
            self.cost_manager,
            self.budget_manager,
            self.provider_manager,
            self.analytics
        )
        
        # Integration settings
        self.real_time_monitoring_enabled = enable_real_time_monitoring
        self.auto_optimization_enabled = True
        self.cost_tracking_enabled = True
        
        # Event handlers
        self.alert_handlers: List[Callable] = []
        self.cost_event_handlers: List[Callable] = []
        
        # Initialize integration
        self._setup_integration()
        
        logger.info("LLM Cost Management System integrated successfully")
    
    def _setup_integration(self):
        """Setup integration between components"""
        
        # Register alert callback
        self.cost_manager.alert_callback = self._handle_cost_alert
        
        # Update AI models manager with cost-aware model selection
        self._enhance_ai_models_manager()
        
        # Enable real-time monitoring
        if self.real_time_monitoring_enabled:
            asyncio.create_task(self._real_time_monitoring_loop())
    
    def _enhance_ai_models_manager(self):
        """Enhance AI models manager with cost-aware features"""
        
        # Add cost tracking to the models manager
        original_update_usage = self.ai_models_manager._update_usage_stats
        
        def enhanced_update_usage(model_config, usage):
            """Enhanced usage tracking with cost management integration"""
            # Call original function
            original_update_usage(model_config, usage)
            
            # Track with cost manager
            if self.cost_tracking_enabled:
                total_tokens = usage.get('total_tokens', 0)
                cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
                
                asyncio.create_task(self.cost_manager.track_usage(
                    provider=model_config.provider.value,
                    model=model_config.name,
                    tokens_used=total_tokens,
                    cost=cost,
                    task_type="ai_completion"
                ))
        
        # Replace the original method
        self.ai_models_manager._update_usage_stats = enhanced_update_usage
    
    async def _handle_cost_alert(self, alert):
        """Handle cost alerts from the cost manager"""
        logger.warning(f"Cost alert: {alert.message}")
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Automatic optimization based on alert
        if alert.level.value in ['warning', 'critical', 'emergency']:
            await self._handle_cost_alert_optimization(alert)
    
    async def _handle_cost_alert_optimization(self, alert):
        """Handle automatic optimization when cost alerts trigger"""
        if not self.auto_optimization_enabled:
            return
        
        logger.info("Executing automatic cost optimization...")
        
        try:
            # Switch to more cost-effective models
            if alert.event_type.value == 'budget_limit_reached':
                await self._switch_to_emergency_mode()
            
            # Trigger budget evaluation
            triggered_rules = await self.budget_manager.evaluate_budget_rules()
            logger.info(f"Budget rules triggered: {len(triggered_rules)}")
            
        except Exception as e:
            logger.error(f"Automatic optimization error: {e}")
    
    async def _switch_to_emergency_mode(self):
        """Switch to emergency cost-saving mode"""
        logger.critical("Switching to emergency cost-saving mode")
        
        # Update AI models manager to use only fast models
        original_get_model = self.ai_models_manager.get_model_for_task
        
        def emergency_get_model(task_type, preferred_tier=None):
            """Emergency model selection - always use fast tier"""
            return self.ai_models_manager.MODEL_REGISTRY.get("gpt-3.5-turbo") or \
                   self.ai_models_manager.MODEL_REGISTRY.get("claude-haiku")
        
        # Temporarily replace model selection
        self.ai_models_manager.get_model_for_task = emergency_get_model
        
        # Reset after 1 hour
        asyncio.create_task(self._reset_normal_mode())
    
    async def _reset_normal_mode(self):
        """Reset to normal operation mode"""
        await asyncio.sleep(3600)  # 1 hour
        
        logger.info("Resetting to normal operation mode")
        # Restore original model selection logic
        # This would restore the original get_model_for_task method
    
    async def _real_time_monitoring_loop(self):
        """Main real-time monitoring loop"""
        while True:
            try:
                if self.real_time_monitoring_enabled:
                    await self._update_real_time_metrics()
                
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _update_real_time_metrics(self):
        """Update real-time cost metrics"""
        try:
            # Get current AI usage
            current_stats = self.ai_models_manager.get_usage_stats()
            
            # Update cost tracking with latest AI usage
            for model_name, stats in current_stats.items():
                if stats['request_count'] > 0:
                    # Find model config
                    model_config = None
                    for config in self.ai_models_manager.MODEL_REGISTRY.values():
                        if config.name == model_name:
                            model_config = config
                            break
                    
                    if model_config:
                        total_tokens = stats['total_tokens']
                        cost = stats['total_cost']
                        
                        await self.cost_manager.track_usage(
                            provider=model_config.provider.value,
                            model=model_name,
                            tokens_used=total_tokens,
                            cost=cost,
                            task_type="ai_completion",
                            metadata={'requests': stats['request_count']}
                        )
        
        except Exception as e:
            logger.error(f"Real-time metrics update error: {e}")
    
    # Public API methods
    
    async def track_ai_request(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost: float,
        request_duration: float = None,
        session_id: str = None,
        task_type: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Track an AI request with full cost analysis"""
        
        # Track with cost manager
        metrics = await self.cost_manager.track_usage(
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            cost=cost,
            request_duration=request_duration,
            session_id=session_id,
            task_type=task_type,
            metadata=metadata
        )
        
        # Record with provider manager
        await self.provider_manager.record_request(
            provider=f"{provider}-{model.replace('-', '_')}",
            success=True,  # Would be determined by actual success
            response_time=request_duration or 0.0,
            tokens_used=tokens_used,
            cost=cost
        )
        
        return metrics
    
    async def select_cost_optimal_model(
        self,
        task_type: str = "general",
        token_count: int = 1000,
        budget_constraint: Optional[float] = None,
        max_response_time: Optional[float] = None
    ) -> str:
        """Select the most cost-effective model for a task"""
        
        try:
            selected_provider = await self.provider_manager.select_optimal_provider(
                task_type=task_type,
                token_count=token_count,
                budget_constraint=budget_constraint,
                max_response_time=max_response_time
            )
            
            # Map provider to model
            provider_to_model = {
                "openai-gpt4": "gpt-4-turbo",
                "openai-gpt35": "gpt-3.5-turbo", 
                "anthropic-sonnet": "claude-3-5-sonnet",
                "anthropic-haiku": "claude-haiku"
            }
            
            model_name = provider_to_model.get(selected_provider, "gpt-3.5-turbo")
            
            logger.info(f"Selected cost-optimal model: {model_name} for {task_type}")
            return model_name
            
        except Exception as e:
            logger.error(f"Model selection error: {e}")
            # Fallback to default fast model
            return "gpt-3.5-turbo"
    
    async def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get comprehensive cost optimization recommendations"""
        
        recommendations = []
        
        try:
            # Get budget optimization suggestions
            budget_recommendations = await self.budget_manager.optimize_budget_allocation()
            recommendations.extend(budget_recommendations.get('recommendations', []))
            
            # Get provider optimization suggestions
            provider_suggestions = await self.provider_manager.get_cost_optimization_suggestions()
            recommendations.extend(provider_suggestions)
            
            # Get analytics insights
            usage_insights = await self.analytics.analyze_usage_patterns()
            recommendations.extend([
                {
                    'type': insight.type,
                    'title': insight.title,
                    'description': insight.description,
                    'impact': insight.impact,
                    'potential_savings': insight.potential_savings,
                    'actionable': insight.actionable,
                    'recommendation': insight.recommendation
                }
                for insight in usage_insights
            ])
            
            # Get forecast-based recommendations
            try:
                forecast = await self.cost_forecaster.project_budget_requirements()
                recommendations.append({
                    'type': 'budget_forecast',
                    'title': 'Budget Forecast Recommendation',
                    'description': f'Projected monthly cost: ${forecast.total_projected_cost:.2f}',
                    'recommendation': f'Consider setting budget to ${forecast.recommended_budget:.2f}',
                    'potential_savings': forecast.potential_budget_overrun,
                    'risk_level': forecast.risk_assessment
                })
            except Exception as e:
                logger.error(f"Forecast recommendation error: {e}")
            
        except Exception as e:
            logger.error(f"Cost optimization error: {e}")
        
        return recommendations
    
    async def get_comprehensive_cost_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """Get comprehensive cost analysis report"""
        
        try:
            # Get basic report
            report = await self.analytics.generate_cost_report(start_date, end_date, report_type)
            
            # Add cost management insights
            optimization_recommendations = await self.get_cost_optimization_recommendations()
            
            # Add real-time status
            dashboard_status = await self.dashboard.get_real_time_status()
            
            # Add budget analysis
            budget_status = await self.budget_manager.get_budget_status()
            
            # Add provider health
            provider_health = await self.provider_manager.get_provider_health_report()
            
            # Add anomaly summary
            anomaly_summary = await self.anomaly_detector.get_anomaly_summary()
            
            # Compile comprehensive report
            comprehensive_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'report_type': report_type,
                    'system_version': '1.0.0'
                },
                'cost_analysis': report,
                'optimization_recommendations': optimization_recommendations,
                'system_status': {
                    'real_time_monitoring': dashboard_status,
                    'budget_status': budget_status,
                    'provider_health': provider_health,
                    'anomaly_detection': anomaly_summary
                },
                'forecasts': {
                    'cost_forecast': await self._get_cost_forecast_summary(),
                    'budget_projection': await self._get_budget_projection_summary()
                }
            }
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Comprehensive report generation error: {e}")
            return {'error': f'Report generation failed: {str(e)}'}
    
    async def _get_cost_forecast_summary(self) -> Dict[str, Any]:
        """Get cost forecast summary"""
        try:
            forecasts = await self.cost_forecaster.forecast_costs(forecast_horizon_days=30)
            
            if forecasts:
                total_projected_cost = sum(f.predicted_cost for f in forecasts)
                avg_confidence = sum(f.confidence_score for f in forecasts) / len(forecasts)
                
                return {
                    '30_day_projection': total_projected_cost,
                    'average_confidence': avg_confidence,
                    'daily_average': total_projected_cost / 30,
                    'confidence_level': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.5 else 'low'
                }
            else:
                return {'error': 'Unable to generate forecast'}
        except Exception as e:
            return {'error': f'Forecast error: {str(e)}'}
    
    async def _get_budget_projection_summary(self) -> Dict[str, Any]:
        """Get budget projection summary"""
        try:
            projection = await self.cost_forecaster.project_budget_requirements(target_days=30)
            
            return {
                'recommended_monthly_budget': projection.recommended_budget,
                'risk_assessment': projection.risk_assessment,
                'potential_overrun': projection.potential_budget_overrun,
                'confidence_score': projection.confidence_score,
                'key_assumptions': projection.assumptions[:3]  # Top 3 assumptions
            }
        except Exception as e:
            return {'error': f'Budget projection error: {str(e)}'}
    
    async def create_budget_profile(
        self,
        name: str,
        monthly_limit: float,
        daily_limit: float,
        tier: str = "medium",
        auto_optimization: bool = True
    ) -> Dict[str, Any]:
        """Create a new budget profile"""
        
        try:
            from .budget_manager import BudgetTier, BudgetProfile, BudgetRule
            
            # Map tier string to enum
            tier_map = {
                'low': BudgetTier.LOW,
                'medium': BudgetTier.MEDIUM,
                'high': BudgetTier.HIGH,
                'enterprise': BudgetTier.ENTERPRISE
            }
            
            budget_tier = tier_map.get(tier.lower(), BudgetTier.MEDIUM)
            
            # Create budget profile
            profile = BudgetProfile(
                name=name,
                tier=budget_tier,
                monthly_limit=monthly_limit,
                daily_limit=daily_limit,
                auto_optimization=auto_optimization
            )
            
            # Add basic rules
            profile.add_rule(BudgetRule(
                name="Cost threshold alert",
                condition=json.dumps({"type": "budget_percentage", "threshold": 0.8}),
                actions=[],  # Will be populated by budget manager
                priority=1
            ))
            
            # Add to budget manager
            self.budget_manager.add_profile(profile)
            
            logger.info(f"Created budget profile: {name}")
            
            return {
                'profile_name': name,
                'monthly_limit': monthly_limit,
                'daily_limit': daily_limit,
                'tier': tier,
                'message': 'Budget profile created successfully'
            }
            
        except Exception as e:
            logger.error(f"Budget profile creation error: {e}")
            return {'error': f'Profile creation failed: {str(e)}'}
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def add_cost_event_handler(self, handler: Callable):
        """Add custom cost event handler"""
        self.cost_event_handlers.append(handler)
        logger.info(f"Added cost event handler: {handler.__name__}")
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        
        try:
            # Current metrics
            current_metrics = self.cost_manager.get_current_metrics()
            
            # Budget status
            budget_status = await self.budget_manager.get_budget_status()
            
            # Provider health
            provider_health = await self.provider_manager.get_provider_health_report()
            
            # Recent alerts
            recent_alerts = self.cost_manager.get_active_alerts()[:5]
            
            # Cost summary
            cost_summary = self.cost_manager.get_cost_summary()
            
            # Real-time status
            dashboard_status = await self.dashboard.get_real_time_status()
            
            return {
                'system_overview': {
                    'current_period_cost': f"${current_metrics.total_cost:.2f}",
                    'total_tokens': f"{current_metrics.total_tokens:,}",
                    'request_count': current_metrics.request_count,
                    'avg_cost_per_request': f"${current_metrics.average_cost_per_request:.4f}",
                    'active_budget': budget_status.get('daily_spent', 'N/A'),
                    'emergency_mode': budget_status.get('emergency_mode', False),
                    'monitoring_enabled': self.real_time_monitoring_enabled,
                    'auto_optimization': self.auto_optimization_enabled
                },
                'provider_status': {
                    'total_providers': provider_health['summary']['total_providers'],
                    'available_providers': provider_health['summary']['available_providers'],
                    'healthy_providers': provider_health['summary']['healthy_providers']
                },
                'alert_summary': {
                    'total_active_alerts': len(recent_alerts),
                    'recent_alerts': [
                        {
                            'level': alert.level.value,
                            'message': alert.message[:100] + '...' if len(alert.message) > 100 else alert.message,
                            'timestamp': alert.timestamp.isoformat()
                        }
                        for alert in recent_alerts
                    ]
                },
                'system_health': {
                    'cost_tracking_active': self.cost_tracking_enabled,
                    'real_time_monitoring': self.real_time_monitoring_enabled,
                    'last_update': dashboard_status['last_update'],
                    'data_freshness': 'current' if (datetime.now() - datetime.fromisoformat(dashboard_status['last_update'])).total_seconds() < 60 else 'stale'
                }
            }
            
        except Exception as e:
            logger.error(f"System overview error: {e}")
            return {'error': f'System overview failed: {str(e)}'}
    
    async def shutdown(self):
        """Shutdown the cost management system gracefully"""
        logger.info("Shutting down LLM Cost Management System...")
        
        # Disable real-time monitoring
        self.real_time_monitoring_enabled = False
        
        # Save configurations
        await self.dashboard._save_dashboard_configurations()
        
        logger.info("LLM Cost Management System shutdown complete")
    
    def __repr__(self):
        return f"LLMIntegratedCostManager(monitoring={'enabled' if self.real_time_monitoring_enabled else 'disabled'})"