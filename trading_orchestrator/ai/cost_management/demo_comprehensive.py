#!/usr/bin/env python3
"""
LLM Cost Management System - Comprehensive Demo

This demo showcases the complete LLM Cost Management System with all features:
- Real-time cost tracking
- Budget management and alerts
- Provider optimization
- Cost analytics and forecasting
- Anomaly detection
- Interactive dashboard
- Integration with AI models

Run this script to see the system in action.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import the cost management system
from cost_management import (
    LLMCostManager,
    LLMIntegratedCostManager,
    BudgetTier,
    AlertLevel,
    CostEventType
)

# Import existing AI models manager (relative import for demo)
try:
    from ai.models.ai_models_manager import AIModelsManager
except ImportError:
    # Mock class for demo when ai module not available
    class AIModelsManager:
        pass

from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class CostManagementDemo:
    """
    Comprehensive demonstration of the LLM Cost Management System
    """
    
    def __init__(self):
        """Initialize the demo with all components"""
        
        # Create data directories
        import os
        os.makedirs("data", exist_ok=True)
        
        # Initialize AI models manager (mock)
        self.ai_models_manager = self._create_mock_ai_models_manager()
        
        # Initialize integrated cost management system
        self.cost_manager_system = LLMIntegratedCostManager(
            ai_models_manager=self.ai_models_manager,
            database_path="data/demo_cost_management.db",
            enable_real_time_monitoring=True
        )
        
        logger.info("LLM Cost Management System Demo initialized")
    
    def _create_mock_ai_models_manager(self) -> AIModelsManager:
        """Create a mock AI models manager for demo purposes"""
        
        class MockAIModelsManager:
            def __init__(self):
                self.MODEL_REGISTRY = {
                    "gpt-4-turbo": type('Config', (), {
                        'name': 'gpt-4-turbo',
                        'provider': type('Provider', (), {'value': 'openai'})(),
                        'cost_per_1k_tokens': 0.01
                    })(),
                    "gpt-3.5-turbo": type('Config', (), {
                        'name': 'gpt-3.5-turbo',
                        'provider': type('Provider', (), {'value': 'openai'})(),
                        'cost_per_1k_tokens': 0.0015
                    })(),
                    "claude-3-5-sonnet": type('Config', (), {
                        'name': 'claude-3-5-sonnet',
                        'provider': type('Provider', (), {'value': 'anthropic'})(),
                        'cost_per_1k_tokens': 0.015
                    })(),
                    "claude-haiku": type('Config', (), {
                        'name': 'claude-haiku',
                        'provider': type('Provider', (), {'value': 'anthropic'})(),
                        'cost_per_1k_tokens': 0.00025
                    })()
                }
                
                self.usage_stats = {}
            
            def get_model_for_task(self, task_type, preferred_tier=None):
                """Mock model selection"""
                models = list(self.MODEL_REGISTRY.keys())
                return self.MODEL_REGISTRY[random.choice(models)]
            
            def get_usage_stats(self):
                """Get current usage statistics"""
                return self.usage_stats.copy()
            
            def _update_usage_stats(self, model_config, usage):
                """Mock usage tracking"""
                model_name = model_config.name
                if model_name not in self.usage_stats:
                    self.usage_stats[model_name] = {
                        'total_tokens': 0,
                        'total_cost': 0.0,
                        'request_count': 0
                    }
                
                self.usage_stats[model_name]['total_tokens'] += usage.get('total_tokens', 0)
                self.usage_stats[model_name]['total_cost'] += usage.get('total_tokens', 0) / 1000 * model_config.cost_per_1k_tokens
                self.usage_stats[model_name]['request_count'] += 1
        
        return MockAIModelsManager()
    
    async def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        
        logger.info("=" * 80)
        logger.info("🚀 LLM COST MANAGEMENT SYSTEM - COMPREHENSIVE DEMO")
        logger.info("=" * 80)
        
        try:
            # Step 1: System Overview
            await self._demo_system_overview()
            
            # Step 2: Basic Cost Tracking
            await self._demo_basic_cost_tracking()
            
            # Step 3: Budget Management
            await self._demo_budget_management()
            
            # Step 4: Provider Optimization
            await self._demo_provider_optimization()
            
            # Step 5: Analytics and Forecasting
            await self._demo_analytics_forecasting()
            
            # Step 6: Anomaly Detection
            await self._demo_anomaly_detection()
            
            # Step 7: Dashboard and Monitoring
            await self._demo_dashboard_monitoring()
            
            # Step 8: Cost Optimization
            await self._demo_cost_optimization()
            
            # Step 9: Integration Features
            await self._demo_integration_features()
            
            # Step 10: Comprehensive Report
            await self._demo_comprehensive_report()
            
            logger.info("=" * 80)
            logger.info("✅ DEMO COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self._cleanup_demo()
    
    async def _demo_system_overview(self):
        """Demonstrate system overview"""
        logger.info("\n📊 SYSTEM OVERVIEW")
        logger.info("-" * 40)
        
        overview = await self.cost_manager_system.get_system_overview()
        
        logger.info(f"Current Cost: {overview['system_overview']['current_period_cost']}")
        logger.info(f"Total Tokens: {overview['system_overview']['total_tokens']}")
        logger.info(f"Request Count: {overview['system_overview']['request_count']}")
        logger.info(f"Emergency Mode: {overview['system_overview']['emergency_mode']}")
        logger.info(f"Monitoring Active: {overview['system_overview']['monitoring_enabled']}")
    
    async def _demo_basic_cost_tracking(self):
        """Demonstrate basic cost tracking"""
        logger.info("\n💰 BASIC COST TRACKING")
        logger.info("-" * 40)
        
        # Simulate various AI requests
        requests = [
            {"provider": "openai", "model": "gpt-4-turbo", "tokens": 1500, "cost": 0.015},
            {"provider": "anthropic", "model": "claude-3-5-sonnet", "tokens": 2000, "cost": 0.03},
            {"provider": "openai", "model": "gpt-3.5-turbo", "tokens": 1000, "cost": 0.0015},
            {"provider": "anthropic", "model": "claude-haiku", "tokens": 800, "cost": 0.0002},
            {"provider": "openai", "model": "gpt-4-turbo", "tokens": 2500, "cost": 0.025}
        ]
        
        for i, request in enumerate(requests, 1):
            metrics = await self.cost_manager_system.track_ai_request(
                provider=request["provider"],
                model=request["model"],
                tokens_used=request["tokens"],
                cost=request["cost"],
                task_type="demo_request",
                session_id=f"demo_session_{i}"
            )
            
            logger.info(f"Request {i}: {request['provider']}/{request['model']} - "
                       f"{request['tokens']} tokens, ${request['cost']:.4f}")
        
        # Show current metrics
        current_metrics = self.cost_manager_system.cost_manager.get_current_metrics()
        logger.info(f"Total Tracked Cost: ${current_metrics.total_cost:.4f}")
        logger.info(f"Total Tracked Tokens: {current_metrics.total_tokens:,}")
    
    async def _demo_budget_management(self):
        """Demonstrate budget management features"""
        logger.info("\n💳 BUDGET MANAGEMENT")
        logger.info("-" * 40)
        
        # Create a custom budget profile
        result = await self.cost_manager_system.create_budget_profile(
            name="Demo Budget",
            monthly_limit=500.0,
            daily_limit=25.0,
            tier="medium",
            auto_optimization=True
        )
        
        logger.info(f"Created Budget Profile: {result}")
        
        # Get budget status
        budget_status = await self.cost_manager_system.budget_manager.get_budget_status()
        logger.info(f"Budget Status: {json.dumps(budget_status, indent=2)}")
        
        # Simulate budget limit reached
        await self._simulate_budget_limit_reached()
    
    async def _demo_provider_optimization(self):
        """Demonstrate provider optimization features"""
        logger.info("\n🔄 PROVIDER OPTIMIZATION")
        logger.info("-" * 40)
        
        # Get provider health report
        provider_health = await self.cost_manager_system.provider_manager.get_provider_health_report()
        logger.info(f"Provider Health Summary:")
        logger.info(f"  Total Providers: {provider_health['summary']['total_providers']}")
        logger.info(f"  Available: {provider_health['summary']['available_providers']}")
        logger.info(f"  Healthy: {provider_health['summary']['healthy_providers']}")
        
        # Demonstrate cost-optimal model selection
        selected_model = await self.cost_manager_system.select_cost_optimal_model(
            task_type="reasoning",
            token_count=1500,
            budget_constraint=0.1
        )
        
        logger.info(f"Selected Cost-Optimal Model: {selected_model}")
        
        # Get cost optimization suggestions
        suggestions = await self.cost_manager_system.provider_manager.get_cost_optimization_suggestions()
        if suggestions:
            logger.info("Cost Optimization Suggestions:")
            for suggestion in suggestions[:3]:  # Show first 3
                logger.info(f"  - {suggestion.get('type', 'Unknown')}: {suggestion.get('message', 'No message')}")
    
    async def _demo_analytics_forecasting(self):
        """Demonstrate analytics and forecasting"""
        logger.info("\n📈 ANALYTICS & FORECASTING")
        logger.info("-" * 40)
        
        # Get cost trends
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        trends = await self.cost_manager_system.analytics.get_cost_trends(start_date, end_date, 'daily')
        logger.info(f"Cost Trends (Last 7 Days): {len(trends)} data points")
        
        # Detect anomalies
        anomalies = await self.cost_manager_system.analytics.detect_cost_anomalies(start_date, end_date)
        logger.info(f"Detected Anomalies: {len(anomalies)}")
        
        for anomaly in anomalies[:3]:  # Show first 3
            logger.info(f"  - {anomaly.title}: {anomaly.description}")
        
        # Generate cost forecast
        try:
            forecasts = await self.cost_manager_system.cost_forecaster.forecast_costs(
                forecast_horizon_days=7
            )
            
            if forecasts:
                total_forecast_cost = sum(f.predicted_cost for f in forecasts)
                logger.info(f"7-Day Cost Forecast: ${total_forecast_cost:.4f}")
                logger.info(f"Average Confidence: {sum(f.confidence_score for f in forecasts) / len(forecasts):.2f}")
        except Exception as e:
            logger.info(f"Forecast demo (simplified): {str(e)}")
    
    async def _demo_anomaly_detection(self):
        """Demonstrate anomaly detection"""
        logger.info("\n🚨 ANOMALY DETECTION")
        logger.info("-" * 40)
        
        # Trigger some anomalies by simulating unusual usage
        await self._simulate_anomaly_scenarios()
        
        # Get anomaly summary
        anomaly_summary = await self.cost_manager_system.anomaly_detector.get_anomaly_summary(hours=24)
        logger.info(f"Anomaly Summary (Last 24h):")
        logger.info(f"  Total Anomalies: {anomaly_summary['total_anomalies']}")
        logger.info(f"  Active Anomalies: {anomaly_summary['active_anomalies']}")
        logger.info(f"  Resolution Rate: {anomaly_summary['resolution_rate_percent']:.1f}%")
        
        if anomaly_summary['most_common_type']:
            logger.info(f"  Most Common Type: {anomaly_summary['most_common_type']}")
    
    async def _demo_dashboard_monitoring(self):
        """Demonstrate dashboard and monitoring"""
        logger.info("\n📱 DASHBOARD & MONITORING")
        logger.info("-" * 40)
        
        # Get dashboard data
        dashboard_data = await self.cost_manager_system.dashboard.get_dashboard_data("Main Overview")
        
        logger.info(f"Dashboard: {dashboard_data['dashboard_info']['name']}")
        logger.info(f"Widgets: {len(dashboard_data['widgets'])}")
        logger.info(f"Last Updated: {dashboard_data['dashboard_info']['last_updated']}")
        
        # Check alert thresholds
        alerts = await self.cost_manager_system.dashboard.check_alert_thresholds()
        logger.info(f"Triggered Alerts: {len(alerts)}")
        
        for alert in alerts[:3]:  # Show first 3
            logger.info(f"  - {alert['severity'].upper()}: {alert['metric']} = {alert['current_value']}")
        
        # Get real-time status
        status = await self.cost_manager_system.dashboard.get_real_time_status()
        logger.info(f"Real-time Status: {json.dumps(status, indent=2)}")
    
    async def _demo_cost_optimization(self):
        """Demonstrate cost optimization features"""
        logger.info("\n⚡ COST OPTIMIZATION")
        logger.info("-" * 40)
        
        # Get optimization recommendations
        recommendations = await self.cost_manager_system.get_cost_optimization_recommendations()
        
        logger.info(f"Optimization Recommendations ({len(recommendations)}):")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            impact = rec.get('impact', 'medium')
            potential_savings = rec.get('potential_savings', 0)
            logger.info(f"  {i}. [{impact.upper()}] {rec.get('title', 'Unknown')}")
            logger.info(f"     {rec.get('description', 'No description')}")
            if potential_savings > 0:
                logger.info(f"     Potential Savings: ${potential_savings:.4f}")
        
        # Simulate optimization in action
        await self._simulate_cost_optimization()
    
    async def _demo_integration_features(self):
        """Demonstrate integration with AI models manager"""
        logger.info("\n🔗 INTEGRATION FEATURES")
        logger.info("-" * 40)
        
        # Test AI model selection with cost awareness
        models_to_test = [
            {"task": "reasoning", "tokens": 2000, "budget": 0.1},
            {"task": "fast_decision", "tokens": 500, "budget": 0.01},
            {"task": "high_frequency", "tokens": 100, "budget": 0.001}
        ]
        
        logger.info("Cost-Aware Model Selection:")
        for test in models_to_test:
            selected = await self.cost_manager_system.select_cost_optimal_model(
                task_type=test["task"],
                token_count=test["tokens"],
                budget_constraint=test["budget"]
            )
            logger.info(f"  {test['task']}: {selected} (budget: ${test['budget']})")
        
        # Add custom alert handler
        def custom_alert_handler(alert):
            logger.info(f"🔔 CUSTOM ALERT: {alert.message}")
        
        self.cost_manager_system.add_alert_handler(custom_alert_handler)
        logger.info("Added custom alert handler")
    
    async def _demo_comprehensive_report(self):
        """Demonstrate comprehensive reporting"""
        logger.info("\n📋 COMPREHENSIVE REPORT")
        logger.info("-" * 40)
        
        # Generate comprehensive report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        report = await self.cost_manager_system.get_comprehensive_cost_report(
            start_date=start_date,
            end_date=end_date,
            report_type="comprehensive"
        )
        
        logger.info("Comprehensive Cost Report Generated:")
        logger.info(f"  Report Period: {report['report_metadata']['period']}")
        logger.info(f"  Total Cost Events: {report['cost_analysis']['summary'].get('request_count', 0)}")
        logger.info(f"  Optimization Recommendations: {len(report.get('optimization_recommendations', []))}")
        logger.info(f"  System Status Sections: {len(report.get('system_status', {}))}")
        
        # Save report to file
        report_file = "data/comprehensive_cost_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_file}")
    
    async def _simulate_budget_limit_reached(self):
        """Simulate a budget limit being reached"""
        logger.info("Simulating budget limit scenario...")
        
        # Create a very low budget limit for demo
        from ai.cost_management.cost_manager import BudgetLimit
        
        low_budget = BudgetLimit(
            name="Demo Emergency Budget",
            limit_amount=0.01,  # Very low limit for demo
            period=timedelta(hours=1),
            alert_thresholds=[0.5, 0.8, 0.95],
            auto_actions={
                "switch_to_faster_models": True,
                "send_emergency_alert": True
            }
        )
        
        self.cost_manager_system.cost_manager.add_budget(low_budget)
        
        # Track a request that will exceed the budget
        await self.cost_manager_system.cost_manager.track_usage(
            provider="openai",
            model="gpt-4-turbo",
            tokens_used=2000,
            cost=0.02,  # Exceeds the $0.01 limit
            task_type="budget_test"
        )
    
    async def _simulate_anomaly_scenarios(self):
        """Simulate various anomaly scenarios"""
        logger.info("Simulating anomaly scenarios...")
        
        # Simulate cost spike
        await self.cost_manager_system.cost_manager.track_usage(
            provider="openai",
            model="gpt-4-turbo",
            tokens_used=5000,  # Large number to trigger spike
            cost=0.05,
            task_type="anomaly_test_spike"
        )
        
        # Simulate unusual request pattern
        for i in range(10):
            await self.cost_manager_system.cost_manager.track_usage(
                provider="anthropic",
                model="claude-3-5-sonnet",
                tokens_used=random.randint(100, 200),
                cost=random.uniform(0.001, 0.003),
                task_type="anomaly_test_pattern"
            )
    
    async def _simulate_cost_optimization(self):
        """Simulate cost optimization in action"""
        logger.info("Simulating cost optimization...")
        
        # Get current provider health
        health_report = await self.cost_manager_system.provider_manager.get_provider_health_report()
        
        if health_report['providers']:
            # Simulate switching to a healthier provider
            logger.info("Optimizing provider selection based on health metrics...")
            
            for provider, stats in health_report['providers'].items():
                logger.info(f"  {provider}: {stats['status']} (score: {stats['health_score']})")
        
        # Simulate budget optimization
        optimization = await self.cost_manager_system.budget_manager.optimize_budget_allocation()
        logger.info(f"Budget Optimization Potential Savings: ${optimization.get('potential_total_savings', 0):.4f}")
    
    async def _cleanup_demo(self):
        """Clean up demo resources"""
        logger.info("Cleaning up demo resources...")
        
        # Shutdown the cost management system
        await self.cost_manager_system.shutdown()
        
        logger.info("Demo cleanup complete")


async def main():
    """Main demo function"""
    try:
        demo = CostManagementDemo()
        await demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LLM COST MANAGEMENT SYSTEM DEMO                          ║
║                                                                              ║
║  This demo showcases the comprehensive LLM Cost Management System with:     ║
║  • Real-time cost tracking and monitoring                                   ║
║  • Advanced budget management and alerts                                    ║
║  • Provider optimization and intelligent routing                            ║
║  • Cost analytics and predictive forecasting                                ║
║  • Anomaly detection and automated responses                                ║
║  • Interactive dashboards and reporting                                     ║
║  • Full integration with existing AI systems                                ║
║                                                                              ║
║  The demo will run through all major features and show real-time            ║
║  cost management in action.                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run the demo
    asyncio.run(main())