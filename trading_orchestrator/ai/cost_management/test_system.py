#!/usr/bin/env python3
"""
Simple test for LLM Cost Management System
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from cost_management.cost_manager import LLMCostManager
    from cost_management.budget_manager import BudgetManager
    from cost_management.provider_manager import ProviderManager
    from cost_management.analytics import CostAnalytics
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test basic functionality
async def test_basic_functionality():
    """Test basic cost management functionality"""
    
    print("\n🧪 Testing LLM Cost Management System")
    print("=" * 50)
    
    try:
        # Test cost manager
        cost_manager = LLMCostManager(database_path="data/test_costs.db")
        print("✅ Cost Manager initialized")
        
        # Test budget manager
        budget_manager = BudgetManager(cost_manager)
        print("✅ Budget Manager initialized")
        
        # Test provider manager
        provider_manager = ProviderManager()
        print("✅ Provider Manager initialized")
        
        # Test analytics
        analytics = CostAnalytics(cost_manager, provider_manager)
        print("✅ Analytics initialized")
        
        # Test basic cost tracking
        metrics = await cost_manager.track_usage(
            provider="openai",
            model="gpt-4-turbo",
            tokens_used=1000,
            cost=0.01,
            task_type="test"
        )
        print(f"✅ Cost tracking successful: ${metrics.total_cost:.4f}")
        
        # Test current metrics
        current_metrics = cost_manager.get_current_metrics()
        print(f"✅ Current metrics: {current_metrics.total_tokens} tokens, ${current_metrics.total_cost:.4f}")
        
        # Test budget creation
        from cost_management.cost_manager import BudgetLimit
        budget = BudgetLimit(
            name="Test Budget",
            limit_amount=100.0,
            period=timedelta(days=1),
            alert_thresholds=[0.5, 0.8, 0.95],
            auto_actions={"switch_to_faster_models": True}
        )
        cost_manager.add_budget(budget)
        print("✅ Budget created successfully")
        
        # Test budget status
        budget_status = await budget_manager.get_budget_status()
        print(f"✅ Budget status retrieved: {len(budget_status)} profiles")
        
        # Test provider health
        health_report = await provider_manager.get_provider_health_report()
        print(f"✅ Provider health report: {health_report['summary']['total_providers']} providers")
        
        # Test cost summary
        cost_summary = cost_manager.get_cost_summary()
        print(f"✅ Cost summary: {cost_summary['current_metrics']['total_cost']} total cost")
        
        print("\n🎉 All tests passed!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run tests
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\n✅ LLM Cost Management System is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)