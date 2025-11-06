#!/usr/bin/env python3
"""
Quick Start Guide - LLM Cost Management System

This script provides a quick demonstration of the LLM Cost Management System
for immediate testing and validation.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_management.cost_manager import LLMCostManager, BudgetLimit
from cost_management.integration import LLMIntegratedCostManager
from cost_management.budget_manager import BudgetTier

async def quick_start_demo():
    """Quick start demonstration"""
    
    print("\n" + "="*70)
    print("🚀 LLM COST MANAGEMENT SYSTEM - QUICK START DEMO")
    print("="*70)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        # 1. Initialize Cost Manager
        print("\n1️⃣ Initializing Cost Manager...")
        cost_manager = LLMCostManager(database_path="data/quick_start_costs.db")
        
        # 2. Add a budget
        daily_budget = BudgetLimit(
            name="Quick Start Daily Budget",
            limit_amount=100.0,
            period=timedelta(days=1),
            alert_thresholds=[0.5, 0.8, 0.95],
            auto_actions={"switch_to_faster_models": True}
        )
        cost_manager.add_budget(daily_budget)
        print("✅ Budget created: $100/day")
        
        # 3. Track some AI requests
        print("\n2️⃣ Tracking AI Requests...")
        
        requests = [
            {"provider": "openai", "model": "gpt-4-turbo", "tokens": 1500, "cost": 0.015},
            {"provider": "anthropic", "model": "claude-3-5-sonnet", "tokens": 2000, "cost": 0.03},
            {"provider": "openai", "model": "gpt-3.5-turbo", "tokens": 1000, "cost": 0.0015},
        ]
        
        for i, req in enumerate(requests, 1):
            metrics = await cost_manager.track_usage(
                provider=req["provider"],
                model=req["model"],
                tokens_used=req["tokens"],
                cost=req["cost"],
                task_type="quick_start_demo"
            )
            print(f"   Request {i}: {req['provider']}/{req['model']} - ${req['cost']:.4f}")
        
        # 4. Get current metrics
        print("\n3️⃣ Current Cost Metrics:")
        metrics = cost_manager.get_current_metrics()
        print(f"   💰 Total Cost: ${metrics.total_cost:.4f}")
        print(f"   🔢 Total Tokens: {metrics.total_tokens:,}")
        print(f"   📊 Requests: {metrics.request_count}")
        print(f"   💵 Avg Cost/Request: ${metrics.average_cost_per_request:.4f}")
        
        # 5. Provider breakdown
        print("\n4️⃣ Cost Breakdown by Provider:")
        for provider, data in metrics.provider_breakdown.items():
            print(f"   📊 {provider}: ${data['total_cost']:.4f} ({data['total_tokens']:,} tokens)")
        
        # 6. Budget status
        print("\n5️⃣ Budget Status:")
        budget_utilization = (metrics.total_cost / daily_budget.limit_amount) * 100
        print(f"   💳 Daily Budget: ${daily_budget.limit_amount:.2f}")
        print(f"   💸 Spent: ${metrics.total_cost:.2f}")
        print(f"   📈 Utilization: {budget_utilization:.1f}%")
        
        if budget_utilization < 50:
            print("   ✅ Status: HEALTHY")
        elif budget_utilization < 80:
            print("   ⚠️  Status: CAUTION")
        else:
            print("   🚨 Status: WARNING")
        
        # 7. Analytics preview
        print("\n6️⃣ Analytics Preview:")
        cost_summary = cost_manager.get_cost_summary()
        print(f"   📊 Total Active Budgets: {len(cost_summary['active_budgets'])}")
        print(f"   📧 Recent Alerts: {len(cost_summary['recent_alerts'])}")
        
        # 8. Cost optimization suggestions
        print("\n7️⃣ Quick Optimization Tips:")
        if metrics.total_tokens > 0:
            avg_cost_per_1k = metrics.cost_per_1k_tokens
            if avg_cost_per_1k > 0.01:
                print("   💡 Consider using GPT-3.5-turbo for cost reduction")
            if avg_cost_per_1k < 0.001:
                print("   💡 You're using cost-effective models well!")
        
        print("\n" + "="*70)
        print("✅ QUICK START DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📚 Next Steps:")
        print("1. Review the comprehensive documentation in README.md")
        print("2. Run the full demo: python demo_comprehensive.py")
        print("3. Explore integration examples in integration_example.py")
        print("4. Check the implementation summary in IMPLEMENTATION_SUMMARY.md")
        
        print("\n🔗 Key Files:")
        print("   • README.md - Complete documentation")
        print("   • IMPLEMENTATION_SUMMARY.md - Technical details")
        print("   • demo_comprehensive.py - Full system demo")
        print("   • integration_example.py - Trading orchestrator integration")
        print("   • test_system.py - System validation tests")
        
        print("\n💡 Quick Commands:")
        print("   python test_system.py           # Run system tests")
        print("   python demo_comprehensive.py    # Run full demo")
        print("   python integration_example.py   # See integration example")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  LLM COST MANAGEMENT SYSTEM                         ║
║                        QUICK START GUIDE                            ║
║                                                                      ║
║  This demo will show you the core features of the LLM               ║
║  Cost Management System in under 60 seconds.                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    success = asyncio.run(quick_start_demo())
    
    if success:
        print("\n🎉 Demo completed successfully! The system is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)