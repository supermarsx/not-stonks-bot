"""
Integration Example - LLM Cost Management with AI Orchestrator

This example shows how to integrate the comprehensive LLM Cost Management System
with the existing AI trading orchestrator for seamless cost tracking and optimization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from ai.orchestrator import AITradingOrchestrator, TradingMode
from ai.models.ai_models_manager import AIModelsManager
from ai.tools.trading_tools import TradingTools

# Import the cost management system
from ai.cost_management import LLMIntegratedCostManager

from loguru import logger


class TradingOrchestratorWithCostManagement:
    """
    Enhanced Trading Orchestrator with integrated cost management
    
    This class wraps the existing AI trading orchestrator and adds comprehensive
    cost management, budget controls, and optimization features.
    """
    
    def __init__(
        self,
        broker_manager,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        cost_tracking_enabled: bool = True,
        budget_limit_daily: float = 100.0,
        budget_limit_monthly: float = 3000.0
    ):
        """
        Initialize enhanced trading orchestrator with cost management
        
        Args:
            broker_manager: Existing broker manager instance
            openai_api_key: OpenAI API key for cost tracking
            anthropic_api_key: Anthropic API key for cost tracking
            cost_tracking_enabled: Enable/disable cost tracking
            budget_limit_daily: Daily budget limit in USD
            budget_limit_monthly: Monthly budget limit in USD
        """
        self.broker_manager = broker_manager
        
        # Initialize AI components (existing)
        self.ai_models = AIModelsManager(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key
        )
        
        self.trading_tools = TradingTools(broker_manager=broker_manager)
        
        # Initialize enhanced orchestrator (existing)
        self.orchestrator = AITradingOrchestrator(
            ai_models_manager=self.ai_models,
            trading_tools=self.trading_tools,
            trading_mode=TradingMode.PAPER  # Default to paper trading
        )
        
        # Initialize cost management system
        if cost_tracking_enabled:
            self.cost_system = LLMIntegratedCostManager(
                ai_models_manager=self.ai_models,
                database_path="data/trading_orchestrator_costs.db",
                enable_real_time_monitoring=True
            )
            
            # Setup trading-specific budget profiles
            self._setup_trading_budgets(budget_limit_daily, budget_limit_monthly)
            
            # Register custom alert handler for trading
            self.cost_system.add_alert_handler(self._handle_cost_alert)
            
            # Track integration
            self.cost_tracking_enabled = True
        else:
            self.cost_system = None
            self.cost_tracking_enabled = False
        
        logger.info(f"Trading Orchestrator with Cost Management initialized (cost_tracking={cost_tracking_enabled})")
    
    def _setup_trading_budgets(self, daily_limit: float, monthly_limit: float):
        """Setup trading-specific budget profiles"""
        
        # Trading budget profile
        asyncio.create_task(self.cost_system.create_budget_profile(
            name="Trading Operations",
            monthly_limit=monthly_limit,
            daily_limit=daily_limit,
            tier="medium",
            auto_optimization=True
        ))
        
        # Emergency trading budget (very conservative)
        asyncio.create_task(self.cost_system.create_budget_profile(
            name="Emergency Trading",
            monthly_limit=monthly_limit * 0.1,  # 10% of normal budget
            daily_limit=daily_limit * 0.1,
            tier="low",
            auto_optimization=True
        ))
        
        # Set active profile
        self.cost_system.budget_manager.set_active_profile("Trading Operations")
        
        logger.info(f"Trading budgets configured: ${daily_limit}/day, ${monthly_limit}/month")
    
    async def _handle_cost_alert(self, alert):
        """Handle cost alerts specific to trading operations"""
        
        if alert.level.value in ['warning', 'critical', 'emergency']:
            logger.warning(f"Trading cost alert: {alert.message}")
            
            # If in emergency mode, switch to paper trading
            if alert.level.value == 'emergency':
                logger.critical("Switching to paper trading due to cost emergency")
                self.orchestrator.trading_mode = TradingMode.PAPER
                
                # Also enable emergency budget profile
                self.cost_system.budget_manager.set_active_profile("Emergency Trading")
    
    async def analyze_market_with_cost_tracking(
        self,
        symbols: list,
        analysis_type: str = 'comprehensive',
        budget_constraint: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced market analysis with cost tracking and optimization
        
        Args:
            symbols: List of symbols to analyze
            analysis_type: Type of analysis (quick, comprehensive, deep)
            budget_constraint: Maximum cost for this analysis
            
        Returns:
            Market analysis results with cost tracking metadata
        """
        
        # Select cost-optimal model based on analysis type and budget
        model_selection = self._get_optimal_model_for_analysis(
            analysis_type, budget_constraint
        )
        
        logger.info(f"Market analysis with {model_selection} (budget: ${budget_constraint or 'unlimited'})")
        
        # Track the analysis start
        analysis_start_time = datetime.now()
        
        try:
            # Perform market analysis using existing orchestrator
            analysis_result = await self.orchestrator.analyze_market(
                symbols=symbols,
                analysis_type=analysis_type,
                use_reasoning_model=(model_selection in ['claude-3-5-sonnet', 'gpt-4-turbo'])
            )
            
            # Calculate costs (simplified estimation)
            estimated_tokens = self._estimate_analysis_tokens(symbols, analysis_type)
            estimated_cost = self._calculate_analysis_cost(model_selection, estimated_tokens)
            
            # Track costs with the cost management system
            if self.cost_tracking_enabled:
                await self.cost_system.track_ai_request(
                    provider=self._get_provider_for_model(model_selection),
                    model=model_selection,
                    tokens_used=estimated_tokens,
                    cost=estimated_cost,
                    request_duration=(datetime.now() - analysis_start_time).total_seconds(),
                    session_id=f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    task_type="market_analysis",
                    metadata={
                        'symbols': symbols,
                        'analysis_type': analysis_type,
                        'symbols_count': len(symbols)
                    }
                )
            
            # Add cost metadata to results
            analysis_result['cost_metadata'] = {
                'model_used': model_selection,
                'estimated_cost': estimated_cost,
                'estimated_tokens': estimated_tokens,
                'cost_per_symbol': estimated_cost / len(symbols),
                'analysis_duration': (datetime.now() - analysis_start_time).total_seconds(),
                'budget_constraint': budget_constraint,
                'cost_tracking_enabled': self.cost_tracking_enabled
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            raise
    
    async def generate_strategy_with_cost_optimization(
        self,
        strategy_type,
        symbols: list,
        max_cost: Optional[float] = 5.0
    ) -> Dict[str, Any]:
        """
        Generate trading strategy with cost optimization
        
        Args:
            strategy_type: Type of strategy to generate
            symbols: List of symbols for the strategy
            max_cost: Maximum cost for strategy generation
            
        Returns:
            Strategy with cost optimization metadata
        """
        
        # Use cost-optimal model for strategy generation
        if max_cost and max_cost < 1.0:
            # Use fast models for low-cost generation
            preferred_model = "gpt-3.5-turbo"
        else:
            # Use reasoning models for complex strategies
            preferred_model = "claude-3-5-sonnet"
        
        logger.info(f"Generating {strategy_type.value} strategy with {preferred_model} (max cost: ${max_cost})")
        
        # Track strategy generation
        gen_start_time = datetime.now()
        
        try:
            # Generate strategy using existing orchestrator
            strategy_result = await self.orchestrator.generate_trading_strategy(
                strategy_type=strategy_type,
                symbols=symbols,
                parameters={}  # Could include cost optimization parameters
            )
            
            # Calculate generation costs
            estimated_tokens = self._estimate_strategy_tokens(len(symbols), strategy_type)
            estimated_cost = self._calculate_generation_cost(preferred_model, estimated_tokens)
            
            # Track costs if under budget
            if self.cost_tracking_enabled and (not max_cost or estimated_cost <= max_cost):
                await self.cost_system.track_ai_request(
                    provider=self._get_provider_for_model(preferred_model),
                    model=preferred_model,
                    tokens_used=estimated_tokens,
                    cost=estimated_cost,
                    request_duration=(datetime.now() - gen_start_time).total_seconds(),
                    session_id=f"strategy_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    task_type="strategy_generation",
                    metadata={
                        'strategy_type': strategy_type.value,
                        'symbols': symbols,
                        'symbols_count': len(symbols)
                    }
                )
            
            # Add cost metadata to strategy
            strategy_result['cost_metadata'] = {
                'model_used': preferred_model,
                'generation_cost': estimated_cost,
                'cost_per_symbol': estimated_cost / len(symbols),
                'max_cost': max_cost,
                'budget_compliant': not max_cost or estimated_cost <= max_cost,
                'generation_duration': (datetime.now() - gen_start_time).total_seconds()
            }
            
            return strategy_result
            
        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            raise
    
    async def execute_ai_trade_with_cost_control(
        self,
        symbol: str,
        side: str,
        reasoning: str,
        max_cost: Optional[float] = 0.1,
        risk_check: bool = True
    ) -> Dict[str, Any]:
        """
        Execute AI trade with cost controls
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            reasoning: AI reasoning for the trade
            max_cost: Maximum cost for trade analysis
            risk_check: Whether to perform risk checks
            
        Returns:
            Trade execution result with cost metadata
        """
        
        # Use fast model for trade execution decisions (cost-efficient)
        trade_model = "gpt-3.5-turbo"
        
        logger.info(f"Executing {side} trade for {symbol} with {trade_model} (max cost: ${max_cost})")
        
        # Track trade execution
        exec_start_time = datetime.now()
        
        try:
            # Execute trade using existing orchestrator
            trade_result = await self.orchestrator.execute_ai_trade(
                symbol=symbol,
                side=side,
                reasoning=reasoning,
                risk_check=risk_check
            )
            
            # Calculate execution costs (should be minimal for trade decisions)
            estimated_tokens = 200  # Small for trade decisions
            estimated_cost = self._calculate_generation_cost(trade_model, estimated_tokens)
            
            # Only track costs if they're significant
            if self.cost_tracking_enabled and estimated_cost > 0.01:  # Track costs > $0.01
                await self.cost_system.track_ai_request(
                    provider=self._get_provider_for_model(trade_model),
                    model=trade_model,
                    tokens_used=estimated_tokens,
                    cost=estimated_cost,
                    request_duration=(datetime.now() - exec_start_time).total_seconds(),
                    session_id=f"trade_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    task_type="trade_execution",
                    metadata={
                        'symbol': symbol,
                        'side': side,
                        'risk_check': risk_check
                    }
                )
            
            # Add cost metadata
            trade_result['cost_metadata'] = {
                'model_used': trade_model,
                'execution_cost': estimated_cost,
                'max_cost': max_cost,
                'budget_compliant': not max_cost or estimated_cost <= max_cost,
                'execution_duration': (datetime.now() - exec_start_time).total_seconds(),
                'cost_tracking_enabled': self.cost_tracking_enabled
            }
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            raise
    
    def _get_optimal_model_for_analysis(self, analysis_type: str, budget: Optional[float]) -> str:
        """Get cost-optimal model for market analysis"""
        
        if budget and budget < 0.5:
            # Low budget - use fastest models
            return "gpt-3.5-turbo"
        elif analysis_type == 'quick':
            # Quick analysis - use fast models
            return "gpt-3.5-turbo"
        elif analysis_type == 'deep':
            # Deep analysis - use reasoning models if budget allows
            if not budget or budget > 2.0:
                return "claude-3-5-sonnet"
            else:
                return "gpt-3.5-turbo"
        else:
            # Comprehensive analysis - balanced choice
            return "claude-3-5-sonnet" if not budget or budget > 1.0 else "gpt-3.5-turbo"
    
    def _estimate_analysis_tokens(self, symbols: list, analysis_type: str) -> int:
        """Estimate tokens for market analysis"""
        
        base_tokens = len(symbols) * 100  # Base tokens per symbol
        
        if analysis_type == 'quick':
            return int(base_tokens * 1.5)
        elif analysis_type == 'comprehensive':
            return int(base_tokens * 3.0)
        elif analysis_type == 'deep':
            return int(base_tokens * 5.0)
        else:
            return int(base_tokens * 2.0)
    
    def _calculate_analysis_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for market analysis"""
        
        cost_per_1k = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-4-turbo": 0.01,
            "claude-3-5-sonnet": 0.015,
            "claude-haiku": 0.00025
        }
        
        return (tokens / 1000) * cost_per_1k.get(model, 0.001)
    
    def _estimate_strategy_tokens(self, symbol_count: int, strategy_type) -> int:
        """Estimate tokens for strategy generation"""
        
        base_tokens = symbol_count * 500  # Base tokens per symbol
        
        # Strategy type multipliers
        multipliers = {
            "MOMENTUM": 2.0,
            "MEAN_REVERSION": 2.5,
            "PAIRS_TRADING": 3.0,
            "AI_DISCRETIONARY": 4.0
        }
        
        multiplier = multipliers.get(str(strategy_type), 2.0)
        return int(base_tokens * multiplier)
    
    def _calculate_generation_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for generation tasks"""
        
        cost_per_1k = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-4-turbo": 0.01,
            "claude-3-5-sonnet": 0.015,
            "claude-haiku": 0.00025
        }
        
        return (tokens / 1000) * cost_per_1k.get(model, 0.001)
    
    def _get_provider_for_model(self, model: str) -> str:
        """Get provider name for model"""
        
        if "gpt" in model:
            return "openai"
        elif "claude" in model:
            return "anthropic"
        else:
            return "unknown"
    
    async def get_trading_cost_summary(self) -> Dict[str, Any]:
        """Get trading-specific cost summary"""
        
        if not self.cost_tracking_enabled:
            return {"error": "Cost tracking is not enabled"}
        
        try:
            # Get system overview
            system_overview = await self.cost_system.get_system_overview()
            
            # Get budget status
            budget_status = await self.cost_system.budget_manager.get_budget_status()
            
            # Get optimization recommendations
            recommendations = await self.cost_system.get_cost_optimization_recommendations()
            
            # Filter recommendations for trading relevance
            trading_recommendations = [
                rec for rec in recommendations
                if any(keyword in rec.get('title', '').lower() 
                      for keyword in ['model', 'provider', 'cost', 'budget'])
            ]
            
            return {
                'trading_cost_summary': {
                    'current_period_cost': system_overview['system_overview']['current_period_cost'],
                    'daily_budget_utilization': budget_status.get('daily_percentage', 'N/A'),
                    'emergency_mode': system_overview['system_overview']['emergency_mode'],
                    'total_requests': system_overview['system_overview']['request_count'],
                    'average_cost_per_request': system_overview['system_overview']['avg_cost_per_request']
                },
                'budget_status': budget_status,
                'trading_recommendations': trading_recommendations[:5],  # Top 5
                'cost_optimization_potential': sum(
                    rec.get('potential_savings', 0) 
                    for rec in trading_recommendations
                ),
                'cost_tracking_enabled': self.cost_tracking_enabled
            }
            
        except Exception as e:
            logger.error(f"Cost summary error: {e}")
            return {"error": f"Failed to get cost summary: {str(e)}"}
    
    async def optimize_trading_costs(self) -> Dict[str, Any]:
        """Optimize trading costs based on current usage patterns"""
        
        if not self.cost_tracking_enabled:
            return {"error": "Cost tracking is not enabled"}
        
        try:
            # Get current usage patterns
            cost_summary = await self.get_trading_cost_summary()
            
            if 'error' in cost_summary:
                return cost_summary
            
            optimizations_applied = []
            
            # Check if we should switch to more cost-effective models
            current_cost_per_request = float(
                cost_summary['trading_cost_summary']['average_cost_per_request'].replace('$', '')
            )
            
            if current_cost_per_request > 0.01:  # If > $0.01 per request
                optimizations_applied.append({
                    'type': 'model_optimization',
                    'action': 'Switch to faster models for routine tasks',
                    'description': 'Current average cost per request is high',
                    'expected_savings': '20-30%'
                })
            
            # Check budget utilization
            daily_percentage = cost_summary['budget_status'].get('daily_percentage', '0%')
            daily_pct = float(daily_percentage.replace('%', ''))
            
            if daily_pct > 80:
                optimizations_applied.append({
                    'type': 'budget_control',
                    'action': 'Enable stricter budget controls',
                    'description': f'Daily budget utilization at {daily_percentage}',
                    'expected_savings': 'Budget compliance'
                })
            
            # Provider optimization
            provider_suggestions = await self.cost_system.provider_manager.get_cost_optimization_suggestions()
            if provider_suggestions:
                optimizations_applied.append({
                    'type': 'provider_optimization',
                    'action': 'Optimize provider selection',
                    'description': f'{len(provider_suggestions)} provider optimization opportunities found',
                    'expected_savings': 'Variable'
                })
            
            return {
                'optimizations_applied': optimizations_applied,
                'current_metrics': cost_summary,
                'optimization_timestamp': datetime.now().isoformat(),
                'total_optimizations': len(optimizations_applied)
            }
            
        except Exception as e:
            logger.error(f"Cost optimization error: {e}")
            return {"error": f"Failed to optimize costs: {str(e)}"}
    
    def get_current_trading_mode(self) -> str:
        """Get current trading mode (enhanced with cost status)"""
        
        base_mode = self.orchestrator.trading_mode.value
        
        if self.cost_tracking_enabled:
            # Check if emergency mode is active
            if hasattr(self, 'cost_system') and self.cost_system.budget_manager.emergency_mode:
                return f"{base_mode} (EMERGENCY MODE - Cost Controls Active)"
            else:
                return f"{base_mode} (Cost Tracked)"
        
        return base_mode
    
    async def shutdown_enhanced_orchestrator(self):
        """Shutdown the enhanced orchestrator with cost management cleanup"""
        
        logger.info("Shutting down enhanced trading orchestrator...")
        
        # Shutdown cost management system
        if self.cost_tracking_enabled:
            await self.cost_system.shutdown()
        
        logger.info("Enhanced trading orchestrator shutdown complete")


# Example usage
async def main():
    """Example of using the enhanced trading orchestrator"""
    
    # Mock broker manager (in real usage, use actual broker manager)
    class MockBrokerManager:
        pass
    
    broker_manager = MockBrokerManager()
    
    # Initialize enhanced orchestrator
    enhanced_orchestrator = TradingOrchestratorWithCostManagement(
        broker_manager=broker_manager,
        openai_api_key="your_openai_key",
        anthropic_api_key="your_anthropic_key",
        cost_tracking_enabled=True,
        budget_limit_daily=50.0,  # $50/day budget
        budget_limit_monthly=1500.0  # $1500/month budget
    )
    
    try:
        # Example 1: Market analysis with cost tracking
        logger.info("=" * 60)
        logger.info("EXAMPLE 1: Market Analysis with Cost Tracking")
        logger.info("=" * 60)
        
        analysis = await enhanced_orchestrator.analyze_market_with_cost_tracking(
            symbols=['AAPL', 'TSLA', 'MSFT'],
            analysis_type='comprehensive',
            budget_constraint=2.0  # Max $2 for this analysis
        )
        
        logger.info(f"Analysis cost: ${analysis['cost_metadata']['estimated_cost']:.4f}")
        logger.info(f"Model used: {analysis['cost_metadata']['model_used']}")
        
        # Example 2: Strategy generation with cost optimization
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 2: Strategy Generation with Cost Optimization")
        logger.info("=" * 60)
        
        from ai.orchestrator import StrategyType
        
        strategy = await enhanced_orchestrator.generate_strategy_with_cost_optimization(
            strategy_type=StrategyType.MOMENTUM,
            symbols=['AAPL', 'MSFT'],
            max_cost=5.0  # Max $5 for strategy generation
        )
        
        logger.info(f"Strategy generation cost: ${strategy['cost_metadata']['generation_cost']:.4f}")
        logger.info(f"Budget compliant: {strategy['cost_metadata']['budget_compliant']}")
        
        # Example 3: Cost summary and optimization
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 3: Cost Summary and Optimization")
        logger.info("=" * 60)
        
        cost_summary = await enhanced_orchestrator.get_trading_cost_summary()
        logger.info(f"Cost summary: {json.dumps(cost_summary, indent=2)}")
        
        optimizations = await enhanced_orchestrator.optimize_trading_costs()
        logger.info(f"Optimizations applied: {optimizations['total_optimizations']}")
        
        # Example 4: Trading mode with cost awareness
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 4: Cost-Aware Trading Mode")
        logger.info("=" * 60)
        
        current_mode = enhanced_orchestrator.get_current_trading_mode()
        logger.info(f"Current trading mode: {current_mode}")
        
    except Exception as e:
        logger.error(f"Example error: {e}")
    
    finally:
        # Cleanup
        await enhanced_orchestrator.shutdown_enhanced_orchestrator()


if __name__ == "__main__":
    import json
    asyncio.run(main())