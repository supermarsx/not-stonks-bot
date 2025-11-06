"""
Slippage Tracker - Execution Quality Analysis

Tracks and analyzes execution quality:
- Slippage calculation and analysis
- Market impact assessment
- Execution benchmark comparisons
- Cost analysis (commissions, spreads, market impact)
- Performance metrics by broker and order type
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from decimal import Decimal

from config.database import get_db
from database.models.trading import Order, Trade, OrderType, OrderSide
from database.models.risk import RiskEvent

logger = logging.getLogger(__name__)


class SlippageType(str, Enum):
    """Types of slippage measurement."""
    ABSOLUTE = "absolute"  # Price difference
    PERCENTAGE = "percentage"  # Percentage difference
    BID_ASK = "bid_ask"  # Bid-ask spread impact
    VWAP = "vwap"  # Volume-weighted average price
    TWAP = "twap"  # Time-weighted average price


class SlippageAnalyzer:
    """
    Comprehensive slippage and execution quality analysis.
    
    Analyzes:
    - Execution slippage vs benchmarks
    - Market impact assessment
    - Broker execution quality
    - Order type performance
    - Time-based execution patterns
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Slippage analysis configuration
        self.benchmark_types = {
            "market_orders": "mid_price",
            "limit_orders": "limit_price",
            "stop_orders": "stop_trigger_price"
        }
        
        # Analysis periods
        self.analysis_periods = {
            "intraday": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }
        
        # Market data cache for benchmarks (simplified)
        self.market_data_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info(f"SlippageAnalyzer initialized for user {self.user_id}")
    
    async def calculate_slippage(self, order_id: int) -> Dict[str, Any]:
        """
        Calculate slippage for a specific order.
        
        Args:
            order_id: ID of order to analyze
            
        Returns:
            Slippage analysis results
        """
        try:
            # Get order and related trades
            order = self.db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                return {"error": "Order not found"}
            
            if order.status != "filled":
                return {"error": "Order not filled"}
            
            if not order.avg_fill_price:
                return {"error": "No fill price available"}
            
            # Get benchmark price
            benchmark_price = await self._get_benchmark_price(order)
            
            if not benchmark_price:
                return {"error": "No benchmark price available"}
            
            # Calculate slippage metrics
            slippage_metrics = await self._calculate_slippage_metrics(order, benchmark_price)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(order.symbol, order.submitted_at)
            
            # Calculate execution quality score
            quality_score = await self._calculate_execution_quality_score(slippage_metrics, market_conditions)
            
            result = {
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "fill_price": order.avg_fill_price,
                "benchmark_price": benchmark_price,
                "slippage": slippage_metrics,
                "market_conditions": market_conditions,
                "execution_quality_score": quality_score,
                "broker_name": order.broker_name,
                "execution_time": (order.filled_at - order.submitted_at).total_seconds() if order.filled_at else None
            }
            
            logger.debug(f"Slippage calculated for order {order_id}: {slippage_metrics.get('percentage', 0):.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Slippage calculation error: {str(e)}")
            return {"error": str(e)}
    
    async def get_slippage_analysis(self, start_date: datetime, end_date: datetime,
                                  symbol: str = None, broker: str = None) -> Dict[str, Any]:
        """
        Get comprehensive slippage analysis for a period.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            symbol: Filter by symbol (optional)
            broker: Filter by broker (optional)
            
        Returns:
            Comprehensive slippage analysis
        """
        try:
            # Build query for filled orders
            query = self.db.query(Order).filter(
                Order.user_id == self.user_id,
                Order.status == "filled",
                Order.submitted_at >= start_date,
                Order.submitted_at <= end_date
            )
            
            if symbol:
                query = query.filter(Order.symbol == symbol)
            
            if broker:
                query = query.filter(Order.broker_name == broker)
            
            orders = query.all()
            
            if not orders:
                return {"message": "No filled orders found for analysis period"}
            
            # Analyze each order
            slippage_results = []
            for order in orders:
                try:
                    slippage_result = await self.calculate_slippage(order.id)
                    if "error" not in slippage_result:
                        slippage_results.append(slippage_result)
                except Exception as e:
                    logger.warning(f"Error analyzing order {order.id}: {str(e)}")
                    continue
            
            if not slippage_results:
                return {"error": "No valid slippage data available"}
            
            # Calculate aggregate metrics
            aggregate_metrics = await self._calculate_aggregate_metrics(slippage_results)
            
            # Analyze by broker
            broker_analysis = await self._analyze_by_broker(slippage_results)
            
            # Analyze by order type
            order_type_analysis = await self._analyze_by_order_type(slippage_results)
            
            # Analyze by symbol
            symbol_analysis = await self._analyze_by_symbol(slippage_results)
            
            # Time-based analysis
            time_analysis = await self._analyze_by_time(slippage_results)
            
            return {
                "analysis_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": (end_date - start_date).days
                },
                "summary": {
                    "total_orders_analyzed": len(slippage_results),
                    "total_volume": sum(r["quantity"] for r in slippage_results),
                    "avg_slippage": aggregate_metrics["avg_percentage_slippage"],
                    "median_slippage": aggregate_metrics["median_percentage_slippage"]
                },
                "aggregate_metrics": aggregate_metrics,
                "broker_analysis": broker_analysis,
                "order_type_analysis": order_type_analysis,
                "symbol_analysis": symbol_analysis,
                "time_analysis": time_analysis,
                "detailed_results": slippage_results[:100]  # Limit to first 100 for performance
            }
            
        except Exception as e:
            logger.error(f"Slippage analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def compare_broker_execution_quality(self, symbol: str = None) -> Dict[str, Any]:
        """
        Compare execution quality across brokers.
        
        Args:
            symbol: Symbol to focus analysis on (optional)
            
        Returns:
            Broker execution quality comparison
        """
        try:
            # Get filled orders from last 30 days
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            query = self.db.query(Order).filter(
                Order.user_id == self.user_id,
                Order.status == "filled",
                Order.submitted_at >= start_date
            )
            
            if symbol:
                query = query.filter(Order.symbol == symbol)
            
            orders = query.all()
            
            if not orders:
                return {"message": "No filled orders found for comparison"}
            
            # Analyze by broker
            broker_data = {}
            
            for order in orders:
                broker_name = order.broker_name
                
                if broker_name not in broker_data:
                    broker_data[broker_name] = {
                        "orders": [],
                        "total_volume": 0,
                        "avg_execution_time": 0,
                        "slippage_data": []
                    }
                
                broker_data[broker_name]["orders"].append(order)
                broker_data[broker_name]["total_volume"] += order.quantity
                
                # Calculate execution time
                if order.filled_at:
                    exec_time = (order.filled_at - order.submitted_at).total_seconds()
                    broker_data[broker_name]["avg_execution_time"] += exec_time
                
                # Calculate slippage if possible
                try:
                    slippage_result = await self.calculate_slippage(order.id)
                    if "error" not in slippage_result and "percentage" in slippage_result["slippage"]:
                        broker_data[broker_name]["slippage_data"].append(
                            slippage_result["slippage"]["percentage"]
                        )
                except:
                    continue
            
            # Calculate broker metrics
            broker_metrics = {}
            
            for broker_name, data in broker_data.items():
                orders_count = len(data["orders"])
                if orders_count == 0:
                    continue
                
                avg_execution_time = data["avg_execution_time"] / orders_count
                slippage_data = data["slippage_data"]
                
                broker_metrics[broker_name] = {
                    "orders_count": orders_count,
                    "total_volume": data["total_volume"],
                    "avg_execution_time": avg_execution_time,
                    "avg_slippage": np.mean(slippage_data) if slippage_data else 0,
                    "median_slippage": np.median(slippage_data) if slippage_data else 0,
                    "slippage_std": np.std(slippage_data) if slippage_data else 0,
                    "slippage_25th": np.percentile(slippage_data, 25) if slippage_data else 0,
                    "slippage_75th": np.percentile(slippage_data, 75) if slippage_data else 0,
                    "quality_score": await self._calculate_broker_quality_score(data, slippage_data)
                }
            
            # Rank brokers
            ranked_brokers = sorted(
                broker_metrics.items(),
                key=lambda x: x[1]["quality_score"],
                reverse=True
            )
            
            return {
                "comparison_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol_filter": symbol
                },
                "broker_metrics": broker_metrics,
                "broker_rankings": [
                    {
                        "rank": i + 1,
                        "broker": broker_name,
                        "quality_score": metrics["quality_score"],
                        "avg_slippage": metrics["avg_slippage"],
                        "avg_execution_time": metrics["avg_execution_time"]
                    }
                    for i, (broker_name, metrics) in enumerate(ranked_brokers)
                ],
                "best_performer": ranked_brokers[0][0] if ranked_brokers else None,
                "worst_performer": ranked_brokers[-1][0] if ranked_brokers else None
            }
            
        except Exception as e:
            logger.error(f"Broker comparison error: {str(e)}")
            return {"error": str(e)}
    
    async def get_market_impact_analysis(self, order_id: int) -> Dict[str, Any]:
        """
        Analyze market impact of a specific order.
        
        Args:
            order_id: ID of order to analyze
            
        Returns:
            Market impact analysis
        """
        try:
            order = self.db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                return {"error": "Order not found"}
            
            # Get pre and post trade market data
            pre_trade_data = await self._get_market_data_snapshot(order.symbol, order.submitted_at, before=True)
            post_trade_data = await self._get_market_data_snapshot(order.symbol, order.filled_at, before=False)
            
            if not pre_trade_data or not post_trade_data:
                return {"error": "Insufficient market data for impact analysis"}
            
            # Calculate market impact metrics
            price_impact = self._calculate_price_impact(pre_trade_data, post_trade_data)
            volume_impact = self._calculate_volume_impact(order, pre_trade_data, post_trade_data)
            
            # Estimate market impact based on order characteristics
            estimated_impact = await self._estimate_market_impact(order, pre_trade_data)
            
            result = {
                "order_id": order_id,
                "symbol": order.symbol,
                "quantity": order.quantity,
                "market_impact": {
                    "price_impact": price_impact,
                    "volume_impact": volume_impact,
                    "estimated_impact": estimated_impact
                },
                "market_conditions": {
                    "pre_trade_price": pre_trade_data.get("price"),
                    "post_trade_price": post_trade_data.get("price"),
                    "pre_trade_volume": pre_trade_data.get("volume", 0),
                    "post_trade_volume": post_trade_data.get("volume", 0)
                },
                "analysis_timestamp": datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Market impact analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _get_benchmark_price(self, order: Order) -> Optional[float]:
        """Get benchmark price for slippage calculation."""
        try:
            symbol = order.symbol
            order_type = order.order_type.value
            side = order.side.value
            
            if order_type == "market":
                # Use mid price for market orders
                return await self._get_mid_price(symbol, order.submitted_at)
            elif order_type == "limit":
                # Use limit price for limit orders
                return order.limit_price
            elif order_type == "stop":
                # Use stop price for stop orders
                return order.stop_price
            else:
                # Fallback to mid price
                return await self._get_mid_price(symbol, order.submitted_at)
                
        except Exception as e:
            logger.error(f"Benchmark price calculation error: {str(e)}")
            return None
    
    async def _calculate_slippage_metrics(self, order: Order, benchmark_price: float) -> Dict[str, Any]:
        """Calculate various slippage metrics."""
        try:
            fill_price = order.avg_fill_price
            side = order.side.value
            
            # Absolute slippage
            absolute_slippage = fill_price - benchmark_price
            
            # Percentage slippage (positive = adverse, negative = favorable)
            percentage_slippage = absolute_slippage / benchmark_price if benchmark_price > 0 else 0
            
            # Adjust sign based on trade side
            if side == "buy":
                # For buy orders, higher price is worse (adverse slippage)
                signed_percentage = percentage_slippage
            else:
                # For sell orders, lower price is worse (adverse slippage)
                signed_percentage = -percentage_slippage
            
            # VWAP-based slippage (if available)
            vwap_slippage = await self._calculate_vwap_slippage(order)
            
            # Time-based analysis
            execution_time = (order.filled_at - order.submitted_at).total_seconds() if order.filled_at else 0
            
            return {
                "absolute": absolute_slippage,
                "percentage": signed_percentage,
                "percentage_bps": signed_percentage * 10000,  # basis points
                "vwap_slippage": vwap_slippage,
                "execution_time_seconds": execution_time,
                "benchmark_price": benchmark_price,
                "fill_price": fill_price
            }
            
        except Exception as e:
            logger.error(f"Slippage metrics calculation error: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_vwap_slippage(self, order: Order) -> Optional[float]:
        """Calculate VWAP-based slippage."""
        try:
            # This would calculate volume-weighted average price for the execution period
            # For now, return None as it's complex to implement without market data
            
            return None
            
        except Exception as e:
            logger.error(f"VWAP slippage calculation error: {str(e)}")
            return None
    
    async def _get_mid_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get mid price (bid + ask) / 2 for symbol at timestamp."""
        try:
            # This would query market data for bid/ask prices
            # For now, return a placeholder based on order limit price
            
            # Cache the mid price for this symbol
            cache_key = f"{symbol}_{timestamp.hour}"
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]
            
            # Placeholder implementation - would get real market data
            estimated_price = 100.0  # Default price
            self.market_data_cache[cache_key] = estimated_price
            
            return estimated_price
            
        except Exception as e:
            logger.error(f"Mid price retrieval error: {str(e)}")
            return None
    
    async def _get_market_conditions(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Get market conditions at specific time."""
        try:
            # This would analyze market volatility, volume, spreads, etc.
            # For now, return simplified conditions
            
            return {
                "volatility": 0.02,  # 2% daily volatility (placeholder)
                "volume_24h": 1000000,  # Placeholder volume
                "bid_ask_spread": 0.01,  # 1 cent spread (placeholder)
                "market_cap": 1000000000,  # Placeholder market cap
                "sector": "technology"  # Placeholder sector
            }
            
        except Exception as e:
            logger.error(f"Market conditions retrieval error: {str(e)}")
            return {}
    
    async def _calculate_execution_quality_score(self, slippage_metrics: Dict[str, Any], 
                                               market_conditions: Dict[str, Any]) -> float:
        """Calculate overall execution quality score (0-100)."""
        try:
            slippage_bps = abs(slippage_metrics.get("percentage_bps", 0))
            execution_time = slippage_metrics.get("execution_time_seconds", 0)
            
            # Base score
            score = 100.0
            
            # Deduct for slippage (exponential penalty for high slippage)
            if slippage_bps > 0:
                slippage_penalty = min(slippage_bps / 10, 50)  # Max 50 point penalty
                score -= slippage_penalty
            
            # Deduct for slow execution
            if execution_time > 300:  # 5 minutes
                time_penalty = min((execution_time - 300) / 10, 20)  # Max 20 point penalty
                score -= time_penalty
            
            # Adjust for market conditions
            volatility = market_conditions.get("volatility", 0.02)
            if volatility > 0.05:  # High volatility
                score -= 5  # Adjustment for difficult market conditions
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {str(e)}")
            return 50.0  # Default score
    
    async def _calculate_aggregate_metrics(self, slippage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics from slippage results."""
        try:
            if not slippage_results:
                return {}
            
            slippage_values = [r["slippage"]["percentage"] for r in slippage_results]
            execution_times = [r["slippage"]["execution_time_seconds"] for r in slippage_results]
            
            return {
                "count": len(slippage_results),
                "avg_percentage_slippage": np.mean(slippage_values),
                "median_percentage_slippage": np.median(slippage_values),
                "std_percentage_slippage": np.std(slippage_values),
                "min_percentage_slippage": np.min(slippage_values),
                "max_percentage_slippage": np.max(slippage_values),
                "avg_execution_time": np.mean(execution_times),
                "median_execution_time": np.median(execution_times),
                "percentile_25": np.percentile(slippage_values, 25),
                "percentile_75": np.percentile(slippage_values, 75),
                "percentile_90": np.percentile(slippage_values, 90),
                "percentile_95": np.percentile(slippage_values, 95)
            }
            
        except Exception as e:
            logger.error(f"Aggregate metrics calculation error: {str(e)}")
            return {}
    
    async def _analyze_by_broker(self, slippage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze slippage by broker."""
        try:
            broker_data = {}
            
            for result in slippage_results:
                broker = result["broker_name"]
                if broker not in broker_data:
                    broker_data[broker] = []
                broker_data[broker].append(result["slippage"]["percentage"])
            
            analysis = {}
            for broker, slippage_values in broker_data.items():
                analysis[broker] = {
                    "count": len(slippage_values),
                    "avg_slippage": np.mean(slippage_values),
                    "median_slippage": np.median(slippage_values),
                    "std_slippage": np.std(slippage_values)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Broker analysis error: {str(e)}")
            return {}
    
    async def _analyze_by_order_type(self, slippage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze slippage by order type."""
        try:
            order_type_data = {}
            
            for result in slippage_results:
                order_type = result["order_type"]
                if order_type not in order_type_data:
                    order_type_data[order_type] = []
                order_type_data[order_type].append(result["slippage"]["percentage"])
            
            analysis = {}
            for order_type, slippage_values in order_type_data.items():
                analysis[order_type] = {
                    "count": len(slippage_values),
                    "avg_slippage": np.mean(slippage_values),
                    "median_slippage": np.median(slippage_values),
                    "std_slippage": np.std(slippage_values)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Order type analysis error: {str(e)}")
            return {}
    
    async def _analyze_by_symbol(self, slippage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze slippage by symbol."""
        try:
            symbol_data = {}
            
            for result in slippage_results:
                symbol = result["symbol"]
                if symbol not in symbol_data:
                    symbol_data[symbol] = []
                symbol_data[symbol].append(result["slippage"]["percentage"])
            
            analysis = {}
            for symbol, slippage_values in symbol_data.items():
                analysis[symbol] = {
                    "count": len(slippage_values),
                    "avg_slippage": np.mean(slippage_values),
                    "median_slippage": np.median(slippage_values),
                    "std_slippage": np.std(slippage_values),
                    "total_volume": sum(r["quantity"] for r in slippage_results if r["symbol"] == symbol)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Symbol analysis error: {str(e)}")
            return {}
    
    async def _analyze_by_time(self, slippage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze slippage by time patterns."""
        try:
            hourly_data = {}
            daily_data = {}
            
            for result in slippage_results:
                # Get execution timestamp (fallback to submission if filled_at is None)
                order_id = result["order_id"]
                order = next((o for o in self.db.query(Order).filter(Order.id == order_id).all() if o.id == order_id), None)
                
                if not order or not order.filled_at:
                    continue
                
                timestamp = order.filled_at
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(result["slippage"]["percentage"])
                
                if day_of_week not in daily_data:
                    daily_data[day_of_week] = []
                daily_data[day_of_week].append(result["slippage"]["percentage"])
            
            # Calculate hourly patterns
            hourly_analysis = {}
            for hour, slippage_values in hourly_data.items():
                hourly_analysis[hour] = {
                    "count": len(slippage_values),
                    "avg_slippage": np.mean(slippage_values),
                    "std_slippage": np.std(slippage_values)
                }
            
            # Calculate daily patterns
            daily_analysis = {}
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for day_of_week, slippage_values in daily_data.items():
                daily_analysis[day_names[day_of_week]] = {
                    "count": len(slippage_values),
                    "avg_slippage": np.mean(slippage_values),
                    "std_slippage": np.std(slippage_values)
                }
            
            return {
                "hourly_patterns": hourly_analysis,
                "daily_patterns": daily_analysis
            }
            
        except Exception as e:
            logger.error(f"Time analysis error: {str(e)}")
            return {}
    
    async def _calculate_broker_quality_score(self, broker_data: Dict, slippage_data: List[float]) -> float:
        """Calculate overall quality score for broker."""
        try:
            if not slippage_data:
                return 50.0  # Neutral score
            
            # Lower average slippage is better
            avg_slippage = np.mean(slippage_data)
            slippage_score = max(0, 100 - abs(avg_slippage) * 10000)  # Convert to reasonable scale
            
            # Execution speed score (if available)
            orders = broker_data["orders"]
            execution_times = [
                (order.filled_at - order.submitted_at).total_seconds()
                for order in orders if order.filled_at
            ]
            
            speed_score = 100
            if execution_times:
                avg_time = np.mean(execution_times)
                if avg_time > 60:  # > 1 minute
                    speed_score = max(0, 100 - (avg_time - 60))
            
            # Combined score
            quality_score = (slippage_score * 0.7) + (speed_score * 0.3)
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.error(f"Broker quality score error: {str(e)}")
            return 50.0
    
    async def _get_market_data_snapshot(self, symbol: str, timestamp: datetime, before: bool = True) -> Optional[Dict[str, Any]]:
        """Get market data snapshot before or after timestamp."""
        try:
            # This would query historical market data
            # For now, return placeholder data
            
            return {
                "price": 100.0,
                "volume": 1000000,
                "bid": 99.99,
                "ask": 100.01,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Market data snapshot error: {str(e)}")
            return None
    
    def _calculate_price_impact(self, pre_trade: Dict[str, Any], post_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate price impact from pre to post trade."""
        try:
            pre_price = pre_trade.get("price", 0)
            post_price = post_trade.get("price", 0)
            
            if pre_price == 0:
                return {"impact": 0, "impact_percentage": 0}
            
            price_impact = post_price - pre_price
            impact_percentage = price_impact / pre_price
            
            return {
                "impact": price_impact,
                "impact_percentage": impact_percentage,
                "impact_bps": impact_percentage * 10000
            }
            
        except Exception as e:
            logger.error(f"Price impact calculation error: {str(e)}")
            return {"impact": 0, "impact_percentage": 0}
    
    def _calculate_volume_impact(self, order: Order, pre_trade: Dict[str, Any], post_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volume impact."""
        try:
            pre_volume = pre_trade.get("volume", 0)
            post_volume = post_trade.get("volume", 0)
            
            # Volume impact as percentage of order size
            volume_impact = order.quantity / max(pre_volume, 1)
            
            return {
                "volume_ratio": volume_impact,
                "is_material": volume_impact > 0.01  # 1% threshold
            }
            
        except Exception as e:
            logger.error(f"Volume impact calculation error: {str(e)}")
            return {"volume_ratio": 0, "is_material": False}
    
    async def _estimate_market_impact(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate market impact based on order characteristics."""
        try:
            # Simplified market impact estimation
            # Real implementation would use sophisticated models
            
            quantity = order.quantity
            order_type = order.order_type.value
            
            # Base impact rate (percentage)
            base_impact = 0.001  # 0.1%
            
            # Adjust for order type
            type_multipliers = {
                "market": 2.0,  # Higher impact
                "limit": 0.5,   # Lower impact
                "stop": 1.5     # Medium impact
            }
            
            multiplier = type_multipliers.get(order_type, 1.0)
            estimated_impact = base_impact * multiplier
            
            return {
                "estimated_impact_percentage": estimated_impact,
                "estimated_impact_bps": estimated_impact * 10000,
                "confidence": 0.5  # Low confidence for simplified model
            }
            
        except Exception as e:
            logger.error(f"Market impact estimation error: {str(e)}")
            return {"estimated_impact_percentage": 0, "confidence": 0}
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
