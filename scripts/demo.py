#!/usr/bin/env python3
"""
Day Trading Orchestrator - Interactive Demo
Demonstrates system features without requiring real broker connections
"""

import asyncio
import sys
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

class TradingDemo:
    """Interactive trading demonstration system"""
    
    def __init__(self):
        self.demo_portfolio = {
            "cash": 100000.0,
            "positions": {},
            "orders": [],
            "trades": [],
            "pnl": 0.0
        }
        
        self.market_data = {
            "AAPL": {"price": 150.25, "change": 1.5, "volume": 50000000},
            "MSFT": {"price": 299.80, "change": -0.8, "volume": 30000000},
            "GOOGL": {"price": 125.40, "change": 2.1, "volume": 25000000},
            "AMZN": {"price": 3180.50, "change": -15.2, "volume": 15000000},
            "TSLA": {"price": 220.80, "change": 8.5, "volume": 45000000},
            "BTC-USD": {"price": 45200.0, "change": 1200.0, "volume": 5000000000},
            "ETH-USD": {"price": 2850.0, "change": -45.0, "volume": 2000000000}
        }
        
        self.demo_strategies = [
            {
                "name": "Mean Reversion",
                "description": "Buy oversold, sell overbought assets",
                "active": True,
                "pnl": 1250.75,
                "trades": 15,
                "win_rate": 73.3
            },
            {
                "name": "Trend Following",
                "description": "Follow momentum in trending markets",
                "active": True,
                "pnl": 890.25,
                "trades": 22,
                "win_rate": 68.2
            },
            {
                "name": "Pairs Trading",
                "description": "Trade correlated asset pairs",
                "active": False,
                "pnl": 456.50,
                "trades": 8,
                "win_rate": 75.0
            }
        ]
        
        self.running = False
        
    def print_banner(self):
        """Print the demo banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           DAY TRADING ORCHESTRATOR DEMO                     ║
    ║                 Interactive Showcase                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🎮 Welcome to the Interactive Demo!
    📊 Explore all features without risking real money
    🤖 See AI-powered trading in action
    ⚡ Experience real-time market simulation
    🛡️  Learn about risk management features
    """
        print(banner)
    
    def print_portfolio(self):
        """Print current portfolio status"""
        print("\n" + "="*60)
        print("📊 PORTFOLIO STATUS")
        print("="*60)
        
        # Portfolio summary
        total_value = self.demo_portfolio["cash"]
        unrealized_pnl = 0
        
        for symbol, position in self.demo_portfolio["positions"].items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol]["price"]
                position_value = position["quantity"] * current_price
                unrealized_pnl += position_value - position["cost"]
                total_value += position_value
        
        print(f"💰 Cash:           ${self.demo_portfolio['cash']:,.2f}")
        print(f"📈 Market Value:   ${total_value:,.2f}")
        print(f"📊 Unrealized P&L: ${unrealized_pnl:+,.2f}")
        print(f"💼 Total Trades:   {len(self.demo_portfolio['trades'])}")
        
        # Positions
        if self.demo_portfolio["positions"]:
            print("\n📋 OPEN POSITIONS:")
            print("-" * 60)
            print(f"{'Symbol':<10} {'Quantity':<10} {'Avg Cost':<12} {'Current':<12} {'P&L':<12}")
            print("-" * 60)
            
            for symbol, position in self.demo_portfolio["positions"].items():
                if symbol in self.market_data:
                    current_price = self.market_data[symbol]["price"]
                    avg_cost = position["cost"] / position["quantity"]
                    pnl = (current_price - avg_cost) * position["quantity"]
                    
                    print(f"{symbol:<10} {position['quantity']:<10} "
                          f"${avg_cost:<11.2f} ${current_price:<11.2f} "
                          f"${pnl:+,.2f}")
    
    def print_market_data(self):
        """Print current market data"""
        print("\n" + "="*60)
        print("📈 MARKET DATA")
        print("="*60)
        
        print(f"{'Symbol':<10} {'Price':<12} {'Change':<12} {'Volume':<15}")
        print("-" * 60)
        
        for symbol, data in self.market_data.items():
            change_indicator = "↑" if data["change"] >= 0 else "↓"
            change_str = f"{change_indicator} ${abs(data['change']):.2f}"
            
            print(f"{symbol:<10} ${data['price']:<11.2f} "
                  f"{change_str:<12} {data['volume']:,<15}")
    
    def print_strategies(self):
        """Print strategy performance"""
        print("\n" + "="*60)
        print("🧠 ACTIVE STRATEGIES")
        print("="*60)
        
        for strategy in self.demo_strategies:
            status = "🟢 ACTIVE" if strategy["active"] else "🔴 INACTIVE"
            print(f"\n{status} {strategy['name']}")
            print(f"   {strategy['description']}")
            print(f"   P&L: ${strategy['pnl']:+,.2f} | "
                  f"Trades: {strategy['trades']} | "
                  f"Win Rate: {strategy['win_rate']:.1f}%")
    
    def print_risk_metrics(self):
        """Print risk management metrics"""
        print("\n" + "="*60)
        print("🛡️  RISK METRICS")
        print("="*60)
        
        # Calculate risk metrics
        total_value = self.demo_portfolio["cash"] + sum(
            pos["cost"] for pos in self.demo_portfolio["positions"].values()
        )
        
        # Position concentration
        max_position_value = 0
        for symbol, position in self.demo_portfolio["positions"].items():
            if symbol in self.market_data:
                position_value = position["quantity"] * self.market_data[symbol]["price"]
                max_position_value = max(max_position_value, position_value)
        
        concentration = (max_position_value / total_value * 100) if total_value > 0 else 0
        
        print(f"💼 Portfolio Value:  ${total_value:,.2f}")
        print(f"📊 Max Position:     {concentration:.1f}% (max: 10%)")
        print(f"⚡ VaR (95%):        ${total_value * 0.02:,.2f}")
        print(f"🔥 Portfolio Heat:   {concentration:.1f}% (max: 25%)")
        print(f"📉 Daily P&L:        ${self.demo_portfolio['pnl']:+,.2f}")
        print(f"🔒 Circuit Breaker:  READY")
    
    def print_ai_insights(self):
        """Print AI-generated market insights"""
        print("\n" + "="*60)
        print("🤖 AI MARKET INSIGHTS")
        print("="*60)
        
        insights = [
            {
                "symbol": "AAPL",
                "sentiment": "BULLISH",
                "confidence": 85,
                "reasoning": "Strong earnings beat, positive guidance, institutional buying"
            },
            {
                "symbol": "BTC-USD",
                "sentiment": "NEUTRAL",
                "confidence": 62,
                "reasoning": "Mixed signals, await breakout confirmation"
            },
            {
                "symbol": "TSLA",
                "sentiment": "BULLISH",
                "confidence": 78,
                "reasoning": "Production ramp-up, positive delivery numbers"
            }
        ]
        
        for insight in insights:
            emoji = "🟢" if insight["sentiment"] == "BULLISH" else "🔴" if insight["sentiment"] == "BEARISH" else "🟡"
            print(f"\n{emoji} {insight['symbol']}: {insight['sentiment']} ({insight['confidence']}%)")
            print(f"   {insight['reasoning']}")
    
    def simulate_order_execution(self, symbol: str, side: str, quantity: int, order_type: str = "market"):
        """Simulate order execution"""
        if symbol not in self.market_data:
            return False, f"Symbol {symbol} not available"
        
        current_price = self.market_data[symbol]["price"]
        
        # Simulate slippage for market orders
        if order_type == "market":
            slippage = random.uniform(-0.01, 0.01) * current_price
            execution_price = current_price + slippage
        else:
            execution_price = current_price  # Simplified for demo
        
        total_cost = execution_price * quantity
        
        if side.lower() == "buy":
            if total_cost > self.demo_portfolio["cash"]:
                return False, "Insufficient cash"
            
            # Update portfolio
            self.demo_portfolio["cash"] -= total_cost
            
            if symbol in self.demo_portfolio["positions"]:
                # Add to existing position
                old_quantity = self.demo_portfolio["positions"][symbol]["quantity"]
                old_cost = self.demo_portfolio["positions"][symbol]["cost"]
                
                new_quantity = old_quantity + quantity
                new_cost = old_cost + total_cost
                
                self.demo_portfolio["positions"][symbol] = {
                    "quantity": new_quantity,
                    "cost": new_cost
                }
            else:
                # New position
                self.demo_portfolio["positions"][symbol] = {
                    "quantity": quantity,
                    "cost": total_cost
                }
            
            message = f"Bought {quantity} {symbol} @ ${execution_price:.2f}"
            
        else:  # sell
            if symbol not in self.demo_portfolio["positions"]:
                return False, f"No position in {symbol}"
            
            position = self.demo_portfolio["positions"][symbol]
            if quantity > position["quantity"]:
                return False, "Insufficient shares"
            
            # Calculate P&L
            avg_cost = position["cost"] / position["quantity"]
            pnl = (execution_price - avg_cost) * quantity
            self.demo_portfolio["pnl"] += pnl
            
            # Update portfolio
            self.demo_portfolio["cash"] += total_cost
            
            if quantity == position["quantity"]:
                # Close position
                del self.demo_portfolio["positions"][symbol]
            else:
                # Partial position
                position["quantity"] -= quantity
                position["cost"] = avg_cost * position["quantity"]
            
            message = f"Sold {quantity} {symbol} @ ${execution_price:.2f} (P&L: ${pnl:+.2f})"
        
        # Record trade
        trade = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": execution_price,
            "timestamp": datetime.now().isoformat(),
            "type": order_type
        }
        self.demo_portfolio["trades"].append(trade)
        
        return True, message
    
    def update_market_data(self):
        """Simulate real-time market data updates"""
        for symbol, data in self.market_data.items():
            # Random price movement
            change_percent = random.uniform(-0.02, 0.02)  # ±2%
            price_change = data["price"] * change_percent
            
            # Apply realistic bounds
            min_price = data["price"] * 0.5
            max_price = data["price"] * 2.0
            new_price = max(min_price, min(max_price, data["price"] + price_change))
            
            data["price"] = round(new_price, 2)
            
            # Update change
            data["change"] = round(data["price"] - (data["price"] - data["change"]), 2)
            
            # Update volume (simulate activity)
            volume_change = random.randint(-500000, 1500000)
            data["volume"] = max(1000000, data["volume"] + volume_change)
    
    async def run_demo(self):
        """Run the interactive demonstration"""
        self.print_banner()
        
        print("\n🎯 Demo Features:")
        print("  • Portfolio simulation with real-time P&L")
        print("  • Market data simulation")
        print("  • AI market insights")
        print("  • Strategy performance tracking")
        print("  • Risk management visualization")
        print("  • Order execution simulation")
        
        print("\n📝 Demo Commands:")
        print("  portfolio  - Show portfolio status")
        print("  market     - Show market data")
        print("  strategies - Show strategy performance")
        print("  risk       - Show risk metrics")
        print("  ai         - Show AI insights")
        print("  buy <symbol> <qty> - Place buy order")
        print("  sell <symbol> <qty> - Place sell order")
        print("  simulate   - Run automated simulation")
        print("  quit       - Exit demo")
        
        self.running = True
        
        while self.running:
            try:
                # Update market data periodically
                self.update_market_data()
                
                # Get user input
                command = input("\n> ").strip().lower()
                
                if not command:
                    continue
                
                # Process commands
                if command == "quit" or command == "exit":
                    self.running = False
                    break
                
                elif command == "portfolio":
                    self.print_portfolio()
                
                elif command == "market":
                    self.print_market_data()
                
                elif command == "strategies":
                    self.print_strategies()
                
                elif command == "risk":
                    self.print_risk_metrics()
                
                elif command == "ai":
                    self.print_ai_insights()
                
                elif command.startswith("buy "):
                    parts = command.split()
                    if len(parts) >= 3:
                        try:
                            symbol = parts[1].upper()
                            quantity = int(parts[2])
                            success, message = self.simulate_order_execution(symbol, "buy", quantity)
                            print(f"📈 {message}")
                        except ValueError:
                            print("❌ Invalid quantity")
                        except Exception as e:
                            print(f"❌ Order failed: {e}")
                    else:
                        print("❌ Usage: buy <symbol> <quantity>")
                
                elif command.startswith("sell "):
                    parts = command.split()
                    if len(parts) >= 3:
                        try:
                            symbol = parts[1].upper()
                            quantity = int(parts[2])
                            success, message = self.simulate_order_execution(symbol, "sell", quantity)
                            print(f"📉 {message}")
                        except ValueError:
                            print("❌ Invalid quantity")
                        except Exception as e:
                            print(f"❌ Order failed: {e}")
                    else:
                        print("❌ Usage: sell <symbol> <quantity>")
                
                elif command == "simulate":
                    await self.run_automated_simulation()
                
                elif command == "help":
                    print("\n📖 Available Commands:")
                    print("  portfolio  - Display portfolio status")
                    print("  market     - Show current market data")
                    print("  strategies - View active trading strategies")
                    print("  risk       - Show risk management metrics")
                    print("  ai         - Display AI market analysis")
                    print("  buy <sym> <qty> - Place buy order (e.g., 'buy AAPL 10')")
                    print("  sell <sym> <qty> - Place sell order (e.g., 'sell TSLA 5')")
                    print("  simulate   - Run automated trading simulation")
                    print("  quit       - Exit the demo")
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Demo interrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Thank you for trying the Day Trading Orchestrator Demo!")
        print("🚀 Ready to start trading with real brokers? Check the installation guide.")
    
    async def run_automated_simulation(self):
        """Run an automated trading simulation"""
        print("\n🤖 Starting Automated Trading Simulation...")
        print("Simulating 30 seconds of trading activity...")
        
        symbols = list(self.market_data.keys())
        total_trades = 0
        
        for i in range(30):  # 30 seconds
            # Randomly decide to trade
            if random.random() < 0.3:  # 30% chance of trade
                symbol = random.choice(symbols)
                side = "buy" if random.random() < 0.6 else "sell"
                quantity = random.randint(1, 10)
                
                success, message = self.simulate_order_execution(symbol, side, quantity)
                if success:
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - {message}")
                    total_trades += 1
            
            # Update market data
            self.update_market_data()
            
            # Show periodic updates
            if (i + 1) % 5 == 0:
                current_value = self.demo_portfolio["cash"]
                for sym, pos in self.demo_portfolio["positions"].items():
                    if sym in self.market_data:
                        current_value += pos["quantity"] * self.market_data[sym]["price"]
                
                print(f"📊 Portfolio Value: ${current_value:,.2f} | "
                      f"Cash: ${self.demo_portfolio['cash']:,.2f} | "
                      f"Trades: {total_trades}")
            
            await asyncio.sleep(1)  # Wait 1 second
        
        print(f"\n✅ Simulation complete! Executed {total_trades} trades")
        self.print_portfolio()

async def main():
    """Main demo runner"""
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Interactive Demo")
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Run non-interactive demo mode")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = TradingDemo()
    
    if args.no_interactive:
        # Run automated demo
        print("🎮 Running Automated Demo...")
        await demo.run_automated_simulation()
    else:
        # Run interactive demo
        await demo.run_demo()

if __name__ == "__main__":
    import argparse
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)