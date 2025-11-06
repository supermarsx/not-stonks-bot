import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore, ChatMessage } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Send, 
  Bot, 
  User, 
  Brain, 
  TrendingUp, 
  TrendingDown,
  Shield,
  Lightbulb,
  AlertCircle,
  Activity,
  MessageSquare,
  Clock
} from 'lucide-react';

export const AIChat: React.FC = () => {
  const { chatMessages, isChatLoading, addChatMessage, setChatLoading, portfolio, strategies } = useTradingStore();
  const [inputMessage, setInputMessage] = useState('');
  const [currentTopic, setCurrentTopic] = useState('general');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  // Initialize with welcome message
  useEffect(() => {
    if (chatMessages.length === 0) {
      const welcomeMessage: ChatMessage = {
        id: 'welcome',
        type: 'assistant',
        content: 'Welcome to MATRIX Trading AI Assistant. I can help you analyze market conditions, review portfolio performance, configure strategies, and provide trading insights. How can I assist you today?',
        timestamp: Date.now()
      };
      addChatMessage(welcomeMessage);
    }
  }, [chatMessages.length, addChatMessage]);

  const quickPrompts = {
    general: [
      'Analyze current market conditions',
      'Review portfolio performance',
      'Identify trading opportunities',
      'Risk assessment recommendations'
    ],
    portfolio: [
      'What is my current P&L?',
      'Analyze position concentrations',
      'Review position performance',
      'Risk-adjusted returns'
    ],
    strategies: [
      'Strategy performance overview',
      'Optimize strategy parameters',
      'Strategy allocation recommendations',
      'Backtesting insights'
    ],
    risk: [
      'Current risk metrics',
      'Portfolio stress testing',
      'Position size recommendations',
      'Risk limit violations'
    ]
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isChatLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: Date.now()
    };

    addChatMessage(userMessage);
    setInputMessage('');
    setChatLoading(true);

    // Simulate AI processing delay
    setTimeout(() => {
      const aiResponse = generateAIResponse(inputMessage);
      addChatMessage(aiResponse);
      setChatLoading(false);
    }, 1500);
  };

  const generateAIResponse = (userInput: string): ChatMessage => {
    const input = userInput.toLowerCase();
    
    // Portfolio-related queries
    if (input.includes('portfolio') || input.includes('p&l') || input.includes('profit')) {
      const totalPnL = portfolio.totalPnL;
      const winRate = portfolio.positions.length > 0 ? 
        portfolio.positions.filter(p => p.pnl > 0).length / portfolio.positions.length * 100 : 0;
      
      return {
        id: Date.now().toString(),
        type: 'assistant',
        content: `PORTFOLIO ANALYSIS:\n\n📊 Total P&L: $${totalPnL.toFixed(2)} (${((totalPnL / portfolio.totalValue) * 100).toFixed(2)}%)\n\n💰 Portfolio Value: $${portfolio.totalValue.toLocaleString()}\n💵 Available Cash: $${portfolio.availableCash.toLocaleString()}\n\n🎯 Win Rate: ${winRate.toFixed(1)}% (${portfolio.positions.filter(p => p.pnl > 0).length}/${portfolio.positions.length} positions)\n\n${totalPnL > 0 ? '✅ Portfolio performing well above market average.' : '⚠️ Portfolio underperforming. Consider rebalancing positions.'}`,
        timestamp: Date.now()
      };
    }

    // Risk-related queries
    if (input.includes('risk') || input.includes('var') || input.includes('drawdown')) {
      return {
        id: Date.now().toString(),
        type: 'assistant',
        content: `RISK ASSESSMENT:\n\n⚡ Value at Risk (95%): $2,500 potential loss in worst 5% of scenarios\n\n📉 Maximum Drawdown: -3.2% (within acceptable limits)\n\n🎯 Risk Utilization: 65% (moderate risk usage)\n\n📊 Sharpe Ratio: 1.8 (good risk-adjusted returns)\n\n🛡️ Recommendation: Current risk levels are appropriate. Consider reducing exposure if drawdown exceeds 5%.`,
        timestamp: Date.now()
      };
    }

    // Strategy-related queries
    if (input.includes('strategy') || input.includes('strategies')) {
      const activeStrategies = strategies.filter(s => s.status === 'running').length;
      const totalPnL = strategies.reduce((sum, s) => sum + s.performance.totalPnL, 0);
      
      return {
        id: Date.now().toString(),
        type: 'assistant',
        content: `STRATEGY OVERVIEW:\n\n🤖 Active Strategies: ${activeStrategies}/${strategies.length}\n\n💰 Total Strategy P&L: $${totalPnL.toFixed(2)}\n\n${strategies.map(s => `📈 ${s.name}: ${s.performance.totalPnL > 0 ? '+' : ''}$${s.performance.totalPnL.toFixed(2)} (${s.performance.winRate.toFixed(1)}% win rate)`).join('\n')}\n\n🎯 Recommendation: ${activeStrategies > 2 ? 'Consider consolidating to focus resources.' : 'Good diversification across strategies.'}`,
        timestamp: Date.now()
      };
    }

    // Market analysis
    if (input.includes('market') || input.includes('analysis') || input.includes('trend')) {
      return {
        id: Date.now().toString(),
        type: 'assistant',
        content: `MARKET ANALYSIS:\n\n📊 Current Market Sentiment: MIXED\n\n📈 Tech Sector: Showing strength with +1.2% average gains\n📉 Volatility Index (VIX): 18.5 (moderate)\n\n🔍 Key Observations:\n• Strong institutional buying in mega-cap tech\n• Increased options activity suggesting potential volatility\n• Treasury yields stabilizing\n\n💡 Trading Recommendations:\n• Consider momentum strategies on strong tech names\n• Monitor earnings announcements this week\n• Prepare for Fed commentary impact\n\n🎯 Market Outlook: Neutral to slightly bullish near-term`,
        timestamp: Date.now()
      };
    }

    // General help
    return {
      id: Date.now().toString(),
      type: 'assistant',
      content: `MATRIX AI ASSISTANT READY\n\nI can help you with:\n\n📊 Portfolio Analysis - Review positions, P&L, and performance metrics\n🤖 Strategy Management - Optimize trading algorithms and parameters\n⚡ Risk Assessment - Monitor VaR, drawdown, and position limits\n📈 Market Insights - Analyze trends, news, and trading opportunities\n🔧 System Configuration - Set up alerts and automation\n\nWhat specific area would you like to explore?`,
      timestamp: Date.now()
    };
  };

  const handleQuickPrompt = (prompt: string) => {
    setInputMessage(prompt);
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400">
            AI TRADING ASSISTANT
          </h1>
          <p className="text-green-600 mt-1">Intelligent insights for your trading operations</p>
        </div>
        <div className="flex items-center gap-2">
          <Brain className="w-6 h-6 text-green-400 animate-pulse" />
          <span className="text-green-400 font-mono">ONLINE</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-3 space-y-4">
          <MatrixCard title="Trading Chat" glow className="h-96 flex flex-col">
            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto space-y-4 p-4 bg-black/30 rounded border border-green-800/30">
              <AnimatePresence>
                {chatMessages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      message.type === 'user' ? 'bg-green-600' : 'bg-blue-600'
                    }`}>
                      {message.type === 'user' ? (
                        <User className="w-4 h-4 text-white" />
                      ) : (
                        <Bot className="w-4 h-4 text-white" />
                      )}
                    </div>
                    
                    <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                      <div className={`p-3 rounded-lg ${
                        message.type === 'user' 
                          ? 'bg-green-600 text-white ml-auto' 
                          : 'bg-gray-800 text-green-400'
                      }`}>
                        <pre className="whitespace-pre-wrap text-sm font-mono leading-relaxed">
                          {message.content}
                        </pre>
                      </div>
                      <div className={`text-xs text-green-600 mt-1 ${
                        message.type === 'user' ? 'text-right' : ''
                      }`}>
                        {formatTimestamp(message.timestamp)}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {/* Loading Indicator */}
              {isChatLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-gray-800 text-green-400 p-3 rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm">AI is analyzing...</span>
                    </div>
                  </div>
                </motion.div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Message Input */}
            <div className="mt-4">
              <div className="flex gap-2">
                <MatrixInput
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask about your portfolio, strategies, or market conditions..."
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  disabled={isChatLoading}
                  className="flex-1"
                />
                <MatrixButton 
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isChatLoading}
                  className="px-6"
                >
                  <Send className="w-4 h-4" />
                </MatrixButton>
              </div>
            </div>
          </MatrixCard>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Quick Actions */}
          <MatrixCard title="Quick Actions" glow>
            <div className="space-y-2">
              {quickPrompts.general.map((prompt, index) => (
                <MatrixButton
                  key={index}
                  variant="secondary"
                  size="sm"
                  onClick={() => handleQuickPrompt(prompt)}
                  className="w-full text-left justify-start text-xs"
                  disabled={isChatLoading}
                >
                  <Lightbulb className="w-3 h-3 mr-2" />
                  {prompt}
                </MatrixButton>
              ))}
            </div>
          </MatrixCard>

          {/* Topic Categories */}
          <MatrixCard title="Topics" glow>
            <div className="space-y-2">
              {Object.entries(quickPrompts).map(([topic, prompts]) => (
                <div key={topic}>
                  <button
                    onClick={() => setCurrentTopic(topic)}
                    className={`w-full text-left p-2 rounded transition-colors ${
                      currentTopic === topic ? 'bg-green-800/50 text-green-400' : 'text-green-600 hover:bg-green-800/20'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {topic === 'portfolio' && <TrendingUp className="w-4 h-4" />}
                      {topic === 'strategies' && <Brain className="w-4 h-4" />}
                      {topic === 'risk' && <Shield className="w-4 h-4" />}
                      {topic === 'general' && <MessageSquare className="w-4 h-4" />}
                      <span className="capitalize font-mono">{topic}</span>
                    </div>
                  </button>
                  
                  {currentTopic === topic && (
                    <div className="ml-6 mt-2 space-y-1">
                      {prompts.map((prompt, index) => (
                        <button
                          key={index}
                          onClick={() => handleQuickPrompt(prompt)}
                          className="block w-full text-left text-xs text-green-600 hover:text-green-400 transition-colors py-1"
                        >
                          • {prompt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </MatrixCard>

          {/* AI Status */}
          <MatrixCard title="AI Status" glow>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-green-600">Status</span>
                <span className="text-green-400 font-mono">ONLINE</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-green-600">Response Time</span>
                <span className="text-green-400 font-mono">1.2s</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-green-600">Knowledge Base</span>
                <span className="text-green-400 font-mono">v2.1</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-green-600">Messages</span>
                <span className="text-green-400 font-mono">{chatMessages.length}</span>
              </div>
            </div>
          </MatrixCard>

          {/* System Insights */}
          <MatrixCard title="Insights" glow>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <Activity className="w-3 h-3 text-green-400" />
                <span className="text-green-600">Market volatility: Moderate</span>
              </div>
              <div className="flex items-center gap-2">
                <AlertCircle className="w-3 h-3 text-yellow-400" />
                <span className="text-green-600">3 position alerts</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="w-3 h-3 text-blue-400" />
                <span className="text-green-600">Next earnings: 2 days</span>
              </div>
            </div>
          </MatrixCard>
        </div>
      </div>
    </div>
  );
};