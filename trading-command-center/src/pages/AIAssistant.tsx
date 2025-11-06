import { Send, Bot, User } from 'lucide-react';
import { useState } from 'react';
import { MatrixCard } from '../components/MatrixCard';
import { GlowingButton } from '../components/GlowingButton';
import toast from 'react-hot-toast';

export default function AIAssistant() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'AI Trading Assistant online. I can help you with market analysis, strategy recommendations, risk assessment, and trade execution. How can I assist you today?',
      timestamp: new Date().toISOString(),
    },
  ]);

  const sendMessage = () => {
    if (!input.trim()) return;

    const userMessage = {
      role: 'user' as const,
      content: input,
      timestamp: new Date().toISOString(),
    };

    const aiResponse = {
      role: 'assistant' as const,
      content: generateAIResponse(input),
      timestamp: new Date().toISOString(),
    };

    setMessages([...messages, userMessage, aiResponse]);
    setInput('');
  };

  const generateAIResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('market') || lowerQuery.includes('analysis')) {
      return 'Based on current market conditions, I see strong momentum in tech stocks (AAPL, NVDA) with positive technical indicators. Would you like me to perform a detailed technical analysis on any specific symbol?';
    }

    if (lowerQuery.includes('risk')) {
      return 'Your current portfolio risk metrics look healthy. Sharpe ratio of 1.85 indicates good risk-adjusted returns. Current drawdown is 2.3%, well within the 5% limit. Position concentration is moderate at 0.23. Would you like recommendations to optimize risk exposure?';
    }

    if (lowerQuery.includes('strategy') || lowerQuery.includes('recommend')) {
      return 'I recommend focusing on the Momentum Swing strategy which has a 67.8% win rate. Mean Reversion strategy is also performing well at 71.1% win rate. Consider pausing the Pairs Trading strategy until market volatility decreases. Would you like me to backtest a new strategy?';
    }

    if (lowerQuery.includes('buy') || lowerQuery.includes('sell') || lowerQuery.includes('trade')) {
      return 'I can help execute trades. Please specify: symbol, quantity, order type (market/limit), and broker. For example: "Buy 100 AAPL at market on Alpaca". I will prepare the order for your review before submission.';
    }

    return 'I understand your query. I can help with: (1) Market analysis and technical indicators, (2) Risk assessment and portfolio optimization, (3) Strategy recommendations and backtesting, (4) Trade execution assistance. What would you like to explore?';
  };

  const quickActions = [
    'Analyze current market trends',
    'Check portfolio risk metrics',
    'Recommend trading strategies',
    'Show top performing positions',
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold matrix-glow-text">AI TRADING ASSISTANT</h1>
        <p className="mt-1 text-sm text-matrix-green/70">
          Powered by GPT-4 and Claude Sonnet for advanced market analysis
        </p>
      </div>

      {/* Chat Interface */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <MatrixCard title="CONVERSATION" glow noPadding>
            <div className="flex h-[600px] flex-col">
              {/* Messages */}
              <div className="flex-1 space-y-4 overflow-y-auto p-4">
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {msg.role === 'assistant' && (
                      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center border-2 border-matrix-green bg-matrix-black">
                        <Bot size={20} className="text-matrix-green" />
                      </div>
                    )}
                    <div
                      className={`max-w-md border-2 p-4 ${
                        msg.role === 'user'
                          ? 'border-matrix-green bg-matrix-dark-green'
                          : 'border-matrix-green/50 bg-matrix-black'
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                      <p className="mt-2 text-xs text-matrix-green/50">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                    {msg.role === 'user' && (
                      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center border-2 border-matrix-green bg-matrix-green">
                        <User size={20} className="text-matrix-black" />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Input */}
              <div className="border-t-2 border-matrix-green p-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask the AI assistant anything..."
                    className="flex-1 border-2 border-matrix-green bg-matrix-black px-3 py-2 text-matrix-green placeholder-matrix-green/30 focus:outline-none focus:ring-2 focus:ring-matrix-green"
                  />
                  <GlowingButton onClick={sendMessage} icon={<Send size={20} />}>
                    SEND
                  </GlowingButton>
                </div>
              </div>
            </div>
          </MatrixCard>
        </div>

        {/* Quick Actions */}
        <div>
          <MatrixCard title="QUICK ACTIONS" glow>
            <div className="space-y-2">
              {quickActions.map((action, index) => (
                <GlowingButton
                  key={index}
                  variant="secondary"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() => {
                    setInput(action);
                    setTimeout(sendMessage, 100);
                  }}
                >
                  {action}
                </GlowingButton>
              ))}
            </div>
          </MatrixCard>

          <div className="mt-4">
            <MatrixCard title="AI CAPABILITIES" glow>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="mb-1 font-bold">Market Analysis</p>
                  <p className="text-matrix-green/70">
                    Technical indicators, sentiment analysis, feature extraction
                  </p>
                </div>
                <div>
                  <p className="mb-1 font-bold">Strategy Testing</p>
                  <p className="text-matrix-green/70">
                    Backtesting, parameter optimization, performance metrics
                  </p>
                </div>
                <div>
                  <p className="mb-1 font-bold">Risk Assessment</p>
                  <p className="text-matrix-green/70">
                    Portfolio analysis, limit checks, exposure monitoring
                  </p>
                </div>
                <div>
                  <p className="mb-1 font-bold">Trade Assistance</p>
                  <p className="text-matrix-green/70">
                    Order recommendations, execution planning, profit/loss estimation
                  </p>
                </div>
              </div>
            </MatrixCard>
          </div>
        </div>
      </div>
    </div>
  );
}
