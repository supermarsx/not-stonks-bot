#!/bin/bash

# Create all pages directly

cat > src/pages/Dashboard.tsx << 'EOF'
import { TrendingUp, DollarSign, Activity, AlertCircle } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Dashboard() {
  const stats = demoData.portfolioStats;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">COMMAND CENTER</h1>
      
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Portfolio Value', value: `$${stats.totalValue.toLocaleString()}`, icon: DollarSign },
          { label: 'Daily P&L', value: `$${stats.dailyPnL.toLocaleString()}`, icon: TrendingUp },
          { label: 'Open Positions', value: stats.openPositions, icon: Activity },
          { label: 'Risk Level', value: stats.riskLevel, icon: AlertCircle },
        ].map((stat) => (
          <div key={stat.label} className="matrix-panel">
            <stat.icon className="mb-2" size={24} />
            <p className="text-2xl font-bold">{stat.value}</p>
            <p className="text-sm text-matrix-medium-green">{stat.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">POSITIONS</h2>
          {demoData.positions.map((pos) => (
            <div key={pos.symbol} className="flex justify-between py-2 border-b border-matrix-dark-green">
              <span>{pos.symbol}</span>
              <span className={pos.pnl >= 0 ? 'positive-value' : 'negative-value'}>
                {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>

        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">RECENT TRADES</h2>
          {demoData.recentTrades.map((trade) => (
            <div key={trade.id} className="flex justify-between py-2 border-b border-matrix-dark-green">
              <div>
                <p className="font-bold">{trade.symbol}</p>
                <p className="text-xs text-matrix-medium-green">{trade.time}</p>
              </div>
              <div className="text-right">
                <p className={trade.side === 'BUY' ? 'positive-value' : 'negative-value'}>
                  {trade.side}
                </p>
                <p className="text-xs">${trade.price}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
EOF

cat > src/pages/Orders.tsx << 'EOF'
import { useState } from 'react';
import { Plus } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Orders() {
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold matrix-glow-text">ORDER MANAGEMENT</h1>
        <button className="matrix-button flex items-center gap-2" onClick={() => setShowForm(!showForm)}>
          <Plus size={20} />
          NEW ORDER
        </button>
      </div>

      {showForm && (
        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">PLACE ORDER</h2>
          <div className="grid grid-cols-2 gap-4">
            <input type="text" placeholder="Symbol" className="matrix-input" />
            <select className="matrix-input">
              <option>Market</option>
              <option>Limit</option>
            </select>
            <input type="number" placeholder="Quantity" className="matrix-input" />
            <input type="number" placeholder="Price" className="matrix-input" />
            <button className="matrix-button col-span-2">SUBMIT</button>
          </div>
        </div>
      )}

      <div className="matrix-panel">
        <h2 className="text-xl font-bold mb-4">OPEN ORDERS</h2>
        <table className="w-full">
          <thead>
            <tr className="border-b-2 border-matrix-green">
              <th className="text-left py-2">Symbol</th>
              <th className="text-left py-2">Type</th>
              <th className="text-left py-2">Side</th>
              <th className="text-left py-2">Quantity</th>
              <th className="text-left py-2">Price</th>
              <th className="text-left py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {demoData.orders.map((order) => (
              <tr key={order.id} className="matrix-data-row">
                <td className="py-3">{order.symbol}</td>
                <td>{order.type}</td>
                <td className={order.side === 'BUY' ? 'positive-value' : 'negative-value'}>
                  {order.side}
                </td>
                <td>{order.quantity}</td>
                <td>${order.price}</td>
                <td className="status-active">{order.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
EOF

cat > src/pages/Strategies.tsx << 'EOF'
import { Play, Pause } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Strategies() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">STRATEGY MANAGEMENT</h1>
      
      <div className="grid gap-4">
        {demoData.strategies.map((strategy) => (
          <div key={strategy.name} className="matrix-panel">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-xl font-bold">{strategy.name}</h3>
                <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-matrix-medium-green">P&L</p>
                    <p className={strategy.pnl >= 0 ? 'positive-value text-lg' : 'negative-value text-lg'}>
                      ${strategy.pnl.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-matrix-medium-green">Trades</p>
                    <p className="text-lg">{strategy.trades}</p>
                  </div>
                  <div>
                    <p className="text-matrix-medium-green">Win Rate</p>
                    <p className="text-lg">{strategy.winRate}%</p>
                  </div>
                </div>
              </div>
              <button className="matrix-button flex items-center gap-2">
                {strategy.status === 'active' ? <Pause size={16} /> : <Play size={16} />}
                {strategy.status.toUpperCase()}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
EOF

cat > src/pages/Brokers.tsx << 'EOF'
import { CheckCircle, XCircle } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Brokers() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">BROKER CONNECTIONS</h1>
      
      <div className="grid gap-4">
        {demoData.brokers.map((broker) => (
          <div key={broker.name} className="matrix-panel">
            <div className="flex justify-between items-center">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-xl font-bold">{broker.name}</h3>
                  {broker.status === 'connected' ? (
                    <CheckCircle className="text-matrix-green" size={20} />
                  ) : (
                    <XCircle className="text-red-500" size={20} />
                  )}
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-matrix-medium-green">Balance</p>
                    <p className="text-lg">${broker.balance.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-matrix-medium-green">Positions</p>
                    <p className="text-lg">{broker.positions}</p>
                  </div>
                </div>
              </div>
              <button className="matrix-button">MANAGE</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
EOF

cat > src/pages/Risk.tsx << 'EOF'
import { AlertTriangle } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Risk() {
  const metrics = demoData.riskMetrics;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">RISK MONITORING</h1>
      
      <div className="grid grid-cols-3 gap-4">
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} className="matrix-panel">
            <p className="text-matrix-medium-green text-sm mb-2">
              {key.replace(/([A-Z])/g, ' $1').toUpperCase()}
            </p>
            <p className="text-2xl font-bold">
              {typeof value === 'number' ? value.toFixed(2) : value}
              {key.includes('Rate') ? '%' : ''}
            </p>
          </div>
        ))}
      </div>

      <div className="matrix-panel">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="text-yellow-500" />
          <h2 className="text-xl font-bold">RISK ALERTS</h2>
        </div>
        <div className="space-y-2">
          <p className="text-matrix-medium-green">No active risk alerts</p>
          <p className="text-sm text-matrix-medium-green">All positions within risk parameters</p>
        </div>
      </div>
    </div>
  );
}
EOF

cat > src/pages/AIAssistant.tsx << 'EOF'
import { Send } from 'lucide-react';
import { useState } from 'react';

export default function AIAssistant() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'AI Trading Assistant online. How can I help you today?' }
  ]);

  const sendMessage = () => {
    if (!input.trim()) return;
    setMessages([...messages, 
      { role: 'user', content: input },
      { role: 'assistant', content: 'Processing your request...' }
    ]);
    setInput('');
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">AI ASSISTANT</h1>
      
      <div className="matrix-panel min-h-[500px] flex flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto scrollbar-matrix mb-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-md p-3 border-2 ${
                msg.role === 'user' 
                  ? 'border-matrix-green bg-matrix-dark-green' 
                  : 'border-matrix-medium-green'
              }`}>
                <p className="text-sm">{msg.content}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask the AI assistant..."
            className="matrix-input flex-1"
          />
          <button onClick={sendMessage} className="matrix-button">
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
EOF

cat > src/pages/Configuration.tsx << 'EOF'
import { Save } from 'lucide-react';

export default function Configuration() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold matrix-glow-text">CONFIGURATION</h1>
      
      <div className="matrix-panel">
        <h2 className="text-xl font-bold mb-4">SYSTEM SETTINGS</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm mb-2">Trading Mode</label>
            <select className="matrix-input w-full">
              <option>Paper Trading</option>
              <option>Live Trading</option>
            </select>
          </div>
          <div>
            <label className="block text-sm mb-2">Max Position Size</label>
            <input type="number" className="matrix-input w-full" placeholder="10000" />
          </div>
          <div>
            <label className="block text-sm mb-2">Daily Loss Limit</label>
            <input type="number" className="matrix-input w-full" placeholder="5000" />
          </div>
          <button className="matrix-button flex items-center gap-2">
            <Save size={20} />
            SAVE SETTINGS
          </button>
        </div>
      </div>
    </div>
  );
}
EOF

echo "All pages created successfully!"

