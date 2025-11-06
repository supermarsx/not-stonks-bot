#!/bin/bash

# Create Layout component
cat > src/components/Layout.tsx << 'EOF'
import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, Bot, Settings, Shield, Wallet, LineChart } from 'lucide-react';

const nav = [
  { to: '/', icon: Activity, label: 'DASHBOARD' },
  { to: '/orders', icon: BarChart3, label: 'ORDERS' },
  { to: '/strategies', icon: LineChart, label: 'STRATEGIES' },
  { to: '/brokers', icon: Wallet, label: 'BROKERS' },
  { to: '/risk', icon: Shield, label: 'RISK' },
  { to: '/ai', icon: Bot, label: 'AI' },
  { to: '/config', icon: Settings, label: 'CONFIG' },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen">
      <aside className="w-48 border-r-2 border-matrix-green bg-matrix-black p-4">
        <h1 className="mb-8 text-center text-xl font-bold matrix-glow-text">
          TRADING<br />MATRIX
        </h1>
        <nav className="space-y-2">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center space-x-3 border-2 px-3 py-2 transition-all ${
                  isActive
                    ? 'border-matrix-green bg-matrix-green text-matrix-black'
                    : 'border-matrix-dark-green hover:border-matrix-green'
                }`
              }
            >
              <Icon size={18} />
              <span className="text-sm">{label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-auto p-6">{children}</main>
    </div>
  );
}
EOF

# Create Dashboard page
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
          { label: 'Portfolio Value', value: `$${stats.totalValue.toLocaleString()}`, icon: DollarSign, change: '+$12,450' },
          { label: 'Daily P&L', value: `$${stats.dailyPnL.toLocaleString()}`, icon: TrendingUp, change: '+2.45%' },
          { label: 'Open Positions', value: stats.openPositions, icon: Activity, change: '8 active' },
          { label: 'Risk Level', value: stats.riskLevel, icon: AlertCircle, change: 'Normal' },
        ].map((stat) => (
          <div key={stat.label} className="matrix-panel">
            <div className="flex items-center justify-between mb-2">
              <stat.icon className="text-matrix-green" size={24} />
              <span className="text-xs text-matrix-medium-green">{stat.change}</span>
            </div>
            <p className="text-2xl font-bold">{stat.value}</p>
            <p className="text-sm text-matrix-medium-green">{stat.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">POSITIONS</h2>
          <div className="space-y-2">
            {demoData.positions.map((pos) => (
              <div key={pos.symbol} className="flex justify-between py-2 border-b border-matrix-dark-green">
                <span>{pos.symbol}</span>
                <span className={pos.pnl >= 0 ? 'positive-value' : 'negative-value'}>
                  {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">RECENT TRADES</h2>
          <div className="space-y-2">
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
    </div>
  );
}
EOF

# Create Orders page
cat > src/pages/Orders.tsx << 'EOF'
import { useState } from 'react';
import { Plus } from 'lucide-react';
import { demoData } from '../utils/demoData';

export default function Orders() {
  const [showOrderForm, setShowOrderForm] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold matrix-glow-text">ORDER MANAGEMENT</h1>
        <button className="matrix-button flex items-center gap-2" onClick={() => setShowOrderForm(!showOrderForm)}>
          <Plus size={20} />
          NEW ORDER
        </button>
      </div>

      {showOrderForm && (
        <div className="matrix-panel">
          <h2 className="text-xl font-bold mb-4">PLACE ORDER</h2>
          <div className="grid grid-cols-2 gap-4">
            <input type="text" placeholder="Symbol" className="matrix-input" />
            <select className="matrix-input">
              <option>Market</option>
              <option>Limit</option>
              <option>Stop</option>
            </select>
            <input type="number" placeholder="Quantity" className="matrix-input" />
            <input type="number" placeholder="Price" className="matrix-input" />
            <button className="matrix-button col-span-2">SUBMIT ORDER</button>
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

echo "Core pages created"
