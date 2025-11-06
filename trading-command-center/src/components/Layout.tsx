import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, Bot, Settings, Shield, Wallet, LineChart, PieChart } from 'lucide-react';

const nav = [
  { to: '/', icon: Activity, label: 'DASHBOARD' },
  { to: '/orders', icon: BarChart3, label: 'ORDERS' },
  { to: '/strategies', icon: LineChart, label: 'STRATEGIES' },
  { to: '/brokers', icon: Wallet, label: 'BROKERS' },
  { to: '/risk', icon: Shield, label: 'RISK' },
  { to: '/analytics', icon: PieChart, label: 'ANALYTICS' },
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
