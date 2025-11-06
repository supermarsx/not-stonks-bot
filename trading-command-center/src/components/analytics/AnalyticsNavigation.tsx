import { NavLink } from 'react-router-dom';
import { BarChart3, Activity, Shield, Target, FileText, TrendingUp } from 'lucide-react';

const analyticsNav = [
  { to: '/analytics/performance', icon: TrendingUp, label: 'PERFORMANCE' },
  { to: '/analytics/execution', icon: Activity, label: 'EXECUTION' },
  { to: '/analytics/risk', icon: Shield, label: 'RISK' },
  { to: '/analytics/optimization', icon: Target, label: 'OPTIMIZATION' },
  { to: '/analytics/reports', icon: FileText, label: 'REPORTS' },
];

export default function AnalyticsNavigation() {
  return (
    <div className="p-4">
      <h2 className="mb-6 text-center text-lg font-bold matrix-glow-text">
        ANALYTICS<br />CENTER
      </h2>
      <nav className="space-y-2">
        {analyticsNav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center space-x-3 border-2 px-3 py-2 transition-all text-sm ${
                isActive
                  ? 'border-matrix-green bg-matrix-green text-matrix-black'
                  : 'border-matrix-dark-green hover:border-matrix-green'
              }`
            }
          >
            <Icon size={16} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
