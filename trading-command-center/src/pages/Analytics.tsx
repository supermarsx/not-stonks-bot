import { Routes, Route, Navigate } from 'react-router-dom';
import PerformanceAnalytics from './analytics/PerformanceAnalytics';
import ExecutionQuality from './analytics/ExecutionQuality';
import RiskAnalytics from './analytics/RiskAnalytics';
import PortfolioOptimization from './analytics/PortfolioOptimization';
import ReportsDashboard from './analytics/ReportsDashboard';
import AnalyticsNavigation from '../components/analytics/AnalyticsNavigation';

export default function Analytics() {
  return (
    <div className="flex h-full">
      <div className="w-64 border-r-2 border-matrix-green bg-matrix-black">
        <AnalyticsNavigation />
      </div>
      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/analytics/performance" replace />} />
          <Route path="/performance" element={<PerformanceAnalytics />} />
          <Route path="/execution" element={<ExecutionQuality />} />
          <Route path="/risk" element={<RiskAnalytics />} />
          <Route path="/optimization" element={<PortfolioOptimization />} />
          <Route path="/reports" element={<ReportsDashboard />} />
        </Routes>
      </div>
    </div>
  );
}