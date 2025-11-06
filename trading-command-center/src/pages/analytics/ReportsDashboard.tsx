import { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { FileText, Download, Calendar, Settings, Send, Eye, Trash2, Plus } from 'lucide-react';
import { MatrixCard } from '../../components/MatrixCard';
import { StatCard } from '../../components/StatCard';
import { GlowingButton } from '../../components/GlowingButton';
import { analyticsApi } from '../../services/analyticsApi';

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: 'performance' | 'risk' | 'execution' | 'optimization';
  last_used: string;
  frequency: string;
}

interface GeneratedReport {
  id: string;
  name: string;
  type: string;
  generated_at: string;
  file_size: string;
  format: string;
  status: 'completed' | 'generating' | 'failed';
}

const reportTemplates: ReportTemplate[] = [
  {
    id: '1',
    name: 'Daily Performance Summary',
    description: 'Comprehensive daily portfolio performance analysis',
    type: 'performance',
    last_used: '2025-11-06',
    frequency: 'Daily',
  },
  {
    id: '2',
    name: 'Weekly Risk Assessment',
    description: 'Portfolio risk metrics and stress testing results',
    type: 'risk',
    last_used: '2025-11-05',
    frequency: 'Weekly',
  },
  {
    id: '3',
    name: 'Monthly Execution Quality',
    description: 'Trade execution analysis and TCA breakdown',
    type: 'execution',
    last_used: '2025-11-01',
    frequency: 'Monthly',
  },
  {
    id: '4',
    name: 'Quarterly Portfolio Review',
    description: 'Complete portfolio optimization and rebalancing analysis',
    type: 'optimization',
    last_used: '2025-11-01',
    frequency: 'Quarterly',
  },
  {
    id: '5',
    name: 'ESG Impact Report',
    description: 'Environmental, social, and governance impact analysis',
    type: 'performance',
    last_used: '2025-10-30',
    frequency: 'Monthly',
  },
];

const generatedReports: GeneratedReport[] = [
  {
    id: '1',
    name: 'Performance_Report_2025-11-06',
    type: 'Performance Analysis',
    generated_at: '2025-11-06 09:15:00',
    file_size: '2.4 MB',
    format: 'PDF',
    status: 'completed',
  },
  {
    id: '2',
    name: 'Risk_Assessment_2025-11-05',
    type: 'Risk Analysis',
    generated_at: '2025-11-05 17:30:00',
    file_size: '1.8 MB',
    format: 'Excel',
    status: 'completed',
  },
  {
    id: '3',
    name: 'Execution_Quality_2025-11-04',
    type: 'Execution Analysis',
    generated_at: '2025-11-04 14:45:00',
    file_size: '3.2 MB',
    format: 'PDF',
    status: 'completed',
  },
  {
    id: '4',
    name: 'Optimization_Report_2025-11-03',
    type: 'Portfolio Optimization',
    generated_at: '2025-11-03 11:20:00',
    file_size: '1.5 MB',
    format: 'JSON',
    status: 'completed',
  },
];

export default function ReportsDashboard() {
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [reportFormat, setReportFormat] = useState('pdf');
  const [scheduledReports, setScheduledReports] = useState<any[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Generate report mutation
  const generateReportMutation = useMutation({
    mutationFn: async (reportData: any) => {
      setIsGenerating(true);
      // Simulate report generation based on type
      const template = reportTemplates.find(t => t.id === selectedTemplate);
      if (!template) throw new Error('Template not found');

      switch (template.type) {
        case 'performance':
          return analyticsApi.generatePerformanceReport({
            portfolio_name: 'Main Portfolio',
            returns: Array.from({ length: 30 }, () => Math.random() * 0.04 - 0.02),
            dates: Array.from({ length: 30 }, (_, i) => {
              const date = new Date();
              date.setDate(date.getDate() - i);
              return date.toISOString().split('T')[0];
            }),
            period: 'Last 30 Days',
          });
        case 'risk':
          return analyticsApi.generateRiskReport({
            portfolio_name: 'Main Portfolio',
            portfolio_weights: {
              'AAPL': 0.25, 'GOOGL': 0.20, 'MSFT': 0.15, 'TSLA': 0.10,
              'AMZN': 0.10, 'NVDA': 0.08, 'META': 0.07, 'NFLX': 0.05,
            },
            portfolio_returns: Array.from({ length: 252 }, () => Math.random() * 0.04 - 0.02),
          });
        case 'execution':
          return analyticsApi.generateExecutionReport({
            portfolio_name: 'Main Portfolio',
            trades: Array.from({ length: 50 }, (_, i) => ({
              id: i,
              symbol: ['AAPL', 'GOOGL', 'MSFT'][i % 3],
              side: i % 2 === 0 ? 'buy' : 'sell',
              quantity: Math.floor(Math.random() * 1000) + 100,
              execution_price: Math.random() * 200 + 50,
              timestamp: new Date().toISOString(),
            })),
          });
        case 'optimization':
          return analyticsApi.generateOptimizationReport({
            current_portfolio: {
              'AAPL': 0.25, 'GOOGL': 0.20, 'MSFT': 0.15, 'TSLA': 0.10,
              'AMZN': 0.10, 'NVDA': 0.08, 'META': 0.07, 'NFLX': 0.05,
            },
            optimization_results: {
              optimal_weights: [0.22, 0.18, 0.16, 0.12, 0.12, 0.10, 0.06, 0.04],
              portfolio_return: 0.15,
              portfolio_volatility: 0.18,
              sharpe_ratio: 0.83,
            },
            optimization_method: 'Mean-Variance',
            expected_returns: [0.12, 0.15, 0.11, 0.25, 0.14, 0.28, 0.16, 0.18],
            asset_names: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
          });
        default:
          throw new Error('Unknown report type');
      }
    },
    onSuccess: (data) => {
      setIsGenerating(false);
      console.log('Report generated successfully:', data);
      // Here you would typically add the new report to the list and show success message
    },
    onError: (error) => {
      setIsGenerating(false);
      console.error('Report generation failed:', error);
    },
  });

  const handleGenerateReport = () => {
    if (!selectedTemplate) return;
    generateReportMutation.mutate({
      template_id: selectedTemplate,
      format: reportFormat,
    });
  };

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'performance': return 'text-green-400';
      case 'risk': return 'text-red-400';
      case 'execution': return 'text-blue-400';
      case 'optimization': return 'text-yellow-400';
      default: return 'text-matrix-green';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'performance': return '📈';
      case 'risk': return '🛡️';
      case 'execution': return '⚡';
      case 'optimization': return '🎯';
      default: return '📄';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold matrix-glow-text">Reports Dashboard</h1>
          <p className="text-matrix-green/70 mt-2">Generate, schedule, and manage analytics reports</p>
        </div>
        <div className="flex space-x-4">
          <GlowingButton variant="secondary">
            <Calendar className="h-4 w-4 mr-2" />
            Schedule Report
          </GlowingButton>
          <GlowingButton>
            <Plus className="h-4 w-4 mr-2" />
            New Template
          </GlowingButton>
        </div>
      </div>

      {/* Report Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          label="Reports Generated"
          value="47"
          icon={FileText}
        />
        <StatCard
          label="Active Templates"
          value="12"
          icon={Settings}
        />
        <StatCard
          label="Total Downloads"
          value="234"
          icon={Download}
        />
        <StatCard
          label="Scheduled Reports"
          value="8"
          icon={Calendar}
        />
      </div>

      {/* Report Generation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Generate New Report" className="p-6">
          <div className="space-y-6">
            <div>
              <title className="block text-sm text-matrix-green/70 mb-3">Select Report Template</title>
              <div className="space-y-2">
                {reportTemplates.map((template) => (
                  <div
                    key={template.id}
                    className={`p-3 border-2 rounded cursor-pointer transition-all ${
                      selectedTemplate === template.id
                        ? 'border-matrix-green bg-matrix-green/10'
                        : 'border-matrix-dark-green hover:border-matrix-green'
                    }`}
                    onClick={() => setSelectedTemplate(template.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold flex items-center">
                          <span className="mr-2">{getTypeIcon(template.type)}</span>
                          {template.name}
                        </h3>
                        <p className="text-sm text-matrix-green/70">{template.description}</p>
                      </div>
                      <div className="text-right">
                        <span className={`text-sm ${getTypeColor(template.type)}`}>
                          {template.type.toUpperCase()}
                        </span>
                        <p className="text-xs text-matrix-green/50">{template.frequency}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <title className="block text-sm text-matrix-green/70 mb-2">Output Format</title>
              <div className="flex space-x-4">
                {['pdf', 'excel', 'json'].map((format) => (
                  <button
                    key={format}
                    className={`px-4 py-2 border-2 rounded transition-all ${
                      reportFormat === format
                        ? 'border-matrix-green bg-matrix-green text-matrix-black'
                        : 'border-matrix-dark-green hover:border-matrix-green'
                    }`}
                    onClick={() => setReportFormat(format)}
                  >
                    {format.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <GlowingButton
              onClick={handleGenerateReport}
              disabled={!selectedTemplate || isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                  Generating Report...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4 mr-2" />
                  Generate Report
                </>
              )}
            </GlowingButton>
          </div>
        </MatrixCard>

        <MatrixCard title="Quick Actions" className="p-6">
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold text-matrix-green mb-3">Popular Templates</h3>
              <div className="space-y-2">
                {reportTemplates.slice(0, 3).map((template) => (
                  <div key={template.id} className="flex justify-between items-center p-2 border border-matrix-green/20 rounded">
                    <div>
                      <span className="font-medium">{template.name}</span>
                      <p className="text-xs text-matrix-green/70">Last used: {template.last_used}</p>
                    </div>
                    <GlowingButton
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        setSelectedTemplate(template.id);
                        handleGenerateReport();
                      }}
                    >
                      Run
                    </GlowingButton>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-matrix-green mb-3">Export Options</h3>
              <div className="grid grid-cols-2 gap-2">
                <GlowingButton variant="secondary" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Bulk Export
                </GlowingButton>
                <GlowingButton variant="secondary" size="sm">
                  <Send className="h-4 w-4 mr-2" />
                  Email Reports
                </GlowingButton>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-matrix-green mb-3">Automation</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-sm">
                  <span>Daily Performance</span>
                  <span className="text-green-400">Active</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span>Weekly Risk Review</span>
                  <span className="text-green-400">Active</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span>Monthly Compliance</span>
                  <span className="text-yellow-400">Paused</span>
                </div>
              </div>
            </div>
          </div>
        </MatrixCard>
      </div>

      {/* Generated Reports */}
      <MatrixCard title="Recent Reports" className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-matrix-green/30">
                <th className="text-left py-3">Report Name</th>
                <th className="text-left py-3">Type</th>
                <th className="text-left py-3">Generated</th>
                <th className="text-left py-3">Size</th>
                <th className="text-left py-3">Format</th>
                <th className="text-left py-3">Status</th>
                <th className="text-left py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {generatedReports.map((report) => (
                <tr key={report.id} className="border-b border-matrix-green/10 hover:bg-matrix-green/5">
                  <td className="py-3 font-mono">{report.name}</td>
                  <td className="py-3">
                    <span className={`${getTypeColor(report.type)} flex items-center`}>
                      <span className="mr-2">{getTypeIcon(report.type.toLowerCase())}</span>
                      {report.type}
                    </span>
                  </td>
                  <td className="py-3 font-mono text-matrix-green/70">
                    {new Date(report.generated_at).toLocaleString()}
                  </td>
                  <td className="py-3 font-mono">{report.file_size}</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-matrix-green/20 rounded text-xs font-mono">
                      {report.format}
                    </span>
                  </td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs ${
                      report.status === 'completed' ? 'bg-green-900 text-green-300' :
                      report.status === 'generating' ? 'bg-yellow-900 text-yellow-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {report.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="flex space-x-2">
                      <button className="text-blue-400 hover:text-blue-300 transition-colors">
                        <Eye className="h-4 w-4" />
                      </button>
                      <button className="text-green-400 hover:text-green-300 transition-colors">
                        <Download className="h-4 w-4" />
                      </button>
                      <button className="text-red-400 hover:text-red-300 transition-colors">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </MatrixCard>

      {/* Scheduled Reports */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MatrixCard title="Scheduled Reports" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-matrix-green">Upcoming</h3>
              <GlowingButton size="sm" variant="secondary">
                <Plus className="h-4 w-4 mr-2" />
                Add Schedule
              </GlowingButton>
            </div>
            <div className="space-y-3">
              <div className="p-3 border border-matrix-green/20 rounded">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold">Daily Performance Summary</h4>
                    <p className="text-sm text-matrix-green/70">Every day at 6:00 PM</p>
                  </div>
                  <span className="text-green-400 text-sm">Today 18:00</span>
                </div>
              </div>
              <div className="p-3 border border-matrix-green/20 rounded">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold">Weekly Risk Assessment</h4>
                    <p className="text-sm text-matrix-green/70">Every Monday at 9:00 AM</p>
                  </div>
                  <span className="text-yellow-400 text-sm">Mon 09:00</span>
                </div>
              </div>
              <div className="p-3 border border-matrix-green/20 rounded">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold">Monthly Portfolio Review</h4>
                    <p className="text-sm text-matrix-green/70">First day of month at 10:00 AM</p>
                  </div>
                  <span className="text-blue-400 text-sm">Dec 1 10:00</span>
                </div>
              </div>
            </div>
          </div>
        </MatrixCard>

        <MatrixCard title="Report Analytics" className="p-6">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-matrix-green mb-3">Usage Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span>Most Generated</span>
                  <span className="font-mono text-green-400">Performance Reports (45%)</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Preferred Format</span>
                  <span className="font-mono text-blue-400">PDF (67%)</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Avg Generation Time</span>
                  <span className="font-mono text-yellow-400">2.3 seconds</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Storage Used</span>
                  <span className="font-mono text-matrix-green">127.4 MB</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-matrix-green mb-3">Recent Activity</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-matrix-green/70">Performance Report generated</span>
                  <span className="text-matrix-green/50">2 min ago</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-matrix-green/70">Risk Assessment downloaded</span>
                  <span className="text-matrix-green/50">1 hour ago</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-matrix-green/70">New template created</span>
                  <span className="text-matrix-green/50">3 hours ago</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-matrix-green/70">Scheduled report sent</span>
                  <span className="text-matrix-green/50">Yesterday</span>
                </div>
              </div>
            </div>
          </div>
        </MatrixCard>
      </div>
    </div>
  );
}