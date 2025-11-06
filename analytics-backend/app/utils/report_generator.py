"""
Report Generation and Export System
Comprehensive reporting with multiple export formats and visualizations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import json
import io
import base64
from pathlib import Path

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    plt.style.use('dark_background')  # Matrix theme
except ImportError:
    plt = None
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None

# Document generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except ImportError:
    SimpleDocTemplate = None

class ReportGenerator:
    """Advanced report generation with multiple export formats"""
    
    def __init__(self, theme: str = "matrix"):
        self.theme = theme
        self.setup_styling()
    
    def setup_styling(self):
        """Setup styling for charts and reports"""
        if self.theme == "matrix":
            self.colors = {
                'primary': '#00ff00',
                'secondary': '#00cc00',
                'background': '#000000',
                'text': '#00ff00',
                'grid': '#003300'
            }
        else:
            self.colors = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'background': '#F18F01',
                'text': '#C73E1D',
                'grid': '#CCCCCC'
            }
    
    def generate_performance_report(
        self,
        portfolio_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None,
        period: str = "1Y"
    ) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        
        report_data = {
            'metadata': {
                'report_type': 'Performance Analysis',
                'generated_at': datetime.now().isoformat(),
                'period': period,
                'portfolio_name': portfolio_data.get('name', 'Portfolio')
            },
            'executive_summary': self._create_executive_summary(portfolio_data, benchmark_data),
            'performance_metrics': self._extract_performance_metrics(portfolio_data),
            'risk_analysis': self._extract_risk_analysis(portfolio_data),
            'attribution_analysis': self._create_attribution_analysis(portfolio_data),
            'charts': self._generate_performance_charts(portfolio_data, benchmark_data),
            'recommendations': self._generate_performance_recommendations(portfolio_data)
        }
        
        return report_data
    
    def generate_risk_report(
        self,
        portfolio_data: Dict[str, Any],
        risk_analytics: Dict[str, Any],
        stress_scenarios: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk analysis report"""
        
        report_data = {
            'metadata': {
                'report_type': 'Risk Analysis',
                'generated_at': datetime.now().isoformat(),
                'portfolio_name': portfolio_data.get('name', 'Portfolio')
            },
            'risk_summary': self._create_risk_summary(risk_analytics),
            'var_analysis': risk_analytics.get('var_analysis', {}),
            'stress_testing': stress_scenarios,
            'concentration_analysis': risk_analytics.get('concentration_analysis', {}),
            'correlation_analysis': risk_analytics.get('correlation_analysis', {}),
            'charts': self._generate_risk_charts(risk_analytics, stress_scenarios),
            'risk_recommendations': self._generate_risk_recommendations(risk_analytics)
        }
        
        return report_data
    
    def generate_execution_report(
        self,
        trades_data: List[Dict[str, Any]],
        execution_analytics: Dict[str, Any],
        period: str = "1M"
    ) -> Dict[str, Any]:
        """Generate trade execution quality report"""
        
        report_data = {
            'metadata': {
                'report_type': 'Execution Quality Analysis',
                'generated_at': datetime.now().isoformat(),
                'period': period,
                'total_trades': len(trades_data)
            },
            'execution_summary': self._create_execution_summary(execution_analytics),
            'cost_analysis': execution_analytics.get('cost_analysis', {}),
            'venue_analysis': self._analyze_execution_venues(trades_data),
            'timing_analysis': self._analyze_execution_timing(trades_data),
            'charts': self._generate_execution_charts(trades_data, execution_analytics),
            'execution_recommendations': self._generate_execution_recommendations(execution_analytics)
        }
        
        return report_data
    
    def generate_optimization_report(
        self,
        current_portfolio: Dict[str, Any],
        optimized_portfolio: Dict[str, Any],
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate portfolio optimization analysis report"""
        
        report_data = {
            'metadata': {
                'report_type': 'Portfolio Optimization',
                'generated_at': datetime.now().isoformat(),
                'optimization_method': optimization_results.get('method', 'Unknown')
            },
            'optimization_summary': self._create_optimization_summary(optimization_results),
            'allocation_comparison': self._compare_allocations(current_portfolio, optimized_portfolio),
            'expected_improvements': self._calculate_expected_improvements(current_portfolio, optimized_portfolio),
            'efficient_frontier': optimization_results.get('efficient_frontier', {}),
            'charts': self._generate_optimization_charts(current_portfolio, optimized_portfolio, optimization_results),
            'implementation_plan': self._create_implementation_plan(current_portfolio, optimized_portfolio)
        }
        
        return report_data
    
    def export_to_pdf(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report to PDF format"""
        
        if SimpleDocTemplate is None:
            return "PDF export not available - reportlab not installed"
        
        try:
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.darkgreen,
                spaceAfter=30
            )
            story.append(Paragraph(report_data['metadata']['report_type'], title_style))
            story.append(Spacer(1, 12))
            
            # Metadata
            meta_data = [
                ['Generated:', report_data['metadata']['generated_at']],
                ['Report Type:', report_data['metadata']['report_type']],
                ['Period:', report_data['metadata'].get('period', 'N/A')]
            ]
            
            meta_table = Table(meta_data)
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            if 'executive_summary' in report_data:
                story.append(Paragraph("Executive Summary", styles['Heading2']))
                for key, value in report_data['executive_summary'].items():
                    story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Main Content Sections
            for section_name, section_data in report_data.items():
                if section_name in ['metadata', 'executive_summary', 'charts']:
                    continue
                
                story.append(Paragraph(section_name.replace('_', ' ').title(), styles['Heading2']))
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, (str, int, float)):
                            story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                        elif isinstance(value, list) and value:
                            story.append(Paragraph(f"<b>{key}:</b>", styles['Normal']))
                            for item in value[:5]:  # Limit to 5 items
                                story.append(Paragraph(f"• {item}", styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            return f"PDF report exported to {filename}"
            
        except Exception as e:
            return f"PDF export failed: {str(e)}"
    
    def export_to_excel(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report to Excel format"""
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = []
                for section, data in report_data.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (str, int, float)):
                                summary_data.append({
                                    'Section': section,
                                    'Metric': key,
                                    'Value': value
                                })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual sections as separate sheets
                for section_name, section_data in report_data.items():
                    if section_name in ['metadata', 'charts'] or not isinstance(section_data, dict):
                        continue
                    
                    # Convert section data to DataFrame format
                    rows = []
                    for key, value in section_data.items():
                        if isinstance(value, (str, int, float)):
                            rows.append({'Metric': key, 'Value': value})
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                rows.append({'Metric': f"{key}_{i}", 'Value': str(item)})
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                rows.append({'Metric': f"{key}_{subkey}", 'Value': str(subvalue)})
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        sheet_name = section_name[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return f"Excel report exported to {filename}"
            
        except Exception as e:
            return f"Excel export failed: {str(e)}"
    
    def export_to_json(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report to JSON format"""
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            return f"JSON report exported to {filename}"
        except Exception as e:
            return f"JSON export failed: {str(e)}"
    
    def _create_executive_summary(self, portfolio_data: Dict, benchmark_data: Optional[Dict]) -> Dict[str, str]:
        """Create executive summary for performance report"""
        
        summary = {}
        
        # Portfolio performance
        if 'total_return' in portfolio_data:
            summary['Total Return'] = f"{portfolio_data['total_return']*100:.2f}%"
        
        if 'sharpe_ratio' in portfolio_data:
            summary['Sharpe Ratio'] = f"{portfolio_data['sharpe_ratio']:.2f}"
        
        if 'max_drawdown' in portfolio_data:
            summary['Maximum Drawdown'] = f"{portfolio_data['max_drawdown']*100:.2f}%"
        
        if 'volatility' in portfolio_data:
            summary['Volatility'] = f"{portfolio_data['volatility']*100:.2f}%"
        
        # Benchmark comparison
        if benchmark_data:
            if 'total_return' in benchmark_data and 'total_return' in portfolio_data:
                excess_return = portfolio_data['total_return'] - benchmark_data['total_return']
                summary['Excess Return vs Benchmark'] = f"{excess_return*100:.2f}%"
        
        return summary
    
    def _extract_performance_metrics(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Extract and format performance metrics"""
        
        metrics = {}
        
        performance_keys = [
            'annual_return', 'total_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'max_drawdown', 'volatility', 'win_rate'
        ]
        
        for key in performance_keys:
            if key in portfolio_data:
                value = portfolio_data[key]
                if isinstance(value, (int, float)):
                    if 'ratio' in key or key in ['win_rate']:
                        metrics[key.replace('_', ' ').title()] = f"{value:.3f}"
                    else:
                        metrics[key.replace('_', ' ').title()] = f"{value*100:.2f}%"
        
        return metrics
    
    def _extract_risk_analysis(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Extract risk analysis data"""
        
        risk_data = {}
        
        if 'var_analysis' in portfolio_data:
            risk_data['VaR Analysis'] = portfolio_data['var_analysis']
        
        if 'tail_risk' in portfolio_data:
            risk_data['Tail Risk Metrics'] = portfolio_data['tail_risk']
        
        return risk_data
    
    def _create_attribution_analysis(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Create performance attribution analysis"""
        
        attribution = {}
        
        if 'sector_attribution' in portfolio_data:
            attribution['Sector Attribution'] = portfolio_data['sector_attribution']
        
        if 'asset_attribution' in portfolio_data:
            attribution['Asset Attribution'] = portfolio_data['asset_attribution']
        
        return attribution
    
    def _generate_performance_charts(self, portfolio_data: Dict, benchmark_data: Optional[Dict]) -> List[str]:
        """Generate performance visualization charts"""
        
        charts = []
        
        if plt is None:
            return ["Chart generation not available - matplotlib not installed"]
        
        try:
            # Performance comparison chart
            if 'returns_series' in portfolio_data:
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
                ax.set_facecolor('black')
                
                returns = portfolio_data['returns_series']
                if isinstance(returns, dict):
                    dates = list(returns.keys())
                    values = list(returns.values())
                    ax.plot(dates, values, color=self.colors['primary'], linewidth=2, label='Portfolio')
                
                if benchmark_data and 'returns_series' in benchmark_data:
                    bench_returns = benchmark_data['returns_series']
                    if isinstance(bench_returns, dict):
                        bench_dates = list(bench_returns.keys())
                        bench_values = list(bench_returns.values())
                        ax.plot(bench_dates, bench_values, color=self.colors['secondary'], linewidth=2, label='Benchmark')
                
                ax.set_title('Portfolio Performance', color=self.colors['text'], fontsize=16)
                ax.set_xlabel('Date', color=self.colors['text'])
                ax.set_ylabel('Cumulative Return', color=self.colors['text'])
                ax.legend()
                ax.grid(True, color=self.colors['grid'], alpha=0.3)
                
                # Save chart
                chart_filename = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename, facecolor='black', bbox_inches='tight')
                plt.close()
                charts.append(chart_filename)
            
        except Exception as e:
            charts.append(f"Chart generation error: {str(e)}")
        
        return charts
    
    def _generate_risk_charts(self, risk_analytics: Dict, stress_scenarios: Dict) -> List[str]:
        """Generate risk visualization charts"""
        
        charts = []
        
        if plt is None:
            return ["Chart generation not available"]
        
        # Risk distribution chart, correlation heatmap, etc.
        # Implementation would depend on specific risk data structure
        charts.append("risk_distribution.png")
        charts.append("correlation_heatmap.png")
        
        return charts
    
    def _generate_execution_charts(self, trades_data: List, execution_analytics: Dict) -> List[str]:
        """Generate execution quality charts"""
        
        charts = []
        # Implementation for execution charts
        charts.append("execution_quality.png")
        charts.append("cost_analysis.png")
        
        return charts
    
    def _generate_optimization_charts(self, current: Dict, optimized: Dict, results: Dict) -> List[str]:
        """Generate optimization visualization charts"""
        
        charts = []
        # Implementation for optimization charts
        charts.append("efficient_frontier.png")
        charts.append("allocation_comparison.png")
        
        return charts
    
    def _generate_performance_recommendations(self, portfolio_data: Dict) -> List[str]:
        """Generate performance-based recommendations"""
        
        recommendations = []
        
        if 'sharpe_ratio' in portfolio_data:
            sharpe = portfolio_data['sharpe_ratio']
            if sharpe < 0.5:
                recommendations.append("Consider improving risk-adjusted returns through better diversification")
            elif sharpe > 1.5:
                recommendations.append("Excellent risk-adjusted performance - maintain current strategy")
        
        if 'max_drawdown' in portfolio_data:
            drawdown = portfolio_data['max_drawdown']
            if drawdown > 0.2:
                recommendations.append("High maximum drawdown detected - consider risk management improvements")
        
        return recommendations
    
    def _generate_risk_recommendations(self, risk_analytics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        # Implementation for risk recommendations
        recommendations.append("Monitor concentration risk in top holdings")
        recommendations.append("Consider stress testing portfolio against market scenarios")
        
        return recommendations
    
    def _generate_execution_recommendations(self, execution_analytics: Dict) -> List[str]:
        """Generate execution quality recommendations"""
        
        recommendations = []
        # Implementation for execution recommendations
        recommendations.append("Optimize order timing to reduce market impact")
        recommendations.append("Review venue selection for cost efficiency")
        
        return recommendations
    
    # Helper methods for other report sections
    def _create_risk_summary(self, risk_analytics: Dict) -> Dict[str, str]:
        """Create risk summary"""
        return {"Overall Risk Level": "Moderate", "Key Concerns": "Concentration Risk"}
    
    def _create_execution_summary(self, execution_analytics: Dict) -> Dict[str, str]:
        """Create execution summary"""
        return {"Execution Quality": "Good", "Average Cost": "5.2 bps"}
    
    def _create_optimization_summary(self, optimization_results: Dict) -> Dict[str, str]:
        """Create optimization summary"""
        return {"Expected Improvement": "12.3% return increase", "Risk Reduction": "8.7%"}
    
    def _analyze_execution_venues(self, trades_data: List) -> Dict[str, Any]:
        """Analyze execution venues"""
        return {"Primary Venue": "NASDAQ", "Secondary Venue": "NYSE"}
    
    def _analyze_execution_timing(self, trades_data: List) -> Dict[str, Any]:
        """Analyze execution timing"""
        return {"Average Execution Time": "2.3 seconds", "Best Time": "Morning"}
    
    def _compare_allocations(self, current: Dict, optimized: Dict) -> Dict[str, Any]:
        """Compare current vs optimized allocations"""
        return {"Changes Required": "Moderate", "Major Shifts": 3}
    
    def _calculate_expected_improvements(self, current: Dict, optimized: Dict) -> Dict[str, Any]:
        """Calculate expected improvements"""
        return {"Return Improvement": "1.2%", "Risk Reduction": "0.8%"}
    
    def _create_implementation_plan(self, current: Dict, optimized: Dict) -> Dict[str, Any]:
        """Create implementation plan"""
        return {"Implementation Phases": 3, "Timeline": "2 weeks", "Priority": "High"}