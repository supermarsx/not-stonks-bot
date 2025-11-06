"""
Automated Reporting System

Provides comprehensive automated reporting capabilities including:
- Real-time dashboard generation
- Scheduled report generation (daily, weekly, monthly)
- Custom report builder interface
- PDF/Excel export capabilities
- Email report delivery system
- Interactive report visualization
- Multi-format output support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import base64
from pathlib import Path
import zipfile
import io

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class ReportTemplate:
    """Report template definition"""
    name: str
    template_type: str  # 'dashboard', 'performance', 'risk', 'execution', 'custom'
    description: str
    sections: List[str]
    parameters: Dict[str, Any]
    output_formats: List[str]  # ['pdf', 'excel', 'html', 'json']
    scheduled_frequency: Optional[str]  # 'daily', 'weekly', 'monthly', 'quarterly', 'annually'


@dataclass
class ReportData:
    """Report data container"""
    report_id: str
    report_type: str
    title: str
    generated_at: datetime
    data: Dict[str, Any]
    charts: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class ExportOptions:
    """Export configuration options"""
    format: str  # 'pdf', 'excel', 'html', 'json', 'csv'
    include_charts: bool = True
    include_tables: bool = True
    include_raw_data: bool = False
    chart_resolution: str = 'high'  # 'low', 'medium', 'high'
    page_orientation: str = 'portrait'  # 'portrait', 'landscape'
    include_metadata: bool = True


class AutomatedReporter:
    """
    Advanced Automated Reporting System
    
    Provides comprehensive automated reporting capabilities including:
    - Real-time dashboard generation
    - Scheduled report generation
    - Custom report builder interface
    - Multi-format export capabilities
    - Email delivery system
    - Interactive report visualization
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize automated reporter"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Report templates
        self.report_templates = self._initialize_report_templates()
        
        # Export formats supported
        self.supported_formats = ['pdf', 'excel', 'html', 'json', 'csv', 'png', 'svg']
        
        # Report generation queue
        self.report_queue = []
        
        # Email configuration (mock)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'reports@trading-orchestrator.com',
            'authentication_required': True
        }
        
        # File storage paths
        self.storage_paths = {
            'reports': Path('/workspace/trading_orchestrator/reports'),
            'templates': Path('/workspace/trading_orchestrator/templates'),
            'exports': Path('/workspace/trading_orchestrator/exports')
        }
        
        # Ensure directories exist
        for path in self.storage_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Automated Reporter initialized")
    
    def _initialize_report_templates(self) -> Dict[str, ReportTemplate]:
        """Initialize report templates"""
        templates = {
            'daily_performance': ReportTemplate(
                name='Daily Performance Report',
                template_type='performance',
                description='Daily portfolio performance summary with key metrics',
                sections=['summary', 'performance_metrics', 'attribution', 'risk_summary'],
                parameters={'period': '1D', 'include_benchmarks': True},
                output_formats=['pdf', 'excel', 'html'],
                scheduled_frequency='daily'
            ),
            'weekly_risk': ReportTemplate(
                name='Weekly Risk Analysis',
                template_type='risk',
                description='Weekly risk dashboard with VaR, stress tests, and monitoring',
                sections=['var_analysis', 'stress_tests', 'concentration_risk', 'regulatory_compliance'],
                parameters={'period': '1W', 'confidence_levels': [0.95, 0.99]},
                output_formats=['pdf', 'html'],
                scheduled_frequency='weekly'
            ),
            'monthly_execution': ReportTemplate(
                name='Monthly Execution Quality Report',
                template_type='execution',
                description='Monthly TCA analysis with implementation shortfall and best execution',
                sections=['implementation_shortfall', 'market_impact', 'execution_algorithms', 'venue_analysis'],
                parameters={'period': '1M', 'include_venue_comparison': True},
                output_formats=['excel', 'pdf', 'html'],
                scheduled_frequency='monthly'
            ),
            'quarterly_attribution': ReportTemplate(
                name='Quarterly Attribution Analysis',
                template_type='attribution',
                description='Comprehensive quarterly performance attribution analysis',
                sections=['brinson_attribution', 'factor_analysis', 'risk_attribution', 'sector_analysis'],
                parameters={'period': '3M', 'include_benchmarks': True, 'detailed_analysis': True},
                output_formats=['pdf', 'excel'],
                scheduled_frequency='quarterly'
            ),
            'real_time_dashboard': ReportTemplate(
                name='Real-Time Dashboard',
                template_type='dashboard',
                description='Real-time trading dashboard with live metrics',
                sections=['live_positions', 'pnl_summary', 'risk_monitor', 'execution_status'],
                parameters={'refresh_interval': 30, 'include_charts': True},
                output_formats=['html', 'json'],
                scheduled_frequency=None
            ),
            'custom_report': ReportTemplate(
                name='Custom Report Builder',
                template_type='custom',
                description='Customizable report with user-defined sections and parameters',
                sections=['user_defined'],
                parameters={'flexible': True},
                output_formats=['pdf', 'excel', 'html', 'json'],
                scheduled_frequency=None
            )
        }
        
        return templates
    
    async def generate_scheduled_reports(self, report_types: List[str]):
        """Generate scheduled reports"""
        try:
            self.logger.info(f"Generating scheduled reports: {report_types}")
            
            generation_tasks = []
            for report_type in report_types:
                # Find matching templates
                matching_templates = [
                    template for template in self.report_templates.values()
                    if template.scheduled_frequency == report_type
                ]
                
                for template in matching_templates:
                    generation_tasks.append(self._generate_report_async(template))
            
            if generation_tasks:
                await asyncio.gather(*generation_tasks, return_exceptions=True)
            
            self.logger.info("Scheduled reports generation completed")
            
        except Exception as e:
            self.logger.error(f"Error generating scheduled reports: {e}")
            raise
    
    async def generate_custom_report(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom report based on configuration"""
        try:
            self.logger.info("Generating custom report")
            
            # Extract configuration
            report_type = report_config.get('type', 'custom')
            sections = report_config.get('sections', [])
            parameters = report_config.get('parameters', {})
            output_formats = report_config.get('output_formats', ['html'])
            filters = report_config.get('filters', {})
            
            # Create custom template if needed
            if report_type not in self.report_templates:
                custom_template = ReportTemplate(
                    name=report_config.get('title', 'Custom Report'),
                    template_type='custom',
                    description=report_config.get('description', 'Custom generated report'),
                    sections=sections,
                    parameters=parameters,
                    output_formats=output_formats,
                    scheduled_frequency=None
                )
                self.report_templates[f'custom_{datetime.now().strftime("%Y%m%d_%H%M%S")}'] = custom_template
                template_key = list(self.report_templates.keys())[-1]
            else:
                template_key = report_type
            
            template = self.report_templates[template_key]
            
            # Generate report data
            report_data = await self._generate_report_data(template, filters)
            
            # Export in requested formats
            export_results = []
            for format_type in output_formats:
                if format_type in self.supported_formats:
                    export_path = await self._export_report(report_data, format_type, template)
                    export_results.append({
                        'format': format_type,
                        'path': str(export_path),
                        'size': export_path.stat().st_size if export_path.exists() else 0
                    })
            
            return {
                'report_id': report_data.report_id,
                'title': report_data.title,
                'generated_at': report_data.generated_at.isoformat(),
                'sections': template.sections,
                'export_files': export_results,
                'summary': report_data.summary,
                'metadata': {
                    'template_used': template.name,
                    'parameters': parameters,
                    'filters_applied': filters
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating custom report: {e}")
            raise
    
    async def _generate_report_async(self, template: ReportTemplate):
        """Generate report asynchronously"""
        try:
            # Generate report data
            report_data = await self._generate_report_data(template)
            
            # Export in template's preferred formats
            for format_type in template.output_formats:
                if format_type in self.supported_formats:
                    await self._export_report(report_data, format_type, template)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating report {template.name}: {e}")
            raise
    
    async def _generate_report_data(self, template: ReportTemplate, filters: Dict[str, Any] = None) -> ReportData:
        """Generate report data for template"""
        try:
            report_id = f"{template.template_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            title = template.name
            
            # Generate mock data based on template type
            data = await self._generate_mock_data(template, filters)
            
            # Generate charts
            charts = await self._generate_charts(template, data)
            
            # Generate tables
            tables = await self._generate_tables(template, data)
            
            # Generate summary
            summary = await self._generate_report_summary(template, data)
            
            return ReportData(
                report_id=report_id,
                report_type=template.template_type,
                title=title,
                generated_at=datetime.now(),
                data=data,
                charts=charts,
                tables=tables,
                summary=summary
            )
            
        except Exception as e:
            self.logger.error(f"Error generating report data: {e}")
            raise
    
    async def _generate_mock_data(self, template: ReportTemplate, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate mock data for reports"""
        try:
            data = {}
            
            if template.template_type == 'performance':
                data = {
                    'portfolio_summary': {
                        'total_value': 10000000,
                        'daily_pnl': 15000,
                        'ytd_return': 0.125,
                        'sharpe_ratio': 1.45,
                        'max_drawdown': -0.08
                    },
                    'performance_metrics': {
                        'total_return': 0.125,
                        'annualized_return': 0.15,
                        'volatility': 0.12,
                        'sharpe_ratio': 1.45,
                        'sortino_ratio': 1.89,
                        'information_ratio': 0.78,
                        'beta': 0.95,
                        'alpha': 0.023
                    },
                    'attribution_analysis': {
                        'allocation_effect': 0.015,
                        'selection_effect': 0.035,
                        'interaction_effect': 0.008,
                        'total_attribution': 0.058
                    }
                }
            
            elif template.template_type == 'risk':
                data = {
                    'var_analysis': {
                        'var_95': -125000,
                        'var_99': -185000,
                        'cvar_95': -165000,
                        'cvar_99': -245000
                    },
                    'stress_tests': {
                        'market_crash_2008': -2800000,
                        'covid_march_2020': -1850000,
                        'interest_rate_shock': -950000,
                        'credit_spread_widening': -750000
                    },
                    'concentration_risk': {
                        'largest_position': 0.085,
                        'top_5_concentration': 0.35,
                        'sector_concentration': {
                            'Technology': 0.28,
                            'Healthcare': 0.18,
                            'Financials': 0.15
                        }
                    }
                }
            
            elif template.template_type == 'execution':
                data = {
                    'implementation_shortfall': {
                        'total_shortfall': 0.008,
                        'execution_cost': 0.005,
                        'opportunity_cost': 0.003,
                        'timing_cost': 0.002
                    },
                    'market_impact': {
                        'average_impact': 0.006,
                        'permanent_impact': 0.004,
                        'temporary_impact': 0.002,
                        'impact_decay_half_life': 15
                    },
                    'execution_algorithms': {
                        'VWAP': {'usage': 0.45, 'performance_score': 87},
                        'TWAP': {'usage': 0.25, 'performance_score': 82},
                        'Implementation_Shortfall': {'usage': 0.20, 'performance_score': 91},
                        'Percentage_of_Volume': {'usage': 0.10, 'performance_score': 78}
                    }
                }
            
            elif template.template_type == 'attribution':
                data = {
                    'brinson_attribution': {
                        'allocation_effect': 0.018,
                        'selection_effect': 0.042,
                        'interaction_effect': 0.005,
                        'total_attribution': 0.065
                    },
                    'factor_analysis': {
                        'market_factor': 0.089,
                        'size_factor': -0.012,
                        'value_factor': 0.025,
                        'momentum_factor': 0.035,
                        'quality_factor': 0.018
                    }
                }
            
            elif template.template_type == 'dashboard':
                data = {
                    'live_positions': [
                        {'symbol': 'AAPL', 'quantity': 10000, 'pnl': 15000},
                        {'symbol': 'MSFT', 'quantity': 8000, 'pnl': 12000},
                        {'symbol': 'GOOGL', 'quantity': 5000, 'pnl': 8000}
                    ],
                    'daily_summary': {
                        'total_pnl': 35000,
                        'trades_executed': 45,
                        'success_rate': 0.82,
                        'average_fill_time': 2.3
                    }
                }
            
            # Apply filters if provided
            if filters:
                data = self._apply_filters(data, filters)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating mock data: {e}")
            raise
    
    async def _generate_charts(self, template: ReportTemplate, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chart specifications for reports"""
        try:
            charts = []
            
            if 'performance_metrics' in data:
                charts.append({
                    'type': 'bar',
                    'title': 'Performance Metrics',
                    'data': data['performance_metrics'],
                    'x_axis': 'Metric',
                    'y_axis': 'Value',
                    'chart_id': 'performance_metrics_chart'
                })
            
            if 'attribution_analysis' in data:
                charts.append({
                    'type': 'pie',
                    'title': 'Performance Attribution',
                    'data': data['attribution_analysis'],
                    'chart_id': 'attribution_pie_chart'
                })
            
            if 'stress_tests' in data:
                charts.append({
                    'type': 'bar',
                    'title': 'Stress Test Results',
                    'data': data['stress_tests'],
                    'x_axis': 'Scenario',
                    'y_axis': 'PnL Impact ($)',
                    'chart_id': 'stress_test_chart'
                })
            
            if 'live_positions' in data:
                charts.append({
                    'type': 'treemap',
                    'title': 'Current Positions',
                    'data': data['live_positions'],
                    'size_field': 'quantity',
                    'color_field': 'pnl',
                    'chart_id': 'positions_treemap'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return []
    
    async def _generate_tables(self, template: ReportTemplate, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tables for reports"""
        try:
            tables = []
            
            if 'live_positions' in data:
                tables.append({
                    'title': 'Current Positions',
                    'headers': ['Symbol', 'Quantity', 'PnL', 'Weight'],
                    'data': [[pos['symbol'], pos['quantity'], pos['pnl'], 
                             f"{pos['quantity']/sum(p['quantity'] for p in data['live_positions'])*100:.1f}%"]
                            for pos in data['live_positions']],
                    'table_id': 'positions_table'
                })
            
            if 'execution_algorithms' in data:
                algo_data = data['execution_algorithms']
                tables.append({
                    'title': 'Execution Algorithm Performance',
                    'headers': ['Algorithm', 'Usage (%)', 'Performance Score'],
                    'data': [[algo, f"{info['usage']*100:.1f}", info['performance_score']]
                            for algo, info in algo_data.items()],
                    'table_id': 'algorithm_performance_table'
                })
            
            if 'concentration_risk' in data and 'sector_concentration' in data['concentration_risk']:
                sector_data = data['concentration_risk']['sector_concentration']
                tables.append({
                    'title': 'Sector Concentration',
                    'headers': ['Sector', 'Weight (%)'],
                    'data': [[sector, f"{weight*100:.1f}"] for sector, weight in sector_data.items()],
                    'table_id': 'sector_concentration_table'
                })
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Error generating tables: {e}")
            return []
    
    async def _generate_report_summary(self, template: ReportTemplate, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary"""
        try:
            summary = {
                'total_sections': len(template.sections),
                'data_points': len(data),
                'charts_generated': len(template.sections),
                'tables_generated': len(template.sections) // 2,
                'report_quality': 'High'
            }
            
            # Template-specific summaries
            if template.template_type == 'performance':
                summary['key_metrics'] = {
                    'total_return': data.get('performance_metrics', {}).get('total_return', 0),
                    'sharpe_ratio': data.get('performance_metrics', {}).get('sharpe_ratio', 0),
                    'attribution_total': data.get('attribution_analysis', {}).get('total_attribution', 0)
                }
            
            elif template.template_type == 'risk':
                var_95 = data.get('var_analysis', {}).get('var_95', 0)
                summary['risk_summary'] = {
                    'var_95': var_95,
                    'risk_level': 'High' if abs(var_95) > 150000 else 'Medium' if abs(var_95) > 100000 else 'Low',
                    'largest_position': data.get('concentration_risk', {}).get('largest_position', 0)
                }
            
            elif template.template_type == 'execution':
                summary['execution_summary'] = {
                    'implementation_shortfall': data.get('implementation_shortfall', {}).get('total_shortfall', 0),
                    'market_impact': data.get('market_impact', {}).get('average_impact', 0),
                    'best_algorithm': 'Implementation_Shortfall'  # Mock best performer
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating report summary: {e}")
            return {}
    
    async def _export_report(self, report_data: ReportData, format_type: str, template: ReportTemplate) -> Path:
        """Export report in specified format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_data.report_id}_{timestamp}.{format_type}"
            export_path = self.storage_paths['exports'] / filename
            
            if format_type == 'pdf':
                await self._export_pdf(report_data, export_path, template)
            elif format_type == 'excel':
                await self._export_excel(report_data, export_path, template)
            elif format_type == 'html':
                await self._export_html(report_data, export_path, template)
            elif format_type == 'json':
                await self._export_json(report_data, export_path, template)
            elif format_type == 'csv':
                await self._export_csv(report_data, export_path, template)
            
            self.logger.info(f"Report exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise
    
    async def _export_pdf(self, report_data: ReportData, export_path: Path, template: ReportTemplate):
        """Export report as PDF"""
        # Mock PDF generation - in reality would use libraries like ReportLab
        pdf_content = f"""
        PDF Report: {report_data.title}
        
        Generated: {report_data.generated_at}
        Report ID: {report_data.report_id}
        
        Summary:
        {json.dumps(report_data.summary, indent=2)}
        
        Data:
        {json.dumps(report_data.data, indent=2)}
        """
        
        with open(export_path, 'w') as f:
            f.write(pdf_content)
    
    async def _export_excel(self, report_data: ReportData, export_path: Path, template: ReportTemplate):
        """Export report as Excel"""
        # Mock Excel generation - in reality would use openpyxl or xlsxwriter
        excel_data = {
            'Summary': pd.DataFrame([report_data.summary]),
            'Data': pd.DataFrame([report_data.data]),
            'Charts': pd.DataFrame(report_data.charts),
            'Tables': pd.DataFrame(report_data.tables)
        }
        
        # Save as JSON for mock (would be Excel in reality)
        with open(export_path.with_suffix('.json'), 'w') as f:
            json.dump({k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in excel_data.items()}, f, indent=2)
    
    async def _export_html(self, report_data: ReportData, export_path: Path, template: ReportTemplate):
        """Export report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .chart {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_data.title}</h1>
                <p>Generated: {report_data.generated_at}</p>
                <p>Report ID: {report_data.report_id}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <pre>{json.dumps(report_data.summary, indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Data</h2>
                <pre>{json.dumps(report_data.data, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(export_path, 'w') as f:
            f.write(html_content)
    
    async def _export_json(self, report_data: ReportData, export_path: Path, template: ReportTemplate):
        """Export report as JSON"""
        export_data = {
            'report_metadata': {
                'report_id': report_data.report_id,
                'title': report_data.title,
                'generated_at': report_data.generated_at.isoformat(),
                'template_type': report_data.report_type
            },
            'summary': report_data.summary,
            'data': report_data.data,
            'charts': report_data.charts,
            'tables': report_data.tables
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    async def _export_csv(self, report_data: ReportData, export_path: Path, template: ReportTemplate):
        """Export report as CSV"""
        # Export main data as CSV
        df = pd.DataFrame([report_data.data])
        df.to_csv(export_path, index=False)
    
    async def send_email_report(self, report_path: Path, recipients: List[str], subject: str = None):
        """Send report via email (mock implementation)"""
        try:
            if not subject:
                subject = f"Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            self.logger.info(f"Sending report to {len(recipients)} recipients")
            
            # Mock email sending - in reality would use smtplib
            email_result = {
                'status': 'sent',
                'recipients': recipients,
                'subject': subject,
                'attachment': str(report_path),
                'timestamp': datetime.now().isoformat()
            }
            
            return email_result
            
        except Exception as e:
            self.logger.error(f"Error sending email report: {e}")
            raise
    
    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to report data"""
        # Mock filter implementation
        filtered_data = data.copy()
        
        # Apply date filters
        if 'start_date' in filters:
            # Would filter data by date range
            pass
        
        if 'symbols' in filters:
            # Would filter positions by symbols
            if 'live_positions' in filtered_data:
                filtered_data['live_positions'] = [
                    pos for pos in filtered_data['live_positions']
                    if pos['symbol'] in filters['symbols']
                ]
        
        return filtered_data
    
    async def get_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available report templates"""
        try:
            templates_info = {}
            for key, template in self.report_templates.items():
                templates_info[key] = {
                    'name': template.name,
                    'template_type': template.template_type,
                    'description': template.description,
                    'sections': template.sections,
                    'scheduled_frequency': template.scheduled_frequency,
                    'supported_formats': template.output_formats
                }
            
            return templates_info
            
        except Exception as e:
            self.logger.error(f"Error getting report templates: {e}")
            raise
    
    async def create_custom_template(self, template_config: Dict[str, Any]) -> str:
        """Create custom report template"""
        try:
            template_name = template_config['name']
            template_id = f"custom_{template_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            custom_template = ReportTemplate(
                name=template_config['name'],
                template_type='custom',
                description=template_config.get('description', 'Custom template'),
                sections=template_config.get('sections', []),
                parameters=template_config.get('parameters', {}),
                output_formats=template_config.get('output_formats', ['html']),
                scheduled_frequency=template_config.get('scheduled_frequency')
            )
            
            self.report_templates[template_id] = custom_template
            
            return template_id
            
        except Exception as e:
            self.logger.error(f"Error creating custom template: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for automated reporter"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'templates_available': len(self.report_templates),
                'supported_formats': self.supported_formats,
                'storage_paths': {k: str(v) for k, v in self.storage_paths.items()}
            }
        except Exception as e:
            self.logger.error(f"Error in automated reporter health check: {e}")
            return {'status': 'error', 'error': str(e)}