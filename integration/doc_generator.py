"""
Documentation Generator
Automatically generates API documentation, user guides, and system documentation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import inspect
import importlib.util
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Automatic documentation generator for the system."""
    
    def __init__(self, output_dir: str = "docs/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Documentation templates
        self.templates = {
            "api_doc": self._get_api_doc_template(),
            "user_guide": self._get_user_guide_template(),
            "deployment_guide": self._get_deployment_guide_template(),
            "troubleshooting": self._get_troubleshooting_template()
        }
        
        # API documentation structure
        self.api_docs = {
            "endpoints": [],
            "models": [],
            "schemas": [],
            "authentication": {},
            "examples": []
        }
    
    def _get_api_doc_template(self) -> str:
        """Get API documentation template."""
        return """# {title}

Generated on: {timestamp}
Version: {version}

## Overview
{description}

## Authentication
{auth_info}

## Endpoints

{endpoints}

## Models

{models}

## Examples

{examples}

## Error Codes

{error_codes}
"""
    
    def _get_user_guide_template(self) -> str:
        """Get user guide template."""
        return """# {title}

Generated on: {timestamp}

## Table of Contents
{toc}

## Getting Started
{getting_started}

## User Interface Guide
{ui_guide}

## Features
{features}

## FAQ
{faq}

## Support
{support}
"""
    
    def _get_deployment_guide_template(self) -> str:
        """Get deployment guide template."""
        return """# Deployment Guide

Generated on: {timestamp}

## Prerequisites
{prerequisites}

## Installation
{installation}

## Configuration
{configuration}

## Production Deployment
{production_deployment}

## Monitoring
{monitoring}

## Troubleshooting
{troubleshooting}
"""
    
    def _get_troubleshooting_template(self) -> str:
        """Get troubleshooting guide template."""
        return """# Troubleshooting Guide

Generated on: {timestamp}

## Common Issues
{common_issues}

## Error Messages
{error_messages}

## Performance Issues
{performance_issues}

## Getting Help
{getting_help}
"""
    
    def scan_api_endpoints(self, project_root: str = ".") -> Dict[str, Any]:
        """Scan project for API endpoints and generate documentation."""
        logger.info("Scanning API endpoints...")
        
        # Scan for FastAPI/Flask routes
        endpoints = self._scan_fastapi_routes(project_root)
        if not endpoints:
            endpoints = self._scan_flask_routes(project_root)
        
        # Scan for models/schemas
        models = self._scan_data_models(project_root)
        
        # Generate API documentation
        api_doc = self._generate_api_documentation(endpoints, models)
        
        return api_doc
    
    def _scan_fastapi_routes(self, project_root: str) -> List[Dict[str, Any]]:
        """Scan for FastAPI routes."""
        routes = []
        project_path = Path(project_root)
        
        # Look for FastAPI files
        fastapi_files = list(project_path.rglob("*.py"))
        
        for file_path in fastapi_files:
            try:
                # This is a simplified approach - would need proper AST parsing in production
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Look for route decorators
                import re
                route_patterns = [
                    r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                    r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                    r'def\s+(\w+)\s*\([^)]*\):\s*\n\s*.*["\']{3}([^"\']*)["\']{3}'
                ]
                
                for pattern in route_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        if len(match) == 2:  # method and path
                            method, path = match
                            routes.append({
                                "method": method.upper(),
                                "path": path,
                                "file": str(file_path),
                                "description": f"Auto-detected {method} {path} endpoint"
                            })
                        elif len(match) == 2:  # function and docstring
                            func_name, description = match
                            # Find corresponding route
                            route_pattern = rf'@.*\.(get|post|put|delete|patch)\(["\']([^"\']+)["\'].*def\s+{func_name}\s*\('
                            route_match = re.search(route_pattern, content, re.MULTILINE)
                            if route_match:
                                routes.append({
                                    "method": route_match.group(1).upper(),
                                    "path": route_match.group(2),
                                    "function": func_name,
                                    "description": description.strip() if description else f"Auto-detected endpoint {func_name}",
                                    "file": str(file_path)
                                })
            
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {str(e)}")
        
        return routes
    
    def _scan_flask_routes(self, project_root: str) -> List[Dict[str, Any]]:
        """Scan for Flask routes."""
        routes = []
        project_path = Path(project_root)
        
        # Look for Flask files
        flask_files = list(project_path.rglob("*.py"))
        
        for file_path in flask_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for Flask route decorators
                import re
                route_pattern = r'@app\.route\(["\']([^"\']+)["\'](?:,\s*methods=\[([^\]]+)\])?\)\s*def\s+(\w+)'
                
                matches = re.findall(route_pattern, content, re.MULTILINE)
                for match in matches:
                    path, methods, func_name = match
                    method_list = [m.strip().upper() for m in methods.split(',')] if methods else ['GET']
                    
                    routes.append({
                        "methods": method_list,
                        "path": path,
                        "function": func_name,
                        "description": f"Auto-detected Flask endpoint {func_name}",
                        "file": str(file_path)
                    })
            
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {str(e)}")
        
        return routes
    
    def _scan_data_models(self, project_root: str) -> List[Dict[str, Any]]:
        """Scan for data models and schemas."""
        models = []
        project_path = Path(project_root)
        
        # Look for model files
        model_files = list(project_path.rglob("*.py"))
        
        for file_path in model_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for common model patterns
                import re
                model_patterns = [
                    r'class\s+(\w+)\(.*\):',
                    r'@dataclass',
                    r'class\s+(\w+)Schema.*:',
                    r'class\s+(\w+)Model.*:'
                ]
                
                for pattern in model_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    for match in matches:
                        model_name = match if isinstance(match, str) else match[0]
                        if model_name and not model_name.startswith('_'):
                            # Extract class definition
                            class_pattern = rf'class\s+{model_name}\s*\([^)]*\):.*?(?=\n\n|\nclass|\Z)'
                            class_match = re.search(class_pattern, content, re.MULTILINE | re.DOTALL)
                            
                            if class_match:
                                class_content = class_match.group(0)
                                
                                # Extract fields
                                field_pattern = r'(\w+):\s*([^\n=]+)'
                                fields = re.findall(field_pattern, class_content)
                                
                                models.append({
                                    "name": model_name,
                                    "fields": [{"name": name, "type": type.strip()} for name, type in fields],
                                    "file": str(file_path),
                                    "description": f"Auto-detected model {model_name}"
                                })
            
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {str(e)}")
        
        return models
    
    def _generate_api_documentation(self, endpoints: List[Dict[str, Any]], 
                                  models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive API documentation."""
        doc = {
            "title": "Trading System API",
            "version": "1.0.0",
            "description": "Comprehensive trading system API documentation",
            "timestamp": datetime.now().isoformat(),
            "endpoints": self._format_endpoints(endpoints),
            "models": self._format_models(models),
            "examples": self._generate_api_examples(endpoints),
            "error_codes": self._generate_error_codes(),
            "auth_info": self._generate_auth_info()
        }
        
        return doc
    
    def _format_endpoints(self, endpoints: List[Dict[str, Any]]) -> str:
        """Format endpoints for documentation."""
        if not endpoints:
            return "No endpoints detected."
        
        formatted = ""
        for endpoint in endpoints:
            formatted += f"### {endpoint['method']} {endpoint['path']}\n\n"
            formatted += f"**Description:** {endpoint.get('description', 'No description available')}\n\n"
            
            if 'function' in endpoint:
                formatted += f"**Function:** `{endpoint['function']}`\n\n"
            
            formatted += f"**File:** `{endpoint['file']}`\n\n"
            
            # Add example request/response
            formatted += "**Example Request:**\n"
            formatted += f"```bash\ncurl -X {endpoint['method']} {endpoint['path']}\n```\n\n"
            formatted += "**Example Response:**\n"
            formatted += "```json\n{}\n```\n\n"
            formatted += "---\n\n"
        
        return formatted
    
    def _format_models(self, models: List[Dict[str, Any]]) -> str:
        """Format models for documentation."""
        if not models:
            return "No models detected."
        
        formatted = ""
        for model in models:
            formatted += f"### {model['name']}\n\n"
            formatted += f"**Description:** {model.get('description', 'No description available')}\n\n"
            formatted += "**Fields:**\n\n"
            
            for field in model['fields']:
                formatted += f"- `{field['name']}`: {field['type']}\n"
            
            formatted += f"\n**File:** `{model['file']}`\n\n"
            formatted += "---\n\n"
        
        return formatted
    
    def _generate_api_examples(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate API examples."""
        examples = []
        
        # Common API examples
        examples.append({
            "title": "Authentication",
            "description": "How to authenticate with the API",
            "code": """curl -X POST /api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "user", "password": "pass"}'"""
        })
        
        examples.append({
            "title": "Get Market Data",
            "description": "Retrieve current market data",
            "code": """curl -X GET /api/market-data \\
  -H "Authorization: Bearer <token>" """
        })
        
        return examples
    
    def _generate_error_codes(self) -> Dict[str, str]:
        """Generate common error codes."""
        return {
            "400": "Bad Request - Invalid request format",
            "401": "Unauthorized - Authentication required",
            "403": "Forbidden - Insufficient permissions",
            "404": "Not Found - Resource does not exist",
            "429": "Too Many Requests - Rate limit exceeded",
            "500": "Internal Server Error - Server error occurred"
        }
    
    def _generate_auth_info(self) -> str:
        """Generate authentication information."""
        return """The API uses token-based authentication. Include the following header in your requests:

```
Authorization: Bearer <your-token>
```

To obtain a token, use the login endpoint with valid credentials.

### Rate Limiting
API requests are rate limited to prevent abuse. If you exceed the rate limit, you will receive a 429 response.
"""
    
    def generate_user_guide(self, components: List[str]) -> Dict[str, Any]:
        """Generate user guide for the system."""
        logger.info("Generating user guide...")
        
        toc = self._generate_table_of_contents(components)
        getting_started = self._generate_getting_started()
        ui_guide = self._generate_ui_guide()
        features = self._generate_features_section()
        faq = self._generate_faq()
        support = self._generate_support_info()
        
        guide = {
            "title": "Trading System User Guide",
            "timestamp": datetime.now().isoformat(),
            "toc": toc,
            "getting_started": getting_started,
            "ui_guide": ui_guide,
            "features": features,
            "faq": faq,
            "support": support
        }
        
        return guide
    
    def _generate_table_of_contents(self, components: List[str]) -> str:
        """Generate table of contents."""
        toc = "1. [Getting Started](#getting-started)\n"
        toc += "2. [User Interface Guide](#user-interface-guide)\n"
        toc += "3. [Features](#features)\n"
        toc += "4. [FAQ](#faq)\n"
        toc += "5. [Support](#support)\n"
        
        for component in components:
            toc += f"   - [{component.title()}](#{component.replace(' ', '-').lower()})\n"
        
        return toc
    
    def _generate_getting_started(self) -> str:
        """Generate getting started section."""
        return """## Welcome to the Trading System

This guide will help you get started with our comprehensive trading platform.

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- Valid trading account credentials

### First Steps
1. **Access the System**: Navigate to your trading platform URL
2. **Log In**: Use your credentials to authenticate
3. **Dashboard**: Familiarize yourself with the main dashboard
4. **Configure Settings**: Set up your preferences and risk parameters

### Key Concepts
- **Orders**: Instructions to buy or sell securities
- **Positions**: Current holdings in your portfolio
- **Risk Management**: Tools to protect your investments
- **Analytics**: Performance and market analysis tools
"""
    
    def _generate_ui_guide(self) -> str:
        """Generate user interface guide."""
        return """## User Interface Guide

### Main Dashboard
The dashboard provides an overview of your trading activity:
- **Portfolio Summary**: Current position values and performance
- **Market Overview**: Real-time market data and trends
- **Recent Activity**: Latest trades and order status

### Trading Interface
- **Order Entry**: Place buy/sell orders with custom parameters
- **Position Management**: View and manage your current positions
- **Order History**: Track all your past orders

### Analytics Tools
- **Performance Charts**: Visualize your trading performance
- **Risk Analysis**: Monitor your portfolio risk metrics
- **Market Research**: Access market data and analysis

### Navigation
- Use the sidebar menu to navigate between sections
- All data updates in real-time
- Tooltips provide additional information
"""
    
    def _generate_features_section(self) -> str:
        """Generate features section."""
        return """## Features

### Automated Trading
- **Strategy Execution**: Run automated trading strategies
- **Risk Controls**: Automatic risk management
- **Backtesting**: Test strategies on historical data

### Risk Management
- **Position Limits**: Set maximum position sizes
- **Stop Loss**: Automatic loss limitation
- **Portfolio Analysis**: Comprehensive risk assessment

### Market Data
- **Real-time Quotes**: Live market data
- **Historical Data**: Access to historical price data
- **Market Analytics**: Advanced market analysis tools

### Reporting
- **Performance Reports**: Detailed performance analysis
- **Trade Reports**: Comprehensive trade history
- **Compliance Reports**: Regulatory compliance tracking
"""
    
    def _generate_faq(self) -> str:
        """Generate FAQ section."""
        return """## Frequently Asked Questions

### General Questions

**Q: How do I place an order?**
A: Navigate to the trading interface, select your security, choose order type (market/limit), specify quantity, and submit.

**Q: What are the system hours?**
A: The system operates during regular market hours (9:30 AM - 4:00 PM EST) for trading. Analytics and portfolio tools are available 24/7.

**Q: How do I check my portfolio performance?**
A: Use the dashboard or portfolio section to view real-time performance metrics and historical performance charts.

### Technical Questions

**Q: What if I lose my internet connection?**
A: The system automatically saves your work. You can log back in from any device to continue.

**Q: Are my transactions secure?**
A: Yes, all transactions are encrypted and secured using industry-standard security protocols.

### Support

**Q: Who do I contact for help?**
A: Use the support section or contact our support team at support@tradingsystem.com

**Q: Is there training available?**
A: Yes, we offer online tutorials and documentation to help you get started.
"""
    
    def _generate_support_info(self) -> str:
        """Generate support information."""
        return """## Support

### Getting Help
- **Documentation**: Comprehensive guides and API documentation
- **Support Portal**: Submit tickets and track requests
- **Live Chat**: Real-time assistance during business hours
- **Email Support**: support@tradingsystem.com

### Training Resources
- **Video Tutorials**: Step-by-step video guides
- **Interactive Demos**: Hands-on learning experiences
- **Webinars**: Regular training sessions
- **Documentation**: Detailed user guides and references

### Contact Information
- **Technical Support**: tech-support@tradingsystem.com
- **General Inquiries**: info@tradingsystem.com
- **Phone Support**: 1-800-TRADING (available during market hours)

### System Status
Check our status page at https://status.tradingsystem.com for real-time system status updates.
"""
    
    def generate_deployment_guide(self, environment: str = "production") -> Dict[str, Any]:
        """Generate deployment guide."""
        logger.info(f"Generating deployment guide for {environment}...")
        
        prerequisites = self._generate_prerequisites()
        installation = self._generate_installation_guide()
        configuration = self._generate_configuration_guide()
        production_deployment = self._generate_production_deployment()
        monitoring = self._generate_monitoring_guide()
        troubleshooting = self._generate_deployment_troubleshooting()
        
        guide = {
            "title": f"{environment.title()} Deployment Guide",
            "timestamp": datetime.now().isoformat(),
            "prerequisites": prerequisites,
            "installation": installation,
            "configuration": configuration,
            "production_deployment": production_deployment,
            "monitoring": monitoring,
            "troubleshooting": troubleshooting
        }
        
        return guide
    
    def _generate_prerequisites(self) -> str:
        """Generate prerequisites section."""
        return """## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Database**: PostgreSQL 12+ or MySQL 8+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 100GB available space minimum

### Required Software
- Docker and Docker Compose
- Nginx (for reverse proxy)
- Redis (for caching and sessions)
- SSL certificates (for HTTPS)

### Network Requirements
- Open ports: 80, 443, 5432 (database), 6379 (Redis)
- Firewall configuration
- Load balancer (for production)

### User Accounts
- System user for application
- Database user with appropriate permissions
- SSL certificate management
"""
    
    def _generate_installation_guide(self) -> str:
        """Generate installation guide."""
        return """## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/trading-system.git
cd trading-system
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Database Setup
```bash
# Create database
createdb trading_system

# Run migrations
python manage.py migrate
```

### 4. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install system dependencies
sudo apt-get install postgresql-client redis-tools nginx
```

### 5. Configure Services
```bash
# Copy service configurations
sudo cp deployment/nginx.conf /etc/nginx/sites-available/trading-system
sudo ln -s /etc/nginx/sites-available/trading-system /etc/nginx/sites-enabled/

# Enable and start services
sudo systemctl enable postgresql redis nginx
sudo systemctl start postgresql redis nginx
```

### 6. SSL Setup
```bash
# Install Let's Encrypt certificates
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```
"""
    
    def _generate_configuration_guide(self) -> str:
        """Generate configuration guide."""
        return """## Configuration

### Environment Variables
Update your `.env` file with the following essential settings:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/trading_system
DATABASE_MAX_CONNECTIONS=20

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com
SECURE_SSL_REDIRECT=True

# API Configuration
API_RATE_LIMIT=1000/hour
API_TIMEOUT=30

# Trading Settings
DEFAULT_CURRENCY=USD
MARKET_DATA_PROVIDER=binance
TRADING_ENABLED=True

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn
```

### Application Settings
- **Database Pool**: Configure connection pooling for optimal performance
- **Caching**: Set up Redis for session storage and caching
- **Security**: Configure HTTPS, CSRF protection, and CORS
- **Logging**: Set appropriate log levels and log rotation

### Performance Tuning
- **Worker Processes**: Configure Gunicorn/uWSGI workers
- **Database Optimization**: Tune database parameters
- **Cache Settings**: Configure Redis memory usage
- **Static Files**: Set up CDN for static assets
"""
    
    def _generate_production_deployment(self) -> str:
        """Generate production deployment guide."""
        return """## Production Deployment

### 1. Pre-Deployment Checklist
- [ ] SSL certificates installed and verified
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Log rotation configured
- [ ] Security scanning completed
- [ ] Load balancer configured
- [ ] Firewall rules updated

### 2. Deploy Application
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run database migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Restart services
sudo systemctl restart trading-system
```

### 3. Health Checks
```bash
# Check application health
curl -f http://localhost/health

# Check database connectivity
python manage.py dbshell

# Check Redis connectivity
redis-cli ping
```

### 4. Load Balancing (Optional)
If using multiple application servers:

```yaml
# nginx.conf upstream configuration
upstream trading_app {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}
```

### 5. Database Scaling
- Configure read replicas for read-heavy workloads
- Set up connection pooling (PgBouncer)
- Monitor database performance

### 6. Monitoring Setup
- Application performance monitoring (APM)
- Database monitoring
- Server resource monitoring
- Log aggregation
- Alert configuration
"""
    
    def _generate_monitoring_guide(self) -> str:
        """Generate monitoring guide."""
        return """## Monitoring

### Application Monitoring
- **Health Endpoints**: `/health` and `/metrics`
- **Performance Metrics**: Response times, throughput
- **Error Rates**: Track application errors
- **Business Metrics**: Trade volumes, user activity

### System Monitoring
- **CPU/Memory Usage**: Server resource utilization
- **Disk Space**: Database and log file storage
- **Network Traffic**: Bandwidth and connection counts
- **Process Health**: Application process status

### Database Monitoring
- **Connection Pool**: Active and idle connections
- **Query Performance**: Slow query identification
- **Replication Lag**: For read replicas
- **Backup Status**: Backup completion and verification

### Security Monitoring
- **Failed Login Attempts**: Authentication failures
- **Unusual Access Patterns**: Potential security threats
- **SSL Certificate Expiry**: Automated certificate renewal
- **Vulnerability Scans**: Regular security assessments

### Alert Configuration
Set up alerts for:
- High error rates (>5%)
- Slow response times (>2 seconds)
- High memory usage (>80%)
- Disk space low (<10% free)
- Database connection failures
- SSL certificate expiry (within 30 days)

### Log Management
- **Centralized Logging**: Aggregate logs from all services
- **Log Retention**: Configure appropriate retention periods
- **Log Analysis**: Search and analyze logs for issues
- **Security Logs**: Monitor for security events
"""
    
    def _generate_deployment_troubleshooting(self) -> str:
        """Generate deployment troubleshooting guide."""
        return """## Deployment Troubleshooting

### Common Issues

#### Application Won't Start
1. Check service status: `sudo systemctl status trading-system`
2. Review application logs: `sudo journalctl -u trading-system`
3. Verify environment variables: `cat .env`
4. Check database connectivity: `python manage.py dbshell`

#### Database Connection Issues
1. Verify database is running: `sudo systemctl status postgresql`
2. Check connection string: `echo $DATABASE_URL`
3. Test connection: `psql $DATABASE_URL`
4. Review connection pool settings

#### High Memory Usage
1. Check process memory: `ps aux | grep trading-system`
2. Review application logs for memory leaks
3. Monitor database connections
4. Consider adding more memory or optimizing queries

#### Slow Performance
1. Check server resources: `top`, `htop`, `iotop`
2. Review database performance
3. Check network connectivity
4. Monitor application metrics

#### SSL Certificate Issues
1. Check certificate status: `sudo certbot certificates`
2. Test SSL configuration: `openssl s_client -connect your-domain.com:443`
3. Renew certificates: `sudo certbot renew`
4. Review nginx configuration

### Recovery Procedures

#### Application Recovery
```bash
# Stop services
sudo systemctl stop trading-system

# Revert to previous version
git checkout <previous-commit>

# Restart services
sudo systemctl start trading-system
```

#### Database Recovery
```bash
# Restore from backup
psql $DATABASE_URL < /backup/latest.sql

# Verify restoration
python manage.py dbshell
```

#### Configuration Rollback
```bash
# Restore previous configuration
git checkout HEAD~1 .env

# Restart services
sudo systemctl restart trading-system
```

### Emergency Contacts
- **System Administrator**: admin@yourcompany.com
- **Database Administrator**: dba@yourcompany.com
- **Security Team**: security@yourcompany.com

### Escalation Procedures
1. **Level 1**: Check logs and basic diagnostics
2. **Level 2**: Restart services and verify connectivity
3. **Level 3**: Contact system administrators
4. **Level 4**: Emergency response team activation
"""
    
    def generate_troubleshooting_guide(self, common_issues: List[str] = None) -> Dict[str, Any]:
        """Generate troubleshooting guide."""
        logger.info("Generating troubleshooting guide...")
        
        if common_issues is None:
            common_issues = [
                "Login issues",
                "Trading functionality",
                "Performance problems",
                "Data sync issues",
                "Report generation"
            ]
        
        common_issues_section = self._format_common_issues(common_issues)
        error_messages = self._generate_error_messages_section()
        performance_issues = self._generate_performance_issues_section()
        getting_help = self._generate_getting_help_section()
        
        guide = {
            "title": "Troubleshooting Guide",
            "timestamp": datetime.now().isoformat(),
            "common_issues": common_issues_section,
            "error_messages": error_messages,
            "performance_issues": performance_issues,
            "getting_help": getting_help
        }
        
        return guide
    
    def _format_common_issues(self, issues: List[str]) -> str:
        """Format common issues section."""
        formatted = ""
        for issue in issues:
            formatted += f"### {issue}\n\n"
            formatted += "**Symptoms:**\n"
            formatted += "- Describe what the user might experience\n"
            formatted += "- List any error messages\n\n"
            formatted += "**Possible Causes:**\n"
            formatted += "- Cause 1\n"
            formatted += "- Cause 2\n\n"
            formatted += "**Solution:**\n"
            formatted += "1. Step-by-step solution\n"
            formatted += "2. Additional steps if needed\n\n"
            formatted += "---\n\n"
        
        return formatted
    
    def _generate_error_messages_section(self) -> str:
        """Generate error messages section."""
        return """## Error Messages

### Authentication Errors

**"Invalid credentials"**
- Check username and password
- Verify account is active
- Reset password if necessary

**"Session expired"**
- Log in again
- Check if browser cookies are enabled
- Clear browser cache

### Trading Errors

**"Insufficient funds"**
- Check account balance
- Verify available buying power
- Reduce order size

**"Market closed"**
- Check market hours
- Try after-hours trading if enabled
- Review holiday calendar

### System Errors

**"Connection timeout"**
- Check internet connection
- Try refreshing the page
- Contact support if persistent

**"Server error"**
- Try again in a few minutes
- Check system status page
- Contact support if urgent
"""
    
    def _generate_performance_issues_section(self) -> str:
        """Generate performance issues section."""
        return """## Performance Issues

### Slow Page Loading
1. **Check Internet Connection**
   - Test connection speed
   - Try different network

2. **Browser Issues**
   - Clear browser cache
   - Disable browser extensions
   - Try incognito/private mode

3. **System Performance**
   - Close unnecessary applications
   - Check available system resources
   - Restart browser

### Data Not Updating
1. **Refresh Page**
   - Press F5 or click refresh button
   - Wait for data to reload

2. **Check Real-time Data**
   - Verify market is open
   - Check connection to data provider

3. **Browser Settings**
   - Enable JavaScript
   - Check for popup blockers
   - Allow notifications

### Chart Display Issues
1. **Browser Compatibility**
   - Use supported browser
   - Update browser to latest version

2. **JavaScript Issues**
   - Enable JavaScript
   - Check for security software blocking scripts

3. **Display Settings**
   - Check screen resolution
   - Adjust zoom level
   - Try fullscreen mode
"""
    
    def _generate_getting_help_section(self) -> str:
        """Generate getting help section."""
        return """## Getting Help

### Self-Service Resources
- **Knowledge Base**: Search for solutions to common issues
- **Video Tutorials**: Step-by-step visual guides
- **Community Forum**: Ask questions and share experiences
- **Documentation**: Comprehensive guides and references

### Direct Support
- **Live Chat**: Available during business hours
- **Email Support**: support@tradingsystem.com
- **Phone Support**: 1-800-TRADING
- **Ticket System**: Submit detailed support requests

### Emergency Support
- **Critical Issues**: 24/7 emergency line
- **System Outages**: Real-time status updates
- **Data Issues**: Priority support for data problems

### Before Contacting Support
1. **Gather Information**
   - Error messages
   - Steps to reproduce
   - Browser/system information
   - Account details

2. **Try Basic Troubleshooting**
   - Refresh page
   - Clear cache
   - Try different browser
   - Check system status

3. **Prepare Details**
   - Time of issue
   - Actions taken
   - Expected vs actual behavior
   - Screenshots if helpful

### Support Hours
- **General Support**: Monday-Friday, 8 AM - 6 PM EST
- **Emergency Support**: 24/7 for critical issues
- **Live Chat**: Monday-Friday, 9 AM - 5 PM EST

### Feedback
We value your feedback to improve our system:
- **Feature Requests**: Submit through support portal
- **Bug Reports**: Use the issue tracking system
- **General Feedback**: feedback@tradingsystem.com
"""
    
    def save_documentation(self, doc_type: str, content: Dict[str, Any]) -> str:
        """Save generated documentation to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if doc_type == "api":
            filename = f"api_documentation_{timestamp}.md"
            content_str = self._format_api_doc(content)
        elif doc_type == "user_guide":
            filename = f"user_guide_{timestamp}.md"
            content_str = self._format_user_guide(content)
        elif doc_type == "deployment":
            filename = f"deployment_guide_{timestamp}.md"
            content_str = self._format_deployment_doc(content)
        elif doc_type == "troubleshooting":
            filename = f"troubleshooting_guide_{timestamp}.md"
            content_str = self._format_troubleshooting_doc(content)
        else:
            filename = f"{doc_type}_documentation_{timestamp}.md"
            content_str = json.dumps(content, indent=2)
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(content_str)
        
        logger.info(f"Documentation saved: {filepath}")
        return str(filepath)
    
    def _format_api_doc(self, content: Dict[str, Any]) -> str:
        """Format API documentation."""
        template = self.templates["api_doc"]
        return template.format(
            title=content["title"],
            timestamp=content["timestamp"],
            version=content["version"],
            description=content["description"],
            auth_info=content["auth_info"],
            endpoints=content["endpoints"],
            models=content["models"],
            examples=self._format_api_examples(content["examples"]),
            error_codes=self._format_error_codes(content["error_codes"])
        )
    
    def _format_user_guide(self, content: Dict[str, Any]) -> str:
        """Format user guide."""
        template = self.templates["user_guide"]
        return template.format(
            title=content["title"],
            timestamp=content["timestamp"],
            toc=content["toc"],
            getting_started=content["getting_started"],
            ui_guide=content["ui_guide"],
            features=content["features"],
            faq=content["faq"],
            support=content["support"]
        )
    
    def _format_deployment_doc(self, content: Dict[str, Any]) -> str:
        """Format deployment documentation."""
        template = self.templates["deployment_guide"]
        return template.format(
            timestamp=content["timestamp"],
            prerequisites=content["prerequisites"],
            installation=content["installation"],
            configuration=content["configuration"],
            production_deployment=content["production_deployment"],
            monitoring=content["monitoring"],
            troubleshooting=content["troubleshooting"]
        )
    
    def _format_troubleshooting_doc(self, content: Dict[str, Any]) -> str:
        """Format troubleshooting documentation."""
        template = self.templates["troubleshooting"]
        return template.format(
            timestamp=content["timestamp"],
            common_issues=content["common_issues"],
            error_messages=content["error_messages"],
            performance_issues=content["performance_issues"],
            getting_help=content["getting_help"]
        )
    
    def _format_api_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format API examples."""
        formatted = ""
        for example in examples:
            formatted += f"### {example['title']}\n\n"
            formatted += f"{example['description']}\n\n"
            formatted += "```bash\n"
            formatted += example['code']
            formatted += "\n```\n\n"
        return formatted
    
    def _format_error_codes(self, error_codes: Dict[str, str]) -> str:
        """Format error codes."""
        formatted = "| Error Code | Description |\n"
        formatted += "|------------|-------------|\n"
        for code, description in error_codes.items():
            formatted += f"| {code} | {description} |\n"
        return formatted
    
    def generate_complete_documentation(self, project_root: str = ".") -> Dict[str, str]:
        """Generate complete documentation package."""
        logger.info("Generating complete documentation package...")
        
        generated_files = {}
        
        # Generate API documentation
        api_doc = self.scan_api_endpoints(api_doc)
        if api_doc:
            api_file = self.save_documentation("api", api_doc)
            generated_files["api"] = api_file
        
        # Generate user guide
        components = ["dashboard", "trading", "analytics", "reports"]
        user_guide = self.generate_user_guide(components)
        user_file = self.save_documentation("user_guide", user_guide)
        generated_files["user_guide"] = user_file
        
        # Generate deployment guide
        deployment_guide = self.generate_deployment_guide("production")
        deploy_file = self.save_documentation("deployment", deployment_guide)
        generated_files["deployment"] = deploy_file
        
        # Generate troubleshooting guide
        troubleshooting_guide = self.generate_troubleshooting_guide()
        troubleshoot_file = self.save_documentation("troubleshooting", troubleshooting_guide)
        generated_files["troubleshooting"] = troubleshoot_file
        
        logger.info(f"Documentation generation complete: {len(generated_files)} files created")
        return generated_files


# Global documentation generator instance
doc_generator = DocumentationGenerator()