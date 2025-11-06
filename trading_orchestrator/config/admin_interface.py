"""
Configuration Admin Interface
Web-based admin interface for managing trading orchestrator configurations
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import asyncio
import uvicorn
import os
import hashlib
from contextlib import asynccontextmanager

from .config_manager import get_config_manager, ComprehensiveConfigManager, ConfigType
from .validator import ConfigValidator
from .version_manager import ConfigVersionManager
from .migration import ConfigMigrationManager
from .encryption import ConfigEncryption
from .audit_logger import get_audit_logger


# Pydantic models for API
class ConfigUpdateRequest(BaseModel):
    config_name: str
    key_path: str
    value: Any
    encrypt: bool = False


class ConfigCreateRequest(BaseModel):
    config_name: str
    config_type: str = "json"
    template_name: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


class ConfigImportRequest(BaseModel):
    config_name: str
    file_path: str
    overwrite: bool = False


class TemplateApplyRequest(BaseModel):
    config_name: str
    template_name: str
    variables: Optional[Dict[str, Any]] = None
    merge: bool = True


class MigrationRequest(BaseModel):
    config_name: str
    target_version: str
    environment: Optional[str] = None
    dry_run: bool = False


class SecurityUpdateRequest(BaseModel):
    config_name: str
    security_level: str = "standard"
    dry_run: bool = False


class AdminConfigResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None


class ConfigStatusResponse(BaseModel):
    config_name: str
    status: str
    last_modified: Optional[str]
    size_bytes: Optional[int]
    encrypted_fields: int
    recent_changes: List[str]
    validation_errors: List[str]
    version_history: List[Dict[str, Any]]


class AuditLogResponse(BaseModel):
    events: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class ConfigAdminInterface:
    """
    Configuration Admin Interface
    
    Provides web-based administration for configuration management:
    - Configuration browsing and editing
    - Real-time monitoring
    - Template management
    - Migration tools
    - Security management
    - Audit logging
    """
    
    def __init__(self, config_dir: str = "./config", port: int = 8080, 
                 host: str = "127.0.0.1", username: str = "admin", 
                 password: str = "admin"):
        self.config_dir = Path(config_dir)
        self.port = port
        self.host = host
        self.username = username
        self.password = password
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Initialize components
        self.config_manager = ComprehensiveConfigManager(str(self.config_dir))
        self.migration_manager = ConfigMigrationManager(self.config_manager)
        self.audit_logger = get_audit_logger(str(self.config_dir / "logs"))
        
        # FastAPI app
        self.app = FastAPI(
            title="Trading Orchestrator Configuration Admin",
            description="Web-based configuration management interface",
            version="1.0.0"
        )
        
        # Security
        self.security = HTTPBasic()
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan management"""
        # Startup
        await self.config_manager.initialize()
        self.logger.info("Configuration admin interface started")
        
        yield
        
        # Shutdown
        self.config_manager.shutdown()
        self.logger.info("Configuration admin interface stopped")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        # Authentication dependency
        async def get_current_user(credentials: HTTPBasicCredentials = Depends(self.security)):
            if credentials.username != self.username:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
            if password_hash != self.password_hash:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            return credentials.username
        
        # Dashboard
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request, user: str = Depends(get_current_user)):
            """Main dashboard"""
            return await self._render_dashboard(request, user)
        
        @self.app.get("/api/status")
        async def get_system_status(user: str = Depends(get_current_user)):
            """Get system status"""
            try:
                status = {
                    "configurations": self.config_manager.get_config_status(),
                    "templates": list(self.config_manager.templates.keys()),
                    "system_health": "healthy",
                    "uptime": "running",
                    "version": "1.0.0"
                }
                return AdminConfigResponse(success=True, message="Status retrieved", data=status)
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Configuration management
        @self.app.get("/api/configs")
        async def list_configurations(user: str = Depends(get_current_user)):
            """List all configurations"""
            try:
                configs = self.config_manager.get_config_status()
                return AdminConfigResponse(success=True, message="Configurations retrieved", data=configs)
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.get("/api/config/{config_name}")
        async def get_configuration(config_name: str, user: str = Depends(get_current_user)):
            """Get configuration details"""
            try:
                config = self.config_manager.get_config(config_name)
                if not config:
                    raise HTTPException(status_code=404, detail=f"Configuration not found: {config_name}")
                
                status = self.config_manager.get_config_status(config_name)
                version_history = self.config_manager.version_manager.get_versions(config_name, limit=10)
                
                response_data = {
                    "config": config,
                    "status": status,
                    "version_history": [v.to_dict() for v in version_history]
                }
                
                return AdminConfigResponse(success=True, message="Configuration retrieved", data=response_data)
            except HTTPException:
                raise
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.post("/api/config")
        async def create_configuration(request: ConfigCreateRequest, user: str = Depends(get_current_user)):
            """Create new configuration"""
            try:
                if request.template_name:
                    # Create from template
                    success = self.config_manager.apply_template(request.template_name, request.variables)
                    if success:
                        # Rename to requested name
                        template_config = self.config_manager.get_config(request.template_name)
                        if template_config:
                            self.config_manager.save_config(request.config_name, template_config)
                            return AdminConfigResponse(success=True, message="Configuration created from template")
                else:
                    # Create empty configuration
                    empty_config = {"version": "1.0.0", "created_by": user, "created_at": datetime.utcnow().isoformat()}
                    self.config_manager.save_config(request.config_name, empty_config)
                    return AdminConfigResponse(success=True, message="Configuration created")
                
                return AdminConfigResponse(success=False, message="Failed to create configuration")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.put("/api/config/{config_name}")
        async def update_configuration(config_name: str, request: ConfigUpdateRequest, 
                                     user: str = Depends(get_current_user)):
            """Update configuration value"""
            try:
                if request.encrypt:
                    # Encrypt the value before saving
                    encrypted_value = self.config_manager.encryption.encrypt(str(request.value))
                    success = self.config_manager.set_config_value(config_name, request.key_path, f"encrypted:{encrypted_value}")
                else:
                    success = self.config_manager.set_config_value(config_name, request.key_path, request.value)
                
                if success:
                    # Log the change
                    self.audit_logger.log_config_changed(config_name, request.key_path, "", request.value, user)
                    return AdminConfigResponse(success=True, message="Configuration updated")
                else:
                    return AdminConfigResponse(success=False, message="Failed to update configuration")
            except Exception as e:
                return AdminConfigLogger(success=False, message=str(e), errors=[str(e)])
        
        @self.app.delete("/api/config/{config_name}")
        async def delete_configuration(config_name: str, user: str = Depends(get_current_user)):
            """Delete configuration"""
            try:
                # Create backup before deletion
                self.config_manager.version_manager.create_backup(config_name)
                
                # Log deletion
                self.audit_logger.log_config_deleted(config_name, user)
                
                # Remove from cache and delete file
                config_file = self.config_dir / f"{config_name}.json"
                if config_file.exists():
                    config_file.unlink()
                
                if config_name in self.config_manager.config_cache:
                    del self.config_manager.config_cache[config_name]
                
                return AdminConfigResponse(success=True, message="Configuration deleted")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Template management
        @self.app.get("/api/templates")
        async def list_templates(user: str = Depends(get_current_user)):
            """List all templates"""
            try:
                templates = {}
                for name, template in self.config_manager.templates.items():
                    templates[name] = {
                        "name": template.name,
                        "environment": template.environment.value,
                        "description": template.description,
                        "variables": template.variables,
                        "version": template.version
                    }
                
                return AdminConfigResponse(success=True, message="Templates retrieved", data=templates)
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.post("/api/templates/apply")
        async def apply_template(request: TemplateApplyRequest, user: str = Depends(get_current_user)):
            """Apply template to configuration"""
            try:
                success = self.config_manager.apply_template(request.template_name, request.variables)
                if success:
                    return AdminConfigResponse(success=True, message="Template applied")
                else:
                    return AdminConfigResponse(success=False, message="Failed to apply template")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Migration tools
        @self.app.post("/api/migrate")
        async def migrate_configuration(request: MigrationRequest, user: str = Depends(get_current_user)):
            """Migrate configuration"""
            try:
                result = self.config_manager.migration_manager.migrate_config(
                    request.config_name, request.target_version, request.environment, request.dry_run
                )
                
                return AdminConfigResponse(
                    success=result.success,
                    message=f"Migration {'completed' if result.success else 'failed'}",
                    data={
                        "changes_applied": result.changes_applied,
                        "errors": result.errors,
                        "warnings": result.warnings,
                        "backup_created": result.backup_created
                    }
                )
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.post("/api/security-update")
        async def apply_security_update(request: SecurityUpdateRequest, user: str = Depends(get_current_user)):
            """Apply security updates"""
            try:
                result = self.config_manager.migration_manager.apply_security_updates(
                    request.config_name, request.security_level, request.dry_run
                )
                
                return AdminConfigResponse(
                    success=result.success,
                    message=f"Security update {'completed' if result.success else 'failed'}",
                    data={
                        "changes_applied": result.changes_applied,
                        "errors": result.errors,
                        "warnings": result.warnings
                    }
                )
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Import/Export
        @self.app.post("/api/export")
        async def export_configuration(config_name: str, include_sensitive: bool = False, 
                                     user: str = Depends(get_current_user)):
            """Export configuration"""
            try:
                export_path = self.config_dir / f"exports" / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                export_path.parent.mkdir(exist_ok=True)
                
                success = self.config_manager.export_config(config_name, str(export_path), include_sensitive)
                if success:
                    self.audit_logger.log_config_exported(config_name, str(export_path), user)
                    return AdminConfigResponse(success=True, message="Configuration exported", data={"export_path": str(export_path)})
                else:
                    return AdminConfigResponse(success=False, message="Failed to export configuration")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.post("/api/import")
        async def import_configuration(request: ConfigImportRequest, user: str = Depends(get_current_user)):
            """Import configuration"""
            try:
                success = self.config_manager.import_config(request.file_path, request.config_name, request.overwrite)
                if success:
                    self.audit_logger.log_config_imported(request.config_name, request.file_path, user)
                    return AdminConfigResponse(success=True, message="Configuration imported")
                else:
                    return AdminConfigResponse(success=False, message="Failed to import configuration")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Validation
        @self.app.post("/api/validate/{config_name}")
        async def validate_configuration(config_name: str, user: str = Depends(get_current_user)):
            """Validate configuration"""
            try:
                errors = self.config_manager.validate_config(config_name)
                if errors:
                    self.audit_logger.log_validation_error(config_name, errors, user)
                
                return AdminConfigResponse(
                    success=len(errors) == 0,
                    message="Validation completed",
                    data={"errors": errors}
                )
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Audit logs
        @self.app.get("/api/audit")
        async def get_audit_logs(page: int = 1, page_size: int = 100, 
                               config_name: str = None, user: str = Depends(get_current_user)):
            """Get audit logs"""
            try:
                events = self.audit_logger.get_events(config_name=config_name, limit=page_size)
                
                # Pagination
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                paginated_events = events[start_idx:end_idx]
                
                total_pages = (len(events) + page_size - 1) // page_size
                
                return AuditLogResponse(
                    events=[event.to_dict() for event in paginated_events],
                    total_count=len(events),
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages
                )
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Version management
        @self.app.get("/api/versions/{config_name}")
        async def get_version_history(config_name: str, user: str = Depends(get_current_user)):
            """Get version history"""
            try:
                versions = self.config_manager.version_manager.get_versions(config_name)
                return AdminConfigResponse(
                    success=True,
                    message="Version history retrieved",
                    data=[v.to_dict() for v in versions]
                )
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        @self.app.post("/api/rollback/{config_name}/{version_id}")
        async def rollback_to_version(config_name: str, version_id: str, user: str = Depends(get_current_user)):
            """Rollback to specific version"""
            try:
                success = self.config_manager.version_manager.rollback(config_name, version_id)
                if success:
                    self.audit_logger.log_config_rolled_back(config_name, "current", version_id, user)
                    return AdminConfigResponse(success=True, message="Configuration rolled back")
                else:
                    return AdminConfigResponse(success=False, message="Failed to rollback configuration")
            except Exception as e:
                return AdminConfigResponse(success=False, message=str(e), errors=[str(e)])
        
        # Real-time monitoring
        @self.app.websocket("/ws/monitor")
        async def websocket_monitor(websocket):
            """WebSocket for real-time monitoring"""
            await websocket.accept()
            
            try:
                while True:
                    # Get current status
                    status = self.config_manager.get_config_status()
                    
                    # Send status update
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Wait before next update
                    await asyncio.sleep(5)
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                await websocket.close()
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "config_manager": "running",
                    "audit_logger": "running"
                }
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))
    
    async def _render_dashboard(self, request: Request, user: str) -> str:
        """Render main dashboard HTML"""
        # Get system status
        try:
            config_status = self.config_manager.get_config_status()
            template_count = len(self.config_manager.templates)
            recent_changes = self.config_manager.get_change_history(limit=10)
            
            status_data = {
                "user": user,
                "configs_count": len(config_status),
                "templates_count": template_count,
                "recent_changes": [c.key_path for c in recent_changes],
                "last_update": datetime.utcnow().isoformat()
            }
            
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Configuration Admin</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background: #333; color: white; padding: 20px; border-radius: 5px; }
                    .stats { display: flex; gap: 20px; margin: 20px 0; }
                    .stat-card { background: #f5f5f5; padding: 20px; border-radius: 5px; flex: 1; }
                    .config-list { background: #fff; border: 1px solid #ddd; border-radius: 5px; margin: 20px 0; }
                    .config-item { padding: 10px; border-bottom: 1px solid #eee; }
                    .config-item:last-child { border-bottom: none; }
                    .status-badge { padding: 2px 8px; border-radius: 3px; font-size: 12px; }
                    .status-valid { background: #d4edda; color: #155724; }
                    .status-invalid { background: #f8d7da; color: #721c24; }
                    .nav { background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
                    .nav a { margin-right: 15px; text-decoration: none; color: #007bff; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Trading Orchestrator Configuration Admin</h1>
                    <p>Welcome, {{ user }}</p>
                </div>
                
                <div class="nav">
                    <a href="/">Dashboard</a>
                    <a href="/configs">Configurations</a>
                    <a href="/templates">Templates</a>
                    <a href="/migrations">Migrations</a>
                    <a href="/audit">Audit Logs</a>
                    <a href="/security">Security</a>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <h3>Configurations</h3>
                        <p>{{ configs_count }} total</p>
                    </div>
                    <div class="stat-card">
                        <h3>Templates</h3>
                        <p>{{ templates_count }} available</p>
                    </div>
                    <div class="stat-card">
                        <h3>Recent Changes</h3>
                        <p>{{ recent_changes|length }} in last 24h</p>
                    </div>
                </div>
                
                <div class="config-list">
                    <h3>Configuration Status</h3>
                    <div class="config-item">
                        <strong>System Status:</strong> 
                        <span class="status-badge status-valid">Running</span>
                    </div>
                    <div class="config-item">
                        <strong>Last Update:</strong> {{ last_update }}
                    </div>
                </div>
                
                <script>
                    // WebSocket connection for real-time updates
                    const ws = new WebSocket(`ws://${window.location.host}/ws/monitor`);
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'status_update') {
                            console.log('Status update:', data.data);
                        }
                    };
                    
                    // Auto-refresh dashboard every 30 seconds
                    setInterval(() => {
                        window.location.reload();
                    }, 30000);
                </script>
            </body>
            </html>
            """
            
            return html_template
            
        except Exception as e:
            self.logger.error(f"Failed to render dashboard: {e}")
            return f"<h1>Error</h1><p>{str(e)}</p>"
    
    def run(self):
        """Run the admin interface server"""
        try:
            self.logger.info(f"Starting configuration admin interface on {self.host}:{self.port}")
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )
        except Exception as e:
            self.logger.error(f"Failed to start admin interface: {e}")
            raise


# CLI interface for admin interface
def create_admin_interface(config_dir: str = "./config", port: int = 8080,
                          host: str = "127.0.0.1", username: str = "admin", 
                          password: str = "admin") -> ConfigAdminInterface:
    """Create and configure admin interface"""
    admin = ConfigAdminInterface(
        config_dir=config_dir,
        port=port,
        host=host,
        username=username,
        password=password
    )
    
    return admin


if __name__ == "__main__":
    # CLI entry point
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Orchestrator Configuration Admin")
    parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    parser.add_argument("--port", type=int, default=8080, help="Admin interface port")
    parser.add_argument("--host", default="127.0.0.1", help="Admin interface host")
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument("--password", default="admin", help="Admin password")
    
    args = parser.parse_args()
    
    admin = create_admin_interface(
        config_dir=args.config_dir,
        port=args.port,
        host=args.host,
        username=args.username,
        password=args.password
    )
    
    admin.run()