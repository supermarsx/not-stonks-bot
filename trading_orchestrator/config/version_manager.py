"""
Configuration Version Management System
Manages configuration versioning, backup, and rollback capabilities
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict
import zipfile


@dataclass
class ConfigVersion:
    """Configuration version information"""
    version_id: str
    config_name: str
    timestamp: datetime
    author: str
    description: str
    checksum: str
    file_path: str
    size_bytes: int
    metadata: Dict[str, Any]
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ConfigVersionManager:
    """
    Configuration Version Manager
    
    Features:
    - Automatic version creation on changes
    - Manual version tagging
    - Version rollback capability
    - Version diff calculation
    - Cleanup old versions
    - Version search and filtering
    - Backup and restore
    """
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.versions_dir = self.config_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.versions_dir / "versions.json"
        self.max_versions_per_config = 50
        self.retention_days = 90
        
        self.logger = logging.getLogger(__name__)
        self._versions: Dict[str, List[ConfigVersion]] = {}
        
        # Load existing versions
        self._load_versions_metadata()
    
    def create_version(self, config_name: str, config_data: Dict[str, Any], 
                      author: str = "system", description: str = "", 
                      tags: List[str] = None) -> str:
        """
        Create new configuration version
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data
            author: Author of the version
            description: Description of changes
            tags: Version tags
            
        Returns:
            Version ID
        """
        try:
            # Generate version ID
            timestamp = datetime.utcnow()
            version_id = self._generate_version_id(config_name, timestamp)
            
            # Calculate checksum
            config_json = json.dumps(config_data, sort_keys=True)
            checksum = hashlib.sha256(config_json.encode()).hexdigest()
            
            # Save version file
            version_file = self.versions_dir / f"{config_name}_{version_id}.json"
            with open(version_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            # Create version object
            version = ConfigVersion(
                version_id=version_id,
                config_name=config_name,
                timestamp=timestamp,
                author=author,
                description=description,
                checksum=checksum,
                file_path=str(version_file),
                size_bytes=version_file.stat().st_size,
                metadata={
                    'created_by': 'config_manager',
                    'creation_method': 'automatic'
                },
                tags=tags or []
            )
            
            # Store version
            if config_name not in self._versions:
                self._versions[config_name] = []
            self._versions[config_name].append(version)
            
            # Cleanup old versions
            self._cleanup_old_versions(config_name)
            
            # Save metadata
            self._save_versions_metadata()
            
            self.logger.info(f"Created configuration version: {config_name} - {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create version for {config_name}: {e}")
            raise
    
    def get_versions(self, config_name: str = None, limit: int = None, 
                    tags: List[str] = None) -> List[ConfigVersion]:
        """
        Get configuration versions
        
        Args:
            config_name: Configuration name filter
            limit: Maximum number of versions to return
            tags: Filter by tags
            
        Returns:
            List of versions
        """
        versions = []
        
        # Get versions for specific config or all configs
        if config_name:
            config_versions = self._versions.get(config_name, [])
            versions.extend(config_versions)
        else:
            for config_versions in self._versions.values():
                versions.extend(config_versions)
        
        # Filter by tags
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def get_version(self, config_name: str, version_id: str) -> Optional[ConfigVersion]:
        """Get specific configuration version"""
        versions = self._versions.get(config_name, [])
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    def load_version(self, config_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Load configuration data from version"""
        version = self.get_version(config_name, version_id)
        if not version:
            return None
        
        try:
            with open(version.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load version {version_id}: {e}")
            return None
    
    def rollback(self, config_name: str, version_id: str, 
                create_backup: bool = True) -> bool:
        """
        Rollback configuration to specified version
        
        Args:
            config_name: Configuration name
            version_id: Target version ID
            create_backup: Create backup before rollback
            
        Returns:
            bool: True if rollback successful
        """
        try:
            # Load version data
            version_data = self.load_version(config_name, version_id)
            if not version_data:
                self.logger.error(f"Version not found: {config_name} - {version_id}")
                return False
            
            # Create backup of current configuration if requested
            if create_backup:
                self.create_version(
                    f"{config_name}_backup",
                    version_data,
                    author="rollback_system",
                    description=f"Backup before rollback to {version_id}",
                    tags=["rollback_backup"]
                )
            
            # Load the current configuration data
            current_config_file = self.config_dir / f"{config_name}.json"
            if current_config_file.exists():
                with open(current_config_file, 'r') as f:
                    current_data = json.load(f)
                
                # Create version of current state
                self.create_version(
                    config_name,
                    current_data,
                    author="rollback_system",
                    description=f"Backup before rollback to {version_id}",
                    tags=["pre_rollback"]
                )
            
            # Restore version data to file
            with open(current_config_file, 'w') as f:
                json.dump(version_data, f, indent=2, default=str)
            
            # Create new version with rollback info
            self.create_version(
                config_name,
                version_data,
                author="rollback_system",
                description=f"Rolled back to version {version_id}",
                tags=["rollback"]
            )
            
            self.logger.info(f"Configuration rolled back: {config_name} - {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback {config_name} to {version_id}: {e}")
            return False
    
    def compare_versions(self, config_name: str, version1_id: str, 
                        version2_id: str) -> Dict[str, Any]:
        """
        Compare two configuration versions
        
        Args:
            config_name: Configuration name
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Dictionary with comparison results
        """
        try:
            version1_data = self.load_version(config_name, version1_id)
            version2_data = self.load_version(config_name, version2_id)
            
            if not version1_data or not version2_data:
                return {"error": "One or both versions not found"}
            
            # Simple diff calculation
            changes = self._calculate_diff(version1_data, version2_data)
            
            return {
                "config_name": config_name,
                "version1": version1_id,
                "version2": version2_id,
                "changes": changes,
                "summary": {
                    "added_fields": len(changes.get("added", [])),
                    "removed_fields": len(changes.get("removed", [])),
                    "modified_fields": len(changes.get("modified", []))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {e}")
            return {"error": str(e)}
    
    def tag_version(self, config_name: str, version_id: str, tags: List[str]) -> bool:
        """Add tags to configuration version"""
        try:
            version = self.get_version(config_name, version_id)
            if not version:
                return False
            
            version.tags.extend(tags)
            version.tags = list(set(version.tags))  # Remove duplicates
            
            self._save_versions_metadata()
            self.logger.info(f"Tagged version: {config_name} - {version_id} with {tags}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to tag version: {e}")
            return False
    
    def search_versions(self, search_terms: List[str], 
                       config_name: str = None) -> List[ConfigVersion]:
        """Search configuration versions"""
        results = []
        versions = self.get_versions(config_name)
        
        for version in versions:
            # Search in description and tags
            searchable_text = " ".join([
                version.description,
                " ".join(version.tags),
                version.author
            ]).lower()
            
            if all(term.lower() in searchable_text for term in search_terms):
                results.append(version)
        
        return results
    
    def export_version(self, config_name: str, version_id: str, 
                      export_path: str, include_metadata: bool = True) -> bool:
        """Export configuration version to file"""
        try:
            version = self.get_version(config_name, version_id)
            if not version:
                return False
            
            # Load version data
            version_data = self.load_version(config_name, version_id)
            if not version_data:
                return False
            
            export_data = {
                "version_info": version.to_dict() if include_metadata else None,
                "config_data": version_data
            }
            
            # Save export
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported version: {config_name} - {version_id} to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export version: {e}")
            return False
    
    def import_version(self, import_path: str, config_name: str = None) -> bool:
        """Import configuration version from file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            config_data = import_data.get("config_data")
            version_info = import_data.get("version_info")
            
            if not config_data:
                return False
            
            # Determine config name
            if config_name is None:
                config_name = version_info.get("config_name") if version_info else "imported_config"
            
            # Import as new version
            self.create_version(
                config_name,
                config_data,
                author=version_info.get("author", "import") if version_info else "import",
                description=f"Imported from {import_path}" + 
                           (f" (was {version_info.get('version_id', 'unknown')})" if version_info else ""),
                tags=["imported"]
            )
            
            self.logger.info(f"Imported version from {import_path} as {config_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import version: {e}")
            return False
    
    def create_backup(self, config_name: str, backup_name: str = None) -> bool:
        """Create backup of configuration and all its versions"""
        try:
            versions = self._versions.get(config_name, [])
            if not versions:
                return False
            
            backup_name = backup_name or f"{config_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir = self.versions_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy current config file
            current_config = self.config_dir / f"{config_name}.json"
            if current_config.exists():
                shutil.copy2(current_config, backup_dir / f"{config_name}.json")
            
            # Copy all version files
            for version in versions:
                if Path(version.file_path).exists():
                    shutil.copy2(version.file_path, backup_dir / Path(version.file_path).name)
            
            # Export version metadata
            export_data = {
                "backup_info": {
                    "config_name": config_name,
                    "backup_name": backup_name,
                    "created_at": datetime.utcnow().isoformat(),
                    "version_count": len(versions)
                },
                "versions": [v.to_dict() for v in versions]
            }
            
            with open(backup_dir / "backup_info.json", 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Create zip backup
            zip_path = self.versions_dir / "backups" / f"{backup_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in backup_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(backup_dir)
                        zf.write(file_path, arcname)
            
            # Clean up directory (keep zip)
            shutil.rmtree(backup_dir)
            
            self.logger.info(f"Created backup: {backup_name} ({len(versions)} versions)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore configuration from backup"""
        try:
            backup_dir = self.versions_dir / "backups" / backup_name
            
            # Check if directory backup exists
            if not backup_dir.exists():
                # Try to find zip backup
                zip_path = self.versions_dir / "backups" / f"{backup_name}.zip"
                if not zip_path.exists():
                    self.logger.error(f"Backup not found: {backup_name}")
                    return False
                
                # Extract zip
                backup_dir = self.versions_dir / "backups" / f"temp_{backup_name}"
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(backup_dir)
            
            # Load backup info
            backup_info_file = backup_dir / "backup_info.json"
            if not backup_info_file.exists():
                self.logger.error(f"Backup info not found in {backup_name}")
                return False
            
            with open(backup_info_file, 'r') as f:
                backup_info = json.load(f)
            
            config_name = backup_info["backup_info"]["config_name"]
            
            # Create backup of current state
            self.create_backup(config_name, f"{config_name}_pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Restore current config file
            current_config = self.config_dir / f"{config_name}.json"
            backup_config = backup_dir / f"{config_name}.json"
            
            if backup_config.exists():
                shutil.copy2(backup_config, current_config)
            
            # Restore version files
            for file_path in backup_dir.glob(f"{config_name}_*.json"):
                if file_path.name != f"{config_name}.json":  # Skip main config file
                    target_path = self.versions_dir / file_path.name
                    shutil.copy2(file_path, target_path)
            
            # Reload versions metadata
            self._load_versions_metadata()
            
            # Clean up temp directory if created
            if backup_name != backup_dir.name:
                shutil.rmtree(backup_dir)
            
            self.logger.info(f"Restored backup: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def cleanup_old_versions(self, config_name: str = None) -> int:
        """Clean up old versions based on retention policy"""
        cleaned_count = 0
        
        try:
            target_configs = [config_name] if config_name else list(self._versions.keys())
            
            for config in target_configs:
                cleaned = self._cleanup_old_versions(config)
                cleaned_count += cleaned
            
            if cleaned_count > 0:
                self._save_versions_metadata()
            
            self.logger.info(f"Cleaned up {cleaned_count} old versions")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version management statistics"""
        total_versions = sum(len(versions) for versions in self._versions.values())
        
        stats = {
            "total_configs": len(self._versions),
            "total_versions": total_versions,
            "configs": {}
        }
        
        for config_name, versions in self._versions.items():
            stats["configs"][config_name] = {
                "version_count": len(versions),
                "oldest_version": versions[-1].timestamp.isoformat() if versions else None,
                "newest_version": versions[0].timestamp.isoformat() if versions else None,
                "total_size_mb": sum(v.size_bytes for v in versions) / (1024 * 1024),
                "tags": list(set(tag for v in versions for tag in v.tags))
            }
        
        return stats
    
    def _generate_version_id(self, config_name: str, timestamp: datetime) -> str:
        """Generate unique version ID"""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(config_name.encode()).hexdigest()[:8]
        return f"{timestamp_str}_{config_hash}"
    
    def _calculate_diff(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, List]:
        """Calculate difference between two configuration dictionaries"""
        added = []
        removed = []
        modified = []
        
        def compare_recursive(obj1, obj2, path=""):
            if isinstance(obj1, dict) and isinstance(obj2, dict):
                # Find added and modified keys
                for key in set(obj1.keys()) | set(obj2.keys()):
                    current_path = f"{path}.{key}" if path else key
                    if key not in obj1:
                        added.append(current_path)
                    elif key not in obj2:
                        removed.append(current_path)
                    elif obj1[key] != obj2[key]:
                        if isinstance(obj1[key], dict) and isinstance(obj2[key], dict):
                            compare_recursive(obj1[key], obj2[key], current_path)
                        else:
                            modified.append({
                                "path": current_path,
                                "old_value": obj1[key],
                                "new_value": obj2[key]
                            })
            else:
                if obj1 != obj2:
                    modified.append({
                        "path": path,
                        "old_value": obj1,
                        "new_value": obj2
                    })
        
        compare_recursive(data1, data2)
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified
        }
    
    def _cleanup_old_versions(self, config_name: str) -> int:
        """Clean up old versions for specific configuration"""
        versions = self._versions.get(config_name, [])
        if not versions:
            return 0
        
        cleaned_count = 0
        
        # Clean up by count limit
        if len(versions) > self.max_versions_per_config:
            versions_to_remove = versions[self.max_versions_per_config:]
            versions[:] = versions[:self.max_versions_per_config]
            
            for version in versions_to_remove:
                try:
                    if Path(version.file_path).exists():
                        Path(version.file_path).unlink()
                    cleaned_count += 1
                except:
                    pass
        
        # Clean up by age
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        versions_to_remove = [v for v in versions if v.timestamp < cutoff_date]
        
        if versions_to_remove:
            for version in versions_to_remove:
                try:
                    if Path(version.file_path).exists():
                        Path(version.file_path).unlink()
                    versions.remove(version)
                    cleaned_count += 1
                except:
                    pass
        
        return cleaned_count
    
    def _load_versions_metadata(self):
        """Load versions metadata from file"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self._versions = {}
            for config_name, version_data_list in metadata.items():
                self._versions[config_name] = [
                    ConfigVersion.from_dict(v_data) for v_data in version_data_list
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to load versions metadata: {e}")
    
    def _save_versions_metadata(self):
        """Save versions metadata to file"""
        try:
            metadata = {}
            for config_name, versions in self._versions.items():
                metadata[config_name] = [v.to_dict() for v in versions]
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save versions metadata: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self._save_versions_metadata()