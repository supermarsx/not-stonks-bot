"""
Log Rotation Manager

Advanced log rotation with automatic archival, compression, and retention policies.
"""

import os
import gzip
import shutil
import glob
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio


class CompressionType(Enum):
    """Compression type enumeration"""
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    NONE = "none"


class ArchiveLevel(Enum):
    """Archive level enumeration"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class LogFileInfo:
    """Log file information"""
    path: Path
    size: int
    created_at: datetime
    modified_at: datetime
    compressed: bool = False
    archived: bool = False


@dataclass
class RetentionPolicy:
    """Log retention policy configuration"""
    max_days: int = 30
    max_size_mb: int = 1000
    compression_enabled: bool = True
    compression_type: str = "gzip"
    archive_enabled: bool = True
    archive_level: str = "month"
    cleanup_enabled: bool = True
    min_free_space_mb: int = 100
    backup_before_cleanup: bool = True
    custom_filter: Optional[Callable[[LogFileInfo], bool]] = None


class LogCompressor:
    """Log compression utility"""
    
    @staticmethod
    def compress_gzip(source_path: Path, target_path: Optional[Path] = None) -> Path:
        """Compress file using gzip"""
        if target_path is None:
            target_path = source_path.with_suffix(source_path.suffix + '.gz')
        
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original file after successful compression
        source_path.unlink()
        return target_path
    
    @staticmethod
    def decompress_gzip(compressed_path: Path, target_path: Optional[Path] = None) -> Path:
        """Decompress gzip file"""
        if target_path is None:
            # Remove .gz suffix from filename
            if compressed_path.suffix == '.gz':
                target_path = compressed_path.with_suffix('')
            else:
                target_path = compressed_path.with_suffix('.decompressed')
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return target_path
    
    @staticmethod
    def compress_bzip2(source_path: Path, target_path: Optional[Path] = None) -> Path:
        """Compress file using bzip2"""
        try:
            import bz2
        except ImportError:
            raise ImportError("bzip2 not available, install with: pip install bz2file")
        
        if target_path is None:
            target_path = source_path.with_suffix(source_path.suffix + '.bz2')
        
        with open(source_path, 'rb') as f_in:
            with bz2.open(target_path, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        source_path.unlink()
        return target_path
    
    @staticmethod
    def compress_lzma(source_path: Path, target_path: Optional[Path] = None) -> Path:
        """Compress file using LZMA"""
        try:
            import lzma
        except ImportError:
            raise ImportError("lzma not available, use gzip compression instead")
        
        if target_path is None:
            target_path = source_path.with_suffix(source_path.suffix + '.xz')
        
        with open(source_path, 'rb') as f_in:
            with lzma.open(target_path, 'wb', preset=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        source_path.unlink()
        return target_path
    
    @classmethod
    def compress_file(cls, 
                     source_path: Path, 
                     compression_type: str = "gzip",
                     target_path: Optional[Path] = None) -> Path:
        """Compress file using specified compression method"""
        compression_type = compression_type.lower()
        
        if compression_type == "gzip":
            return cls.compress_gzip(source_path, target_path)
        elif compression_type == "bzip2":
            return cls.compress_bzip2(source_path, target_path)
        elif compression_type == "lzma":
            return cls.compress_lzma(source_path, target_path)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")


class LogArchiver:
    """Log archiving utility"""
    
    def __init__(self, archive_base_path: Path):
        self.archive_base_path = Path(archive_base_path)
        self.archive_base_path.mkdir(parents=True, exist_ok=True)
    
    def archive_by_day(self, log_files: List[LogFileInfo]) -> List[Path]:
        """Archive files organized by day"""
        archived_files = []
        grouped_files = {}
        
        # Group files by day
        for log_file in log_files:
            day_key = log_file.created_at.strftime("%Y-%m-%d")
            if day_key not in grouped_files:
                grouped_files[day_key] = []
            grouped_files[day_key].append(log_file)
        
        # Create archive for each day
        for day_key, files in grouped_files.items():
            archive_path = self.archive_base_path / "daily" / day_key
            archive_path.mkdir(parents=True, exist_ok=True)
            
            for log_file in files:
                target_path = archive_path / log_file.path.name
                shutil.move(str(log_file.path), str(target_path))
                archived_files.append(target_path)
        
        return archived_files
    
    def archive_by_week(self, log_files: List[LogFileInfo]) -> List[Path]:
        """Archive files organized by week"""
        archived_files = []
        grouped_files = {}
        
        # Group files by week
        for log_file in log_files:
            # Get ISO week number
            iso_week = log_file.created_at.isocalendar()
            week_key = f"{iso_week.year}-W{iso_week.week:02d}"
            
            if week_key not in grouped_files:
                grouped_files[week_key] = []
            grouped_files[week_key].append(log_file)
        
        # Create archive for each week
        for week_key, files in grouped_files.items():
            archive_path = self.archive_base_path / "weekly" / week_key
            archive_path.mkdir(parents=True, exist_ok=True)
            
            for log_file in files:
                target_path = archive_path / log_file.path.name
                shutil.move(str(log_file.path), str(target_path))
                archived_files.append(target_path)
        
        return archived_files
    
    def archive_by_month(self, log_files: List[LogFileInfo]) -> List[Path]:
        """Archive files organized by month"""
        archived_files = []
        grouped_files = {}
        
        # Group files by month
        for log_file in log_files:
            month_key = log_file.created_at.strftime("%Y-%m")
            
            if month_key not in grouped_files:
                grouped_files[month_key] = []
            grouped_files[month_key].append(log_file)
        
        # Create archive for each month
        for month_key, files in grouped_files.items():
            archive_path = self.archive_base_path / "monthly" / month_key
            archive_path.mkdir(parents=True, exist_ok=True)
            
            for log_file in files:
                target_path = archive_path / log_file.path.name
                shutil.move(str(log_file.path), str(target_path))
                archived_files.append(target_path)
        
        return archived_files
    
    def archive_files(self, 
                     log_files: List[LogFileInfo], 
                     archive_level: str = "month") -> List[Path]:
        """Archive files using specified level"""
        if archive_level == "day":
            return self.archive_by_day(log_files)
        elif archive_level == "week":
            return self.archive_by_week(log_files)
        elif archive_level == "month":
            return self.archive_by_month(log_files)
        else:
            raise ValueError(f"Unsupported archive level: {archive_level}")


class LogRetentionManager:
    """Log retention and cleanup manager"""
    
    def __init__(self, retention_policies: Dict[str, RetentionPolicy] = None):
        self.retention_policies = retention_policies or self._default_policies()
        self.logger = logging.getLogger(__name__)
    
    def _default_policies(self) -> Dict[str, RetentionPolicy]:
        """Default retention policies"""
        return {
            "general": RetentionPolicy(max_days=30, max_size_mb=1000),
            "error": RetentionPolicy(max_days=90, max_size_mb=500, compression_enabled=True),
            "audit": RetentionPolicy(max_days=365, max_size_mb=2000, compression_enabled=True),
            "security": RetentionPolicy(max_days=365, max_size_mb=500, compression_enabled=True),
            "performance": RetentionPolicy(max_days=7, max_size_mb=100, compression_enabled=True),
        }
    
    def should_retain_file(self, log_file: LogFileInfo, policy_name: str) -> bool:
        """Determine if log file should be retained based on policy"""
        policy = self.retention_policies.get(policy_name, self.retention_policies["general"])
        
        # Check age
        age_days = (datetime.now() - log_file.modified_at).days
        if age_days > policy.max_days:
            return False
        
        # Check size (approximate)
        size_mb = log_file.size / (1024 * 1024)
        if size_mb > policy.max_size_mb:
            return False
        
        # Custom filter
        if policy.custom_filter and not policy.custom_filter(log_file):
            return False
        
        return True
    
    def get_outdated_files(self, 
                          log_files: List[LogFileInfo], 
                          policy_name: str) -> List[LogFileInfo]:
        """Get files that should be cleaned up based on retention policy"""
        policy = self.retention_policies.get(policy_name, self.retention_policies["general"])
        outdated_files = []
        
        for log_file in log_files:
            # Check age
            age_days = (datetime.now() - log_file.modified_at).days
            if age_days > policy.max_days:
                outdated_files.append(log_file)
                continue
            
            # Check size
            size_mb = log_file.size / (1024 * 1024)
            if size_mb > policy.max_size_mb:
                outdated_files.append(log_file)
                continue
            
            # Custom filter
            if policy.custom_filter and not policy.custom_filter(log_file):
                outdated_files.append(log_file)
        
        return outdated_files
    
    def clean_up_old_files(self, 
                          log_files: List[LogFileInfo], 
                          policy_name: str,
                          backup_enabled: bool = True) -> Tuple[List[Path], int]:
        """Clean up old log files based on retention policy"""
        policy = self.retention_policies.get(policy_name, self.retention_policies["general"])
        outdated_files = self.get_outdated_files(log_files, policy_name)
        
        if not outdated_files:
            return [], 0
        
        cleaned_files = []
        total_size_saved = 0
        
        # Backup before cleanup if enabled
        if backup_enabled and policy.backup_before_cleanup:
            self._backup_files(outdated_files)
        
        for log_file in outdated_files:
            try:
                total_size_saved += log_file.size
                log_file.path.unlink()
                cleaned_files.append(log_file.path)
            except Exception as e:
                self.logger.error(f"Failed to delete {log_file.path}: {e}")
        
        self.logger.info(f"Cleaned up {len(cleaned_files)} files, saved {total_size_saved / (1024*1024):.2f} MB")
        return cleaned_files, total_size_saved
    
    def _backup_files(self, log_files: List[LogFileInfo]):
        """Backup files before cleanup"""
        backup_dir = Path("logs/backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for log_file in log_files:
            try:
                backup_path = backup_dir / log_file.path.name
                shutil.copy2(log_file.path, backup_path)
            except Exception as e:
                self.logger.error(f"Failed to backup {log_file.path}: {e}")


class LogRotationManager:
    """Comprehensive log rotation manager with automation"""
    
    def __init__(self, 
                 log_directory: str,
                 archive_directory: Optional[str] = None,
                 retention_policies: Optional[Dict[str, RetentionPolicy]] = None,
                 auto_rotation: bool = True,
                 rotation_interval: int = 3600):  # 1 hour default
        self.log_directory = Path(log_directory)
        self.archive_directory = Path(archive_directory or f"{log_directory}/archives")
        self.retention_manager = LogRetentionManager(retention_policies)
        self.compressor = LogCompressor()
        self.archiver = LogArchiver(self.archive_directory)
        
        self.auto_rotation = auto_rotation
        self.rotation_interval = rotation_interval
        self._rotation_thread = None
        self._stop_rotation = False
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        
        if self.auto_rotation:
            self._start_auto_rotation()
    
    def _start_auto_rotation(self):
        """Start automatic log rotation in background thread"""
        if self._rotation_thread and self._rotation_thread.is_alive():
            return
        
        self._stop_rotation = False
        self._rotation_thread = threading.Thread(
            target=self._rotation_loop, 
            daemon=True
        )
        self._rotation_thread.start()
        self.logger.info("Started automatic log rotation")
    
    def _rotation_loop(self):
        """Main rotation loop"""
        while not self._stop_rotation:
            try:
                self.rotate_logs()
                time.sleep(self.rotation_interval)
            except Exception as e:
                self.logger.error(f"Error in rotation loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def stop_auto_rotation(self):
        """Stop automatic log rotation"""
        self._stop_rotation = True
        if self._rotation_thread and self._rotation_thread.is_alive():
            self._rotation_thread.join(timeout=5)
        self.logger.info("Stopped automatic log rotation")
    
    def scan_log_files(self, pattern: str = "*") -> List[LogFileInfo]:
        """Scan for log files in directory"""
        log_files = []
        
        # Find all log files matching pattern
        file_patterns = [
            f"**/{pattern}",
            f"**/*{pattern}",
            f"**/*{pattern}*"
        ]
        
        for file_pattern in file_patterns:
            for file_path in self.log_directory.glob(file_pattern):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        log_file = LogFileInfo(
                            path=file_path,
                            size=stat.st_size,
                            created_at=datetime.fromtimestamp(stat.st_ctime),
                            modified_at=datetime.fromtimestamp(stat.st_mtime),
                            compressed=file_path.suffix in ['.gz', '.bz2', '.xz'],
                            archived=file_path.parent != self.log_directory
                        )
                        log_files.append(log_file)
                    except Exception as e:
                        self.logger.warning(f"Failed to stat {file_path}: {e}")
        
        return sorted(log_files, key=lambda x: x.modified_at)
    
    def rotate_logs(self, 
                   pattern: str = "*",
                   compression_enabled: bool = True,
                   archive_enabled: bool = True,
                   retention_policy: str = "general") -> Dict[str, int]:
        """Perform log rotation"""
        self.logger.info("Starting log rotation")
        
        log_files = self.scan_log_files(pattern)
        rotation_results = {
            'compressed': 0,
            'archived': 0,
            'deleted': 0,
            'errors': 0
        }
        
        # Compress large log files
        if compression_enabled:
            large_files = [f for f in log_files if f.size > 10 * 1024 * 1024 and not f.compressed]  # 10MB
            for log_file in large_files:
                try:
                    self.compressor.compress_file(log_file.path, "gzip")
                    rotation_results['compressed'] += 1
                    log_file.compressed = True
                except Exception as e:
                    self.logger.error(f"Failed to compress {log_file.path}: {e}")
                    rotation_results['errors'] += 1
        
        # Archive old log files
        if archive_enabled:
            try:
                archived_files = self.archiver.archive_files(log_files, "month")
                rotation_results['archived'] = len(archived_files)
                
                # Update archived flag
                for log_file in log_files:
                    if log_file.path.parent != self.log_directory:
                        log_file.archived = True
            except Exception as e:
                self.logger.error(f"Failed to archive log files: {e}")
                rotation_results['errors'] += 1
        
        # Clean up old files based on retention policy
        try:
            cleaned_files, size_saved = self.retention_manager.clean_up_old_files(
                log_files, retention_policy
            )
            rotation_results['deleted'] = len(cleaned_files)
            self.logger.info(f"Log rotation completed: {rotation_results}")
        except Exception as e:
            self.logger.error(f"Failed to clean up old files: {e}")
            rotation_results['errors'] += 1
        
        return rotation_results
    
    def ensure_disk_space(self, min_free_mb: int = 100) -> bool:
        """Ensure sufficient disk space for log operations"""
        try:
            import shutil
            free_space = shutil.disk_usage(self.log_directory).free / (1024 * 1024)
            
            if free_space < min_free_mb:
                self.logger.warning(f"Low disk space: {free_space:.2f} MB free")
                
                # Aggressive cleanup
                old_logs = self.retention_manager.get_outdated_files(
                    self.scan_log_files(), "general"
                )
                self.retention_manager.clean_up_old_files(old_logs, "general")
                
                # Check space again
                free_space = shutil.disk_usage(self.log_directory).free / (1024 * 1024)
                if free_space < min_free_mb:
                    self.logger.error(f"Insufficient disk space: {free_space:.2f} MB free")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            return False
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics"""
        log_files = self.scan_log_files()
        
        # Calculate statistics
        total_files = len(log_files)
        total_size = sum(f.size for f in log_files)
        compressed_count = sum(1 for f in log_files if f.compressed)
        archived_count = sum(1 for f in log_files if f.archived)
        
        # Age distribution
        age_distribution = {
            'today': 0,
            'week': 0,
            'month': 0,
            'older': 0
        }
        
        now = datetime.now()
        for log_file in log_files:
            age_days = (now - log_file.modified_at).days
            if age_days == 0:
                age_distribution['today'] += 1
            elif age_days <= 7:
                age_distribution['week'] += 1
            elif age_days <= 30:
                age_distribution['month'] += 1
            else:
                age_distribution['older'] += 1
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'compressed_files': compressed_count,
            'archived_files': archived_count,
            'age_distribution': age_distribution,
            'log_directory': str(self.log_directory),
            'archive_directory': str(self.archive_directory),
            'auto_rotation_enabled': self.auto_rotation
        }
    
    def manual_rotation(self, 
                       pattern: str = "*",
                       force_compression: bool = False,
                       force_archiving: bool = False,
                       force_cleanup: bool = False) -> Dict[str, Any]:
        """Perform manual log rotation with specific options"""
        self.logger.info("Starting manual log rotation")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'pattern': pattern,
            'files_processed': 0,
            'operations': []
        }
        
        log_files = self.scan_log_files(pattern)
        results['files_processed'] = len(log_files)
        
        # Force compression
        if force_compression:
            compression_results = []
            for log_file in log_files:
                if not log_file.compressed:
                    try:
                        compressed_path = self.compressor.compress_file(log_file.path, "gzip")
                        compression_results.append(str(compressed_path))
                    except Exception as e:
                        self.logger.error(f"Failed to compress {log_file.path}: {e}")
            
            results['operations'].append({
                'type': 'compression',
                'files_compressed': len(compression_results),
                'files': compression_results
            })
        
        # Force archiving
        if force_archiving:
            try:
                archived_files = self.archiver.archive_files(log_files, "month")
                results['operations'].append({
                    'type': 'archiving',
                    'files_archived': len(archived_files),
                    'archived_to': str(self.archive_directory)
                })
            except Exception as e:
                self.logger.error(f"Failed to archive files: {e}")
        
        # Force cleanup
        if force_cleanup:
            try:
                cleaned_files, size_saved = self.retention_manager.clean_up_old_files(
                    log_files, "general"
                )
                results['operations'].append({
                    'type': 'cleanup',
                    'files_deleted': len(cleaned_files),
                    'size_saved_mb': size_saved / (1024 * 1024)
                })
            except Exception as e:
                self.logger.error(f"Failed to cleanup files: {e}")
        
        self.logger.info(f"Manual rotation completed: {results}")
        return results
    
    def shutdown(self):
        """Shutdown the rotation manager"""
        if self.auto_rotation:
            self.stop_auto_rotation()
        self.logger.info("Log rotation manager shutdown")


# Global rotation manager instance
_rotation_manager = None

def get_rotation_manager(log_directory: str = "logs",
                        archive_directory: Optional[str] = None,
                        auto_rotation: bool = True) -> LogRotationManager:
    """Get global rotation manager instance"""
    global _rotation_manager
    if _rotation_manager is None:
        _rotation_manager = LogRotationManager(
            log_directory=log_directory,
            archive_directory=archive_directory,
            auto_rotation=auto_rotation
        )
    return _rotation_manager


# Utility functions
def setup_log_rotation(config: Dict[str, Any]) -> LogRotationManager:
    """Setup log rotation with configuration"""
    rotation_manager = LogRotationManager(
        log_directory=config.get('log_directory', 'logs'),
        archive_directory=config.get('archive_directory'),
        retention_policies=config.get('retention_policies'),
        auto_rotation=config.get('auto_rotation', True),
        rotation_interval=config.get('rotation_interval', 3600)
    )
    
    return rotation_manager