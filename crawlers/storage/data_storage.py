"""
Data Storage and Retrieval System
Handles storage, retrieval, and management of crawled data
"""

import asyncio
import logging
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
from pathlib import Path
import aiofiles
import aiofiles.os
import hashlib
import gzip
import pickle

from ..base.base_crawler import CrawlResult


class DataStorage:
    """Manages data storage and retrieval"""
    
    def __init__(self, storage_path: str = "./data"):
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        
        # Create storage directories
        self.storage_path.mkdir(exist_ok=True)
        self._create_storage_structure()
        
        # SQLite database for metadata and indexing
        self.db_path = self.storage_path / "crawler_metadata.db"
        self._init_database()
    
    def _create_storage_structure(self):
        """Create organized storage structure"""
        directories = [
            "market_data",
            "news", 
            "social_media",
            "economic",
            "patterns",
            "raw",
            "processed",
            "temp",
            "archive"
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawl_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crawler_name TEXT NOT NULL,
                crawl_timestamp DATETIME NOT NULL,
                data_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                record_count INTEGER,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                execution_time REAL,
                metadata TEXT,
                hash TEXT,
                compressed BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_crawler_time 
            ON crawl_records (crawler_name, crawl_timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_type 
            ON crawl_records (data_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success 
            ON crawl_records (success)
        """)
        
        conn.commit()
        conn.close()
    
    async def store_crawl_result(self, crawler_name: str, result: CrawlResult, 
                               data_category: str = "general") -> str:
        """Store crawl result to file and database"""
        try:
            timestamp = result.timestamp
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"{crawler_name}_{timestamp_str}.json"
            file_path = self.storage_path / data_category / filename
            
            # Convert result to serializable format
            result_data = {
                'crawler_name': crawler_name,
                'timestamp': result.timestamp.isoformat(),
                'success': result.success,
                'crawl_duration': result.crawl_duration,
                'error_message': result.error_message,
                'metadata': result.metadata or {},
                'source': result.source,
                'data': result.data
            }
            
            # Calculate file hash
            data_json = json.dumps(result_data['data'], default=str)
            file_hash = hashlib.md5(data_json.encode()).hexdigest()
            
            # Store data
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(result_data, indent=2, default=str))
            
            # Get file size
            file_size = await aiofiles.os.path.getsize(file_path)
            
            # Store metadata in database
            await self._store_metadata({
                'crawler_name': crawler_name,
                'crawl_timestamp': timestamp,
                'data_type': data_category,
                'file_path': str(file_path),
                'file_size': file_size,
                'record_count': len(result.data) if isinstance(result.data, (list, dict)) else 1,
                'success': result.success,
                'error_message': result.error_message,
                'execution_time': result.crawl_duration,
                'metadata': result.metadata,
                'hash': file_hash,
                'compressed': False
            })
            
            self.logger.info(f"Stored crawl result: {filename}")
            return str(file_path)
        
        except Exception as e:
            self.logger.error(f"Error storing crawl result: {e}")
            raise
    
    async def _store_metadata(self, metadata: Dict[str, Any]):
        """Store metadata in SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO crawl_records 
            (crawler_name, crawl_timestamp, data_type, file_path, file_size, 
             record_count, success, error_message, execution_time, metadata, hash, compressed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['crawler_name'],
            metadata['crawl_timestamp'],
            metadata['data_type'],
            metadata['file_path'],
            metadata['file_size'],
            metadata['record_count'],
            metadata['success'],
            metadata['error_message'],
            metadata['execution_time'],
            json.dumps(metadata['metadata'], default=str),
            metadata['hash'],
            metadata['compressed']
        ))
        
        conn.commit()
        conn.close()
    
    async def retrieve_data(self, crawler_name: str, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None, 
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve crawl data by crawler and date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT crawler_name, crawl_timestamp, data_type, file_path, 
                       file_size, record_count, success, execution_time, metadata
                FROM crawl_records 
                WHERE crawler_name = ?
            """
            params = [crawler_name]
            
            if start_date:
                query += " AND crawl_timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND crawl_timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY crawl_timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            records = cursor.fetchall()
            conn.close()
            
            # Load data from files
            results = []
            for record in records:
                try:
                    file_path = record[3]
                    async with aiofiles.open(file_path, 'r') as f:
                        data = json.loads(await f.read())
                    results.append(data)
                except Exception as e:
                    self.logger.error(f"Error loading data from {file_path}: {e}")
                    continue
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return []
    
    async def get_latest_data(self, crawler_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get latest data from a crawler"""
        results = await self.retrieve_data(crawler_name, limit=limit)
        return results[0] if results else None
    
    async def search_data(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search crawl data using various parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build search query
            base_query = """
                SELECT crawler_name, crawl_timestamp, data_type, file_path, 
                       file_size, record_count, success, metadata
                FROM crawl_records 
                WHERE 1=1
            """
            params = []
            
            # Add filters
            if 'crawler_name' in query_params:
                base_query += " AND crawler_name = ?"
                params.append(query_params['crawler_name'])
            
            if 'data_type' in query_params:
                base_query += " AND data_type = ?"
                params.append(query_params['data_type'])
            
            if 'start_date' in query_params:
                base_query += " AND crawl_timestamp >= ?"
                params.append(query_params['start_date'])
            
            if 'end_date' in query_params:
                base_query += " AND crawl_timestamp <= ?"
                params.append(query_params['end_date'])
            
            if 'success' in query_params:
                base_query += " AND success = ?"
                params.append(query_params['success'])
            
            if 'min_execution_time' in query_params:
                base_query += " AND execution_time >= ?"
                params.append(query_params['min_execution_time'])
            
            base_query += " ORDER BY crawl_timestamp DESC"
            
            if 'limit' in query_params:
                base_query += " LIMIT ?"
                params.append(query_params['limit'])
            
            cursor.execute(base_query, params)
            records = cursor.fetchall()
            conn.close()
            
            # Load data from files
            results = []
            for record in records:
                try:
                    file_path = record[3]
                    async with aiofiles.open(file_path, 'r') as f:
                        data = json.loads(await f.read())
                    results.append(data)
                except Exception as e:
                    self.logger.error(f"Error loading data from {file_path}: {e}")
                    continue
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error searching data: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_records,
                    COUNT(CASE WHEN success = 0 THEN 1 END) as failed_records,
                    SUM(file_size) as total_size,
                    AVG(execution_time) as avg_execution_time,
                    MIN(crawl_timestamp) as earliest_record,
                    MAX(crawl_timestamp) as latest_record
                FROM crawl_records
            """)
            
            overall_stats = cursor.fetchone()
            
            # Statistics by crawler
            cursor.execute("""
                SELECT 
                    crawler_name,
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_runs,
                    AVG(execution_time) as avg_execution_time,
                    SUM(file_size) as total_size
                FROM crawl_records
                GROUP BY crawler_name
                ORDER BY total_runs DESC
            """)
            
            crawler_stats = cursor.fetchall()
            
            # Statistics by data type
            cursor.execute("""
                SELECT 
                    data_type,
                    COUNT(*) as total_records,
                    SUM(file_size) as total_size
                FROM crawl_records
                GROUP BY data_type
                ORDER BY total_records DESC
            """)
            
            data_type_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'overall': {
                    'total_records': overall_stats[0],
                    'successful_records': overall_stats[1],
                    'failed_records': overall_stats[2],
                    'total_size_bytes': overall_stats[3],
                    'total_size_mb': round(overall_stats[3] / (1024 * 1024), 2) if overall_stats[3] else 0,
                    'avg_execution_time': round(overall_stats[4], 2) if overall_stats[4] else 0,
                    'earliest_record': overall_stats[5],
                    'latest_record': overall_stats[6]
                },
                'by_crawler': [
                    {
                        'crawler_name': stat[0],
                        'total_runs': stat[1],
                        'successful_runs': stat[2],
                        'success_rate': round((stat[2] / stat[1] * 100), 2) if stat[1] > 0 else 0,
                        'avg_execution_time': round(stat[3], 2) if stat[3] else 0,
                        'total_size_mb': round(stat[4] / (1024 * 1024), 2) if stat[4] else 0
                    }
                    for stat in crawler_stats
                ],
                'by_data_type': [
                    {
                        'data_type': stat[0],
                        'total_records': stat[1],
                        'total_size_mb': round(stat[2] / (1024 * 1024), 2) if stat[2] else 0
                    }
                    for stat in data_type_stats
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def compress_old_data(self, days_old: int = 30):
        """Compress data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, crawl_timestamp 
                FROM crawl_records 
                WHERE crawl_timestamp < ? AND compressed = FALSE
            """, (cutoff_date,))
            
            records = cursor.fetchall()
            
            for file_path, crawl_timestamp in records:
                try:
                    file_path = Path(file_path)
                    if not file_path.exists():
                        continue
                    
                    # Compress file
                    compressed_path = file_path.with_suffix('.json.gz')
                    
                    async with aiofiles.open(file_path, 'rb') as f_in:
                        async with aiofiles.open(compressed_path, 'wb') as f_out:
                            async for chunk in f_in:
                                await f_out.write(chunk)
                    
                    # Remove original file
                    await aiofiles.os.remove(file_path)
                    
                    # Update database
                    cursor.execute("""
                        UPDATE crawl_records 
                        SET file_path = ?, compressed = TRUE 
                        WHERE file_path = ?
                    """, (str(compressed_path), str(file_path)))
                
                except Exception as e:
                    self.logger.error(f"Error compressing {file_path}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Compressed {len(records)} old files")
        
        except Exception as e:
            self.logger.error(f"Error compressing old data: {e}")
    
    async def cleanup_old_data(self, days_old: int = 90):
        """Remove data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path 
                FROM crawl_records 
                WHERE crawl_timestamp < ?
            """, (cutoff_date,))
            
            records = cursor.fetchall()
            
            for (file_path,) in records:
                try:
                    file_path = Path(file_path)
                    
                    # Remove compressed file if exists
                    compressed_path = file_path.with_suffix('.json.gz')
                    if compressed_path.exists():
                        await aiofiles.os.remove(compressed_path)
                    
                    # Remove original file if exists
                    if file_path.exists():
                        await aiofiles.os.remove(file_path)
                
                except Exception as e:
                    self.logger.error(f"Error removing {file_path}: {e}")
                    continue
            
            # Delete old records from database
            cursor.execute("DELETE FROM crawl_records WHERE crawl_timestamp < ?", (cutoff_date,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up data older than {days_old} days")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def export_data(self, crawler_name: str, start_date: datetime, 
                         end_date: datetime, output_format: str = "json") -> str:
        """Export data in various formats"""
        try:
            data = await self.retrieve_data(
                crawler_name, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if not data:
                return ""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "json":
                # Export as JSON
                export_path = self.storage_path / "exports" / f"{crawler_name}_{timestamp}.json"
                export_path.parent.mkdir(exist_ok=True)
                
                export_data = {
                    'export_info': {
                        'crawler_name': crawler_name,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'export_timestamp': datetime.now().isoformat(),
                        'record_count': len(data)
                    },
                    'data': data
                }
                
                async with aiofiles.open(export_path, 'w') as f:
                    await f.write(json.dumps(export_data, indent=2, default=str))
                
                return str(export_path)
            
            elif output_format.lower() == "csv":
                # Export as CSV
                export_path = self.storage_path / "exports" / f"{crawler_name}_{timestamp}.csv"
                export_path.parent.mkdir(exist_ok=True)
                
                # Convert to DataFrame and export
                if isinstance(data[0]['data'], list):
                    df_data = []
                    for record in data:
                        if isinstance(record['data'], list):
                            for item in record['data']:
                                df_data.append({**record, 'data': item})
                        else:
                            df_data.append(record)
                    
                    df = pd.DataFrame(df_data)
                    df.to_csv(export_path, index=False)
                
                return str(export_path)
            
            else:
                raise ValueError(f"Unsupported export format: {output_format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            raise


class DataMonitor:
    """Monitors data quality and storage health"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            'max_execution_time': 300,  # 5 minutes
            'min_data_freshness': 3600,  # 1 hour
            'min_success_rate': 0.8,  # 80%
            'max_error_rate': 0.2,  # 20%
            'min_data_points': 1
        }
    
    async def check_data_quality(self, crawler_name: str) -> Dict[str, Any]:
        """Check data quality for a specific crawler"""
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)
            
            records = await self.storage.search_data({
                'crawler_name': crawler_name,
                'start_date': start_date,
                'end_date': end_date,
                'limit': 100
            })
            
            if not records:
                return {
                    'status': 'no_data',
                    'message': 'No data found for the specified period'
                }
            
            # Analyze quality metrics
            total_records = len(records)
            successful_records = len([r for r in records if r['success']])
            failed_records = total_records - successful_records
            
            success_rate = successful_records / total_records if total_records > 0 else 0
            error_rate = failed_records / total_records if total_records > 0 else 1
            
            # Execution time analysis
            execution_times = [r['crawl_duration'] for r in records if r['crawl_duration']]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            max_execution_time = max(execution_times) if execution_times else 0
            
            # Data freshness
            latest_record = max(records, key=lambda x: x['timestamp'])
            latest_timestamp = datetime.fromisoformat(latest_record['timestamp'])
            data_freshness = (datetime.now() - latest_timestamp).total_seconds()
            
            # Data completeness
            total_data_points = 0
            for record in records:
                if record['data']:
                    if isinstance(record['data'], (list, dict)):
                        total_data_points += len(record['data'])
                    else:
                        total_data_points += 1
            
            # Determine status
            status = 'healthy'
            issues = []
            
            if success_rate < self.thresholds['min_success_rate']:
                status = 'degraded'
                issues.append(f"Low success rate: {success_rate:.2%}")
            
            if error_rate > self.thresholds['max_error_rate']:
                status = 'degraded'
                issues.append(f"High error rate: {error_rate:.2%}")
            
            if avg_execution_time > self.thresholds['max_execution_time']:
                status = 'degraded'
                issues.append(f"High execution time: {avg_execution_time:.1f}s")
            
            if data_freshness > self.thresholds['min_data_freshness']:
                status = 'degraded'
                issues.append(f"Data not fresh: {data_freshness/60:.1f} minutes old")
            
            if total_data_points < self.thresholds['min_data_points']:
                status = 'degraded'
                issues.append("Insufficient data points")
            
            return {
                'status': status,
                'crawler_name': crawler_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'total_records': total_records,
                    'successful_records': successful_records,
                    'failed_records': failed_records,
                    'success_rate': success_rate,
                    'error_rate': error_rate,
                    'avg_execution_time': avg_execution_time,
                    'max_execution_time': max_execution_time,
                    'data_freshness_seconds': data_freshness,
                    'total_data_points': total_data_points
                },
                'issues': issues,
                'thresholds': self.thresholds
            }
        
        except Exception as e:
            self.logger.error(f"Error checking data quality: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'crawler_name': crawler_name
            }
    
    async def check_storage_health(self) -> Dict[str, Any]:
        """Check overall storage health"""
        try:
            stats = await self.storage.get_statistics()
            
            # Check storage space (would need actual disk usage in production)
            storage_path = self.storage.storage_path
            total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
            
            # Check database integrity
            conn = sqlite3.connect(self.storage.db_path)
            cursor = conn.cursor()
            
            # Check for orphaned records
            cursor.execute("""
                SELECT COUNT(*) FROM crawl_records cr 
                WHERE NOT EXISTS (
                    SELECT 1 FROM crawl_records cr2 WHERE cr2.file_path = cr.file_path
                )
            """)
            orphaned_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Determine status
            status = 'healthy'
            issues = []
            
            if stats['overall']['success_rate'] < 0.7:
                status = 'degraded'
                issues.append("Low overall success rate")
            
            if orphaned_count > 0:
                status = 'degraded'
                issues.append(f"{orphaned_count} orphaned records found")
            
            # Storage size warning (example: > 1GB)
            if total_size > 1024 * 1024 * 1024:  # 1GB
                issues.append(f"Large storage usage: {total_size / (1024*1024):.1f} MB")
            
            return {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'storage_path': str(storage_path),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'orphaned_records': orphaned_count,
                'statistics': stats,
                'issues': issues
            }
        
        except Exception as e:
            self.logger.error(f"Error checking storage health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            # Check all crawlers
            crawler_names = set()
            for record in await self.storage.search_data({'limit': 1000}):
                crawler_names.add(record['crawler_name'])
            
            crawler_reports = {}
            for crawler_name in crawler_names:
                crawler_reports[crawler_name] = await self.check_data_quality(crawler_name)
            
            storage_health = await self.check_storage_health()
            
            # Overall status
            all_statuses = [report['status'] for report in crawler_reports.values()]
            if 'error' in all_statuses:
                overall_status = 'error'
            elif 'degraded' in all_statuses:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            return {
                'report_timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'crawler_reports': crawler_reports,
                'storage_health': storage_health,
                'summary': {
                    'total_crawlers': len(crawler_reports),
                    'healthy_crawlers': len([r for r in crawler_reports.values() if r['status'] == 'healthy']),
                    'degraded_crawlers': len([r for r in crawler_reports.values() if r['status'] == 'degraded']),
                    'error_crawlers': len([r for r in crawler_reports.values() if r['status'] == 'error'])
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }