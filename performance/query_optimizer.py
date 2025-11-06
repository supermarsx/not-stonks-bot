"""
Database Query Optimizer

Comprehensive database query optimization with analysis, index suggestions,
and performance tracking for trading system queries.
"""

import time
import logging
import re
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import json


class QueryType(Enum):
    """Database query type enumeration"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    UNKNOWN = "unknown"


class IndexType(Enum):
    """Database index type enumeration"""
    PRIMARY_KEY = "primary_key"
    UNIQUE = "unique"
    NON_UNIQUE = "non_unique"
    COMPOSITE = "composite"
    PARTIAL = "partial"


class OptimizationType(Enum):
    """Query optimization type enumeration"""
    ADD_INDEX = "add_index"
    REWRITE_QUERY = "rewrite_query",
    ADD_COMPOSITE_INDEX = "add_composite_index",
    PARTITION_TABLE = "partition_table",
    USE_COVERING_INDEX = "use_covering_index",
    LIMIT_QUERY = "limit_query",
    USE_CACHE = "use_cache"


@dataclass
class QueryPattern:
    """Query pattern analysis result"""
    query_hash: str
    original_query: str
    query_type: QueryType
    table: str
    columns: List[str]
    where_clauses: List[str]
    join_clauses: List[str]
    order_by: List[str]
    group_by: List[str]
    limit: Optional[int] = None
    execution_time_ms: float = 0
    frequency: int = 1
    last_executed: Optional[datetime] = None
    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexRecommendation:
    """Index recommendation result"""
    table: str
    columns: List[str]
    index_type: IndexType
    priority: str  # "high", "medium", "low"
    estimated_improvement: float
    estimated_cost: float
    rationale: str
    sql_create_statement: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryAnalysis:
    """Complete query analysis result"""
    query: str
    query_pattern: QueryPattern
    performance_metrics: Dict[str, Any]
    index_recommendations: List[IndexRecommendation]
    optimization_opportunities: List[Dict[str, Any]]
    potential_issues: List[str]
    execution_plan: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'query_pattern': self.query_pattern.to_dict(),
            'performance_metrics': self.performance_metrics,
            'index_recommendations': [rec.to_dict() for rec in self.index_recommendations],
            'optimization_opportunities': self.optimization_opportunities,
            'potential_issues': self.potential_issues,
            'execution_plan': self.execution_plan
        }


class SQLParser:
    """SQL query parser and analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # SQL keywords for pattern recognition
        self.sql_keywords = {
            'select': ['select', 'distinct'],
            'from': ['from', 'join', 'left join', 'right join', 'inner join', 'outer join'],
            'where': ['where', 'and', 'or'],
            'order': ['order by'],
            'group': ['group by'],
            'limit': ['limit', 'offset'],
            'update': ['update'],
            'insert': ['insert', 'insert into'],
            'delete': ['delete', 'delete from']
        }
    
    def parse_query(self, query: str) -> QueryPattern:
        """Parse SQL query into structured components"""
        query = query.strip().lower()
        query_hash = self._generate_query_hash(query)
        
        # Extract query type
        query_type = self._extract_query_type(query)
        
        # Extract components
        table = self._extract_table(query)
        columns = self._extract_columns(query)
        where_clauses = self._extract_where_clauses(query)
        join_clauses = self._extract_join_clauses(query)
        order_by = self._extract_order_by(query)
        group_by = self._extract_group_by(query)
        limit = self._extract_limit(query)
        
        return QueryPattern(
            query_hash=query_hash,
            original_query=query,
            query_type=query_type,
            table=table,
            columns=columns,
            where_clauses=where_clauses,
            join_clauses=join_clauses,
            order_by=order_by,
            group_by=group_by,
            limit=limit
        )
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query pattern recognition"""
        import hashlib
        # Normalize query by removing whitespace and converting to lowercase
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _extract_query_type(self, query: str) -> QueryType:
        """Extract query type"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith('create'):
            return QueryType.CREATE
        elif query_lower.startswith('alter'):
            return QueryType.ALTER
        elif query_lower.startswith('drop'):
            return QueryType.DROP
        else:
            return QueryType.UNKNOWN
    
    def _extract_table(self, query: str) -> str:
        """Extract primary table name"""
        table_patterns = [
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'insert\s+(?:into\s+)?([a-zA-Z_][a-zA-Z0-9_]*)',
            r'delete\s+(?:from\s+)?([a-zA-Z_][a-zA-Z0-9_]*)',
            r'into\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in table_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from SELECT"""
        if not query.lower().startswith('select'):
            return []
        
        # Find SELECT and FROM
        select_match = re.search(r'select\s+(.+?)\s+from', query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return []
        
        columns_part = select_match.group(1).strip()
        
        # Split columns (simple approach, could be improved for complex cases)
        columns = []
        for col in columns_part.split(','):
            col = col.strip()
            # Remove table aliases and functions
            col = re.sub(r'^[a-zA-Z_][a-zA-Z0-9_]*\.', '', col)  # Remove table prefix
            col = re.sub(r'\s*as\s+[a-zA-Z_][a-zA-Z0-9_]*', '', col, flags=re.IGNORECASE)  # Remove alias
            col = re.sub(r'\(.*?\)', '', col)  # Remove function calls
            columns.append(col.strip())
        
        return [col for col in columns if col and col != '*']
    
    def _extract_where_clauses(self, query: str) -> List[str]:
        """Extract WHERE clause conditions"""
        where_match = re.search(r'where\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+limit|\s+;|$)', 
                               query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR for individual conditions
            conditions = re.split(r'\s+(?:and|or)\s+', where_clause, flags=re.IGNORECASE)
            return [cond.strip() for cond in conditions if cond.strip()]
        return []
    
    def _extract_join_clauses(self, query: str) -> List[str]:
        """Extract JOIN clauses"""
        join_patterns = [
            r'(left\s+join|right\s+join|inner\s+join|outer\s+join|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(left\s+join|right\s+join|inner\s+join|outer\s+join|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        joins = []
        for pattern in join_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            joins.extend([f"{match[0]} {match[1]}" for match in matches])
        
        return joins
    
    def _extract_order_by(self, query: str) -> List[str]:
        """Extract ORDER BY columns"""
        order_match = re.search(r'order\s+by\s+(.+?)(?:\s+limit|\s+;|$)', 
                               query, re.IGNORECASE | re.DOTALL)
        if order_match:
            order_clause = order_match.group(1).strip()
            columns = [col.strip() for col in order_clause.split(',')]
            return columns
        return []
    
    def _extract_group_by(self, query: str) -> List[str]:
        """Extract GROUP BY columns"""
        group_match = re.search(r'group\s+by\s+(.+?)(?:\s+having|\s+order\s+by|\s+limit|\s+;|$)', 
                               query, re.IGNORECASE | re.DOTALL)
        if group_match:
            group_clause = group_match.group(1).strip()
            columns = [col.strip() for col in group_clause.split(',')]
            return columns
        return []
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract LIMIT value"""
        limit_match = re.search(r'limit\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            return int(limit_match.group(1))
        return None


class IndexAnalyzer:
    """Database index analysis and recommendation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Index recommendation rules
        self.index_rules = [
            {
                'name': 'Primary Key Auto-created',
                'priority': 'high',
                'condition': lambda pattern: pattern.query_type == QueryType.SELECT and pattern.table != 'unknown',
                'recommendation': lambda pattern: self._recommend_primary_key(pattern)
            },
            {
                'name': 'WHERE Clause Index',
                'priority': 'high',
                'condition': lambda pattern: pattern.where_clauses,
                'recommendation': lambda pattern: self._recommend_where_index(pattern)
            },
            {
                'name': 'ORDER BY Index',
                'priority': 'medium',
                'condition': lambda pattern: pattern.order_by,
                'recommendation': lambda pattern: self._recommend_order_index(pattern)
            },
            {
                'name': 'GROUP BY Index',
                'priority': 'medium',
                'condition': lambda pattern: pattern.group_by,
                'recommendation': lambda pattern: self._recommend_group_index(pattern)
            },
            {
                'name': 'Composite Index',
                'priority': 'medium',
                'condition': lambda pattern: len(pattern.where_clauses) > 1,
                'recommendation': lambda pattern: self._recommend_composite_index(pattern)
            },
            {
                'name': 'JOIN Column Index',
                'priority': 'high',
                'condition': lambda pattern: pattern.join_clauses,
                'recommendation': lambda pattern: self._recommend_join_index(pattern)
            }
        ]
    
    def analyze_indexes(self, query_pattern: QueryPattern, 
                       table_stats: Dict[str, Any] = None) -> List[IndexRecommendation]:
        """Analyze query and recommend indexes"""
        recommendations = []
        table_stats = table_stats or {}
        
        for rule in self.index_rules:
            try:
                if rule['condition'](query_pattern):
                    recommendation = rule['recommendation'](query_pattern)
                    if recommendation:
                        recommendations.append(recommendation)
            except Exception as e:
                self.logger.error(f"Error applying index rule {rule['name']}: {e}")
        
        # Remove duplicates and sort by priority
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        return self._sort_by_priority(unique_recommendations)
    
    def _recommend_primary_key(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend primary key index"""
        estimated_improvement = 95.0  # Significant improvement for primary key
        
        return IndexRecommendation(
            table=pattern.table,
            columns=['id'],  # Assuming 'id' as primary key
            index_type=IndexType.PRIMARY_KEY,
            priority='high',
            estimated_improvement=estimated_improvement,
            estimated_cost=1.0,
            rationale="Primary key provides unique row identification and is essential for table performance",
            sql_create_statement=f"ALTER TABLE {pattern.table} ADD PRIMARY KEY (id);"
        )
    
    def _recommend_where_index(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend index for WHERE clause columns"""
        # Extract column names from WHERE clauses
        where_columns = []
        for clause in pattern.where_clauses:
            # Simple column extraction (could be improved)
            col_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]', clause)
            if col_match:
                where_columns.append(col_match.group(1))
        
        if not where_columns:
            return None
        
        columns_str = ', '.join(where_columns)
        estimated_improvement = 80.0 if len(where_columns) == 1 else 90.0
        
        return IndexRecommendation(
            table=pattern.table,
            columns=where_columns,
            index_type=IndexType.NON_UNIQUE,
            priority='high',
            estimated_improvement=estimated_improvement,
            estimated_cost=len(where_columns),
            rationale=f"Index on WHERE clause columns ({', '.join(where_columns)}) for faster filtering",
            sql_create_statement=f"CREATE INDEX idx_{pattern.table}_{'_'.join(where_columns)} ON {pattern.table} ({columns_str});"
        )
    
    def _recommend_order_index(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend index for ORDER BY columns"""
        if not pattern.order_by:
            return None
        
        order_columns = []
        for col in pattern.order_by:
            # Remove ASC/DESC and table aliases
            col = re.sub(r'\s+(asc|desc)', '', col, flags=re.IGNORECASE)
            col = col.strip()
            if '.' in col:
                col = col.split('.')[1]  # Remove table prefix
            order_columns.append(col)
        
        columns_str = ', '.join(order_columns)
        estimated_improvement = 70.0
        
        return IndexRecommendation(
            table=pattern.table,
            columns=order_columns,
            index_type=IndexType.NON_UNIQUE,
            priority='medium',
            estimated_improvement=estimated_improvement,
            estimated_cost=len(order_columns),
            rationale=f"Index on ORDER BY columns ({columns_str}) for faster sorting",
            sql_create_statement=f"CREATE INDEX idx_{pattern.table}_order_{'_'.join(order_columns)} ON {pattern.table} ({columns_str});"
        )
    
    def _recommend_group_index(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend index for GROUP BY columns"""
        if not pattern.group_by:
            return None
        
        group_columns = [col.strip() for col in pattern.group_by if col.strip()]
        columns_str = ', '.join(group_columns)
        estimated_improvement = 60.0
        
        return IndexRecommendation(
            table=pattern.table,
            columns=group_columns,
            index_type=IndexType.NON_UNIQUE,
            priority='medium',
            estimated_improvement=estimated_improvement,
            estimated_cost=len(group_columns),
            rationale=f"Index on GROUP BY columns ({columns_str}) for faster aggregation",
            sql_create_statement=f"CREATE INDEX idx_{pattern.table}_group_{'_'.join(group_columns)} ON {pattern.table} ({columns_str});"
        )
    
    def _recommend_composite_index(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend composite index for multiple WHERE conditions"""
        where_columns = []
        for clause in pattern.where_clauses:
            col_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]', clause)
            if col_match:
                where_columns.append(col_match.group(1))
        
        if len(where_columns) < 2:
            return None
        
        columns_str = ', '.join(where_columns)
        estimated_improvement = 85.0
        
        return IndexRecommendation(
            table=pattern.table,
            columns=where_columns,
            index_type=IndexType.COMPOSITE,
            priority='high',
            estimated_improvement=estimated_improvement,
            estimated_cost=len(where_columns) * 1.5,
            rationale=f"Composite index on {columns_str} for optimal performance with multiple filter conditions",
            sql_create_statement=f"CREATE INDEX idx_{pattern.table}_composite_{'_'.join(where_columns)} ON {pattern.table} ({columns_str});"
        )
    
    def _recommend_join_index(self, pattern: QueryPattern) -> IndexRecommendation:
        """Recommend index for JOIN columns"""
        join_columns = []
        for join in pattern.join_clauses:
            # Extract join conditions
            on_match = re.search(r'on\s+([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)', 
                               join, re.IGNORECASE)
            if on_match:
                # Get column from the current table (assume it's the first one)
                join_columns.append(on_match.group(2))
        
        if not join_columns:
            return None
        
        columns_str = ', '.join(join_columns)
        estimated_improvement = 75.0
        
        return IndexRecommendation(
            table=pattern.table,
            columns=join_columns,
            index_type=IndexType.NON_UNIQUE,
            priority='high',
            estimated_improvement=estimated_improvement,
            estimated_cost=len(join_columns),
            rationale=f"Index on JOIN column(s) ({columns_str}) for faster table joins",
            sql_create_statement=f"CREATE INDEX idx_{pattern.table}_join_{'_'.join(join_columns)} ON {pattern.table} ({columns_str});"
        )
    
    def _deduplicate_recommendations(self, recommendations: List[IndexRecommendation]) -> List[IndexRecommendation]:
        """Remove duplicate index recommendations"""
        unique_recommendations = []
        seen_signatures = set()
        
        for rec in recommendations:
            # Create signature based on table and columns
            signature = (rec.table, tuple(rec.columns))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _sort_by_priority(self, recommendations: List[IndexRecommendation]) -> List[IndexRecommendation]:
        """Sort recommendations by priority"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(recommendations, key=lambda x: priority_order.get(x.priority, 3))


class QueryOptimizer:
    """Main query optimization engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sql_parser = SQLParser()
        self.index_analyzer = IndexAnalyzer()
        
        # Query performance tracking
        self.query_history = deque(maxlen=10000)
        self.query_patterns = {}
        self.slow_queries = deque(maxlen=1000)
        
        # Optimization rules
        self.optimization_rules = [
            self._optimize_missing_limit,
            self._optimize_select_star,
            self._optimize_unnecessary_distinct,
            self._optimize_or_conditions,
            self._optimize_subqueries,
            self._optimize_join_order
        ]
    
    def analyze_query(self, 
                     query: str, 
                     execution_time_ms: float = None,
                     connection: sqlite3.Connection = None) -> QueryAnalysis:
        """Analyze query and provide optimization recommendations"""
        start_time = time.time()
        
        # Parse query
        query_pattern = self.sql_parser.parse_query(query)
        
        # Track execution time if provided
        if execution_time_ms is not None:
            query_pattern.execution_time_ms = execution_time_ms
            query_pattern.last_executed = datetime.now()
        
        # Get performance metrics
        performance_metrics = self._analyze_performance(query_pattern)
        
        # Get index recommendations
        table_stats = self._get_table_statistics(connection) if connection else {}
        index_recommendations = self.index_analyzer.analyze_indexes(query_pattern, table_stats)
        
        # Find optimization opportunities
        optimization_opportunities = self._find_optimization_opportunities(query_pattern)
        
        # Identify potential issues
        potential_issues = self._identify_potential_issues(query_pattern)
        
        # Get execution plan if connection available
        execution_plan = None
        if connection:
            execution_plan = self._get_execution_plan(query, connection)
        
        analysis_time = (time.time() - start_time) * 1000
        
        analysis = QueryAnalysis(
            query=query,
            query_pattern=query_pattern,
            performance_metrics=performance_metrics,
            index_recommendations=index_recommendations,
            optimization_opportunities=optimization_opportunities,
            potential_issues=potential_issues,
            execution_plan=execution_plan
        )
        
        # Store analysis
        self.query_history.append(analysis)
        self._update_query_patterns(query_pattern)
        
        self.logger.debug(f"Query analysis completed in {analysis_time:.2f}ms")
        return analysis
    
    def _analyze_performance(self, query_pattern: QueryPattern) -> Dict[str, Any]:
        """Analyze query performance characteristics"""
        metrics = {
            'query_type': query_pattern.query_type.value,
            'table': query_pattern.table,
            'has_where': bool(query_pattern.where_clauses),
            'has_join': bool(query_pattern.join_clauses),
            'has_order_by': bool(query_pattern.order_by),
            'has_group_by': bool(query_pattern.group_by),
            'has_limit': query_pattern.limit is not None,
            'complexity_score': self._calculate_complexity_score(query_pattern)
        }
        
        # Performance indicators
        if query_pattern.execution_time_ms > 0:
            metrics.update({
                'execution_time_ms': query_pattern.execution_time_ms,
                'performance_rating': self._rate_performance(query_pattern.execution_time_ms),
                'is_slow': query_pattern.execution_time_ms > 1000
            })
        
        # Efficiency indicators
        metrics.update({
            'estimated_scan_efficiency': self._estimate_scan_efficiency(query_pattern),
            'index_usage_potential': self._estimate_index_usage(query_pattern)
        })
        
        return metrics
    
    def _calculate_complexity_score(self, pattern: QueryPattern) -> float:
        """Calculate query complexity score"""
        score = 0.0
        
        # Base complexity by query type
        type_scores = {
            QueryType.SELECT: 1.0,
            QueryType.INSERT: 0.5,
            QueryType.UPDATE: 0.8,
            QueryType.DELETE: 0.6
        }
        score += type_scores.get(pattern.query_type, 1.0)
        
        # Add complexity for joins
        score += len(pattern.join_clauses) * 0.5
        
        # Add complexity for WHERE clauses
        score += len(pattern.where_clauses) * 0.3
        
        # Add complexity for GROUP BY and ORDER BY
        score += len(pattern.group_by) * 0.4
        score += len(pattern.order_by) * 0.4
        
        # Add complexity for columns selected
        score += len(pattern.columns) * 0.1
        
        return min(score, 10.0)  # Cap at 10
    
    def _rate_performance(self, execution_time_ms: float) -> str:
        """Rate query performance"""
        if execution_time_ms < 100:
            return "excellent"
        elif execution_time_ms < 500:
            return "good"
        elif execution_time_ms < 1000:
            return "fair"
        elif execution_time_ms < 5000:
            return "poor"
        else:
            return "very_poor"
    
    def _estimate_scan_efficiency(self, pattern: QueryPattern) -> str:
        """Estimate table scan efficiency"""
        if not pattern.where_clauses and not pattern.join_clauses:
            return "low"  # Full table scan likely
        
        if pattern.table == "unknown":
            return "unknown"
        
        # Simple heuristic based on WHERE clauses
        where_score = len(pattern.where_clauses)
        
        if where_score >= 3:
            return "high"
        elif where_score >= 2:
            return "medium"
        else:
            return "medium"
    
    def _estimate_index_usage(self, pattern: QueryPattern) -> str:
        """Estimate potential for index usage"""
        potential = 0
        
        if pattern.where_clauses:
            potential += len(pattern.where_clauses)
        
        if pattern.join_clauses:
            potential += len(pattern.join_clauses)
        
        if pattern.order_by:
            potential += 1
        
        if pattern.group_by:
            potential += 1
        
        if potential >= 4:
            return "high"
        elif potential >= 2:
            return "medium"
        else:
            return "low"
    
    def _find_optimization_opportunities(self, pattern: QueryPattern) -> List[Dict[str, Any]]:
        """Find query optimization opportunities"""
        opportunities = []
        
        for rule in self.optimization_rules:
            try:
                opportunity = rule(pattern)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                self.logger.error(f"Error in optimization rule: {e}")
        
        return opportunities
    
    def _optimize_missing_limit(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Suggest adding LIMIT for SELECT queries without pagination"""
        if (pattern.query_type == QueryType.SELECT and 
            not pattern.limit and 
            pattern.table != "unknown"):
            
            return {
                'type': 'add_limit',
                'description': 'Consider adding LIMIT clause to prevent excessive data retrieval',
                'priority': 'medium',
                'suggestion': 'Add LIMIT with appropriate value based on use case',
                'estimated_improvement': 'Reduced memory usage and faster response times'
            }
        return None
    
    def _optimize_select_star(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Suggest avoiding SELECT * for performance"""
        if pattern.columns == ['*']:
            return {
                'type': 'avoid_select_star',
                'description': 'SELECT * retrieves all columns, which may be inefficient',
                'priority': 'medium',
                'suggestion': 'Specify only needed columns explicitly',
                'estimated_improvement': 'Reduced I/O and memory usage'
            }
        return None
    
    def _optimize_unnecessary_distinct(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Suggest removing unnecessary DISTINCT"""
        if (pattern.query_type == QueryType.SELECT and 
            '*' not in pattern.columns and 
            len(pattern.columns) <= 2):
            
            return {
                'type': 'remove_distinct',
                'description': 'DISTINCT may be unnecessary if few columns are selected',
                'priority': 'low',
                'suggestion': 'Consider removing DISTINCT if duplicates are acceptable',
                'estimated_improvement': 'Faster query execution'
            }
        return None
    
    def _optimize_or_conditions(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Optimize OR conditions that might benefit from UNION"""
        or_clauses = [clause for clause in pattern.where_clauses if ' or ' in clause.lower()]
        
        if or_clauses and len(or_clauses) > 1:
            return {
                'type': 'optimize_or_conditions',
                'description': 'Multiple OR conditions may be optimized with UNION',
                'priority': 'low',
                'suggestion': 'Consider splitting OR conditions into UNION queries',
                'estimated_improvement': 'Potentially faster execution with proper indexes'
            }
        return None
    
    def _optimize_subqueries(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Identify subquery optimization opportunities"""
        if 'select' in pattern.original_query and ' where ' in pattern.original_query.lower():
            return {
                'type': 'optimize_subquery',
                'description': 'Subqueries might be optimized with JOINs',
                'priority': 'medium',
                'suggestion': 'Consider converting subqueries to JOINs where appropriate',
                'estimated_improvement': 'Better query execution plan'
            }
        return None
    
    def _optimize_join_order(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Suggest join order optimization"""
        if len(pattern.join_clauses) > 2:
            return {
                'type': 'optimize_join_order',
                'description': 'Multiple joins may benefit from explicit join order hints',
                'priority': 'low',
                'suggestion': 'Consider optimizing join order for better performance',
                'estimated_improvement': 'Reduced query execution time'
            }
        return None
    
    def _identify_potential_issues(self, pattern: QueryPattern) -> List[str]:
        """Identify potential query issues"""
        issues = []
        
        # Check for missing WHERE clause on UPDATE/DELETE
        if pattern.query_type in [QueryType.UPDATE, QueryType.DELETE]:
            if not pattern.where_clauses:
                issues.append(f"UPDATE/DELETE without WHERE clause on {pattern.table} - risky operation")
        
        # Check for missing LIMIT on large SELECT
        if (pattern.query_type == QueryType.SELECT and 
            not pattern.limit and 
            len(pattern.columns) > 10):
            issues.append("Large SELECT without LIMIT - may retrieve excessive data")
        
        # Check for complex WHERE conditions
        if len(pattern.where_clauses) > 5:
            issues.append(f"Complex WHERE clause with {len(pattern.where_clauses)} conditions - consider indexing")
        
        # Check for missing indexes on frequently filtered columns
        if pattern.where_clauses:
            issues.append("WHERE clause detected - ensure appropriate indexes exist")
        
        return issues
    
    def _get_table_statistics(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        """Get table statistics for better recommendations"""
        stats = {}
        try:
            cursor = connection.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get index information
                    cursor.execute(f"PRAGMA index_list({table})")
                    indexes = cursor.fetchall()
                    
                    stats[table] = {
                        'row_count': row_count,
                        'indexes': len(indexes)
                    }
                except Exception:
                    continue  # Skip tables with issues
                    
        except Exception as e:
            self.logger.error(f"Error getting table statistics: {e}")
        
        return stats
    
    def _get_execution_plan(self, query: str, connection: sqlite3.Connection) -> Dict[str, Any]:
        """Get query execution plan"""
        try:
            cursor = connection.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            plan_rows = cursor.fetchall()
            
            plan = {
                'steps': [],
                'query': query
            }
            
            for row in plan_rows:
                plan['steps'].append({
                    'select_id': row[0],
                    'order': row[1],
                    'from': row[2],
                    'detail': row[3]
                })
            
            return plan
        except Exception as e:
            self.logger.error(f"Error getting execution plan: {e}")
            return {'error': str(e)}
    
    def _update_query_patterns(self, pattern: QueryPattern):
        """Update query pattern statistics"""
        if pattern.query_hash in self.query_patterns:
            existing = self.query_patterns[pattern.query_hash]
            existing.frequency += 1
            existing.last_executed = pattern.last_executed or datetime.now()
            
            # Update execution time if available
            if pattern.execution_time_ms > 0:
                existing.execution_time_ms = (
                    (existing.execution_time_ms * (existing.frequency - 1) + 
                     pattern.execution_time_ms) / existing.frequency
                )
        else:
            pattern.frequency = 1
            self.query_patterns[pattern.query_hash] = pattern
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.query_history:
            return {'message': 'No queries analyzed yet'}
        
        # Aggregate statistics
        total_queries = len(self.query_history)
        slow_queries = [q for q in self.query_history 
                       if q.query_pattern.execution_time_ms > 1000]
        avg_execution_time = statistics.mean(
            [q.query_pattern.execution_time_ms for q in self.query_history 
             if q.query_pattern.execution_time_ms > 0]
        ) if self.query_history else 0
        
        # Most frequent patterns
        frequent_patterns = sorted(
            self.query_patterns.values(),
            key=lambda x: x.frequency,
            reverse=True
        )[:10]
        
        # Index recommendations summary
        all_recommendations = []
        for analysis in self.query_history[-100:]:  # Last 100 queries
            all_recommendations.extend(analysis.index_recommendations)
        
        # Group recommendations by table
        recommendations_by_table = defaultdict(list)
        for rec in all_recommendations:
            recommendations_by_table[rec.table].append(rec)
        
        # Performance trends
        execution_times = [q.query_pattern.execution_time_ms for q in self.query_history[-50:]
                         if q.query_pattern.execution_time_ms > 0]
        performance_trend = 'stable'
        if len(execution_times) > 10:
            recent_avg = statistics.mean(execution_times[-10:])
            older_avg = statistics.mean(execution_times[-20:-10])
            if recent_avg > older_avg * 1.2:
                performance_trend = 'degrading'
            elif recent_avg < older_avg * 0.8:
                performance_trend = 'improving'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_queries_analyzed': total_queries,
                'slow_queries_count': len(slow_queries),
                'average_execution_time_ms': avg_execution_time,
                'performance_trend': performance_trend
            },
            'most_frequent_patterns': [
                {
                    'query_hash': p.query_hash,
                    'frequency': p.frequency,
                    'table': p.table,
                    'query_type': p.query_type.value,
                    'avg_execution_time_ms': p.execution_time_ms
                }
                for p in frequent_patterns
            ],
            'index_recommendations_summary': {
                table: [
                    {
                        'columns': rec.columns,
                        'priority': rec.priority,
                        'estimated_improvement': rec.estimated_improvement,
                        'sql': rec.sql_create_statement
                    }
                    for rec in recommendations
                ]
                for table, recommendations in recommendations_by_table.items()
            },
            'optimization_opportunities': [
                {
                    'type': opp['type'],
                    'description': opp['description'],
                    'priority': opp['priority'],
                    'count': len([o for o in self.query_history if any(
                        optimization.get('type') == opp['type'] 
                        for optimization in o.optimization_opportunities
                    )])
                }
                for opp in [
                    {'type': 'add_limit', 'description': 'Add LIMIT clause', 'priority': 'medium'},
                    {'type': 'avoid_select_star', 'description': 'Avoid SELECT *', 'priority': 'medium'},
                    {'type': 'optimize_subquery', 'description': 'Optimize subqueries', 'priority': 'medium'}
                ]
            ],
            'slow_queries': [
                {
                    'query': analysis.query,
                    'execution_time_ms': analysis.query_pattern.execution_time_ms,
                    'table': analysis.query_pattern.table
                }
                for analysis in slow_queries[-10:]  # Last 10 slow queries
            ]
        }
    
    def get_slow_queries_report(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get report of slowest queries"""
        slow_queries = sorted(
            self.query_history,
            key=lambda x: x.query_pattern.execution_time_ms or 0,
            reverse=True
        )[:limit]
        
        return [
            {
                'query': analysis.query,
                'execution_time_ms': analysis.query_pattern.execution_time_ms,
                'table': analysis.query_pattern.table,
                'query_type': analysis.query_pattern.query_type.value,
                'recommendations_count': len(analysis.index_recommendations),
                'complexity_score': analysis.performance_metrics.get('complexity_score', 0),
                'potential_issues': analysis.potential_issues
            }
            for analysis in slow_queries
        ]


# Global query optimizer instance
_query_optimizer = None

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance"""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    return _query_optimizer


def initialize_query_optimizer() -> QueryOptimizer:
    """Initialize query optimizer"""
    return QueryOptimizer()