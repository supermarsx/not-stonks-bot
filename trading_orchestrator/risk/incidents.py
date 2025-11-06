"""
Incident Management System

Provides comprehensive incident tracking and postmortem analysis:
- Incident creation and tracking
- Postmortem analysis
- Root cause investigation
- Action item tracking
- Incident metrics and reporting
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from config.database import get_db

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKER = "blocker"


class IncidentStatus(str, Enum):
    """Incident status types."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentPriority(str, Enum):
    """Incident priority levels."""
    P1 = "p1"  # Highest
    P2 = "p2"  # High
    P3 = "p3"  # Medium
    P4 = "p4"  # Low


class IncidentManager:
    """
    Manages trading incidents and postmortem analysis.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Incident categories
        self.incident_categories = {
            "trading_error": "Trading execution errors",
            "risk_breach": "Risk limit breaches",
            "system_failure": "System or connectivity failures",
            "data_issue": "Market data or feed issues",
            "compliance": "Compliance or regulatory issues",
            "strategy_failure": "Trading strategy failures",
            "performance": "Performance degradation",
            "security": "Security incidents"
        }
        
        logger.info(f"IncidentManager initialized for user {self.user_id}")
    
    async def create_incident(self, title: str, description: str, 
                            severity: IncidentSeverity = IncidentSeverity.MEDIUM,
                            category: str = "trading_error",
                            event_type: str = None,
                            data: Dict[str, Any] = None) -> int:
        """
        Create a new incident.
        
        Args:
            title: Incident title/summary
            description: Detailed incident description
            severity: Incident severity level
            category: Incident category
            event_type: Related event type (optional)
            data: Additional incident data
            
        Returns:
            Created incident ID
        """
        try:
            incident = Incident(
                user_id=self.user_id,
                title=title,
                description=description,
                severity=severity,
                status=IncidentStatus.OPEN,
                category=category,
                priority=await self._determine_priority(severity),
                event_type=event_type,
                incident_data=data or {},
                detected_at=datetime.now(),
                created_at=datetime.now()
            )
            
            self.db.add(incident)
            self.db.commit()
            
            logger.info(f"Incident created for user {self.user_id}: {title} (Severity: {severity.value})")
            
            # Log the incident creation
            await self._log_incident_event(incident.id, "incident_created", {
                "title": title,
                "severity": severity.value,
                "category": category
            })
            
            return incident.id
            
        except Exception as e:
            logger.error(f"Incident creation error: {str(e)}")
            raise
    
    async def update_incident_status(self, incident_id: int, status: IncidentStatus,
                                   notes: str = None, updated_by: str = "system") -> bool:
        """
        Update incident status and add notes.
        
        Args:
            incident_id: ID of incident to update
            status: New incident status
            notes: Status update notes
            updated_by: Who is updating the incident
            
        Returns:
            Success status
        """
        try:
            incident = self.db.query(Incident).filter(
                and_(
                    Incident.id == incident_id,
                    Incident.user_id == self.user_id
                )
            ).first()
            
            if not incident:
                logger.warning(f"Incident {incident_id} not found for user {self.user_id}")
                return False
            
            # Update status
            incident.status = status
            incident.updated_at = datetime.now()
            
            # Add status history
            status_update = IncidentStatusHistory(
                incident_id=incident_id,
                status=status,
                notes=notes,
                updated_by=updated_by,
                timestamp=datetime.now()
            )
            
            self.db.add(status_update)
            self.db.commit()
            
            logger.info(f"Incident {incident_id} status updated to {status.value} by {updated_by}")
            
            # Auto-resolve timestamp if status is resolved
            if status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                incident.resolved_at = datetime.now()
                self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Incident status update error: {str(e)}")
            return False
    
    async def add_incident_note(self, incident_id: int, note: str, 
                              note_type: str = "general",
                              created_by: str = "system") -> bool:
        """
        Add note to incident.
        
        Args:
            incident_id: ID of incident
            note: Note content
            note_type: Type of note (general, investigation, resolution, etc.)
            created_by: Who created the note
            
        Returns:
            Success status
        """
        try:
            incident = self.db.query(Incident).filter(
                and_(
                    Incident.id == incident_id,
                    Incident.user_id == self.user_id
                )
            ).first()
            
            if not incident:
                return False
            
            # Add note
            incident_note = IncidentNote(
                incident_id=incident_id,
                note_type=note_type,
                note_content=note,
                created_by=created_by,
                created_at=datetime.now()
            )
            
            self.db.add(incident_note)
            self.db.commit()
            
            logger.info(f"Note added to incident {incident_id} by {created_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"Incident note addition error: {str(e)}")
            return False
    
    async def create_postmortem(self, incident_id: int, root_cause: str,
                              timeline: List[Dict[str, Any]],
                              impact_assessment: Dict[str, Any],
                              remediation_steps: List[Dict[str, Any]],
                              prevention_measures: List[str],
                              lessons_learned: str,
                              created_by: str = "system") -> bool:
        """
        Create postmortem analysis for an incident.
        
        Args:
            incident_id: ID of incident
            root_cause: Root cause analysis
            timeline: Event timeline
            impact_assessment: Impact assessment details
            remediation_steps: Steps taken to resolve
            prevention_measures: Measures to prevent recurrence
            lessons_learned: Key lessons learned
            created_by: Who created the postmortem
            
        Returns:
            Success status
        """
        try:
            incident = self.db.query(Incident).filter(
                and_(
                    Incident.id == incident_id,
                    Incident.user_id == self.user_id
                )
            ).first()
            
            if not incident:
                return False
            
            # Create postmortem
            postmortem = IncidentPostmortem(
                incident_id=incident_id,
                root_cause=root_cause,
                timeline=timeline,
                impact_assessment=impact_assessment,
                remediation_steps=remediation_steps,
                prevention_measures=prevention_measures,
                lessons_learned=lessons_learned,
                created_by=created_by,
                created_at=datetime.now()
            )
            
            # Mark incident as postmortem completed
            incident.status = IncidentStatus.CLOSED
            incident.has_postmortem = True
            
            self.db.add(postmortem)
            self.db.commit()
            
            logger.info(f"Postmortem created for incident {incident_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Postmortem creation error: {str(e)}")
            return False
    
    async def get_incidents(self, start_date: datetime = None, end_date: datetime = None,
                          status: IncidentStatus = None, severity: IncidentSeverity = None,
                          category: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents with optional filtering.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            status: Filter by status
            severity: Filter by severity
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of incident records
        """
        try:
            # Build query
            query = self.db.query(Incident).filter(
                Incident.user_id == self.user_id
            )
            
            if start_date:
                query = query.filter(Incident.detected_at >= start_date)
            if end_date:
                query = query.filter(Incident.detected_at <= end_date)
            if status:
                query = query.filter(Incident.status == status)
            if severity:
                query = query.filter(Incident.severity == severity)
            if category:
                query = query.filter(Incident.category == category)
            
            incidents = query.order_by(desc(Incident.detected_at)).limit(limit).all()
            
            result = []
            for incident in incidents:
                # Get related notes
                notes = self.db.query(IncidentNote).filter(
                    IncidentNote.incident_id == incident.id
                ).order_by(IncidentNote.created_at).all()
                
                # Get postmortem if exists
                postmortem = self.db.query(IncidentPostmortem).filter(
                    IncidentPostmortem.incident_id == incident.id
                ).first()
                
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity.value,
                    "status": incident.status.value,
                    "priority": incident.priority.value,
                    "category": incident.category,
                    "event_type": incident.event_type,
                    "detected_at": incident.detected_at,
                    "resolved_at": incident.resolved_at,
                    "created_at": incident.created_at,
                    "has_postmortem": incident.has_postmortem,
                    "incident_data": incident.incident_data or {},
                    "notes_count": len(notes),
                    "latest_note": notes[-1].note_content[:200] + "..." if notes else None
                }
                
                if postmortem:
                    incident_data["postmortem"] = {
                        "root_cause": postmortem.root_cause,
                        "lessons_learned": postmortem.lessons_learned,
                        "created_at": postmortem.created_at
                    }
                
                result.append(incident_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Incident retrieval error: {str(e)}")
            return []
    
    async def get_incident_metrics(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """
        Get incident metrics for a date range.
        
        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            
        Returns:
            Incident metrics and statistics
        """
        try:
            # Get all incidents in range
            incidents = self.db.query(Incident).filter(
                and_(
                    Incident.user_id == self.user_id,
                    Incident.detected_at >= start_date,
                    Incident.detected_at <= end_date
                )
            ).all()
            
            # Calculate metrics
            total_incidents = len(incidents)
            
            by_severity = {}
            by_status = {}
            by_category = {}
            by_priority = {}
            
            resolution_times = []
            open_incidents = 0
            
            for incident in incidents:
                # Count by severity
                severity = incident.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # Count by status
                status = incident.status.value
                by_status[status] = by_status.get(status, 0) + 1
                
                # Count by category
                category = incident.category
                by_category[category] = by_category.get(category, 0) + 1
                
                # Count by priority
                priority = incident.priority.value
                by_priority[priority] = by_priority.get(priority, 0) + 1
                
                # Track resolution time
                if incident.resolved_at:
                    resolution_time = (incident.resolved_at - incident.detected_at).total_seconds() / 3600  # hours
                    resolution_times.append(resolution_time)
                else:
                    open_incidents += 1
            
            # Calculate statistics
            metrics = {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": (end_date - start_date).days + 1
                },
                "summary": {
                    "total_incidents": total_incidents,
                    "open_incidents": open_incidents,
                    "resolved_incidents": total_incidents - open_incidents,
                    "incidents_with_postmortems": sum(1 for i in incidents if i.has_postmortem),
                    "avg_resolution_time_hours": sum(resolution_times) / len(resolution_times) if resolution_times else 0,
                    "median_resolution_time_hours": self._calculate_median(resolution_times) if resolution_times else 0
                },
                "distribution": {
                    "by_severity": by_severity,
                    "by_status": by_status,
                    "by_category": by_category,
                    "by_priority": by_priority
                },
                "trends": await self._calculate_incident_trends(incidents)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Incident metrics calculation error: {str(e)}")
            return {"error": str(e)}
    
    async def generate_incident_report(self, start_date: datetime, 
                                     end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive incident report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Detailed incident report
        """
        try:
            incidents = await self.get_incidents(start_date, end_date)
            metrics = await self.get_incident_metrics(start_date, end_date)
            
            report = {
                "report_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "generated_at": datetime.now()
                },
                "executive_summary": {
                    "total_incidents": metrics["summary"]["total_incidents"],
                    "critical_incidents": metrics["distribution"]["by_severity"].get("critical", 0),
                    "open_incidents": metrics["summary"]["open_incidents"],
                    "avg_resolution_time": f"{metrics['summary']['avg_resolution_time_hours']:.1f} hours"
                },
                "metrics": metrics,
                "incidents": incidents,
                "top_categories": sorted(
                    metrics["distribution"]["by_category"].items(),
                    key=lambda x: x[1], reverse=True
                )[:5],
                "recommendations": await self._generate_incident_recommendations(incidents, metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Incident report generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _determine_priority(self, severity: IncidentSeverity) -> IncidentPriority:
        """Determine incident priority based on severity."""
        priority_mapping = {
            IncidentSeverity.CRITICAL: IncidentPriority.P1,
            IncidentSeverity.HIGH: IncidentPriority.P1,
            IncidentSeverity.MEDIUM: IncidentPriority.P2,
            IncidentSeverity.LOW: IncidentPriority.P3
        }
        
        return priority_mapping.get(severity, IncidentPriority.P4)
    
    async def _log_incident_event(self, incident_id: int, event_type: str, 
                                event_data: Dict[str, Any]) -> None:
        """Log incident-related events."""
        try:
            # This would integrate with the audit logger
            logger.info(f"Incident event: {event_type} for incident {incident_id}")
            
        except Exception as e:
            logger.error(f"Incident event logging error: {str(e)}")
    
    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    async def _calculate_incident_trends(self, incidents: List) -> Dict[str, Any]:
        """Calculate incident trends over time."""
        try:
            daily_counts = {}
            
            for incident in incidents:
                date = incident.detected_at.date()
                daily_counts[date] = daily_counts.get(date, 0) + 1
            
            return {
                "daily_distribution": daily_counts,
                "peak_day": max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None,
                "trend_direction": self._calculate_trend_direction(daily_counts)
            }
            
        except Exception as e:
            logger.error(f"Incident trends calculation error: {str(e)}")
            return {}
    
    def _calculate_trend_direction(self, daily_counts: Dict) -> str:
        """Calculate trend direction for incident frequency."""
        try:
            if not daily_counts:
                return "stable"
            
            dates = sorted(daily_counts.keys())
            if len(dates) < 2:
                return "stable"
            
            # Simple trend calculation
            first_half = dates[:len(dates)//2]
            second_half = dates[len(dates)//2:]
            
            first_half_avg = sum(daily_counts[d] for d in first_half) / len(first_half)
            second_half_avg = sum(daily_counts[d] for d in second_half) / len(second_half)
            
            if second_half_avg > first_half_avg * 1.1:
                return "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "stable"
    
    async def _generate_incident_recommendations(self, incidents: List[Dict[str, Any]], 
                                               metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on incident analysis."""
        recommendations = []
        
        try:
            # High incident count recommendation
            if metrics["summary"]["total_incidents"] > 50:
                recommendations.append("High incident frequency detected - review system stability and processes")
            
            # Critical incidents recommendation
            if metrics["distribution"]["by_severity"].get("critical", 0) > 0:
                recommendations.append("Critical incidents occurred - immediate attention required for system reliability")
            
            # Long resolution times recommendation
            if metrics["summary"]["avg_resolution_time_hours"] > 24:
                recommendations.append("Long average resolution times - improve incident response procedures")
            
            # Category-specific recommendations
            top_category = max(metrics["distribution"]["by_category"].items(), 
                             key=lambda x: x[1]) if metrics["distribution"]["by_category"] else None
            
            if top_category and top_category[1] > metrics["summary"]["total_incidents"] * 0.3:
                category_name = top_category[0]
                recommendations.append(f"High frequency of {category_name} incidents - focus improvement efforts in this area")
            
            # Open incidents recommendation
            if metrics["summary"]["open_incidents"] > 5:
                recommendations.append("Multiple open incidents - prioritize resolution to prevent escalation")
            
            # Postmortem completion recommendation
            completion_rate = (metrics["summary"]["incidents_with_postmortems"] / 
                             max(metrics["summary"]["total_incidents"], 1)) * 100
            
            if completion_rate < 50:
                recommendations.append("Low postmortem completion rate - improve incident analysis processes")
                
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
        
        return recommendations
    
    def close(self):
        """Cleanup resources."""
        self.db.close()


# Incident models (would normally be in database/models/, but included here for completeness)
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text, Float
from sqlalchemy.sql import func
from config.database import Base


class Incident(Base):
    """Incident tracking table."""
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Incident details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    priority = Column(String(20), nullable=False)
    
    # Additional data
    event_type = Column(String(50), nullable=True)
    incident_data = Column(JSON, default=dict)
    has_postmortem = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Incident(id={self.id}, title='{self.title}', severity='{self.severity}', status='{self.status}')>"


class IncidentNote(Base):
    """Incident notes and comments."""
    __tablename__ = "incident_notes"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, nullable=False, index=True)
    
    # Note details
    note_type = Column(String(50), default="general", nullable=False)
    note_content = Column(Text, nullable=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    
    def __repr__(self):
        return f"<IncidentNote(id={self.id}, incident_id={self.incident_id}, type='{self.note_type}')>"


class IncidentStatusHistory(Base):
    """Incident status change history."""
    __tablename__ = "incident_status_history"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, nullable=False, index=True)
    
    # Status change details
    status = Column(String(20), nullable=False)
    notes = Column(Text, nullable=True)
    updated_by = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    def __repr__(self):
        return f"<IncidentStatusHistory(id={self.id}, incident_id={self.incident_id}, status='{self.status}')>"


class IncidentPostmortem(Base):
    """Incident postmortem analysis."""
    __tablename__ = "incident_postmortems"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, nullable=False, index=True)
    
    # Postmortem content
    root_cause = Column(Text, nullable=False)
    timeline = Column(JSON, default=list)
    impact_assessment = Column(JSON, default=dict)
    remediation_steps = Column(JSON, default=list)
    prevention_measures = Column(JSON, default=list)
    lessons_learned = Column(Text, nullable=False)
    
    # Metadata
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    
    def __repr__(self):
        return f"<IncidentPostmortem(id={self.id}, incident_id={self.incident_id})>"
