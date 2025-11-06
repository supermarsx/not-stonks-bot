"""
Risk Alerting and Escalation System

This module implements comprehensive risk alerting and escalation procedures,
including real-time risk alerts, escalation procedures, and multi-channel notifications.

Features:
- Real-time risk alerts
- Multi-level escalation procedures
- Multi-channel notifications (email, SMS, webhooks, dashboard)
- Alert suppression and deduplication
- Alert routing and assignment
- Critical alert handling
- Alert analytics and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import json
import smtplib
import aiohttp
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

from trading_orchestrator.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Alert categories."""
    RISK_LIMIT = "risk_limit"
    POSITION = "position"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    COMPLIANCE = "compliance"
    SYSTEM = "system"
    TRADING = "trading"
    FINANCIAL = "financial"


class AlertStatus(Enum):
    """Alert status states."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


class EscalationLevel(Enum):
    """Escalation levels."""
    LEVEL_1 = 1  # Team lead
    LEVEL_2 = 2  # Department head
    LEVEL_3 = 3  # Senior management
    LEVEL_4 = 4  # Executive team
    LEVEL_5 = 5  # Board/C-level


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRecipient:
    """Alert recipient configuration."""
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user: Optional[str] = None
    teams_user: Optional[str] = None
    active: bool = True
    timezone: str = "UTC"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    conditions: Dict[str, Any]
    recipients: List[str]  # User IDs
    channels: List[NotificationChannel]
    suppression_time_minutes: int = 60  # Prevent duplicate alerts
    escalation_delay_minutes: int = 15  # Auto-escalate if not acknowledged
    enabled: bool = True


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    rule_id: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    source: str
    account_id: Optional[str] = None
    user_id: Optional[str] = None
    asset_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_level: Optional[EscalationLevel] = None


class Alert(Base):
    """Database model for risk alerts."""
    __tablename__ = "risk_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), nullable=False, index=True, unique=True)
    rule_id = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)
    account_id = Column(String(100), nullable=True)
    user_id = Column(String(100), nullable=True)
    asset_symbol = Column(String(20), nullable=True)
    metadata = Column(Text, nullable=True)  # JSON data
    status = Column(String(20), default="new", nullable=False)
    acknowledged_by = Column(String(100), nullable=True)
    resolved_by = Column(String(100), nullable=True)
    escalated = Column(Boolean, default=False, nullable=False)
    escalation_level = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, onupdate=func.now())


class AlertNotification(Base):
    """Database model for alert notifications."""
    __tablename__ = "alert_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), nullable=False, index=True)
    channel = Column(String(20), nullable=False)
    recipient = Column(String(100), nullable=False)
    recipient_contact = Column(String(200), nullable=True)
    status = Column(String(20), nullable=False)
    sent_at = Column(DateTime, server_default=func.now())
    delivered_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)


class AlertEscalation(Base):
    """Database model for alert escalations."""
    __tablename__ = "alert_escalations"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), nullable=False, index=True)
    from_level = Column(Integer, nullable=False)
    to_level = Column(Integer, nullable=False)
    triggered_by = Column(String(100), nullable=False)
    reason = Column(String(200), nullable=False)
    escalated_at = Column(DateTime, server_default=func.now())
    resolved = Column(Boolean, default=False, nullable=False)


class RiskAlertingManager:
    """Manager for risk alerts and escalation."""
    
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.db_manager = db_manager
        self.config = config
        self._alert_rules: Dict[str, AlertRule] = {}
        self._recipients: Dict[str, AlertRecipient] = {}
        self._active_alerts: Dict[str, RiskAlert] = {}
        self._suppression_cache: Dict[str, datetime] = {}
        self._notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Initialize notification handlers
        self._setup_notification_handlers()
    
    def _setup_notification_handlers(self) -> None:
        """Setup notification delivery handlers."""
        self._notification_handlers = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.SMS: self._send_sms,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.DASHBOARD: self._send_dashboard,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.TEAMS: self._send_teams,
            NotificationChannel.PAGERDUTY: self._send_pagerduty
        }
    
    async def initialize(self) -> None:
        """Initialize the alerting manager."""
        logger.info("Initializing risk alerting manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Load alert rules
            await self._load_alert_rules()
            
            # Load recipients
            await self._load_recipients()
            
            # Start alert monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("Risk alerting manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize alerting manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for alerts."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _load_alert_rules(self) -> None:
        """Load alert rules from configuration."""
        # This would load from a configuration file or database
        await self._initialize_default_rules()
    
    async def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        # Position limit violation rule
        position_rule = AlertRule(
            rule_id="POSITION_LIMIT_VIOLATION",
            name="Position Limit Violation",
            category=AlertCategory.POSITION,
            severity=AlertSeverity.HIGH,
            conditions={"violation_type": "position_limit"},
            recipients=["risk_team_lead"],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        )
        self._alert_rules[position_rule.rule_id] = position_rule
        
        # Critical drawdown rule
        drawdown_rule = AlertRule(
            rule_id="CRITICAL_DRAWDOWN",
            name="Critical Portfolio Drawdown",
            category=AlertCategory.RISK_LIMIT,
            severity=AlertSeverity.CRITICAL,
            conditions={"drawdown_threshold": 0.10},  # 10%
            recipients=["risk_team_lead", "portfolio_manager", "cio"],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK],
            escalation_delay_minutes=5
        )
        self._alert_rules[drawdown_rule.rule_id] = drawdown_rule
        
        # System error rule
        system_rule = AlertRule(
            rule_id="SYSTEM_ERROR",
            name="Critical System Error",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.EMERGENCY,
            conditions={"error_level": "critical"},
            recipients=["tech_team", "cto"],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PAGERDUTY],
            suppression_time_minutes=30
        )
        self._alert_rules[system_rule.rule_id] = system_rule
        
        logger.info("Loaded default alert rules")
    
    async def _load_recipients(self) -> None:
        """Load alert recipients."""
        # This would load from user database/config
        await self._initialize_default_recipients()
    
    async def _initialize_default_recipients(self) -> None:
        """Initialize default recipients."""
        recipients = [
            AlertRecipient(
                user_id="risk_team_lead",
                name="Risk Team Lead",
                email="risk-lead@company.com",
                phone="+1234567890",
                slack_user="@risk_lead"
            ),
            AlertRecipient(
                user_id="portfolio_manager",
                name="Portfolio Manager",
                email="pm@company.com",
                phone="+1234567891",
                slack_user="@pm"
            ),
            AlertRecipient(
                user_id="cio",
                name="Chief Investment Officer",
                email="cio@company.com",
                phone="+1234567892"
            ),
            AlertRecipient(
                user_id="tech_team",
                name="Technology Team",
                email="tech@company.com",
                slack_user="#tech-alerts"
            ),
            AlertRecipient(
                user_id="cto",
                name="Chief Technology Officer",
                email="cto@company.com",
                phone="+1234567893"
            )
        ]
        
        for recipient in recipients:
            self._recipients[recipient.user_id] = recipient
        
        logger.info("Loaded default alert recipients")
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Start alert escalation task
        asyncio.create_task(self._alert_escalation_monitor())
        
        # Start notification retry task
        asyncio.create_task(self._notification_retry_monitor())
        
        logger.info("Started alert monitoring tasks")
    
    async def create_alert(self, rule_id: str, title: str, message: str, 
                          source: str, metadata: Dict[str, Any] = None,
                          category: Optional[AlertCategory] = None,
                          severity: Optional[AlertSeverity] = None,
                          account_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          asset_symbol: Optional[str] = None) -> str:
        """Create a new risk alert."""
        try:
            # Get alert rule
            rule = self._alert_rules.get(rule_id)
            if not rule or not rule.enabled:
                logger.warning(f"Unknown or disabled alert rule: {rule_id}")
                return ""
            
            # Generate unique alert ID
            alert_id = f"{rule_id}_{int(datetime.utcnow().timestamp() * 1000)}"
            
            # Check suppression
            if await self._is_alert_suppressed(rule_id, metadata):
                logger.info(f"Alert {alert_id} suppressed due to recent duplicate")
                return ""
            
            # Create alert
            alert = RiskAlert(
                alert_id=alert_id,
                rule_id=rule_id,
                category=category or rule.category,
                severity=severity or rule.severity,
                title=title,
                message=message,
                source=source,
                account_id=account_id,
                user_id=user_id,
                asset_symbol=asset_symbol,
                metadata=metadata or {}
            )
            
            # Save to database
            await self._save_alert_to_db(alert)
            
            # Add to active alerts
            self._active_alerts[alert_id] = alert
            
            # Send notifications
            await self._send_alert_notifications(alert, rule)
            
            # Schedule auto-escalation
            if rule.escalation_delay_minutes > 0:
                await self._schedule_auto_escalation(alert_id, rule)
            
            logger.warning(f"Risk alert created: {alert_id} - {title}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert {alert_id}: {e}")
            return ""
    
    async def _is_alert_suppressed(self, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed (deduplication)."""
        # Create suppression key from rule and relevant metadata
        relevant_keys = sorted([k for k in metadata.keys() if k in ['account_id', 'asset_symbol']])
        suppression_key = f"{rule_id}:{':'.join(str(metadata.get(k, '')) for k in relevant_keys)}"
        
        if suppression_key in self._suppression_cache:
            last_alert_time = self._suppression_cache[suppression_key]
            rule = self._alert_rules.get(rule_id)
            if rule and (datetime.utcnow() - last_alert_time).total_seconds() < rule.suppression_time_minutes * 60:
                return True
        
        # Update suppression cache
        self._suppression_cache[suppression_key] = datetime.utcnow()
        
        return False
    
    async def _save_alert_to_db(self, alert: RiskAlert) -> None:
        """Save alert to database."""
        async with self.db_manager.get_session() as session:
            alert_record = Alert(
                alert_id=alert.alert_id,
                rule_id=alert.rule_id,
                category=alert.category.value,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                source=alert.source,
                account_id=alert.account_id,
                user_id=alert.user_id,
                asset_symbol=alert.asset_symbol,
                metadata=json.dumps(alert.metadata),
                status="new"
            )
            session.add(alert_record)
            await session.commit()
    
    async def _send_alert_notifications(self, alert: RiskAlert, rule: AlertRule) -> None:
        """Send alert notifications via configured channels."""
        for channel in rule.channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                for recipient_id in rule.recipients:
                    await handler(alert, recipient_id, channel)
    
    async def _send_email(self, alert: RiskAlert, recipient_id: str, 
                        channel: NotificationChannel) -> None:
        """Send email notification."""
        try:
            recipient = self._recipients.get(recipient_id)
            if not recipient or not recipient.email:
                logger.warning(f"No email for recipient {recipient_id}")
                return
            
            # In production, this would use proper email service
            logger.info(f"Would send email to {recipient.email}: {alert.title}")
            
            # Record notification attempt
            await self._record_notification(alert.alert_id, channel.value, 
                                          recipient_id, recipient.email, "sent")
            
        except Exception as e:
            logger.error(f"Error sending email to {recipient_id}: {e}")
            await self._record_notification(alert.alert_id, channel.value, 
                                          recipient_id, recipient.email if recipient else "", 
                                          "failed", str(e))
    
    async def _send_sms(self, alert: RiskAlert, recipient_id: str,
                       channel: NotificationChannel) -> None:
        """Send SMS notification."""
        try:
            recipient = self._recipients.get(recipient_id)
            if not recipient or not recipient.phone:
                logger.warning(f"No phone for recipient {recipient_id}")
                return
            
            # In production, this would use SMS service like Twilio
            logger.info(f"Would send SMS to {recipient.phone}: {alert.title}")
            
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, recipient.phone, "sent")
            
        except Exception as e:
            logger.error(f"Error sending SMS to {recipient_id}: {e}")
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, recipient.phone if recipient else "",
                                          "failed", str(e))
    
    async def _send_webhook(self, alert: RiskAlert, recipient_id: str,
                          channel: NotificationChannel) -> None:
        """Send webhook notification."""
        try:
            webhook_url = self.config.get('webhook_urls', {}).get(recipient_id)
            if not webhook_url:
                logger.warning(f"No webhook URL for {recipient_id}")
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "category": alert.category.value,
                "timestamp": alert.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent to {recipient_id}")
                        await self._record_notification(alert.alert_id, channel.value,
                                                      recipient_id, webhook_url, "sent")
                    else:
                        raise Exception(f"HTTP {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending webhook to {recipient_id}: {e}")
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, webhook_url or "", "failed", str(e))
    
    async def _send_dashboard(self, alert: RiskAlert, recipient_id: str,
                            channel: NotificationChannel) -> None:
        """Send dashboard notification."""
        # Dashboard notifications are typically real-time via WebSocket
        logger.info(f"Dashboard notification: {alert.title}")
        await self._record_notification(alert.alert_id, channel.value,
                                      recipient_id, "dashboard", "sent")
    
    async def _send_slack(self, alert: RiskAlert, recipient_id: str,
                        channel: NotificationChannel) -> None:
        """Send Slack notification."""
        try:
            recipient = self._recipients.get(recipient_id)
            # In production, this would use Slack API
            logger.info(f"Would send Slack message: {alert.title}")
            
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, "slack", "sent")
            
        except Exception as e:
            logger.error(f"Error sending Slack message to {recipient_id}: {e}")
    
    async def _send_teams(self, alert: RiskAlert, recipient_id: str,
                        channel: NotificationChannel) -> None:
        """Send Teams notification."""
        try:
            recipient = self._recipients.get(recipient_id)
            # In production, this would use Teams API
            logger.info(f"Would send Teams message: {alert.title}")
            
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, "teams", "sent")
            
        except Exception as e:
            logger.error(f"Error sending Teams message to {recipient_id}: {e}")
    
    async def _send_pagerduty(self, alert: RiskAlert, recipient_id: str,
                            channel: NotificationChannel) -> None:
        """Send PagerDuty notification."""
        try:
            # In production, this would use PagerDuty API
            logger.info(f"Would trigger PagerDuty: {alert.title}")
            
            await self._record_notification(alert.alert_id, channel.value,
                                          recipient_id, "pagerduty", "sent")
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification to {recipient_id}: {e}")
    
    async def _record_notification(self, alert_id: str, channel: str, recipient: str,
                                 contact_info: str, status: str, error: str = None) -> None:
        """Record notification attempt in database."""
        async with self.db_manager.get_session() as session:
            notification = AlertNotification(
                alert_id=alert_id,
                channel=channel,
                recipient=recipient,
                recipient_contact=contact_info,
                status=status,
                error_message=error
            )
            session.add(notification)
            await session.commit()
    
    async def _schedule_auto_escalation(self, alert_id: str, rule: AlertRule) -> None:
        """Schedule automatic escalation for unacknowledged alerts."""
        if rule.escalation_delay_minutes > 0:
            await asyncio.sleep(rule.escalation_delay_minutes * 60)
            
            # Check if alert is still unacknowledged
            alert = self._active_alerts.get(alert_id)
            if alert and not alert.acknowledged_at:
                await self.escalate_alert(alert_id, EscalationLevel.LEVEL_1, "Auto-escalation timeout")
    
    async def _alert_escalation_monitor(self) -> None:
        """Monitor alerts for auto-escalation."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                for alert_id, alert in list(self._active_alerts.items()):
                    if not alert.acknowledged_at and not alert.escalated:
                        # Check if escalation timeout reached
                        time_since_creation = (current_time - alert.created_at).total_seconds()
                        rule = self._alert_rules.get(alert.rule_id)
                        
                        if rule and time_since_creation > rule.escalation_delay_minutes * 60:
                            await self.escalate_alert(alert_id, EscalationLevel.LEVEL_1, "Timeout")
                            
            except Exception as e:
                logger.error(f"Error in escalation monitor: {e}")
    
    async def _notification_retry_monitor(self) -> None:
        """Monitor failed notifications for retry."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                async with self.db_manager.get_session() as session:
                    failed_notifications = await session.execute(
                        AlertNotification.__table__.select()
                        .where(
                            AlertNotification.status == "failed",
                            AlertNotification.retry_count < AlertNotification.max_retries
                        )
                    )
                    
                    for notification in failed_notifications:
                        await self._retry_notification(notification)
                        
            except Exception as e:
                logger.error(f"Error in notification retry monitor: {e}")
    
    async def _retry_notification(self, notification: AlertNotification) -> None:
        """Retry a failed notification."""
        # This would implement retry logic
        logger.info(f"Retrying notification {notification.id}")
    
    async def escalate_alert(self, alert_id: str, level: EscalationLevel, 
                           reason: str, escalated_by: str = "system") -> bool:
        """Escalate an alert to a higher level."""
        try:
            alert = self._active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert not found for escalation: {alert_id}")
                return False
            
            if alert.escalated:
                logger.warning(f"Alert already escalated: {alert_id}")
                return False
            
            # Update alert
            alert.escalated = True
            alert.escalation_level = level
            
            # Update database
            async with self.db_manager.get_session() as session:
                await session.execute(
                    Alert.__table__.update()
                    .where(Alert.alert_id == alert_id)
                    .values(escalated=True, escalation_level=level.value, status="escalated")
                )
                
                # Record escalation
                escalation = AlertEscalation(
                    alert_id=alert_id,
                    from_level=alert.escalation_level.value if alert.escalation_level else 0,
                    to_level=level.value,
                    triggered_by=escalated_by,
                    reason=reason
                )
                session.add(escalation)
                await session.commit()
            
            # Send escalation notifications to higher-level recipients
            await self._send_escalation_notifications(alert, level)
            
            logger.warning(f"Alert escalated: {alert_id} to level {level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert_id}: {e}")
            return False
    
    async def _send_escalation_notifications(self, alert: RiskAlert, level: EscalationLevel) -> None:
        """Send escalation notifications based on level."""
        # Define escalation recipients by level
        level_recipients = {
            EscalationLevel.LEVEL_1: ["department_head"],
            EscalationLevel.LEVEL_2: ["senior_management"],
            EscalationLevel.LEVEL_3: ["executive_team"],
            EscalationLevel.LEVEL_4: ["c_level"],
            EscalationLevel.LEVEL_5: ["board"]
        }
        
        recipients = level_recipients.get(level, [])
        
        for recipient_id in recipients:
            # Send to all channels for critical escalation
            for channel in [NotificationChannel.EMAIL, NotificationChannel.SMS]:
                handler = self._notification_handlers.get(channel)
                if handler:
                    await handler(alert, recipient_id, channel)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, 
                              notes: str = "") -> bool:
        """Acknowledge an alert."""
        try:
            alert = self._active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
            
            alert.acknowledged_at = datetime.utcnow()
            
            # Update database
            async with self.db_manager.get_session() as session:
                await session.execute(
                    Alert.__table__.update()
                    .where(Alert.alert_id == alert_id)
                    .values(
                        status="acknowledged",
                        acknowledged_by=acknowledged_by,
                        acknowledged_at=datetime.utcnow()
                    )
                )
                await session.commit()
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, 
                          notes: str = "") -> bool:
        """Resolve an alert."""
        try:
            alert = self._active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False
            
            alert.resolved_at = datetime.utcnow()
            
            # Remove from active alerts
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]
            
            # Update database
            async with self.db_manager.get_session() as session:
                await session.execute(
                    Alert.__table__.update()
                    .where(Alert.alert_id == alert_id)
                    .values(
                        status="resolved",
                        resolved_by=resolved_by,
                        resolved_at=datetime.utcnow()
                    )
                )
                await session.commit()
            
            logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def get_alert_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            async with self.db_manager.get_session() as session:
                alerts = await session.execute(
                    Alert.__table__.select()
                    .where(Alert.created_at >= cutoff_time)
                )
                
                alerts_list = alerts.fetchall()
            
            # Group by severity and status
            by_severity = {}
            by_status = {}
            total_count = len(alerts_list)
            
            for alert in alerts_list:
                severity = alert.severity
                status = alert.status
                
                by_severity[severity] = by_severity.get(severity, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1
            
            return {
                "total_alerts": total_count,
                "by_severity": by_severity,
                "by_status": by_status,
                "time_period_hours": hours_back,
                "active_alerts": len(self._active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {}
    
    async def get_alerting_status(self) -> Dict[str, Any]:
        """Get overall alerting system status."""
        return {
            "active_rules": len([r for r in self._alert_rules.values() if r.enabled]),
            "active_recipients": len([r for r in self._recipients.values() if r.active]),
            "active_alerts": len(self._active_alerts),
            "notification_channels": len(NotificationChannel),
            "last_update": datetime.utcnow().isoformat()
        }


# Factory function for creating alerting manager
async def create_alerting_manager(db_manager: DatabaseManager, 
                                config: Dict[str, Any]) -> RiskAlertingManager:
    """Create a configured alerting manager."""
    manager = RiskAlertingManager(db_manager, config)
    await manager.initialize()
    
    return manager


# Predefined alert rules for common scenarios
COMMON_ALERT_RULES = {
    "position_limit_violation": AlertRule(
        rule_id="POSITION_LIMIT_VIOLATION",
        name="Position Limit Violation",
        category=AlertCategory.POSITION,
        severity=AlertSeverity.HIGH,
        conditions={"violation_type": "position_limit"},
        recipients=["risk_team_lead"],
        channels=[NotificationChannel.EMAIL]
    ),
    
    "drawdown_alert": AlertRule(
        rule_id="DRAWDOWN_ALERT",
        name="Portfolio Drawdown Alert",
        category=AlertCategory.RISK_LIMIT,
        severity=AlertSeverity.MEDIUM,
        conditions={"drawdown_threshold": 0.05},  # 5%
        recipients=["portfolio_manager"],
        channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD]
    ),
    
    "system_error": AlertRule(
        rule_id="SYSTEM_ERROR",
        name="Critical System Error",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.CRITICAL,
        conditions={"error_level": "critical"},
        recipients=["tech_team", "cto"],
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PAGERDUTY]
    )
}