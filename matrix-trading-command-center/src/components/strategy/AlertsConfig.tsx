import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  Bell, 
  Shield, 
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Info,
  CheckCircle,
  Mail,
  Phone,
  MessageSquare,
  Settings,
  Save,
  Plus,
  Trash2,
  Copy,
  Play,
  Pause,
  RotateCcw,
  Download,
  Upload,
  Clock,
  DollarSign,
  BarChart3,
  Target,
  Activity,
  Zap,
  Volume2,
  VolumeX,
  Eye,
  Filter,
  Calendar,
  Globe,
  Smartphone,
  Monitor,
  Wifi
} from 'lucide-react';

interface AlertRule {
  id: string;
  name: string;
  description: string;
  type: 'performance' | 'risk' | 'position' | 'market' | 'system' | 'custom';
  condition: {
    metric: string;
    operator: 'greater_than' | 'less_than' | 'equals' | 'not_equals' | 'crosses_above' | 'crosses_below';
    value: number | string;
    timeframe?: string;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  channels: NotificationChannel[];
  cooldown: number; // minutes
  lastTriggered?: Date;
  triggers: number;
  cooldownActive: boolean;
  customScript?: string;
}

interface NotificationChannel {
  id: string;
  name: string;
  type: 'email' | 'sms' | 'slack' | 'discord' | 'webhook' | 'push' | 'telegram';
  enabled: boolean;
  config: {
    email?: string;
    phone?: string;
    webhookUrl?: string;
    webhookHeaders?: Record<string, string>;
    channel?: string;
    botToken?: string;
    chatId?: string;
  };
}

interface AlertPreferences {
  global: {
    enableAll: boolean;
    muteAll: boolean;
    defaultCooldown: number;
    quietHours: {
      enabled: boolean;
      start: string;
      end: string;
      timezone: string;
    };
  };
  escalation: {
    enabled: boolean;
    levels: EscalationLevel[];
    timeoutMinutes: number;
  };
  grouping: {
    enabled: boolean;
    timeWindow: number; // minutes
    maxAlerts: number;
  };
  analytics: {
    trackTriggers: boolean;
    trackChannels: boolean;
    retentionDays: number;
  };
}

interface EscalationLevel {
  level: number;
  condition: string;
  channels: string[];
  delay: number; // minutes
}

const AlertsConfig: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'rules' | 'channels' | 'preferences' | 'history'>('rules');
  const [alertRules, setAlertRules] = useState<AlertRule[]>([
    {
      id: '1',
      name: 'High Drawdown Alert',
      description: 'Triggered when portfolio drawdown exceeds threshold',
      type: 'risk',
      condition: {
        metric: 'max_drawdown',
        operator: 'greater_than',
        value: 0.10
      },
      severity: 'high',
      enabled: true,
      channels: ['1', '2'],
      cooldown: 30,
      triggers: 0,
      cooldownActive: false
    },
    {
      id: '2',
      name: 'Positive P&L Threshold',
      description: 'Alert when daily profit exceeds target',
      type: 'performance',
      condition: {
        metric: 'daily_pnl',
        operator: 'greater_than',
        value: 1000
      },
      severity: 'medium',
      enabled: true,
      channels: ['1'],
      cooldown: 60,
      triggers: 0,
      cooldownActive: false
    },
    {
      id: '3',
      name: 'Position Size Warning',
      description: 'Alert when position exceeds allocation limit',
      type: 'position',
      condition: {
        metric: 'position_size',
        operator: 'greater_than',
        value: 0.20
      },
      severity: 'medium',
      enabled: false,
      channels: ['2'],
      cooldown: 15,
      triggers: 0,
      cooldownActive: false
    }
  ]);

  const [notificationChannels, setNotificationChannels] = useState<NotificationChannel[]>([
    {
      id: '1',
      name: 'Email Alerts',
      type: 'email',
      enabled: true,
      config: {
        email: 'trader@example.com'
      }
    },
    {
      id: '2',
      name: 'SMS Alerts',
      type: 'sms',
      enabled: true,
      config: {
        phone: '+1234567890'
      }
    },
    {
      id: '3',
      name: 'Slack Integration',
      type: 'slack',
      enabled: false,
      config: {
        webhookUrl: 'https://hooks.slack.com/services/...',
        channel: '#trading-alerts'
      }
    },
    {
      id: '4',
      name: 'Discord Bot',
      type: 'discord',
      enabled: false,
      config: {
        webhookUrl: 'https://discord.com/api/webhooks/...',
        channel: 'trading-alerts'
      }
    },
    {
      id: '5',
      name: 'Telegram Bot',
      type: 'telegram',
      enabled: false,
      config: {
        botToken: '1234567890:ABCdef...',
        chatId: '@your_channel'
      }
    }
  ]);

  const [preferences, setPreferences] = useState<AlertPreferences>({
    global: {
      enableAll: true,
      muteAll: false,
      defaultCooldown: 30,
      quietHours: {
        enabled: true,
        start: '22:00',
        end: '08:00',
        timezone: 'UTC'
      }
    },
    escalation: {
      enabled: true,
      levels: [
        {
          level: 1,
          condition: 'High Severity Alert',
          channels: ['1'],
          delay: 0
        },
        {
          level: 2,
          condition: 'Critical Alert (5 min)',
          channels: ['1', '2'],
          delay: 5
        },
        {
          level: 3,
          condition: 'Critical Alert (15 min)',
          channels: ['1', '2', '3'],
          delay: 15
        }
      ],
      timeoutMinutes: 60
    },
    grouping: {
      enabled: true,
      timeWindow: 5,
      maxAlerts: 10
    },
    analytics: {
      trackTriggers: true,
      trackChannels: true,
      retentionDays: 90
    }
  });

  const [isTestingAlerts, setIsTestingAlerts] = useState(false);
  const [testAlert, setTestAlert] = useState<AlertRule | null>(null);

  const tabs = [
    { id: 'rules', label: 'Alert Rules', icon: Bell },
    { id: 'channels', label: 'Channels', icon: MessageSquare },
    { id: 'preferences', label: 'Preferences', icon: Settings },
    { id: 'history', label: 'History', icon: Clock }
  ];

  const updateAlertRule = (id: string, updates: Partial<AlertRule>) => {
    setAlertRules(prev => prev.map(rule => 
      rule.id === id ? { ...rule, ...updates } : rule
    ));
  };

  const updateNotificationChannel = (id: string, updates: Partial<NotificationChannel>) => {
    setNotificationChannels(prev => prev.map(channel => 
      channel.id === id ? { ...channel, ...updates } : channel
    ));
  };

  const updatePreferences = (section: keyof AlertPreferences, updates: any) => {
    setPreferences(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        ...updates
      }
    }));
  };

  const handleTestAlert = (rule: AlertRule) => {
    setTestAlert(rule);
    setIsTestingAlerts(true);
    
    setTimeout(() => {
      setIsTestingAlerts(false);
      setTestAlert(null);
    }, 3000);
  };

  const renderAlertRules = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">Alert Rules</h3>
        <MatrixButton
          onClick={() => {
            const newRule: AlertRule = {
              id: Date.now().toString(),
              name: 'New Alert Rule',
              description: 'Custom alert rule',
              type: 'custom',
              condition: {
                metric: 'custom_metric',
                operator: 'greater_than',
                value: 0
              },
              severity: 'medium',
              enabled: false,
              channels: [],
              cooldown: preferences.global.defaultCooldown,
              triggers: 0,
              cooldownActive: false
            };
            setAlertRules(prev => [...prev, newRule]);
          }}
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Rule
        </MatrixButton>
      </div>

      <div className="space-y-4">
        {alertRules.map((rule) => (
          <MatrixCard key={rule.id} className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className={`
                  w-3 h-3 rounded-full
                  ${rule.enabled ? 'bg-green-400' : 'bg-gray-600'}
                `} />
                <div>
                  <h4 className="font-medium text-white">{rule.name}</h4>
                  <p className="text-sm text-gray-400">{rule.description}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <select
                  value={rule.severity}
                  onChange={(e) => updateAlertRule(rule.id, { severity: e.target.value as any })}
                  className="text-sm px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
                <MatrixButton
                  size="sm"
                  onClick={() => handleTestAlert(rule)}
                  disabled={isTestingAlerts}
                >
                  {isTestingAlerts && testAlert?.id === rule.id ? (
                    <Activity className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                </MatrixButton>
                <MatrixButton
                  variant="destructive"
                  size="sm"
                  onClick={() => setAlertRules(prev => prev.filter(r => r.id !== rule.id))}
                >
                  <Trash2 className="w-4 h-4" />
                </MatrixButton>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Condition</label>
                <div className="flex space-x-2">
                  <select
                    value={rule.condition.metric}
                    onChange={(e) => updateAlertRule(rule.id, {
                      condition: { ...rule.condition, metric: e.target.value }
                    })}
                    className="flex-1 text-sm px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white"
                  >
                    <option value="max_drawdown">Max Drawdown</option>
                    <option value="daily_pnl">Daily P&L</option>
                    <option value="position_size">Position Size</option>
                    <option value="sharpe_ratio">Sharpe Ratio</option>
                    <option value="volatility">Volatility</option>
                    <option value="win_rate">Win Rate</option>
                  </select>
                  <select
                    value={rule.condition.operator}
                    onChange={(e) => updateAlertRule(rule.id, {
                      condition: { ...rule.condition, operator: e.target.value as any }
                    })}
                    className="flex-1 text-sm px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white"
                  >
                    <option value="greater_than">></option>
                    <option value="less_than"><</option>
                    <option value="equals">=</option>
                    <option value="not_equals">!=</option>
                    <option value="crosses_above">Crosses Above</option>
                    <option value="crosses_below">Crosses Below</option>
                  </select>
                  <input
                    type="text"
                    value={rule.condition.value}
                    onChange={(e) => updateAlertRule(rule.id, {
                      condition: { ...rule.condition, value: e.target.value }
                    })}
                    className="flex-1 text-sm px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white"
                    placeholder="Value"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-300 mb-1">Cooldown (min)</label>
                <MatrixInput
                  type="number"
                  value={rule.cooldown}
                  onChange={(e) => updateAlertRule(rule.id, { cooldown: parseInt(e.target.value) })}
                  className="w-full text-sm"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-300 mb-1">Channels</label>
                <div className="flex flex-wrap gap-1">
                  {notificationChannels.map((channel) => (
                    <button
                      key={channel.id}
                      onClick={() => {
                        const channels = rule.channels.includes(channel.id)
                          ? rule.channels.filter(id => id !== channel.id)
                          : [...rule.channels, channel.id];
                        updateAlertRule(rule.id, { channels });
                      }}
                      className={`
                        text-xs px-2 py-1 rounded
                        ${rule.channels.includes(channel.id)
                          ? 'bg-cyan-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }
                      `}
                    >
                      {channel.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={rule.enabled}
                    onChange={(e) => updateAlertRule(rule.id, { enabled: e.target.checked })}
                    className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
                  />
                  <span className="text-sm text-gray-300">Enabled</span>
                </label>
                <span className="text-sm text-gray-400">
                  Triggered: {rule.triggers} times
                </span>
                {rule.cooldownActive && (
                  <span className="text-sm text-yellow-400 flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    Cooling down
                  </span>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <select
                  value={rule.type}
                  onChange={(e) => updateAlertRule(rule.id, { type: e.target.value as any })}
                  className="text-sm px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white"
                >
                  <option value="performance">Performance</option>
                  <option value="risk">Risk</option>
                  <option value="position">Position</option>
                  <option value="market">Market</option>
                  <option value="system">System</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
            </div>
          </MatrixCard>
        ))}
      </div>
    </div>
  );

  const renderNotificationChannels = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">Notification Channels</h3>
        <MatrixButton
          onClick={() => {
            const newChannel: NotificationChannel = {
              id: Date.now().toString(),
              name: 'New Channel',
              type: 'webhook',
              enabled: false,
              config: {}
            };
            setNotificationChannels(prev => [...prev, newChannel]);
          }}
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Channel
        </MatrixButton>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {notificationChannels.map((channel) => (
          <MatrixCard key={channel.id} className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className={`
                  w-3 h-3 rounded-full
                  ${channel.enabled ? 'bg-green-400' : 'bg-gray-600'}
                `} />
                <div>
                  <h4 className="font-medium text-white flex items-center">
                    {channel.type === 'email' && <Mail className="w-4 h-4 mr-2" />}
                    {channel.type === 'sms' && <Phone className="w-4 h-4 mr-2" />}
                    {channel.type === 'slack' && <MessageSquare className="w-4 h-4 mr-2" />}
                    {channel.type === 'discord' && <MessageSquare className="w-4 h-4 mr-2" />}
                    {channel.type === 'telegram' && <MessageSquare className="w-4 h-4 mr-2" />}
                    {channel.type === 'webhook' && <Globe className="w-4 h-4 mr-2" />}
                    {channel.type === 'push' && <Bell className="w-4 h-4 mr-2" />}
                    {channel.name}
                  </h4>
                  <p className="text-sm text-gray-400 capitalize">{channel.type} notification</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <MatrixButton
                  size="sm"
                  onClick={() => {
                    // Test channel
                    console.log(`Testing ${channel.name}`);
                  }}
                >
                  <Play className="w-4 h-4" />
                </MatrixButton>
                <MatrixButton
                  variant="destructive"
                  size="sm"
                  onClick={() => setNotificationChannels(prev => prev.filter(c => c.id !== channel.id))}
                >
                  <Trash2 className="w-4 h-4" />
                </MatrixButton>
              </div>
            </div>

            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={channel.enabled}
                  onChange={(e) => updateNotificationChannel(channel.id, { enabled: e.target.checked })}
                  className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
                />
                <span className="text-sm text-gray-300">Enabled</span>
              </label>

              {channel.type === 'email' && (
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Email Address</label>
                  <MatrixInput
                    type="email"
                    value={channel.config.email || ''}
                    onChange={(e) => updateNotificationChannel(channel.id, {
                      config: { ...channel.config, email: e.target.value }
                    })}
                    className="w-full text-sm"
                    placeholder="user@example.com"
                  />
                </div>
              )}

              {channel.type === 'sms' && (
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Phone Number</label>
                  <MatrixInput
                    type="tel"
                    value={channel.config.phone || ''}
                    onChange={(e) => updateNotificationChannel(channel.id, {
                      config: { ...channel.config, phone: e.target.value }
                    })}
                    className="w-full text-sm"
                    placeholder="+1234567890"
                  />
                </div>
              )}

              {(channel.type === 'slack' || channel.type === 'discord' || channel.type === 'webhook') && (
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Webhook URL</label>
                  <MatrixInput
                    type="url"
                    value={channel.config.webhookUrl || ''}
                    onChange={(e) => updateNotificationChannel(channel.id, {
                      config: { ...channel.config, webhookUrl: e.target.value }
                    })}
                    className="w-full text-sm"
                    placeholder="https://hooks.slack.com/services/..."
                  />
                </div>
              )}

              {(channel.type === 'slack' || channel.type === 'discord') && (
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Channel</label>
                  <MatrixInput
                    type="text"
                    value={channel.config.channel || ''}
                    onChange={(e) => updateNotificationChannel(channel.id, {
                      config: { ...channel.config, channel: e.target.value }
                    })}
                    className="w-full text-sm"
                    placeholder="#channel-name"
                  />
                </div>
              )}

              {channel.type === 'telegram' && (
                <>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Bot Token</label>
                    <MatrixInput
                      type="text"
                      value={channel.config.botToken || ''}
                      onChange={(e) => updateNotificationChannel(channel.id, {
                        config: { ...channel.config, botToken: e.target.value }
                      })}
                      className="w-full text-sm"
                      placeholder="1234567890:ABCdef..."
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Chat ID</label>
                    <MatrixInput
                      type="text"
                      value={channel.config.chatId || ''}
                      onChange={(e) => updateNotificationChannel(channel.id, {
                        config: { ...channel.config, chatId: e.target.value }
                      })}
                      className="w-full text-sm"
                      placeholder="@channel_name"
                    />
                  </div>
                </>
              )}
            </div>
          </MatrixCard>
        ))}
      </div>
    </div>
  );

  const renderPreferences = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Global Settings</h4>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="enableAll"
                checked={preferences.global.enableAll}
                onChange={(e) => updatePreferences('global', { enableAll: e.target.checked })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="enableAll" className="text-sm text-gray-300">Enable All Alerts</label>
            </div>
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="muteAll"
                checked={preferences.global.muteAll}
                onChange={(e) => updatePreferences('global', { muteAll: e.target.checked })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="muteAll" className="text-sm text-gray-300">Mute All Alerts</label>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Default Cooldown (minutes)</label>
              <MatrixInput
                type="number"
                value={preferences.global.defaultCooldown}
                onChange={(e) => updatePreferences('global', { defaultCooldown: parseInt(e.target.value) })}
                className="w-full text-sm"
              />
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Quiet Hours</h4>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="quietHoursEnabled"
                checked={preferences.global.quietHours.enabled}
                onChange={(e) => updatePreferences('global', {
                  quietHours: { ...preferences.global.quietHours, enabled: e.target.checked }
                })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="quietHoursEnabled" className="text-sm text-gray-300">Enable Quiet Hours</label>
            </div>
            {preferences.global.quietHours.enabled && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Start Time</label>
                    <MatrixInput
                      type="time"
                      value={preferences.global.quietHours.start}
                      onChange={(e) => updatePreferences('global', {
                        quietHours: { ...preferences.global.quietHours, start: e.target.value }
                      })}
                      className="w-full text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">End Time</label>
                    <MatrixInput
                      type="time"
                      value={preferences.global.quietHours.end}
                      onChange={(e) => updatePreferences('global', {
                        quietHours: { ...preferences.global.quietHours, end: e.target.value }
                      })}
                      className="w-full text-sm"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Timezone</label>
                  <select
                    value={preferences.global.quietHours.timezone}
                    onChange={(e) => updatePreferences('global', {
                      quietHours: { ...preferences.global.quietHours, timezone: e.target.value }
                    })}
                    className="w-full text-sm px-3 py-2 bg-gray-900 border border-gray-600 rounded text-white"
                  >
                    <option value="UTC">UTC</option>
                    <option value="America/New_York">Eastern Time</option>
                    <option value="America/Chicago">Central Time</option>
                    <option value="America/Denver">Mountain Time</option>
                    <option value="America/Los_Angeles">Pacific Time</option>
                    <option value="Europe/London">London</option>
                    <option value="Europe/Paris">Paris</option>
                    <option value="Asia/Tokyo">Tokyo</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Alert Grouping</h4>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="groupingEnabled"
                checked={preferences.grouping.enabled}
                onChange={(e) => updatePreferences('grouping', { enabled: e.target.checked })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="groupingEnabled" className="text-sm text-gray-300">Enable Alert Grouping</label>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Time Window (minutes)</label>
              <MatrixInput
                type="number"
                value={preferences.grouping.timeWindow}
                onChange={(e) => updatePreferences('grouping', { timeWindow: parseInt(e.target.value) })}
                className="w-full text-sm"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Max Alerts per Group</label>
              <MatrixInput
                type="number"
                value={preferences.grouping.maxAlerts}
                onChange={(e) => updatePreferences('grouping', { maxAlerts: parseInt(e.target.value) })}
                className="w-full text-sm"
              />
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Analytics</h4>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="trackTriggers"
                checked={preferences.analytics.trackTriggers}
                onChange={(e) => updatePreferences('analytics', { trackTriggers: e.target.checked })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="trackTriggers" className="text-sm text-gray-300">Track Alert Triggers</label>
            </div>
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="trackChannels"
                checked={preferences.analytics.trackChannels}
                onChange={(e) => updatePreferences('analytics', { trackChannels: e.target.checked })}
                className="w-4 h-4 text-cyan-400 bg-gray-900 border-gray-600 rounded focus:ring-cyan-400"
              />
              <label htmlFor="trackChannels" className="text-sm text-gray-300">Track Channel Performance</label>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Retention Period (days)</label>
              <MatrixInput
                type="number"
                value={preferences.analytics.retentionDays}
                onChange={(e) => updatePreferences('analytics', { retentionDays: parseInt(e.target.value) })}
                className="w-full text-sm"
              />
            </div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-lg font-semibold text-white mb-4">Escalation Rules</h4>
        <div className="space-y-3">
          {preferences.escalation.levels.map((level, index) => (
            <div key={level.level} className="flex items-center space-x-3 p-3 bg-gray-800 rounded">
              <span className="text-sm text-gray-300 w-16">Level {level.level}</span>
              <MatrixInput
                type="text"
                value={level.condition}
                onChange={(e) => {
                  const newLevels = [...preferences.escalation.levels];
                  newLevels[index].condition = e.target.value;
                  updatePreferences('escalation', { levels: newLevels });
                }}
                className="flex-1 text-sm"
                placeholder="Condition"
              />
              <MatrixInput
                type="number"
                value={level.delay}
                onChange={(e) => {
                  const newLevels = [...preferences.escalation.levels];
                  newLevels[index].delay = parseInt(e.target.value);
                  updatePreferences('escalation', { levels: newLevels });
                }}
                className="w-20 text-sm"
                placeholder="Delay"
              />
              <MatrixButton
                variant="destructive"
                size="sm"
                onClick={() => {
                  const newLevels = preferences.escalation.levels.filter((_, i) => i !== index);
                  updatePreferences('escalation', { levels: newLevels });
                }}
              >
                <Trash2 className="w-4 h-4" />
              </MatrixButton>
            </div>
          ))}
          <MatrixButton
            variant="secondary"
            onClick={() => {
              const newLevel: EscalationLevel = {
                level: preferences.escalation.levels.length + 1,
                condition: '',
                channels: [],
                delay: 0
              };
              updatePreferences('escalation', {
                levels: [...preferences.escalation.levels, newLevel]
              });
            }}
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Escalation Level
          </MatrixButton>
        </div>
      </div>
    </div>
  );

  const renderHistory = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">Alert History</h3>
        <div className="flex items-center space-x-2">
          <MatrixButton variant="secondary" size="sm">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </MatrixButton>
          <MatrixButton variant="secondary" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export
          </MatrixButton>
        </div>
      </div>

      <div className="space-y-2">
        {/* Mock alert history */}
        {[
          { time: '2 minutes ago', rule: 'High Drawdown Alert', severity: 'high', channel: 'Email', status: 'delivered' },
          { time: '15 minutes ago', rule: 'Position Size Warning', severity: 'medium', channel: 'SMS', status: 'delivered' },
          { time: '1 hour ago', rule: 'Positive P&L Threshold', severity: 'medium', channel: 'Slack', status: 'delivered' },
          { time: '3 hours ago', rule: 'High Drawdown Alert', severity: 'high', channel: 'Email', status: 'failed' },
          { time: '1 day ago', rule: 'Position Size Warning', severity: 'medium', channel: 'Discord', status: 'delivered' }
        ].map((alert, index) => (
          <MatrixCard key={index} className="p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`
                  w-2 h-2 rounded-full
                  ${alert.severity === 'critical' ? 'bg-red-500' :
                    alert.severity === 'high' ? 'bg-orange-500' :
                    alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'}
                `} />
                <div>
                  <span className="text-sm text-white">{alert.rule}</span>
                  <div className="flex items-center space-x-2 text-xs text-gray-400">
                    <span>{alert.time}</span>
                    <span>•</span>
                    <span>{alert.channel}</span>
                    <span>•</span>
                    <span className={`
                      ${alert.status === 'delivered' ? 'text-green-400' : 'text-red-400'}
                    `}>
                      {alert.status}
                    </span>
                  </div>
                </div>
              </div>
              <span className={`
                text-xs px-2 py-1 rounded
                ${alert.severity === 'critical' ? 'bg-red-900 text-red-200' :
                  alert.severity === 'high' ? 'bg-orange-900 text-orange-200' :
                  alert.severity === 'medium' ? 'bg-yellow-900 text-yellow-200' : 'bg-blue-900 text-blue-200'}
              `}>
                {alert.severity}
              </span>
            </div>
          </MatrixCard>
        ))}
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'rules':
        return renderAlertRules();
      case 'channels':
        return renderNotificationChannels();
      case 'preferences':
        return renderPreferences();
      case 'history':
        return renderHistory();
      default:
        return null;
    }
  };

  return (
    <div className="p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <MatrixCard className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Bell className="w-6 h-6 text-cyan-400" />
              <h2 className="text-xl font-bold text-white">Alert Management</h2>
            </div>
            <div className="flex items-center space-x-2">
              <MatrixButton
                onClick={() => {
                  // Test all alerts
                  console.log('Testing all alerts');
                }}
                variant="secondary"
              >
                <Play className="w-4 h-4 mr-2" />
                Test All
              </MatrixButton>
              <MatrixButton variant="secondary" size="sm">
                <Download className="w-4 h-4" />
              </MatrixButton>
              <MatrixButton variant="secondary" size="sm">
                <Upload className="w-4 h-4" />
              </MatrixButton>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 mb-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    flex flex-col items-center p-3 rounded-lg transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400/30'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                    }
                  `}
                >
                  <Icon className="w-5 h-5 mb-1" />
                  <span className="text-xs text-center leading-tight">{tab.label}</span>
                </button>
              );
            })}
          </div>

          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <MatrixCard className="p-4 bg-gray-800/50">
              {renderTabContent()}
            </MatrixCard>
          </motion.div>

          <div className="mt-6">
            <MatrixButton
              onClick={() => console.log('Alert config saved', { alertRules, notificationChannels, preferences })}
              className="bg-cyan-600 hover:bg-cyan-700"
            >
              <Save className="w-4 h-4 mr-2" />
              Save Configuration
            </MatrixButton>
          </div>
        </MatrixCard>
      </motion.div>
    </div>
  );
};

export default AlertsConfig;