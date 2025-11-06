import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Bell, 
  Mail, 
  MessageSquare, 
  Phone,
  Settings,
  AlertTriangle,
  CheckCircle,
  Plus,
  Trash2,
  Copy,
  TestTube,
  Clock,
  Target,
  Volume2,
  Vibrate
} from 'lucide-react';

interface NotificationChannel {
  id: string;
  name: string;
  type: 'email' | 'sms' | 'webhook' | 'push' | 'slack' | 'discord' | 'telegram';
  enabled: boolean;
  config: {
    endpoint?: string;
    apiKey?: string;
    recipients: string[];
    template?: string;
    webhookUrl?: string;
    phoneNumber?: string;
    email?: string;
  };
  triggers: string[];
  filters: {
    minLevel: string;
    categories: string[];
    symbols?: string[];
  };
  schedule: {
    enabled: boolean;
    quietHours: {
      enabled: boolean;
      start: string;
      end: string;
      timezone: string;
    };
    cooldown: number; // minutes between similar notifications
  };
  metrics: {
    sentToday: number;
    successRate: number;
    avgResponseTime: number;
    lastSent: string;
  };
}

interface NotificationSettings {
  enabled: boolean;
  globalEnabled: boolean;
  defaultLevel: 'info' | 'warning' | 'error' | 'critical';
  batchNotifications: boolean;
  batchInterval: number; // seconds
  retryAttempts: number;
  retryDelay: number;
  escapeHtml: boolean;
  includeStackTrace: boolean;
}

interface NotificationTemplate {
  id: string;
  name: string;
  subject: string;
  body: string;
  variables: string[];
  category: string;
}

const NotificationConfig: React.FC = () => {
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings>({
    enabled: true,
    globalEnabled: true,
    defaultLevel: 'warning',
    batchNotifications: true,
    batchInterval: 60,
    retryAttempts: 3,
    retryDelay: 5,
    escapeHtml: true,
    includeStackTrace: false
  });

  const [notificationChannels, setNotificationChannels] = useState<NotificationChannel[]>([
    {
      id: 'email_default',
      name: 'Email Notifications',
      type: 'email',
      enabled: true,
      config: {
        email: 'trader@example.com',
        recipients: ['trader@example.com'],
        template: 'default'
      },
      triggers: ['trading_errors', 'risk_alerts', 'system_critical'],
      filters: {
        minLevel: 'warning',
        categories: ['trading', 'risk', 'system']
      },
      schedule: {
        enabled: true,
        quietHours: {
          enabled: true,
          start: '22:00',
          end: '08:00',
          timezone: 'UTC'
        },
        cooldown: 30
      },
      metrics: {
        sentToday: 12,
        successRate: 0.95,
        avgResponseTime: 2.3,
        lastSent: new Date().toISOString()
      }
    },
    {
      id: 'webhook_default',
      name: 'Slack Webhook',
      type: 'slack',
      enabled: false,
      config: {
        webhookUrl: '',
        recipients: ['#trading-alerts']
      },
      triggers: ['all_errors', 'risk_alerts'],
      filters: {
        minLevel: 'error',
        categories: ['trading', 'risk']
      },
      schedule: {
        enabled: false,
        quietHours: {
          enabled: false,
          start: '22:00',
          end: '08:00',
          timezone: 'UTC'
        },
        cooldown: 15
      },
      metrics: {
        sentToday: 0,
        successRate: 0,
        avgResponseTime: 0,
        lastSent: ''
      }
    }
  ]);

  const [templates] = useState<NotificationTemplate[]>([
    {
      id: 'default',
      name: 'Default',
      subject: '[Trading System] {{level}}: {{title}}',
      body: `Alert Level: {{level}}\nTitle: {{title}}\nMessage: {{message}}\nTimestamp: {{timestamp}}\nSource: {{source}}`,
      variables: ['level', 'title', 'message', 'timestamp', 'source'],
      category: 'general'
    },
    {
      id: 'trading_error',
      name: 'Trading Error',
      subject: '[Trading Error] {{symbol}} - {{error_type}}',
      body: `A trading error occurred:\n\nSymbol: {{symbol}}\nError Type: {{error_type}}\nMessage: {{message}}\nOrder ID: {{order_id}}\nTimestamp: {{timestamp}}`,
      variables: ['symbol', 'error_type', 'message', 'order_id', 'timestamp'],
      category: 'trading'
    },
    {
      id: 'risk_alert',
      name: 'Risk Alert',
      subject: '[Risk Alert] {{alert_type}} - {{risk_level}}',
      body: `Risk Management Alert:\n\nAlert Type: {{alert_type}}\nRisk Level: {{risk_level}}\nMessage: {{message}}\nCurrent Exposure: {{exposure}}\nRecommendation: {{recommendation}}`,
      variables: ['alert_type', 'risk_level', 'message', 'exposure', 'recommendation'],
      category: 'risk'
    }
  ]);

  const [activeChannel, setActiveChannel] = useState<string>('email_default');
  const [isTesting, setIsTesting] = useState<string | null>(null);
  const [showTemplateModal, setShowTemplateModal] = useState(false);
  const [saving, setSaving] = useState(false);

  const triggerTypes = [
    'all_errors',
    'trading_errors',
    'risk_alerts',
    'system_critical',
    'position_opened',
    'position_closed',
    'profit_target_reached',
    'stop_loss_triggered',
    'connection_lost',
    'api_errors'
  ];

  const logLevels = [
    { value: 'info', label: 'Info' },
    { value: 'warning', label: 'Warning' },
    { value: 'error', label: 'Error' },
    { value: 'critical', label: 'Critical' }
  ];

  const categories = [
    'trading', 'risk', 'system', 'broker', 'strategy', 'ai', 'performance', 'security'
  ];

  const channelTypes = [
    { value: 'email', label: 'Email', icon: Mail },
    { value: 'sms', label: 'SMS', icon: Phone },
    { value: 'webhook', label: 'Webhook', icon: Settings },
    { value: 'push', label: 'Push Notification', icon: Bell },
    { value: 'slack', label: 'Slack', icon: MessageSquare },
    { value: 'discord', label: 'Discord', icon: MessageSquare },
    { value: 'telegram', label: 'Telegram', icon: MessageSquare }
  ];

  useEffect(() => {
    loadNotificationConfiguration();
  }, []);

  const loadNotificationConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getNotificationConfig();
      if (config) {
        setNotificationSettings(config.settings || notificationSettings);
        setNotificationChannels(config.channels || notificationChannels);
      }
    } catch (error) {
      console.error('Failed to load notification configuration:', error);
    }
  };

  const saveNotificationConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.saveNotificationConfig({
        settings: notificationSettings,
        channels: notificationChannels
      });
    } catch (error) {
      console.error('Failed to save notification configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const addNotificationChannel = () => {
    const newChannel: NotificationChannel = {
      id: `channel_${Date.now()}`,
      name: 'New Channel',
      type: 'email',
      enabled: false,
      config: {
        recipients: []
      },
      triggers: [],
      filters: {
        minLevel: 'warning',
        categories: []
      },
      schedule: {
        enabled: true,
        quietHours: {
          enabled: true,
          start: '22:00',
          end: '08:00',
          timezone: 'UTC'
        },
        cooldown: 30
      },
      metrics: {
        sentToday: 0,
        successRate: 0,
        avgResponseTime: 0,
        lastSent: ''
      }
    };
    setNotificationChannels([...notificationChannels, newChannel]);
    setActiveChannel(newChannel.id);
  };

  const updateNotificationChannel = (id: string, updates: Partial<NotificationChannel>) => {
    setNotificationChannels(channels => 
      channels.map(channel => channel.id === id ? { ...channel, ...updates } : channel)
    );
  };

  const deleteNotificationChannel = (id: string) => {
    setNotificationChannels(channels => channels.filter(channel => channel.id !== id));
    if (activeChannel === id) {
      const remaining = notificationChannels.filter(c => c.id !== id);
      setActiveChannel(remaining.length > 0 ? remaining[0].id : '');
    }
  };

  const duplicateChannel = (channel: NotificationChannel) => {
    const duplicated: NotificationChannel = {
      ...channel,
      id: `channel_${Date.now()}`,
      name: `${channel.name} (Copy)`,
      enabled: false,
      metrics: {
        sentToday: 0,
        successRate: 0,
        avgResponseTime: 0,
        lastSent: ''
      }
    };
    setNotificationChannels([...notificationChannels, duplicated]);
    setActiveChannel(duplicated.id);
  };

  const testNotification = async (channel: NotificationChannel) => {
    setIsTesting(channel.id);
    try {
      // Simulate notification test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      updateNotificationChannel(channel.id, {
        metrics: {
          ...channel.metrics,
          sentToday: channel.metrics.sentToday + 1,
          lastSent: new Date().toISOString(),
          successRate: Math.min(1, channel.metrics.successRate + 0.05)
        }
      });
    } catch (error) {
      console.error('Failed to test notification:', error);
    } finally {
      setIsTesting(null);
    }
  };

  const getTypeIcon = (type: string) => {
    const channelType = channelTypes.find(ct => ct.value === type);
    return channelType ? <channelType.icon className="h-4 w-4" /> : <Bell className="h-4 w-4" />;
  };

  const getStatusBadge = (enabled: boolean) => {
    return (
      <Badge variant={enabled ? 'default' : 'outline'}>
        {enabled ? 'Active' : 'Inactive'}
      </Badge>
    );
  };

  const activeChannelData = notificationChannels.find(c => c.id === activeChannel);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Configuration
          </CardTitle>
          <CardDescription>
            Configure alert channels and notification templates
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={notificationSettings.enabled}
                  onCheckedChange={(checked) => setNotificationSettings(prev => ({ ...prev, enabled: checked }))}
                />
                <Label>Notification System {notificationSettings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant={notificationSettings.globalEnabled ? 'default' : 'outline'}>
                {notificationSettings.globalEnabled ? 'Global Active' : 'Global Inactive'}
              </Badge>
            </div>
            <Button onClick={saveNotificationConfiguration} disabled={saving}>
              <Settings className="h-4 w-4 mr-2" />
              {saving ? 'Saving...' : 'Save Configuration'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Channels</p>
                <p className="text-2xl font-bold">{notificationChannels.filter(c => c.enabled).length}</p>
              </div>
              <Bell className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Sent Today</p>
                <p className="text-2xl font-bold">
                  {notificationChannels.reduce((sum, c) => sum + c.metrics.sentToday, 0)}
                </p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold">
                  {((notificationChannels.reduce((sum, c) => sum + c.metrics.successRate, 0) / notificationChannels.length) * 100).toFixed(1)}%
                </p>
              </div>
              <Target className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Response</p>
                <p className="text-2xl font-bold">
                  {(notificationChannels.reduce((sum, c) => sum + c.metrics.avgResponseTime, 0) / notificationChannels.length).toFixed(1)}s
                </p>
              </div>
              <Clock className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="channels" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="channels">Channels</TabsTrigger>
          <TabsTrigger value="triggers">Triggers</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="channels" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Channels List */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Notification Channels</CardTitle>
                    <CardDescription>
                      Manage your notification delivery methods
                    </CardDescription>
                  </div>
                  <Button onClick={addNotificationChannel} size="sm">
                    Add Channel
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {notificationChannels.map((channel) => (
                  <div
                    key={channel.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      activeChannel === channel.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setActiveChannel(channel.id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(channel.type)}
                        <h3 className="font-medium">{channel.name}</h3>
                        {getStatusBadge(channel.enabled)}
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            testNotification(channel);
                          }}
                          disabled={isTesting === channel.id || !channel.enabled}
                        >
                          <TestTube className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            duplicateChannel(channel);
                          }}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteNotificationChannel(channel.id);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Type:</span>
                        <div className="font-medium capitalize">{channel.type}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Recipients:</span>
                        <div className="font-medium">{channel.config.recipients.length}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Triggers:</span>
                        <div className="font-medium">{channel.triggers.length}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Sent Today:</span>
                        <div className="font-medium">{channel.metrics.sentToday}</div>
                      </div>
                    </div>

                    <div className="mt-2 flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {channel.filters.minLevel} level
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        Success: {(channel.metrics.successRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}

                {notificationChannels.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Bell className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No notification channels configured</p>
                    <Button onClick={addNotificationChannel} variant="outline" className="mt-2" size="sm">
                      Add your first channel
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Channel Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Channel Configuration</CardTitle>
                <CardDescription>
                  {activeChannelData ? 'Configure channel settings and parameters' : 'Select a channel to configure'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeChannelData ? (
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="channel-name">Channel Name</Label>
                      <Input
                        id="channel-name"
                        value={activeChannelData.name}
                        onChange={(e) => updateNotificationChannel(activeChannelData.id, { name: e.target.value })}
                      />
                    </div>

                    <div>
                      <Label htmlFor="channel-type">Channel Type</Label>
                      <Select
                        value={activeChannelData.type}
                        onValueChange={(value: any) => updateNotificationChannel(activeChannelData.id, { type: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {channelTypes.map((type) => (
                            <SelectItem key={type.value} value={type.value}>
                              {type.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable Channel</Label>
                        <p className="text-sm text-muted-foreground">
                          Activate this notification channel
                        </p>
                      </div>
                      <Switch
                        checked={activeChannelData.enabled}
                        onCheckedChange={(checked) => updateNotificationChannel(activeChannelData.id, { enabled: checked })}
                      />
                    </div>

                    <Separator />

                    {/* Type-specific configuration */}
                    {(activeChannelData.type === 'email' || activeChannelData.type === 'sms') && (
                      <div>
                        <Label htmlFor="recipients">Recipients</Label>
                        <Textarea
                          id="recipients"
                          value={activeChannelData.config.recipients.join('\n')}
                          onChange={(e) => {
                            const recipients = e.target.value.split('\n').filter(r => r.trim());
                            updateNotificationChannel(activeChannelData.id, {
                              config: { ...activeChannelData.config, recipients }
                            });
                          }}
                          placeholder="Enter email addresses or phone numbers (one per line)"
                          rows={3}
                        />
                      </div>
                    )}

                    {activeChannelData.type === 'slack' || activeChannelData.type === 'discord' || activeChannelData.type === 'telegram' || activeChannelData.type === 'webhook' ? (
                      <div>
                        <Label htmlFor="webhook-url">
                          {activeChannelData.type === 'webhook' ? 'Webhook URL' : 'Channel/Endpoint'}
                        </Label>
                        <Input
                          id="webhook-url"
                          value={activeChannelData.config.webhookUrl || ''}
                          onChange={(e) => updateNotificationChannel(activeChannelData.id, {
                            config: { ...activeChannelData.config, webhookUrl: e.target.value }
                          })}
                          placeholder="Enter webhook URL or channel identifier"
                        />
                      </div>
                    )}

                    {activeChannelData.type !== 'webhook' && (
                      <>
                        <div>
                          <Label htmlFor="template">Template</Label>
                          <Select
                            value={activeChannelData.config.template || 'default'}
                            onValueChange={(value) => updateNotificationChannel(activeChannelData.id, {
                              config: { ...activeChannelData.config, template: value }
                            })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {templates.map((template) => (
                                <SelectItem key={template.id} value={template.id}>
                                  {template.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label htmlFor="api-key">API Key (Optional)</Label>
                          <Input
                            id="api-key"
                            type="password"
                            value={activeChannelData.config.apiKey || ''}
                            onChange={(e) => updateNotificationChannel(activeChannelData.id, {
                              config: { ...activeChannelData.config, apiKey: e.target.value }
                            })}
                            placeholder="Enter API key if required"
                          />
                        </div>
                      </>
                    )}

                    {/* Filters */}
                    <Separator />
                    <div>
                      <h3 className="text-sm font-medium mb-3">Filters</h3>
                      
                      <div>
                        <Label>Minimum Level</Label>
                        <Select
                          value={activeChannelData.filters.minLevel}
                          onValueChange={(value) => updateNotificationChannel(activeChannelData.id, {
                            filters: { ...activeChannelData.filters, minLevel: value }
                          })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {logLevels.map((level) => (
                              <SelectItem key={level.value} value={level.value}>
                                {level.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="mt-3">
                        <Label>Categories</Label>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          {categories.map((category) => (
                            <div key={category} className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                id={`category-${category}`}
                                checked={activeChannelData.filters.categories.includes(category)}
                                onChange={(e) => {
                                  const categories = activeChannelData.filters.categories;
                                  const newCategories = e.target.checked
                                    ? [...categories, category]
                                    : categories.filter(c => c !== category);
                                  updateNotificationChannel(activeChannelData.id, {
                                    filters: { ...activeChannelData.filters, categories: newCategories }
                                  });
                                }}
                                className="rounded"
                              />
                              <Label htmlFor={`category-${category}`} className="text-sm capitalize">
                                {category}
                              </Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Select a channel to configure its settings</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="triggers" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Notification Triggers</CardTitle>
              <CardDescription>
                Configure when notifications should be sent
              </CardDescription>
            </CardHeader>
            <CardContent>
              {activeChannelData && (
                <div className="space-y-4">
                  <div>
                    <Label>Active Triggers</Label>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2">
                      {triggerTypes.map((trigger) => (
                        <div key={trigger} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id={`trigger-${trigger}`}
                            checked={activeChannelData.triggers.includes(trigger)}
                            onChange={(e) => {
                              const triggers = activeChannelData.triggers;
                              const newTriggers = e.target.checked
                                ? [...triggers, trigger]
                                : triggers.filter(t => t !== trigger);
                              updateNotificationChannel(activeChannelData.id, { triggers: newTriggers });
                            }}
                            className="rounded"
                          />
                          <Label htmlFor={`trigger-${trigger}`} className="text-sm">
                            {trigger.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </Label>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h3 className="text-sm font-medium mb-3">Schedule Settings</h3>
                    
                    <div className="flex items-center justify-between mb-3">
                      <Label>Quiet Hours</Label>
                      <Switch
                        checked={activeChannelData.schedule.quietHours.enabled}
                        onCheckedChange={(checked) => updateNotificationChannel(activeChannelData.id, {
                          schedule: {
                            ...activeChannelData.schedule,
                            quietHours: {
                              ...activeChannelData.schedule.quietHours,
                              enabled: checked
                            }
                          }
                        })}
                      />
                    </div>

                    {activeChannelData.schedule.quietHours.enabled && (
                      <div className="grid grid-cols-2 gap-4 mb-3">
                        <div>
                          <Label>Start Time</Label>
                          <Input
                            type="time"
                            value={activeChannelData.schedule.quietHours.start}
                            onChange={(e) => updateNotificationChannel(activeChannelData.id, {
                              schedule: {
                                ...activeChannelData.schedule,
                                quietHours: {
                                  ...activeChannelData.schedule.quietHours,
                                  start: e.target.value
                                }
                              }
                            })}
                          />
                        </div>
                        <div>
                          <Label>End Time</Label>
                          <Input
                            type="time"
                            value={activeChannelData.schedule.quietHours.end}
                            onChange={(e) => updateNotificationChannel(activeChannelData.id, {
                              schedule: {
                                ...activeChannelData.schedule,
                                quietHours: {
                                  ...activeChannelData.schedule.quietHours,
                                  end: e.target.value
                                }
                              }
                            })}
                          />
                        </div>
                      </div>
                    )}

                    <div>
                      <Label>Cooldown Period: {activeChannelData.schedule.cooldown} minutes</Label>
                      <Slider
                        value={[activeChannelData.schedule.cooldown]}
                        onValueChange={([value]) => updateNotificationChannel(activeChannelData.id, {
                          schedule: { ...activeChannelData.schedule, cooldown: value }
                        })}
                        min={1}
                        max={120}
                        step={5}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="templates" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Notification Templates</CardTitle>
                  <CardDescription>
                    Manage notification message templates and formatting
                  </CardDescription>
                </div>
                <Button onClick={() => setShowTemplateModal(true)} size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  New Template
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {templates.map((template) => (
                  <div key={template.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{template.name}</h3>
                      <Badge variant="outline">{template.category}</Badge>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-muted-foreground">Subject:</span>
                        <div className="font-mono bg-gray-50 p-2 rounded">{template.subject}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Body:</span>
                        <div className="font-mono bg-gray-50 p-2 rounded text-xs">{template.body}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Variables:</span>
                        <div className="flex gap-1 mt-1">
                          {template.variables.map((variable) => (
                            <Badge key={variable} variant="secondary" className="text-xs">
                              {`{{${variable}}}`}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Global Settings</CardTitle>
                <CardDescription>
                  Configure notification system behavior
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Global Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable/disable all notifications globally
                    </p>
                  </div>
                  <Switch
                    checked={notificationSettings.globalEnabled}
                    onCheckedChange={(checked) => setNotificationSettings(prev => ({ ...prev, globalEnabled: checked }))}
                  />
                </div>

                <div>
                  <Label htmlFor="default-level">Default Alert Level</Label>
                  <Select
                    value={notificationSettings.defaultLevel}
                    onValueChange={(value: any) => setNotificationSettings(prev => ({ ...prev, defaultLevel: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {logLevels.map((level) => (
                        <SelectItem key={level.value} value={level.value}>
                          {level.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Batch Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Group similar notifications together
                    </p>
                  </div>
                  <Switch
                    checked={notificationSettings.batchNotifications}
                    onCheckedChange={(checked) => setNotificationSettings(prev => ({ ...prev, batchNotifications: checked }))}
                  />
                </div>

                <div>
                  <Label>Batch Interval: {notificationSettings.batchInterval}s</Label>
                  <Slider
                    value={[notificationSettings.batchInterval]}
                    onValueChange={([value]) => setNotificationSettings(prev => ({ ...prev, batchInterval: value }))}
                    min={30}
                    max={300}
                    step={10}
                    className="mt-2"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Delivery Settings</CardTitle>
                <CardDescription>
                  Configure notification delivery and retry behavior
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Retry Attempts</Label>
                  <Input
                    type="number"
                    value={notificationSettings.retryAttempts}
                    onChange={(e) => setNotificationSettings(prev => ({ ...prev, retryAttempts: Number(e.target.value) }))}
                  />
                </div>

                <div>
                  <Label>Retry Delay: {notificationSettings.retryDelay}s</Label>
                  <Slider
                    value={[notificationSettings.retryDelay]}
                    onValueChange={([value]) => setNotificationSettings(prev => ({ ...prev, retryDelay: value }))}
                    min={1}
                    max={60}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Escape HTML</Label>
                    <p className="text-sm text-muted-foreground">
                      Escape HTML in notification messages
                    </p>
                  </div>
                  <Switch
                    checked={notificationSettings.escapeHtml}
                    onCheckedChange={(checked) => setNotificationSettings(prev => ({ ...prev, escapeHtml: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Include Stack Trace</Label>
                    <p className="text-sm text-muted-foreground">
                      Include stack traces in error notifications
                    </p>
                  </div>
                  <Switch
                    checked={notificationSettings.includeStackTrace}
                    onCheckedChange={(checked) => setNotificationSettings(prev => ({ ...prev, includeStackTrace: checked }))}
                  />
                </div>

                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Adjust retry settings carefully to avoid spam notifications while ensuring critical alerts are delivered.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default NotificationConfig;