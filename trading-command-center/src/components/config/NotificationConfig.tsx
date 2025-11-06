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
  TestTube 
} from 'lucide-react';

export interface NotificationChannel {
  id: string;
  name: string;
  type: 'email' | 'sms' | 'webhook' | 'slack' | 'telegram' | 'discord';
  enabled: boolean;
  config: {
    url?: string;
    webhook?: string;
    phone?: string;
    email?: string;
    channel?: string;
    username?: string;
    token?: string;
  };
}

export interface NotificationTemplate {
  id: string;
  name: string;
  subject: string;
  content: string;
  variables: string[];
}

export interface NotificationSettings {
  global: {
    enabled: boolean;
    quietHours: {
      enabled: boolean;
      start: string;
      end: string;
      timezone: string;
    };
  };
  channels: NotificationChannel[];
  templates: NotificationTemplate[];
  alertRules: {
    id: string;
    name: string;
    event: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    enabled: boolean;
    channels: string[];
    templateId: string;
  }[];
  rateLimit: {
    enabled: boolean;
    maxPerHour: number;
    maxPerDay: number;
  };
}

const defaultNotificationSettings: NotificationSettings = {
  global: {
    enabled: true,
    quietHours: {
      enabled: false,
      start: '22:00',
      end: '08:00',
      timezone: 'UTC'
    }
  },
  channels: [
    {
      id: '1',
      name: 'Email Notifications',
      type: 'email',
      enabled: true,
      config: {
        email: 'admin@example.com'
      }
    },
    {
      id: '2',
      name: 'Slack Alerts',
      type: 'slack',
      enabled: false,
      config: {
        webhook: '',
        channel: '#alerts'
      }
    }
  ],
  templates: [
    {
      id: '1',
      name: 'Trade Alert',
      subject: 'Trade Alert: {{symbol}} {{action}}',
      content: 'A {{action}} order for {{quantity}} shares of {{symbol}} has been executed at ${{price}}.',
      variables: ['action', 'symbol', 'quantity', 'price']
    },
    {
      id: '2',
      name: 'Error Alert',
      subject: 'System Error: {{errorType}}',
      content: 'An error occurred: {{errorMessage}}. Please check the system logs for details.',
      variables: ['errorType', 'errorMessage']
    }
  ],
  alertRules: [
    {
      id: '1',
      name: 'Trade Executed',
      event: 'trade_executed',
      severity: 'low',
      enabled: true,
      channels: ['1'],
      templateId: '1'
    },
    {
      id: '2',
      name: 'System Error',
      event: 'system_error',
      severity: 'critical',
      enabled: true,
      channels: ['1', '2'],
      templateId: '2'
    }
  ],
  rateLimit: {
    enabled: true,
    maxPerHour: 100,
    maxPerDay: 500
  }
};

export default function NotificationConfig() {
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings>(defaultNotificationSettings);
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [testResults, setTestResults] = useState<{[key: string]: 'success' | 'error' | null}>({});
  const [newTemplate, setNewTemplate] = useState<Partial<NotificationTemplate>>({});

  const handleGlobalChange = (field: string, value: any) => {
    setNotificationSettings(prev => ({
      ...prev,
      global: {
        ...prev.global,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleQuietHoursChange = (field: string, value: any) => {
    setNotificationSettings(prev => ({
      ...prev,
      global: {
        ...prev.global,
        quietHours: {
          ...prev.global.quietHours,
          [field]: value
        }
      }
    }));
    setHasChanges(true);
  };

  const handleChannelToggle = (channelId: string, enabled: boolean) => {
    setNotificationSettings(prev => ({
      ...prev,
      channels: prev.channels.map(channel =>
        channel.id === channelId ? { ...channel, enabled } : channel
      )
    }));
    setHasChanges(true);
  };

  const handleChannelConfigChange = (channelId: string, field: string, value: any) => {
    setNotificationSettings(prev => ({
      ...prev,
      channels: prev.channels.map(channel =>
        channel.id === channelId 
          ? { ...channel, config: { ...channel.config, [field]: value } }
          : channel
      )
    }));
    setHasChanges(true);
  };

  const handleTemplateEdit = (templateId: string, field: string, value: any) => {
    setNotificationSettings(prev => ({
      ...prev,
      templates: prev.templates.map(template =>
        template.id === templateId ? { ...template, [field]: value } : template
      )
    }));
    setHasChanges(true);
  };

  const handleAddTemplate = () => {
    if (newTemplate.name && newTemplate.subject && newTemplate.content) {
      const template: NotificationTemplate = {
        id: Date.now().toString(),
        name: newTemplate.name!,
        subject: newTemplate.subject!,
        content: newTemplate.content!,
        variables: newTemplate.variables || []
      };
      
      setNotificationSettings(prev => ({
        ...prev,
        templates: [...prev.templates, template]
      }));
      
      setNewTemplate({});
      setHasChanges(true);
    }
  };

  const handleDeleteTemplate = (templateId: string) => {
    setNotificationSettings(prev => ({
      ...prev,
      templates: prev.templates.filter(template => template.id !== templateId)
    }));
    setHasChanges(true);
  };

  const handleTestChannel = async (channelId: string) => {
    setTestResults(prev => ({ ...prev, [channelId]: null }));
    
    // Simulate test
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Randomly succeed/fail for demo
    const success = Math.random() > 0.3;
    setTestResults(prev => ({ ...prev, [channelId]: success ? 'success' : 'error' }));
  };

  const handleSave = async () => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save notification settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getChannelIcon = (type: string) => {
    switch (type) {
      case 'email': return <Mail className="h-4 w-4" />;
      case 'sms': return <Phone className="h-4 w-4" />;
      case 'webhook': return <MessageSquare className="h-4 w-4" />;
      case 'slack': return <MessageSquare className="h-4 w-4" />;
      case 'telegram': return <MessageSquare className="h-4 w-4" />;
      case 'discord': return <MessageSquare className="h-4 w-4" />;
      default: return <Bell className="h-4 w-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'bg-blue-100 text-blue-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Bell className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Notification Configuration</h2>
        </div>
        <Button onClick={handleSave} disabled={!hasChanges || isLoading}>
          {isLoading ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
          ) : (
            <Settings className="h-4 w-4 mr-2" />
          )}
          Save Changes
        </Button>
      </div>

      <Tabs defaultValue="channels" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="channels">Channels</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="rules">Alert Rules</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="channels" className="space-y-4">
          {notificationSettings.channels.map((channel) => (
            <Card key={channel.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center space-x-2">
                    {getChannelIcon(channel.type)}
                    <span>{channel.name}</span>
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    <Badge variant={channel.enabled ? 'default' : 'secondary'}>
                      {channel.enabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleTestChannel(channel.id)}
                    >
                      <TestTube className="h-4 w-4 mr-2" />
                      Test
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={channel.enabled}
                    onCheckedChange={(checked) => handleChannelToggle(channel.id, checked)}
                  />
                  <Label>Enable this channel</Label>
                </div>

                {channel.enabled && (
                  <div className="space-y-4">
                    {/* Channel-specific configuration */}
                    {channel.type === 'email' && (
                      <div className="space-y-2">
                        <Label>Email Address</Label>
                        <Input
                          type="email"
                          value={channel.config.email || ''}
                          onChange={(e) => handleChannelConfigChange(channel.id, 'email', e.target.value)}
                          placeholder="admin@example.com"
                        />
                      </div>
                    )}

                    {(channel.type === 'slack' || channel.type === 'discord') && (
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label>Webhook URL</Label>
                          <Input
                            value={channel.config.webhook || ''}
                            onChange={(e) => handleChannelConfigChange(channel.id, 'webhook', e.target.value)}
                            placeholder="https://hooks.slack.com/services/..."
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Channel</Label>
                          <Input
                            value={channel.config.channel || ''}
                            onChange={(e) => handleChannelConfigChange(channel.id, 'channel', e.target.value)}
                            placeholder="#alerts"
                          />
                        </div>
                      </div>
                    )}

                    {channel.type === 'sms' && (
                      <div className="space-y-2">
                        <Label>Phone Number</Label>
                        <Input
                          value={channel.config.phone || ''}
                          onChange={(e) => handleChannelConfigChange(channel.id, 'phone', e.target.value)}
                          placeholder="+1234567890"
                        />
                      </div>
                    )}

                    {channel.type === 'webhook' && (
                      <div className="space-y-2">
                        <Label>Webhook URL</Label>
                        <Input
                          value={channel.config.url || ''}
                          onChange={(e) => handleChannelConfigChange(channel.id, 'url', e.target.value)}
                          placeholder="https://your-webhook-url.com"
                        />
                      </div>
                    )}

                    {/* Test results */}
                    {testResults[channel.id] && (
                      <Alert variant={testResults[channel.id] === 'success' ? 'default' : 'destructive'}>
                        {testResults[channel.id] === 'success' ? (
                          <CheckCircle className="h-4 w-4" />
                        ) : (
                          <AlertTriangle className="h-4 w-4" />
                        )}
                        <AlertDescription>
                          {testResults[channel.id] === 'success' 
                            ? 'Test notification sent successfully!' 
                            : 'Test notification failed. Please check your configuration.'}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="templates" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Create New Template</CardTitle>
              <CardDescription>
                Create a new notification template with customizable variables
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Template Name</Label>
                  <Input
                    value={newTemplate.name || ''}
                    onChange={(e) => setNewTemplate(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="Template Name"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Subject</Label>
                  <Input
                    value={newTemplate.subject || ''}
                    onChange={(e) => setNewTemplate(prev => ({ ...prev, subject: e.target.value }))}
                    placeholder="Notification Subject"
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label>Content</Label>
                <Textarea
                  value={newTemplate.content || ''}
                  onChange={(e) => setNewTemplate(prev => ({ ...prev, content: e.target.value }))}
                  placeholder="Notification content with {{variables}}"
                  rows={4}
                />
              </div>
              
              <Button onClick={handleAddTemplate}>
                <Plus className="h-4 w-4 mr-2" />
                Add Template
              </Button>
            </CardContent>
          </Card>

          <Separator />

          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Existing Templates</h3>
            {notificationSettings.templates.map((template) => (
              <Card key={template.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{template.name}</CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeleteTemplate(template.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Subject</Label>
                    <Input
                      value={template.subject}
                      onChange={(e) => handleTemplateEdit(template.id, 'subject', e.target.value)}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Content</Label>
                    <Textarea
                      value={template.content}
                      onChange={(e) => handleTemplateEdit(template.id, 'content', e.target.value)}
                      rows={3}
                    />
                  </div>
                  
                  {template.variables.length > 0 && (
                    <div className="space-y-2">
                      <Label>Variables</Label>
                      <div className="flex flex-wrap gap-2">
                        {template.variables.map((variable) => (
                          <Badge key={variable} variant="outline">
                            {variable}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="rules" className="space-y-4">
          {notificationSettings.alertRules.map((rule) => (
            <Card key={rule.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center space-x-2">
                    <span>{rule.name}</span>
                    <Badge className={getSeverityColor(rule.severity)}>
                      {rule.severity}
                    </Badge>
                  </CardTitle>
                  <Switch
                    checked={rule.enabled}
                    onCheckedChange={(checked) => {
                      setNotificationSettings(prev => ({
                        ...prev,
                        alertRules: prev.alertRules.map(r =>
                          r.id === rule.id ? { ...r, enabled: checked } : r
                        )
                      }));
                      setHasChanges(true);
                    }}
                  />
                </div>
                <CardDescription>
                  Triggers on: {rule.event}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Template</Label>
                  <Select
                    value={rule.templateId}
                    onValueChange={(value) => {
                      setNotificationSettings(prev => ({
                        ...prev,
                        alertRules: prev.alertRules.map(r =>
                          r.id === rule.id ? { ...r, templateId: value } : r
                        )
                      }));
                      setHasChanges(true);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {notificationSettings.templates.map(template => (
                        <SelectItem key={template.id} value={template.id}>
                          {template.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Notification Channels</Label>
                  <div className="space-y-2">
                    {notificationSettings.channels.map(channel => (
                      <div key={channel.id} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={rule.channels.includes(channel.id)}
                          onChange={(e) => {
                            const updatedChannels = e.target.checked
                              ? [...rule.channels, channel.id]
                              : rule.channels.filter(id => id !== channel.id);
                            
                            setNotificationSettings(prev => ({
                              ...prev,
                              alertRules: prev.alertRules.map(r =>
                                r.id === rule.id ? { ...r, channels: updatedChannels } : r
                              )
                            }));
                            setHasChanges(true);
                          }}
                        />
                        <Label className="text-sm">{channel.name}</Label>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Global Settings</CardTitle>
              <CardDescription>
                Configure global notification behavior and limits
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={notificationSettings.global.enabled}
                  onCheckedChange={(checked) => handleGlobalChange('enabled', checked)}
                />
                <Label>Enable Notifications</Label>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="text-sm font-semibold flex items-center space-x-2">
                  <Bell className="h-4 w-4" />
                  <span>Quiet Hours</span>
                </h4>
                
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={notificationSettings.global.quietHours.enabled}
                      onCheckedChange={(checked) => handleQuietHoursChange('enabled', checked)}
                    />
                    <Label>Enable Quiet Hours</Label>
                  </div>
                  
                  {notificationSettings.global.quietHours.enabled && (
                    <div className="grid grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <Label>Start Time</Label>
                        <Input
                          type="time"
                          value={notificationSettings.global.quietHours.start}
                          onChange={(e) => handleQuietHoursChange('start', e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>End Time</Label>
                        <Input
                          type="time"
                          value={notificationSettings.global.quietHours.end}
                          onChange={(e) => handleQuietHoursChange('end', e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Timezone</Label>
                        <Select
                          value={notificationSettings.global.quietHours.timezone}
                          onValueChange={(value) => handleQuietHoursChange('timezone', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="UTC">UTC</SelectItem>
                            <SelectItem value="America/New_York">Eastern Time</SelectItem>
                            <SelectItem value="America/Chicago">Central Time</SelectItem>
                            <SelectItem value="America/Denver">Mountain Time</SelectItem>
                            <SelectItem value="America/Los_Angeles">Pacific Time</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="text-sm font-semibold flex items-center space-x-2">
                  <Settings className="h-4 w-4" />
                  <span>Rate Limiting</span>
                </h4>
                
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={notificationSettings.rateLimit.enabled}
                      onCheckedChange={(checked) => {
                        setNotificationSettings(prev => ({
                          ...prev,
                          rateLimit: { ...prev.rateLimit, enabled: checked }
                        }));
                        setHasChanges(true);
                      }}
                    />
                    <Label>Enable Rate Limiting</Label>
                  </div>
                  
                  {notificationSettings.rateLimit.enabled && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Max per Hour</Label>
                        <Input
                          type="number"
                          value={notificationSettings.rateLimit.maxPerHour}
                          onChange={(e) => {
                            setNotificationSettings(prev => ({
                              ...prev,
                              rateLimit: { ...prev.rateLimit, maxPerHour: parseInt(e.target.value) }
                            }));
                            setHasChanges(true);
                          }}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Max per Day</Label>
                        <Input
                          type="number"
                          value={notificationSettings.rateLimit.maxPerDay}
                          onChange={(e) => {
                            setNotificationSettings(prev => ({
                              ...prev,
                              rateLimit: { ...prev.rateLimit, maxPerDay: parseInt(e.target.value) }
                            }));
                            setHasChanges(true);
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}