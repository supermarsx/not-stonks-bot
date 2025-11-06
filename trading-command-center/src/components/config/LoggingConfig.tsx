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
import { Progress } from '@/components/ui/progress';
import { 
  FileText, 
  Database, 
  Cloud, 
  AlertCircle,
  Settings,
  Activity,
  Search,
  Download,
  Trash2,
  Filter,
  Eye,
  Clock,
  HardDrive,
  Wifi
} from 'lucide-react';

interface LogDestination {
  id: string;
  name: string;
  type: 'console' | 'file' | 'database' | 'remote' | 'elasticsearch';
  enabled: boolean;
  config: {
    path?: string;
    format: 'json' | 'text' | 'csv';
    maxSize: number;
    maxFiles: number;
    compression: boolean;
    endpoint?: string;
    apiKey?: string;
  };
  filters: {
    minLevel: string;
    categories: string[];
    excludePatterns: string[];
  };
  performance: {
    logCount: number;
    totalSize: number;
    lastWrite: string;
    errorCount: number;
  };
}

interface LogSettings {
  enabled: boolean;
  level: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  format: 'json' | 'text' | 'pretty';
  timestamp: boolean;
  colorize: boolean;
  bufferSize: number;
  flushInterval: number;
  enableMetrics: boolean;
  performanceLogging: boolean;
  structuredLogging: boolean;
}

interface LogMetrics {
  totalLogs: number;
  logRate: number; // logs per second
  storageUsed: number;
  storageLimit: number;
  errorsToday: number;
  avgLogSize: number;
  retentionDays: number;
}

const LoggingConfig: React.FC = () => {
  const [logSettings, setLogSettings] = useState<LogSettings>({
    enabled: true,
    level: 'info',
    format: 'json',
    timestamp: true,
    colorize: false,
    bufferSize: 1000,
    flushInterval: 5,
    enableMetrics: true,
    performanceLogging: true,
    structuredLogging: true
  });

  const [logDestinations, setLogDestinations] = useState<LogDestination[]>([
    {
      id: 'console_default',
      name: 'Console Output',
      type: 'console',
      enabled: true,
      config: {
        format: 'text',
        maxSize: 0,
        maxFiles: 0,
        compression: false
      },
      filters: {
        minLevel: 'info',
        categories: ['trading', 'risk', 'system'],
        excludePatterns: []
      },
      performance: {
        logCount: 1247,
        totalSize: 0,
        lastWrite: new Date().toISOString(),
        errorCount: 0
      }
    },
    {
      id: 'file_default',
      name: 'Application Logs',
      type: 'file',
      enabled: true,
      config: {
        path: './logs/app.log',
        format: 'json',
        maxSize: 100, // MB
        maxFiles: 30,
        compression: true
      },
      filters: {
        minLevel: 'debug',
        categories: ['all'],
        excludePatterns: []
      },
      performance: {
        logCount: 8934,
        totalSize: 45.6, // MB
        lastWrite: new Date().toISOString(),
        errorCount: 12
      }
    },
    {
      id: 'database_default',
      name: 'Database Storage',
      type: 'database',
      enabled: false,
      config: {
        format: 'json',
        maxSize: 0,
        maxFiles: 0,
        compression: false
      },
      filters: {
        minLevel: 'info',
        categories: ['trading', 'risk'],
        excludePatterns: []
      },
      performance: {
        logCount: 0,
        totalSize: 0,
        lastWrite: '',
        errorCount: 0
      }
    }
  ]);

  const [logMetrics, setLogMetrics] = useState<LogMetrics>({
    totalLogs: 15632,
    logRate: 2.3,
    storageUsed: 234.7, // MB
    storageLimit: 1000, // MB
    errorsToday: 24,
    avgLogSize: 1.2, // KB
    retentionDays: 30
  });

  const [activeDestination, setActiveDestination] = useState<string>('console_default');
  const [searchQuery, setSearchQuery] = useState('');
  const [logFilter, setLogFilter] = useState('all');
  const [saving, setSaving] = useState(false);

  const logLevels = [
    { value: 'debug', label: 'Debug', description: 'Detailed debugging information' },
    { value: 'info', label: 'Info', description: 'General information messages' },
    { value: 'warning', label: 'Warning', description: 'Warning messages' },
    { value: 'error', label: 'Error', description: 'Error messages' },
    { value: 'critical', label: 'Critical', description: 'Critical system errors' }
  ];

  const logCategories = [
    'trading', 'risk', 'strategy', 'broker', 'ai', 'system', 'performance', 'security', 'database', 'network'
  ];

  useEffect(() => {
    loadLoggingConfiguration();
  }, []);

  const loadLoggingConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getLoggingConfig();
      if (config) {
        setLogSettings(config.settings || logSettings);
        setLogDestinations(config.destinations || logDestinations);
        setLogMetrics(config.metrics || logMetrics);
      }
    } catch (error) {
      console.error('Failed to load logging configuration:', error);
    }
  };

  const saveLoggingConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.saveLoggingConfig({
        settings: logSettings,
        destinations: logDestinations,
        metrics: logMetrics
      });
    } catch (error) {
      console.error('Failed to save logging configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const addLogDestination = () => {
    const newDestination: LogDestination = {
      id: `dest_${Date.now()}`,
      name: 'New Destination',
      type: 'file',
      enabled: false,
      config: {
        format: 'json',
        maxSize: 50,
        maxFiles: 10,
        compression: false
      },
      filters: {
        minLevel: 'info',
        categories: ['all'],
        excludePatterns: []
      },
      performance: {
        logCount: 0,
        totalSize: 0,
        lastWrite: '',
        errorCount: 0
      }
    };
    setLogDestinations([...logDestinations, newDestination]);
    setActiveDestination(newDestination.id);
  };

  const updateLogDestination = (id: string, updates: Partial<LogDestination>) => {
    setLogDestinations(destinations => 
      destinations.map(dest => dest.id === id ? { ...dest, ...updates } : dest)
    );
  };

  const deleteLogDestination = (id: string) => {
    setLogDestinations(destinations => destinations.filter(dest => dest.id !== id));
    if (activeDestination === id) {
      const remaining = logDestinations.filter(d => d.id !== id);
      setActiveDestination(remaining.length > 0 ? remaining[0].id : '');
    }
  };

  const clearLogs = async (destinationId: string) => {
    try {
      await window.electronAPI?.clearLogs(destinationId);
      const destination = logDestinations.find(d => d.id === destinationId);
      if (destination) {
        updateLogDestination(destinationId, {
          performance: {
            ...destination.performance,
            logCount: 0,
            totalSize: 0,
            lastWrite: new Date().toISOString(),
            errorCount: 0
          }
        });
      }
    } catch (error) {
      console.error('Failed to clear logs:', error);
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'console':
        return <Settings className="h-4 w-4" />;
      case 'file':
        return <FileText className="h-4 w-4" />;
      case 'database':
        return <Database className="h-4 w-4" />;
      case 'remote':
        return <Cloud className="h-4 w-4" />;
      case 'elasticsearch':
        return <Search className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'debug':
        return 'text-gray-600';
      case 'info':
        return 'text-blue-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      case 'critical':
        return 'text-red-800';
      default:
        return 'text-gray-600';
    }
  };

  const activeDestinationData = logDestinations.find(d => d.id === activeDestination);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Logging & Monitoring Configuration
          </CardTitle>
          <CardDescription>
            Configure log levels, destinations, and monitoring settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={logSettings.enabled}
                  onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, enabled: checked }))}
                />
                <Label>Logging System {logSettings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant="outline">
                {logSettings.level.toUpperCase()} Level
              </Badge>
            </div>
            <Button onClick={saveLoggingConfiguration} disabled={saving}>
              <Settings className="h-4 w-4 mr-2" />
              {saving ? 'Saving...' : 'Save Configuration'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Logs</p>
                <p className="text-2xl font-bold">{logMetrics.totalLogs.toLocaleString()}</p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Log Rate</p>
                <p className="text-2xl font-bold">{logMetrics.logRate}/s</p>
              </div>
              <Clock className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Storage Used</p>
                <p className="text-2xl font-bold">{logMetrics.storageUsed.toFixed(1)}MB</p>
                <Progress value={(logMetrics.storageUsed / logMetrics.storageLimit) * 100} className="mt-1 h-1" />
              </div>
              <HardDrive className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Errors Today</p>
                <p className="text-2xl font-bold text-red-600">{logMetrics.errorsToday}</p>
              </div>
              <AlertCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="destinations" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="destinations">Destinations</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
          <TabsTrigger value="filters">Filters</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        <TabsContent value="destinations" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Destinations List */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Log Destinations</CardTitle>
                    <CardDescription>
                      Configure where logs are stored and sent
                    </CardDescription>
                  </div>
                  <Button onClick={addLogDestination} size="sm">
                    Add Destination
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {logDestinations.map((destination) => (
                  <div
                    key={destination.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      activeDestination === destination.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setActiveDestination(destination.id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(destination.type)}
                        <h3 className="font-medium">{destination.name}</h3>
                        <Badge variant={destination.enabled ? 'default' : 'outline'}>
                          {destination.enabled ? 'Active' : 'Inactive'}
                        </Badge>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            clearLogs(destination.id);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteLogDestination(destination.id);
                          }}
                        >
                          <Download className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Type:</span>
                        <div className="font-medium capitalize">{destination.type}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Log Count:</span>
                        <div className="font-medium">{destination.performance.logCount}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Size:</span>
                        <div className="font-medium">{destination.performance.totalSize.toFixed(1)}MB</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Errors:</span>
                        <div className="font-medium">{destination.performance.errorCount}</div>
                      </div>
                    </div>

                    <div className="mt-2 text-xs text-muted-foreground">
                      Last write: {destination.performance.lastWrite ? new Date(destination.performance.lastWrite).toLocaleString() : 'Never'}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Destination Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Destination Configuration</CardTitle>
                <CardDescription>
                  {activeDestinationData ? 'Configure destination settings and parameters' : 'Select a destination to configure'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeDestinationData ? (
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="destination-name">Name</Label>
                      <Input
                        id="destination-name"
                        value={activeDestinationData.name}
                        onChange={(e) => updateLogDestination(activeDestinationData.id, { name: e.target.value })}
                      />
                    </div>

                    <div>
                      <Label htmlFor="destination-type">Type</Label>
                      <Select
                        value={activeDestinationData.type}
                        onValueChange={(value: any) => updateLogDestination(activeDestinationData.id, { type: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="console">Console Output</SelectItem>
                          <SelectItem value="file">File</SelectItem>
                          <SelectItem value="database">Database</SelectItem>
                          <SelectItem value="remote">Remote Server</SelectItem>
                          <SelectItem value="elasticsearch">Elasticsearch</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable Destination</Label>
                        <p className="text-sm text-muted-foreground">
                          Send logs to this destination
                        </p>
                      </div>
                      <Switch
                        checked={activeDestinationData.enabled}
                        onCheckedChange={(checked) => updateLogDestination(activeDestinationData.id, { enabled: checked })}
                      />
                    </div>

                    <Separator />

                    <div>
                      <Label>Log Format</Label>
                      <Select
                        value={activeDestinationData.config.format}
                        onValueChange={(value: any) => updateLogDestination(activeDestinationData.id, {
                          config: { ...activeDestinationData.config, format: value }
                        })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="json">JSON</SelectItem>
                          <SelectItem value="text">Plain Text</SelectItem>
                          <SelectItem value="csv">CSV</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {(activeDestinationData.type === 'file' || activeDestinationData.type === 'remote') && (
                      <>
                        <div>
                          <Label htmlFor="file-path">File Path</Label>
                          <Input
                            id="file-path"
                            value={activeDestinationData.config.path || ''}
                            onChange={(e) => updateLogDestination(activeDestinationData.id, {
                              config: { ...activeDestinationData.config, path: e.target.value }
                            })}
                            placeholder="./logs/app.log"
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label>Max Size (MB)</Label>
                            <Input
                              type="number"
                              value={activeDestinationData.config.maxSize}
                              onChange={(e) => updateLogDestination(activeDestinationData.id, {
                                config: { ...activeDestinationData.config, maxSize: Number(e.target.value) }
                              })}
                            />
                          </div>
                          <div>
                            <Label>Max Files</Label>
                            <Input
                              type="number"
                              value={activeDestinationData.config.maxFiles}
                              onChange={(e) => updateLogDestination(activeDestinationData.id, {
                                config: { ...activeDestinationData.config, maxFiles: Number(e.target.value) }
                              })}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between">
                          <Label>Compression</Label>
                          <Switch
                            checked={activeDestinationData.config.compression}
                            onCheckedChange={(checked) => updateLogDestination(activeDestinationData.id, {
                              config: { ...activeDestinationData.config, compression: checked }
                            })}
                          />
                        </div>
                      </>
                    )}

                    {(activeDestinationData.type === 'remote' || activeDestinationData.type === 'elasticsearch') && (
                      <>
                        <Separator />
                        <div>
                          <Label htmlFor="endpoint">Endpoint URL</Label>
                          <Input
                            id="endpoint"
                            value={activeDestinationData.config.endpoint || ''}
                            onChange={(e) => updateLogDestination(activeDestinationData.id, {
                              config: { ...activeDestinationData.config, endpoint: e.target.value }
                            })}
                            placeholder="https://api.example.com/logs"
                          />
                        </div>
                        <div>
                          <Label htmlFor="api-key">API Key</Label>
                          <Input
                            id="api-key"
                            type="password"
                            value={activeDestinationData.config.apiKey || ''}
                            onChange={(e) => updateLogDestination(activeDestinationData.id, {
                              config: { ...activeDestinationData.config, apiKey: e.target.value }
                            })}
                            placeholder="Enter API key"
                          />
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Select a destination to configure its settings</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Global Settings</CardTitle>
                <CardDescription>
                  Configure global logging behavior
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="log-level">Minimum Log Level</Label>
                  <Select
                    value={logSettings.level}
                    onValueChange={(value: any) => setLogSettings(prev => ({ ...prev, level: value }))}
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
                  <p className="text-xs text-muted-foreground mt-1">
                    {logLevels.find(l => l.value === logSettings.level)?.description}
                  </p>
                </div>

                <div>
                  <Label htmlFor="log-format">Log Format</Label>
                  <Select
                    value={logSettings.format}
                    onValueChange={(value: any) => setLogSettings(prev => ({ ...prev, format: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="json">JSON</SelectItem>
                      <SelectItem value="text">Plain Text</SelectItem>
                      <SelectItem value="pretty">Pretty Print</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <Label>Show Timestamps</Label>
                    <Switch
                      checked={logSettings.timestamp}
                      onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, timestamp: checked }))}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Colorize Output</Label>
                    <Switch
                      checked={logSettings.colorize}
                      onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, colorize: checked }))}
                    />
                  </div>
                </div>

                <Separator />

                <div>
                  <Label>Buffer Size: {logSettings.bufferSize}</Label>
                  <Slider
                    value={[logSettings.bufferSize]}
                    onValueChange={([value]) => setLogSettings(prev => ({ ...prev, bufferSize: value }))}
                    min={100}
                    max={10000}
                    step={100}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Flush Interval: {logSettings.flushInterval}s</Label>
                  <Slider
                    value={[logSettings.flushInterval]}
                    onValueChange={([value]) => setLogSettings(prev => ({ ...prev, flushInterval: value }))}
                    min={1}
                    max={60}
                    step={1}
                    className="mt-2"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Advanced Features</CardTitle>
                <CardDescription>
                  Enable additional logging capabilities
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Metrics Logging</Label>
                    <p className="text-sm text-muted-foreground">
                      Log system performance metrics
                    </p>
                  </div>
                  <Switch
                    checked={logSettings.enableMetrics}
                    onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, enableMetrics: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Performance Logging</Label>
                    <p className="text-sm text-muted-foreground">
                      Log function execution times
                    </p>
                  </div>
                  <Switch
                    checked={logSettings.performanceLogging}
                    onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, performanceLogging: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Structured Logging</Label>
                    <p className="text-sm text-muted-foreground">
                      Use structured data in logs
                    </p>
                  </div>
                  <Switch
                    checked={logSettings.structuredLogging}
                    onCheckedChange={(checked) => setLogSettings(prev => ({ ...prev, structuredLogging: checked }))}
                  />
                </div>

                <Separator />

                <div>
                  <Label htmlFor="retention-days">Retention Period (days)</Label>
                  <Input
                    id="retention-days"
                    type="number"
                    value={logMetrics.retentionDays}
                    onChange={(e) => setLogMetrics(prev => ({ ...prev, retentionDays: Number(e.target.value) }))}
                  />
                </div>

                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Longer retention periods will increase storage requirements but provide more historical data.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="filters" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Log Filters</CardTitle>
              <CardDescription>
                Configure what logs are captured and processed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {activeDestinationData && (
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="min-level">Minimum Log Level</Label>
                    <Select
                      value={activeDestinationData.filters.minLevel}
                      onValueChange={(value) => updateLogDestination(activeDestinationData.id, {
                        filters: { ...activeDestinationData.filters, minLevel: value }
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

                  <div>
                    <Label>Log Categories</Label>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2">
                      {logCategories.map((category) => (
                        <div key={category} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id={`category-${category}`}
                            checked={activeDestinationData.filters.categories.includes(category) || activeDestinationData.filters.categories.includes('all')}
                            onChange={(e) => {
                              const categories = activeDestinationData.filters.categories;
                              const newCategories = e.target.checked
                                ? [...categories.filter(c => c !== 'all'), category]
                                : categories.filter(c => c !== category && c !== 'all');
                              updateLogDestination(activeDestinationData.id, {
                                filters: { ...activeDestinationData.filters, categories: newCategories }
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

                  <div>
                    <Label htmlFor="exclude-patterns">Exclude Patterns</Label>
                    <Textarea
                      id="exclude-patterns"
                      value={activeDestinationData.filters.excludePatterns.join('\n')}
                      onChange={(e) => {
                        const patterns = e.target.value.split('\n').filter(p => p.trim());
                        updateLogDestination(activeDestinationData.id, {
                          filters: { ...activeDestinationData.filters, excludePatterns: patterns }
                        });
                      }}
                      placeholder="Enter patterns to exclude (one per line)"
                      rows={3}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Real-time Logs</CardTitle>
                <CardDescription>
                  Live log stream and search
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      placeholder="Search logs..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    <Select value={logFilter} onValueChange={setLogFilter}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Levels</SelectItem>
                        {logLevels.map((level) => (
                          <SelectItem key={level.value} value={level.value}>
                            {level.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button variant="outline" size="icon">
                      <Filter className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto">
                    <div className="space-y-1">
                      <div className="text-gray-400">[2024-01-15 10:30:15] INFO [trading] Position opened: AAPL</div>
                      <div className="text-blue-400">[2024-01-15 10:30:20] DEBUG [strategy] Calculating momentum indicators</div>
                      <div className="text-yellow-400">[2024-01-15 10:30:25] WARNING [broker] High latency detected: 250ms</div>
                      <div className="text-gray-400">[2024-01-15 10:30:30] INFO [risk] Risk check passed</div>
                      <div className="text-green-400">[2024-01-15 10:30:35] INFO [trading] Order executed successfully</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Log Analytics</CardTitle>
                <CardDescription>
                  Log statistics and insights
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Avg Log Size</div>
                    <div className="text-2xl font-bold">{logMetrics.avgLogSize}KB</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Retention</div>
                    <div className="text-2xl font-bold">{logMetrics.retentionDays}d</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Storage Usage</span>
                    <span>{((logMetrics.storageUsed / logMetrics.storageLimit) * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={(logMetrics.storageUsed / logMetrics.storageLimit) * 100} className="h-2" />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Log Rate</span>
                    <span>{logMetrics.logRate} logs/sec</span>
                  </div>
                  <Progress value={(logMetrics.logRate / 10) * 100} className="h-2" />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Error Rate</span>
                    <span>{((logMetrics.errorsToday / logMetrics.totalLogs) * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={(logMetrics.errorsToday / logMetrics.totalLogs) * 100} className="h-2" />
                </div>

                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    All systems are logging normally with no critical issues detected.
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

export default LoggingConfig;