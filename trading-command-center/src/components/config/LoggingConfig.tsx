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
  Filter 
} from 'lucide-react';

export interface LogFile {
  id: string;
  name: string;
  size: number;
  lastModified: Date;
  level: 'debug' | 'info' | 'warn' | 'error';
  path: string;
}

export interface LoggingSettings {
  global: {
    enabled: boolean;
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text' | 'xml';
    timestamp: boolean;
  };
  file: {
    enabled: boolean;
    path: string;
    maxSize: number;
    maxFiles: number;
    rotation: 'size' | 'time' | 'never';
    compression: boolean;
  };
  database: {
    enabled: boolean;
    connection: string;
    table: string;
    batchSize: number;
  };
  cloud: {
    enabled: boolean;
    provider: 'aws' | 'gcp' | 'azure';
    bucket: string;
    region: string;
    accessKey: string;
    retentionDays: number;
  };
  filters: {
    enabled: boolean;
    excludePatterns: string[];
    includePatterns: string[];
    sensitiveData: boolean;
  };
  retention: {
    enabled: boolean;
    autoDelete: boolean;
    maxAge: number;
    maxSize: number;
  };
}

const defaultLoggingSettings: LoggingSettings = {
  global: {
    enabled: true,
    level: 'info',
    format: 'json',
    timestamp: true
  },
  file: {
    enabled: true,
    path: './logs',
    maxSize: 100,
    maxFiles: 10,
    rotation: 'size',
    compression: true
  },
  database: {
    enabled: false,
    connection: '',
    table: 'logs',
    batchSize: 100
  },
  cloud: {
    enabled: false,
    provider: 'aws',
    bucket: '',
    region: 'us-east-1',
    accessKey: '',
    retentionDays: 30
  },
  filters: {
    enabled: true,
    excludePatterns: ['*.tmp', '*.log.bak'],
    includePatterns: ['*.log', '*.txt'],
    sensitiveData: true
  },
  retention: {
    enabled: true,
    autoDelete: true,
    maxAge: 30,
    maxSize: 1000
  }
};

const sampleLogFiles: LogFile[] = [
  {
    id: '1',
    name: 'application.log',
    size: 1024 * 1024 * 5, // 5MB
    lastModified: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    level: 'info',
    path: './logs/application.log'
  },
  {
    id: '2',
    name: 'error.log',
    size: 1024 * 1024 * 2, // 2MB
    lastModified: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
    level: 'error',
    path: './logs/error.log'
  },
  {
    id: '3',
    name: 'debug.log',
    size: 1024 * 1024 * 10, // 10MB
    lastModified: new Date(Date.now() - 1000 * 60 * 10), // 10 minutes ago
    level: 'debug',
    path: './logs/debug.log'
  }
];

export default function LoggingConfig() {
  const [loggingSettings, setLoggingSettings] = useState<LoggingSettings>(defaultLoggingSettings);
  const [logFiles, setLogFiles] = useState<LogFile[]>(sampleLogFiles);
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [selectedLog, setSelectedLog] = useState<LogFile | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterLevel, setFilterLevel] = useState<string>('all');
  const [logViewer, setLogViewer] = useState<string>('');
  const [diskUsage, setDiskUsage] = useState({
    used: 45.2,
    total: 100,
    percentage: 45.2
  });

  useEffect(() => {
    if (selectedLog) {
      // Simulate loading log content
      const sampleContent = [
        `${new Date().toISOString()} [INFO] Application started successfully`,
        `${new Date(Date.now() - 1000).toISOString()} [DEBUG] Loading configuration from file`,
        `${new Date(Date.now() - 2000).toISOString()} [WARN] API response time is high: 2500ms`,
        `${new Date(Date.now() - 3000).toISOString()} [ERROR] Database connection failed, retrying in 5 seconds`,
        `${new Date(Date.now() - 4000).toISOString()} [INFO] Database connection restored`
      ].join('\n');
      setLogViewer(sampleContent);
    }
  }, [selectedLog]);

  const handleGlobalChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      global: {
        ...prev.global,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleFileChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      file: {
        ...prev.file,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleDatabaseChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      database: {
        ...prev.database,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleCloudChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      cloud: {
        ...prev.cloud,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleFiltersChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      filters: {
        ...prev.filters,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleRetentionChange = (field: string, value: any) => {
    setLoggingSettings(prev => ({
      ...prev,
      retention: {
        ...prev.retention,
        [field]: value
      }
    }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save logging settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadLog = (logFile: LogFile) => {
    // Simulate download
    const element = document.createElement('a');
    const file = new Blob([logViewer], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = logFile.name;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleDeleteLog = (logFile: LogFile) => {
    setLogFiles(prev => prev.filter(file => file.id !== logFile.id));
    if (selectedLog?.id === logFile.id) {
      setSelectedLog(null);
      setLogViewer('');
    }
  };

  const handleSearch = () => {
    if (!selectedLog) return;
    
    // Simulate search in log content
    const matches = logViewer.split('\n').filter(line => 
      line.toLowerCase().includes(searchQuery.toLowerCase())
    );
    
    if (matches.length > 0) {
      setLogViewer(matches.join('\n'));
    }
  };

  const filteredLogFiles = logFiles.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         file.path.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesLevel = filterLevel === 'all' || file.level === filterLevel;
    return matchesSearch && matchesLevel;
  });

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'debug': return 'bg-gray-100 text-gray-800';
      case 'info': return 'bg-blue-100 text-blue-800';
      case 'warn': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FileText className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Logging Configuration</h2>
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

      {/* Disk Usage Alert */}
      {diskUsage.percentage > 80 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Disk usage is at {diskUsage.percentage}%. Consider cleaning up old log files or increasing disk space.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm text-muted-foreground">Total Log Files</p>
                <p className="text-2xl font-bold">{logFiles.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm text-muted-foreground">Active Logs</p>
                <p className="text-2xl font-bold">
                  {logFiles.filter(f => Date.now() - f.lastModified.getTime() < 24 * 60 * 60 * 1000).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Cloud className="h-5 w-5 text-purple-600" />
              <div>
                <p className="text-sm text-muted-foreground">Disk Usage</p>
                <div className="flex items-center space-x-2">
                  <Progress value={diskUsage.percentage} className="flex-1" />
                  <span className="text-sm font-medium">{diskUsage.percentage}%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="viewer" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="viewer">Log Viewer</TabsTrigger>
          <TabsTrigger value="configuration">Configuration</TabsTrigger>
          <TabsTrigger value="storage">Storage</TabsTrigger>
          <TabsTrigger value="filters">Filters</TabsTrigger>
        </TabsList>

        <TabsContent value="viewer" className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            <div className="col-span-1 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Search & Filter</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Search</Label>
                    <div className="flex space-x-2">
                      <Input
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search logs..."
                      />
                      <Button variant="outline" size="icon" onClick={handleSearch}>
                        <Search className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Log Level</Label>
                    <Select value={filterLevel} onValueChange={setFilterLevel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Levels</SelectItem>
                        <SelectItem value="debug">Debug</SelectItem>
                        <SelectItem value="info">Info</SelectItem>
                        <SelectItem value="warn">Warning</SelectItem>
                        <SelectItem value="error">Error</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <Label>Log Files</Label>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {filteredLogFiles.map((file) => (
                        <div
                          key={file.id}
                          className={`p-3 border rounded cursor-pointer transition-colors ${
                            selectedLog?.id === file.id ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                          }`}
                          onClick={() => setSelectedLog(file)}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium text-sm">{file.name}</span>
                            <Badge className={getLogLevelColor(file.level)} variant="secondary">
                              {file.level}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            {formatFileSize(file.size)}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {file.lastModified.toLocaleString()}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <div className="col-span-3">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                      <FileText className="h-5 w-5" />
                      <span>
                        {selectedLog ? selectedLog.name : 'Select a log file'}
                      </span>
                    </CardTitle>
                    {selectedLog && (
                      <div className="flex space-x-2">
                        <Button variant="outline" size="sm" onClick={() => handleDownloadLog(selectedLog)}>
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => handleDeleteLog(selectedLog)}>
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </Button>
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {selectedLog ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Level:</span>
                          <Badge className={getLogLevelColor(selectedLog.level)} variant="secondary" ml-2>
                            {selectedLog.level}
                          </Badge>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Size:</span>
                          <span className="ml-2">{formatFileSize(selectedLog.size)}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Modified:</span>
                          <span className="ml-2">{selectedLog.lastModified.toLocaleString()}</span>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="bg-black text-green-400 p-4 rounded font-mono text-sm max-h-96 overflow-y-auto">
                        <pre className="whitespace-pre-wrap">{logViewer}</pre>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Select a log file from the list to view its contents</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="configuration" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Global Settings</CardTitle>
              <CardDescription>
                Configure basic logging behavior
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={loggingSettings.global.enabled}
                  onCheckedChange={(checked) => handleGlobalChange('enabled', checked)}
                />
                <Label>Enable Logging</Label>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Log Level</Label>
                  <Select
                    value={loggingSettings.global.level}
                    onValueChange={(value) => handleGlobalChange('level', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="debug">Debug</SelectItem>
                      <SelectItem value="info">Info</SelectItem>
                      <SelectItem value="warn">Warning</SelectItem>
                      <SelectItem value="error">Error</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Format</Label>
                  <Select
                    value={loggingSettings.global.format}
                    onValueChange={(value) => handleGlobalChange('format', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="json">JSON</SelectItem>
                      <SelectItem value="text">Text</SelectItem>
                      <SelectItem value="xml">XML</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center space-x-2 pt-6">
                  <Switch
                    checked={loggingSettings.global.timestamp}
                    onCheckedChange={(checked) => handleGlobalChange('timestamp', checked)}
                  />
                  <Label>Include Timestamp</Label>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="storage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>File Storage</CardTitle>
              <CardDescription>
                Configure local file logging
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={loggingSettings.file.enabled}
                  onCheckedChange={(checked) => handleFileChange('enabled', checked)}
                />
                <Label>Enable File Logging</Label>
              </div>

              {loggingSettings.file.enabled && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Log Directory</Label>
                    <Input
                      value={loggingSettings.file.path}
                      onChange={(e) => handleFileChange('path', e.target.value)}
                      placeholder="./logs"
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Max File Size (MB)</Label>
                      <Input
                        type="number"
                        value={loggingSettings.file.maxSize}
                        onChange={(e) => handleFileChange('maxSize', parseInt(e.target.value))}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Max Files</Label>
                      <Input
                        type="number"
                        value={loggingSettings.file.maxFiles}
                        onChange={(e) => handleFileChange('maxFiles', parseInt(e.target.value))}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Rotation</Label>
                      <Select
                        value={loggingSettings.file.rotation}
                        onValueChange={(value) => handleFileChange('rotation', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="size">By Size</SelectItem>
                          <SelectItem value="time">By Time</SelectItem>
                          <SelectItem value="never">Never</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={loggingSettings.file.compression}
                      onCheckedChange={(checked) => handleFileChange('compression', checked)}
                    />
                    <Label>Compress Old Files</Label>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Database Storage</CardTitle>
              <CardDescription>
                Store logs in a database for querying and analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={loggingSettings.database.enabled}
                  onCheckedChange={(checked) => handleDatabaseChange('enabled', checked)}
                />
                <Label>Enable Database Logging</Label>
              </div>

              {loggingSettings.database.enabled && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Connection String</Label>
                    <Input
                      value={loggingSettings.database.connection}
                      onChange={(e) => handleDatabaseChange('connection', e.target.value)}
                      placeholder="postgresql://user:pass@localhost/dbname"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Table Name</Label>
                      <Input
                        value={loggingSettings.database.table}
                        onChange={(e) => handleDatabaseChange('table', e.target.value)}
                        placeholder="logs"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input
                        type="number"
                        value={loggingSettings.database.batchSize}
                        onChange={(e) => handleDatabaseChange('batchSize', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="filters" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Content Filters</CardTitle>
              <CardDescription>
                Configure log filtering and data protection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={loggingSettings.filters.enabled}
                  onCheckedChange={(checked) => handleFiltersChange('enabled', checked)}
                />
                <Label>Enable Filters</Label>
              </div>

              {loggingSettings.filters.enabled && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Exclude Patterns</Label>
                    <Textarea
                      value={loggingSettings.filters.excludePatterns.join('\n')}
                      onChange={(e) => handleFiltersChange('excludePatterns', e.target.value.split('\n'))}
                      placeholder="*.tmp&#10;*.log.bak&#10;/secret/"
                      rows={3}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Include Patterns</Label>
                    <Textarea
                      value={loggingSettings.filters.includePatterns.join('\n')}
                      onChange={(e) => handleFiltersChange('includePatterns', e.target.value.split('\n'))}
                      placeholder="*.log&#10;*.txt&#10;/logs/"
                      rows={3}
                    />
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={loggingSettings.filters.sensitiveData}
                      onCheckedChange={(checked) => handleFiltersChange('sensitiveData', checked)}
                    />
                    <Label>Filter Sensitive Data</Label>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Retention Policy</CardTitle>
              <CardDescription>
                Configure automatic log cleanup and archival
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={loggingSettings.retention.enabled}
                  onCheckedChange={(checked) => handleRetentionChange('enabled', checked)}
                />
                <Label>Enable Retention Policy</Label>
              </div>

              {loggingSettings.retention.enabled && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={loggingSettings.retention.autoDelete}
                      onCheckedChange={(checked) => handleRetentionChange('autoDelete', checked)}
                    />
                    <Label>Auto Delete Old Logs</Label>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Max Age (days)</Label>
                      <Input
                        type="number"
                        value={loggingSettings.retention.maxAge}
                        onChange={(e) => handleRetentionChange('maxAge', parseInt(e.target.value))}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Max Total Size (MB)</Label>
                      <Input
                        type="number"
                        value={loggingSettings.retention.maxSize}
                        onChange={(e) => handleRetentionChange('maxSize', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}