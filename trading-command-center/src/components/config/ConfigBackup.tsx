import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { 
  Archive, 
  Cloud, 
  Download, 
  Upload,
  Clock,
  RotateCcw,
  Trash2,
  Settings,
  CheckCircle,
  AlertTriangle,
  XCircle,
  HardDrive,
  Shield,
  RefreshCw,
  Plus,
  FolderOpen
} from 'lucide-react';

interface BackupConfig {
  autoBackup: boolean;
  schedule: 'daily' | 'weekly' | 'monthly';
  time: string; // HH:MM format
  retentionDays: number;
  includeSecrets: boolean;
  compress: boolean;
  encrypt: boolean;
  cloudSync: boolean;
  cloudProvider: 'local' | 's3' | 'gdrive' | 'onedrive';
  cloudPath: string;
  maxBackups: number;
}

interface BackupRecord {
  id: string;
  name: string;
  timestamp: string;
  size: number; // bytes
  type: 'manual' | 'scheduled' | 'auto';
  status: 'completed' | 'failed' | 'in_progress' | 'corrupted';
  checksum: string;
  path: string;
  encrypted: boolean;
  compressed: boolean;
  containsSecrets: boolean;
  description?: string;
}

interface RestoreResult {
  success: boolean;
  restoredItems: number;
  totalItems: number;
  errors: string[];
  warnings: string[];
  timestamp: string;
}

const ConfigBackup: React.FC = () => {
  const [backupConfig, setBackupConfig] = useState<BackupConfig>({
    autoBackup: true,
    schedule: 'daily',
    time: '02:00',
    retentionDays: 30,
    includeSecrets: false,
    compress: true,
    encrypt: true,
    cloudSync: false,
    cloudProvider: 'local',
    cloudPath: './backups',
    maxBackups: 10
  });

  const [backupRecords, setBackupRecords] = useState<BackupRecord[]>([
    {
      id: 'backup_1',
      name: 'Daily Backup - 2024-01-15',
      timestamp: new Date(Date.now() - 86400000).toISOString(),
      size: 2048576, // 2MB
      type: 'scheduled',
      status: 'completed',
      checksum: 'sha256:abc123...',
      path: './backups/daily-2024-01-15.bak',
      encrypted: true,
      compressed: true,
      containsSecrets: false
    },
    {
      id: 'backup_2',
      name: 'Manual Backup - Pre Update',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      size: 1536000, // 1.5MB
      type: 'manual',
      status: 'completed',
      checksum: 'sha256:def456...',
      path: './backups/manual-pre-update.bak',
      encrypted: true,
      compressed: false,
      containsSecrets: false,
      description: 'Backup before system update'
    },
    {
      id: 'backup_3',
      name: 'Daily Backup - 2024-01-14',
      timestamp: new Date(Date.now() - 172800000).toISOString(),
      size: 1898496, // 1.8MB
      type: 'scheduled',
      status: 'completed',
      checksum: 'sha256:ghi789...',
      path: './backups/daily-2024-01-14.bak',
      encrypted: true,
      compressed: true,
      containsSecrets: false
    }
  ]);

  const [isCreatingBackup, setIsCreatingBackup] = useState(false);
  const [isRestoringBackup, setIsRestoringBackup] = useState(false);
  const [selectedBackup, setSelectedBackup] = useState<string | null>(null);
  const [restoreResult, setRestoreResult] = useState<RestoreResult | null>(null);
  const [backupProgress, setBackupProgress] = useState(0);
  const [restoreProgress, setRestoreProgress] = useState(0);

  useEffect(() => {
    loadBackupConfiguration();
    // Set up backup schedule check
    const interval = setInterval(checkBackupSchedule, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  const loadBackupConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getBackupConfig();
      if (config) {
        setBackupConfig(config);
      }
    } catch (error) {
      console.error('Failed to load backup configuration:', error);
    }
  };

  const saveBackupConfiguration = async () => {
    try {
      await window.electronAPI?.saveBackupConfig(backupConfig);
      alert('Backup configuration saved');
    } catch (error) {
      console.error('Failed to save backup configuration:', error);
      alert('Failed to save backup configuration');
    }
  };

  const createBackup = async (type: 'manual' | 'scheduled' = 'manual') => {
    setIsCreatingBackup(true);
    setBackupProgress(0);

    try {
      // Simulate backup creation process
      for (let i = 0; i <= 100; i += 10) {
        setBackupProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      const config = await window.electronAPI?.getAllConfiguration();
      
      // Create backup data
      let backupData = {
        ...config,
        _backup: true,
        _version: '1.0.0',
        _timestamp: new Date().toISOString(),
        _type: type,
        _config: backupConfig
      };

      // Remove secrets if not included
      if (!backupConfig.includeSecrets) {
        backupData = sanitizeSecrets(backupData);
      }

      // Compress if enabled
      if (backupConfig.compress) {
        backupData = await window.electronAPI?.compressData(JSON.stringify(backupData));
      }

      // Encrypt if enabled
      if (backupConfig.encrypt) {
        backupData = await window.electronAPI?.encryptData(backupData);
      }

      // Save backup file
      const backupName = `${type}-${new Date().toISOString().split('T')[0]}.bak`;
      const backupPath = `${backupConfig.cloudPath}/${backupName}`;
      
      await window.electronAPI?.createBackup(backupData, backupPath);

      // Calculate checksum
      const checksum = await window.electronAPI?.calculateChecksum(backupData);

      // Add to records
      const newBackup: BackupRecord = {
        id: `backup_${Date.now()}`,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Backup - ${new Date().toLocaleDateString()}`,
        timestamp: new Date().toISOString(),
        size: JSON.stringify(backupData).length,
        type,
        status: 'completed',
        checksum: checksum || 'unknown',
        path: backupPath,
        encrypted: backupConfig.encrypt,
        compressed: backupConfig.compress,
        containsSecrets: backupConfig.includeSecrets
      };

      setBackupRecords(prev => [newBackup, ...prev]);

      // Clean up old backups if needed
      if (backupRecords.length >= backupConfig.maxBackups) {
        const sortedRecords = [...backupRecords].sort((a, b) => 
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
        
        const recordsToDelete = sortedRecords.slice(backupConfig.maxBackups);
        for (const record of recordsToDelete) {
          await deleteBackup(record.id);
        }
      }

    } catch (error) {
      console.error('Backup creation failed:', error);
      
      // Add failed backup record
      const failedBackup: BackupRecord = {
        id: `backup_${Date.now()}`,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Backup - Failed`,
        timestamp: new Date().toISOString(),
        size: 0,
        type,
        status: 'failed',
        checksum: '',
        path: '',
        encrypted: false,
        compressed: false,
        containsSecrets: false
      };
      
      setBackupRecords(prev => [failedBackup, ...prev]);
      
      alert('Backup creation failed: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsCreatingBackup(false);
      setBackupProgress(0);
    }
  };

  const restoreBackup = async (backupId: string) => {
    setIsRestoringBackup(true);
    setRestoreProgress(0);
    setRestoreResult(null);

    try {
      const backup = backupRecords.find(b => b.id === backupId);
      if (!backup) {
        throw new Error('Backup not found');
      }

      // Load backup data
      setRestoreProgress(20);
      const backupData = await window.electronAPI?.loadBackup(backup.path);

      if (!backupData) {
        throw new Error('Failed to load backup data');
      }

      // Decrypt if needed
      setRestoreProgress(40);
      let data = backupData;
      if (backup.encrypted) {
        data = await window.electronAPI?.decryptData(backupData);
      }

      // Decompress if needed
      setRestoreProgress(60);
      if (backup.compressed) {
        data = await window.electronAPI?.decompressData(data);
      }

      const config = typeof data === 'string' ? JSON.parse(data) : data;
      
      // Validate backup
      setRestoreProgress(80);
      if (!config._backup) {
        throw new Error('Invalid backup file format');
      }

      // Apply configuration
      await window.electronAPI?.restoreConfiguration(config);

      setRestoreProgress(100);

      setRestoreResult({
        success: true,
        restoredItems: Object.keys(config).length - 3, // Subtract metadata fields
        totalItems: Object.keys(config).length - 3,
        errors: [],
        warnings: [],
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Backup restoration failed:', error);
      
      setRestoreResult({
        success: false,
        restoredItems: 0,
        totalItems: 0,
        errors: [error instanceof Error ? error.message : 'Restoration failed'],
        warnings: [],
        timestamp: new Date().toISOString()
      });
    } finally {
      setIsRestoringBackup(false);
      setRestoreProgress(0);
    }
  };

  const deleteBackup = async (backupId: string) => {
    try {
      const backup = backupRecords.find(b => b.id === backupId);
      if (backup) {
        await window.electronAPI?.deleteBackup(backup.path);
        setBackupRecords(prev => prev.filter(b => b.id !== backupId));
      }
    } catch (error) {
      console.error('Failed to delete backup:', error);
      alert('Failed to delete backup');
    }
  };

  const sanitizeSecrets = (config: any): any => {
    const sanitized = JSON.parse(JSON.stringify(config));
    
    // Remove API keys and secrets from brokers
    if (sanitized.brokers) {
      sanitized.brokers = sanitized.brokers.map((broker: any) => ({
        ...broker,
        config: {
          ...broker.config,
          apiKey: broker.config?.apiKey ? '***REDACTED***' : undefined,
          secretKey: broker.config?.secretKey ? '***REDACTED***' : undefined
        }
      }));
    }

    // Remove AI provider API keys
    if (sanitized.aiSettings?.providers) {
      sanitized.aiSettings.providers = sanitized.aiSettings.providers.map((provider: any) => ({
        ...provider,
        config: {
          ...provider.config,
          apiKey: provider.config?.apiKey ? '***REDACTED***' : undefined
        }
      }));
    }

    return sanitized;
  };

  const checkBackupSchedule = () => {
    if (!backupConfig.autoBackup) return;

    const now = new Date();
    const [scheduledHour, scheduledMinute] = backupConfig.time.split(':').map(Number);
    
    if (now.getHours() === scheduledHour && now.getMinutes() === scheduledMinute) {
      // Check if we've already created a backup today
      const today = new Date().toISOString().split('T')[0];
      const todaysBackups = backupRecords.filter(b => 
        b.timestamp.startsWith(today) && b.type === 'scheduled'
      );
      
      if (todaysBackups.length === 0) {
        createBackup('scheduled');
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'in_progress':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'corrupted':
        return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      completed: 'default',
      failed: 'destructive',
      in_progress: 'secondary',
      corrupted: 'destructive'
    };

    const colors: Record<string, string> = {
      completed: 'bg-green-500',
      failed: 'bg-red-500',
      in_progress: 'bg-blue-500',
      corrupted: 'bg-orange-500'
    };

    return (
      <Badge variant={variants[status] || 'outline'} className="gap-1">
        <div className={`h-2 w-2 rounded-full ${colors[status] || 'bg-gray-400'}`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Archive className="h-5 w-5" />
          Configuration Backup & Restore
        </CardTitle>
        <CardDescription>
          Manage automatic backups, restore configurations, and version control
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Backup Configuration */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <h3 className="text-lg font-medium">Backup Settings</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Automatic Backups</Label>
                  <p className="text-sm text-muted-foreground">
                    Schedule automatic backups
                  </p>
                </div>
                <Switch
                  checked={backupConfig.autoBackup}
                  onCheckedChange={(checked) => setBackupConfig(prev => ({ ...prev, autoBackup: checked }))}
                />
              </div>

              {backupConfig.autoBackup && (
                <>
                  <div>
                    <Label htmlFor="schedule">Schedule</Label>
                    <Select
                      value={backupConfig.schedule}
                      onValueChange={(value: any) => setBackupConfig(prev => ({ ...prev, schedule: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="backup-time">Backup Time</Label>
                    <Input
                      id="backup-time"
                      type="time"
                      value={backupConfig.time}
                      onChange={(e) => setBackupConfig(prev => ({ ...prev, time: e.target.value }))}
                    />
                  </div>
                </>
              )}

              <div>
                <Label>Retention Period: {backupConfig.retentionDays} days</Label>
                <Input
                  type="number"
                  value={backupConfig.retentionDays}
                  onChange={(e) => setBackupConfig(prev => ({ ...prev, retentionDays: Number(e.target.value) }))}
                  min={1}
                  max={365}
                />
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Include Secrets</Label>
                  <p className="text-sm text-muted-foreground">
                    Backup API keys and passwords
                  </p>
                </div>
                <Switch
                  checked={backupConfig.includeSecrets}
                  onCheckedChange={(checked) => setBackupConfig(prev => ({ ...prev, includeSecrets: checked }))}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Compression</Label>
                  <p className="text-sm text-muted-foreground">
                    Compress backup files
                  </p>
                </div>
                <Switch
                  checked={backupConfig.compress}
                  onCheckedChange={(checked) => setBackupConfig(prev => ({ ...prev, compress: checked }))}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Encryption</Label>
                  <p className="text-sm text-muted-foreground">
                    Encrypt backup files
                  </p>
                </div>
                <Switch
                  checked={backupConfig.encrypt}
                  onCheckedChange={(checked) => setBackupConfig(prev => ({ ...prev, encrypt: checked }))}
                />
              </div>

              <div>
                <Label>Max Backups: {backupConfig.maxBackups}</Label>
                <Input
                  type="number"
                  value={backupConfig.maxBackups}
                  onChange={(e) => setBackupConfig(prev => ({ ...prev, maxBackups: Number(e.target.value) }))}
                  min={1}
                  max={100}
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Next scheduled backup: {backupConfig.autoBackup ? `${backupConfig.schedule} at ${backupConfig.time}` : 'Disabled'}
            </p>
            <Button onClick={saveBackupConfiguration} size="sm">
              <Shield className="h-4 w-4 mr-2" />
              Save Settings
            </Button>
          </div>
        </div>

        <Separator />

        {/* Manual Backup */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            <h3 className="text-lg font-medium">Manual Backup</h3>
          </div>

          <div className="flex gap-2">
            <Button 
              onClick={() => createBackup('manual')} 
              disabled={isCreatingBackup}
              className="flex-1"
            >
              {isCreatingBackup ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Archive className="h-4 w-4 mr-2" />
              )}
              {isCreatingBackup ? 'Creating Backup...' : 'Create Manual Backup'}
            </Button>
          </div>

          {isCreatingBackup && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Backup Progress</span>
                <span>{backupProgress}%</span>
              </div>
              <Progress value={backupProgress} className="h-2" />
            </div>
          )}
        </div>

        <Separator />

        {/* Backup History */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              <h3 className="text-lg font-medium">Backup History</h3>
            </div>
            <div className="text-sm text-muted-foreground">
              {backupRecords.length} backup{backupRecords.length !== 1 ? 's' : ''} available
            </div>
          </div>

          <div className="space-y-2">
            {backupRecords.map((backup) => (
              <div key={backup.id} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(backup.status)}
                    <div>
                      <h4 className="font-medium">{backup.name}</h4>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span>{new Date(backup.timestamp).toLocaleString()}</span>
                        <span>•</span>
                        <span>{formatFileSize(backup.size)}</span>
                        <span>•</span>
                        <Badge variant="outline" className="text-xs">
                          {backup.type}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {getStatusBadge(backup.status)}
                    
                    <div className="flex gap-1">
                      {backup.status === 'completed' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => restoreBackup(backup.id)}
                          disabled={isRestoringBackup}
                        >
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      )}
                      
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          // Download backup
                          window.electronAPI?.downloadBackup(backup.path, backup.name + '.bak');
                        }}
                      >
                        <Download className="h-3 w-3" />
                      </Button>
                      
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => deleteBackup(backup.id)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  {backup.encrypted && (
                    <Badge variant="outline" className="text-xs">
                      <Shield className="h-3 w-3 mr-1" />
                      Encrypted
                    </Badge>
                  )}
                  {backup.compressed && (
                    <Badge variant="outline" className="text-xs">
                      <Archive className="h-3 w-3 mr-1" />
                      Compressed
                    </Badge>
                  )}
                  {backup.containsSecrets && (
                    <Badge variant="outline" className="text-xs">
                      <Key className="h-3 w-3 mr-1" />
                      Contains Secrets
                    </Badge>
                  )}
                  <span>Checksum: {backup.checksum}</span>
                </div>

                {backup.description && (
                  <p className="text-sm text-muted-foreground mt-2">{backup.description}</p>
                )}
              </div>
            ))}
          </div>

          {backupRecords.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <Archive className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No backups available</p>
              <p className="text-sm">Create your first backup to get started</p>
            </div>
          )}
        </div>

        {/* Restore Progress */}
        {isRestoringBackup && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Restore Progress</span>
              <span>{restoreProgress}%</span>
            </div>
            <Progress value={restoreProgress} className="h-2" />
          </div>
        )}

        {/* Restore Results */}
        {restoreResult && (
          <div className="space-y-3">
            <Separator />
            <div className="flex items-center gap-2">
              {restoreResult.success ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
              <h3 className="text-lg font-medium">
                {restoreResult.success ? 'Restore Successful' : 'Restore Failed'}
              </h3>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 border rounded">
                <div className="text-muted-foreground">Restored Items</div>
                <div className="text-xl font-bold text-green-600">
                  {restoreResult.restoredItems} / {restoreResult.totalItems}
                </div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-muted-foreground">Timestamp</div>
                <div className="text-sm">
                  {new Date(restoreResult.timestamp).toLocaleString()}
                </div>
              </div>
            </div>

            {restoreResult.errors.length > 0 && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  <div className="font-medium mb-1">Errors:</div>
                  <ul className="list-disc list-inside space-y-1">
                    {restoreResult.errors.map((error, index) => (
                      <li key={index} className="text-sm">{error}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {restoreResult.warnings.length > 0 && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <div className="font-medium mb-1">Warnings:</div>
                  <ul className="list-disc list-inside space-y-1">
                    {restoreResult.warnings.map((warning, index) => (
                      <li key={index} className="text-sm">{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <Button 
              variant="outline" 
              onClick={() => setRestoreResult(null)}
              className="w-full"
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Storage Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="p-3 border rounded">
            <div className="text-muted-foreground">Total Storage Used</div>
            <div className="text-xl font-bold">
              {formatFileSize(backupRecords.reduce((sum, b) => sum + b.size, 0))}
            </div>
          </div>
          <div className="p-3 border rounded">
            <div className="text-muted-foreground">Available Backups</div>
            <div className="text-xl font-bold">
              {backupRecords.filter(b => b.status === 'completed').length}
            </div>
          </div>
          <div className="p-3 border rounded">
            <div className="text-muted-foreground">Oldest Backup</div>
            <div className="text-sm">
              {backupRecords.length > 0 
                ? new Date(Math.min(...backupRecords.map(b => new Date(b.timestamp).getTime()))).toLocaleDateString()
                : 'None'
              }
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ConfigBackup;