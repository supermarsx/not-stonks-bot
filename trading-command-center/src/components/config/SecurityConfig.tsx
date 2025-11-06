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
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  Lock, 
  Key, 
  Eye, 
  EyeOff, 
  Settings, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Users 
} from 'lucide-react';

export interface SecuritySettings {
  authentication: {
    enabled: boolean;
    method: 'local' | 'oauth' | 'saml' | 'ldap';
    sessionTimeout: number;
    maxLoginAttempts: number;
    lockoutDuration: number;
  };
  encryption: {
    algorithm: 'aes256' | 'chacha20' | 'twofish';
    keyRotation: boolean;
    keyRotationDays: number;
  };
  accessControl: {
    rbacEnabled: boolean;
    defaultRole: 'read' | 'write' | 'admin';
    requireMFA: boolean;
  };
  auditLog: {
    enabled: boolean;
    retentionDays: number;
    level: 'minimal' | 'normal' | 'verbose';
  };
}

const defaultSecuritySettings: SecuritySettings = {
  authentication: {
    enabled: true,
    method: 'local',
    sessionTimeout: 30,
    maxLoginAttempts: 5,
    lockoutDuration: 15
  },
  encryption: {
    algorithm: 'aes256',
    keyRotation: true,
    keyRotationDays: 90
  },
  accessControl: {
    rbacEnabled: true,
    defaultRole: 'read',
    requireMFA: false
  },
  auditLog: {
    enabled: true,
    retentionDays: 30,
    level: 'normal'
  }
};

export default function SecurityConfig() {
  const [securitySettings, setSecuritySettings] = useState<SecuritySettings>(defaultSecuritySettings);
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [validationResults, setValidationResults] = useState<{
    isValid: boolean;
    errors: string[];
    warnings: string[];
  }>({ isValid: true, errors: [], warnings: [] });

  useEffect(() => {
    validateSecuritySettings();
  }, [securitySettings]);

  const validateSecuritySettings = () => {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Authentication validation
    if (securitySettings.authentication.sessionTimeout < 5) {
      errors.push('Session timeout should be at least 5 minutes');
    }
    if (securitySettings.authentication.maxLoginAttempts < 3) {
      errors.push('Maximum login attempts should be at least 3');
    }
    if (securitySettings.authentication.lockoutDuration < 5) {
      warnings.push('Lockout duration is quite short (less than 15 minutes)');
    }

    // Encryption validation
    if (securitySettings.encryption.keyRotationDays > 365) {
      warnings.push('Key rotation interval is quite long (over 1 year)');
    }

    // Access control validation
    if (!securitySettings.accessControl.rbacEnabled && securitySettings.accessControl.requireMFA) {
      warnings.push('MFA requires RBAC to be enabled for proper user role management');
    }

    // Audit log validation
    if (securitySettings.auditLog.level === 'verbose' && securitySettings.auditLog.retentionDays > 30) {
      warnings.push('Verbose audit logging with long retention may consume significant storage');
    }

    setValidationResults({
      isValid: errors.length === 0,
      errors,
      warnings
    });
  };

  const handleSecurityChange = (category: keyof SecuritySettings, field: string, value: any) => {
    setSecuritySettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
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
      console.error('Failed to save security settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSecuritySettings(defaultSecuritySettings);
    setHasChanges(true);
  };

  const generateApiKey = () => {
    const newApiKey = 'sk-' + Math.random().toString(36).substr(2, 32);
    setApiKey(newApiKey);
  };

  const copyApiKey = () => {
    navigator.clipboard.writeText(apiKey);
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Shield className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Security Configuration</h2>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={handleReset}>
            <Settings className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave} disabled={!hasChanges || isLoading || !validationResults.isValid}>
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
            ) : (
              <Lock className="h-4 w-4 mr-2" />
            )}
            Save Changes
          </Button>
        </div>
      </div>

      {/* Validation Results */}
      {!validationResults.isValid && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {validationResults.errors.join(', ')}
          </AlertDescription>
        </Alert>
      )}
      
      {validationResults.warnings.length > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {validationResults.warnings.join(', ')}
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="authentication" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="authentication">Authentication</TabsTrigger>
          <TabsTrigger value="encryption">Encryption</TabsTrigger>
          <TabsTrigger value="access-control">Access Control</TabsTrigger>
          <TabsTrigger value="audit-log">Audit Log</TabsTrigger>
        </TabsList>

        <TabsContent value="authentication" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Users className="h-5 w-5" />
                <span>Authentication Settings</span>
              </CardTitle>
              <CardDescription>
                Configure user authentication and session management
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={securitySettings.authentication.enabled}
                  onCheckedChange={(checked) => handleSecurityChange('authentication', 'enabled', checked)}
                />
                <Label>Enable Authentication</Label>
              </div>
              
              <div className="space-y-2">
                <Label>Authentication Method</Label>
                <Select
                  value={securitySettings.authentication.method}
                  onValueChange={(value: 'local' | 'oauth' | 'saml' | 'ldap') => 
                    handleSecurityChange('authentication', 'method', value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="local">Local</SelectItem>
                    <SelectItem value="oauth">OAuth</SelectItem>
                    <SelectItem value="saml">SAML</SelectItem>
                    <SelectItem value="ldap">LDAP</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Session Timeout (minutes)</Label>
                  <Input
                    type="number"
                    value={securitySettings.authentication.sessionTimeout}
                    onChange={(e) => handleSecurityChange('authentication', 'sessionTimeout', parseInt(e.target.value))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Max Login Attempts</Label>
                  <Input
                    type="number"
                    value={securitySettings.authentication.maxLoginAttempts}
                    onChange={(e) => handleSecurityChange('authentication', 'maxLoginAttempts', parseInt(e.target.value))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Lockout Duration (minutes)</Label>
                  <Input
                    type="number"
                    value={securitySettings.authentication.lockoutDuration}
                    onChange={(e) => handleSecurityChange('authentication', 'lockoutDuration', parseInt(e.target.value))}
                  />
                </div>
              </div>

              <Separator />
              
              <div className="space-y-4">
                <h4 className="text-sm font-semibold flex items-center space-x-2">
                  <Key className="h-4 w-4" />
                  <span>API Keys</span>
                </h4>
                
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={generateApiKey}>
                    <Key className="h-4 w-4 mr-2" />
                    Generate API Key
                  </Button>
                  {apiKey && (
                    <Button variant="outline" onClick={copyApiKey}>
                      Copy Key
                    </Button>
                  )}
                </div>
                
                {apiKey && (
                  <div className="space-y-2">
                    <Label>Generated API Key</Label>
                    <div className="flex space-x-2">
                      <Input
                        value={showApiKey ? apiKey : '••••••••••••••••••••••••••••••••'}
                        readOnly
                        className="font-mono"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setShowApiKey(!showApiKey)}
                      >
                        {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="encryption" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Lock className="h-5 w-5" />
                <span>Encryption Settings</span>
              </CardTitle>
              <CardDescription>
                Configure encryption algorithms and key management
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Encryption Algorithm</Label>
                <Select
                  value={securitySettings.encryption.algorithm}
                  onValueChange={(value: 'aes256' | 'chacha20' | 'twofish') => 
                    handleSecurityChange('encryption', 'algorithm', value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="aes256">AES-256 (Recommended)</SelectItem>
                    <SelectItem value="chacha20">ChaCha20</SelectItem>
                    <SelectItem value="twofish">Twofish</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={securitySettings.encryption.keyRotation}
                    onCheckedChange={(checked) => handleSecurityChange('encryption', 'keyRotation', checked)}
                  />
                  <Label>Enable Key Rotation</Label>
                </div>
                <p className="text-sm text-muted-foreground">
                  Automatically rotate encryption keys after specified interval
                </p>
              </div>

              {securitySettings.encryption.keyRotation && (
                <div className="space-y-2">
                  <Label>Key Rotation Interval (days)</Label>
                  <Input
                    type="number"
                    value={securitySettings.encryption.keyRotationDays}
                    onChange={(e) => handleSecurityChange('encryption', 'keyRotationDays', parseInt(e.target.value))}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access-control" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Access Control</span>
              </CardTitle>
              <CardDescription>
                Configure role-based access control and permissions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={securitySettings.accessControl.rbacEnabled}
                    onCheckedChange={(checked) => handleSecurityChange('accessControl', 'rbacEnabled', checked)}
                  />
                  <Label>Enable Role-Based Access Control</Label>
                </div>
                <p className="text-sm text-muted-foreground">
                  Enables granular permissions based on user roles
                </p>
              </div>

              <div className="space-y-2">
                <Label>Default Role for New Users</Label>
                <Select
                  value={securitySettings.accessControl.defaultRole}
                  onValueChange={(value: 'read' | 'write' | 'admin') => 
                    handleSecurityChange('accessControl', 'defaultRole', value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="read">Read Only</SelectItem>
                    <SelectItem value="write">Read/Write</SelectItem>
                    <SelectItem value="admin">Administrator</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={securitySettings.accessControl.requireMFA}
                    onCheckedChange={(checked) => handleSecurityChange('accessControl', 'requireMFA', checked)}
                  />
                  <Label>Require Multi-Factor Authentication</Label>
                </div>
                <p className="text-sm text-muted-foreground">
                  Forces users to use MFA for additional security
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit-log" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Audit Log</span>
              </CardTitle>
              <CardDescription>
                Configure audit logging and retention policies
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={securitySettings.auditLog.enabled}
                    onCheckedChange={(checked) => handleSecurityChange('auditLog', 'enabled', checked)}
                  />
                  <Label>Enable Audit Logging</Label>
                </div>
                <p className="text-sm text-muted-foreground">
                  Logs all user actions and system events
                </p>
              </div>

              <div className="space-y-2">
                <Label>Audit Log Level</Label>
                <Select
                  value={securitySettings.auditLog.level}
                  onValueChange={(value: 'minimal' | 'normal' | 'verbose') => 
                    handleSecurityChange('auditLog', 'level', value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="minimal">Minimal (Only critical events)</SelectItem>
                    <SelectItem value="normal">Normal (Standard events)</SelectItem>
                    <SelectItem value="verbose">Verbose (All events including details)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Retention Period (days)</Label>
                <Input
                  type="number"
                  value={securitySettings.auditLog.retentionDays}
                  onChange={(e) => handleSecurityChange('auditLog', 'retentionDays', parseInt(e.target.value))}
                />
                <p className="text-sm text-muted-foreground">
                  How long to keep audit logs before automatic deletion
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Status indicator */}
      <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
        <div className="flex items-center space-x-2">
          {validationResults.isValid ? (
            <CheckCircle className="h-5 w-5 text-green-600" />
          ) : (
            <XCircle className="h-5 w-5 text-red-600" />
          )}
          <span className="text-sm font-medium">
            {validationResults.isValid ? 'Configuration is valid' : 'Configuration has errors'}
          </span>
        </div>
        
        {hasChanges && (
          <Badge variant="outline" className="text-amber-600">
            Unsaved Changes
          </Badge>
        )}
      </div>
    </div>
  );
}