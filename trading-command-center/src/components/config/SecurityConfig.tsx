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
  Users,
  Fingerprint,
  Smartphone,
  Globe,
  Server,
  Database,
  Activity
} from 'lucide-react';

interface AuthenticationMethod {
  id: string;
  name: string;
  type: 'password' | 'biometric' | '2fa' | 'oauth' | 'api_key' | 'certificate';
  enabled: boolean;
  required: boolean;
  config: {
    minLength?: number;
    requireSpecial?: boolean;
    expireDays?: number;
    issuer?: string;
    algorithm?: string;
  };
  metrics: {
    successRate: number;
    avgResponseTime: number;
    lastUsed: string;
    failedAttempts: number;
  };
}

interface SecuritySettings {
  enabled: boolean;
  sessionTimeout: number; // minutes
  maxFailedAttempts: number;
  lockoutDuration: number; // minutes
  enforceStrongPasswords: boolean;
  require2FA: boolean;
  allowRememberMe: boolean;
  enableAuditLogging: boolean;
  enableIPWhitelist: boolean;
  enableGeoBlocking: boolean;
  trustedIPs: string[];
  blockedIPs: string[];
}

interface EncryptionSettings {
  enabled: boolean;
  algorithm: 'AES-256' | 'RSA-2048' | 'AES-128' | 'ChaCha20-Poly1305';
  keyRotation: boolean;
  keyRotationPeriod: number; // days
  encryptAtRest: boolean;
  encryptInTransit: boolean;
  enableHsm: boolean;
}

interface SecurityAudit {
  id: string;
  timestamp: string;
  event: string;
  user: string;
  ip: string;
  result: 'success' | 'failure' | 'blocked';
  details: string;
}

const SecurityConfig: React.FC = () => {
  const [securitySettings, setSecuritySettings] = useState<SecuritySettings>({
    enabled: true,
    sessionTimeout: 30,
    maxFailedAttempts: 5,
    lockoutDuration: 15,
    enforceStrongPasswords: true,
    require2FA: false,
    allowRememberMe: true,
    enableAuditLogging: true,
    enableIPWhitelist: false,
    enableGeoBlocking: false,
    trustedIPs: ['127.0.0.1', '::1'],
    blockedIPs: []
  });

  const [encryptionSettings, setEncryptionSettings] = useState<EncryptionSettings>({
    enabled: true,
    algorithm: 'AES-256',
    keyRotation: true,
    keyRotationPeriod: 90,
    encryptAtRest: true,
    encryptInTransit: true,
    enableHsm: false
  });

  const [authMethods, setAuthMethods] = useState<AuthenticationMethod[]>([
    {
      id: 'password',
      name: 'Password Authentication',
      type: 'password',
      enabled: true,
      required: true,
      config: {
        minLength: 8,
        requireSpecial: true
      },
      metrics: {
        successRate: 0.987,
        avgResponseTime: 0.23,
        lastUsed: new Date().toISOString(),
        failedAttempts: 3
      }
    },
    {
      id: '2fa',
      name: 'Two-Factor Authentication',
      type: '2fa',
      enabled: false,
      required: false,
      config: {
        issuer: 'Trading System'
      },
      metrics: {
        successRate: 0.995,
        avgResponseTime: 1.45,
        lastUsed: '',
        failedAttempts: 1
      }
    },
    {
      id: 'api_key',
      name: 'API Key Authentication',
      type: 'api_key',
      enabled: true,
      required: false,
      config: {
        expireDays: 365,
        algorithm: 'SHA-256'
      },
      metrics: {
        successRate: 0.999,
        avgResponseTime: 0.05,
        lastUsed: new Date().toISOString(),
        failedAttempts: 0
      }
    }
  ]);

  const [securityAudit, setSecurityAudit] = useState<SecurityAudit[]>([
    {
      id: '1',
      timestamp: new Date().toISOString(),
      event: 'user_login',
      user: 'admin',
      ip: '192.168.1.100',
      result: 'success',
      details: 'Successful password login'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      event: 'api_access',
      user: 'trading_bot',
      ip: '10.0.0.5',
      result: 'success',
      details: 'API key authentication successful'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      event: 'failed_login',
      user: 'unknown',
      ip: '203.0.113.1',
      result: 'failure',
      details: 'Invalid credentials provided'
    }
  ]);

  const [activeMethod, setActiveMethod] = useState<string>('password');
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [testing2FA, setTesting2FA] = useState(false);

  useEffect(() => {
    loadSecurityConfiguration();
  }, []);

  const loadSecurityConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getSecurityConfig();
      if (config) {
        setSecuritySettings(config.settings || securitySettings);
        setEncryptionSettings(config.encryption || encryptionSettings);
        setAuthMethods(config.authMethods || authMethods);
        setSecurityAudit(config.audit || securityAudit);
      }
    } catch (error) {
      console.error('Failed to load security configuration:', error);
    }
  };

  const saveSecurityConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.saveSecurityConfig({
        settings: securitySettings,
        encryption: encryptionSettings,
        authMethods,
        audit: securityAudit
      });
    } catch (error) {
      console.error('Failed to save security configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const updateAuthMethod = (id: string, updates: Partial<AuthenticationMethod>) => {
    setAuthMethods(methods => 
      methods.map(method => method.id === id ? { ...method, ...updates } : method)
    );
  };

  const test2FA = async () => {
    setTesting2FA(true);
    try {
      // Simulate 2FA test
      await new Promise(resolve => setTimeout(resolve, 2000));
      console.log('2FA test completed');
    } finally {
      setTesting2FA(false);
    }
  };

  const generateAPIKey = () => {
    const key = `api_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return key;
  };

  const rotateEncryptionKeys = async () => {
    try {
      await window.electronAPI?.rotateEncryptionKeys();
      alert('Encryption keys rotated successfully');
    } catch (error) {
      console.error('Failed to rotate encryption keys:', error);
      alert('Failed to rotate encryption keys');
    }
  };

  const getMethodIcon = (type: string) => {
    switch (type) {
      case 'password':
        return <Lock className="h-4 w-4" />;
      case 'biometric':
        return <Fingerprint className="h-4 w-4" />;
      case '2fa':
        return <Smartphone className="h-4 w-4" />;
      case 'oauth':
        return <Globe className="h-4 w-4" />;
      case 'api_key':
        return <Key className="h-4 w-4" />;
      case 'certificate':
        return <Shield className="h-4 w-4" />;
      default:
        return <Key className="h-4 w-4" />;
    }
  };

  const getEventIcon = (event: string) => {
    switch (event) {
      case 'user_login':
        return <Users className="h-4 w-4 text-blue-500" />;
      case 'api_access':
        return <Key className="h-4 w-4 text-green-500" />;
      case 'failed_login':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'data_access':
        return <Database className="h-4 w-4 text-purple-500" />;
      default:
        return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  const getResultBadge = (result: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      success: 'default',
      failure: 'destructive',
      blocked: 'destructive'
    };

    const colors: Record<string, string> = {
      success: 'bg-green-500',
      failure: 'bg-red-500',
      blocked: 'bg-red-700'
    };

    return (
      <Badge variant={variants[result] || 'outline'} className="gap-1">
        <div className={`h-2 w-2 rounded-full ${colors[result] || 'bg-gray-400'}`} />
        {result.charAt(0).toUpperCase() + result.slice(1)}
      </Badge>
    );
  };

  const activeMethodData = authMethods.find(m => m.id === activeMethod);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Security & Authentication Configuration
          </CardTitle>
          <CardDescription>
            Configure authentication methods, encryption, and security policies
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={securitySettings.enabled}
                  onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, enabled: checked }))}
                />
                <Label>Security System {securitySettings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant={encryptionSettings.enabled ? 'default' : 'outline'}>
                Encryption {encryptionSettings.enabled ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={rotateEncryptionKeys} size="sm">
                <Lock className="h-4 w-4 mr-2" />
                Rotate Keys
              </Button>
              <Button onClick={saveSecurityConfiguration} disabled={saving}>
                <Settings className="h-4 w-4 mr-2" />
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Security Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Security Level</p>
                <p className="text-2xl font-bold text-green-600">High</p>
              </div>
              <Shield className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Sessions</p>
                <p className="text-2xl font-bold">3</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Failed Attempts</p>
                <p className="text-2xl font-bold text-red-600">12</p>
              </div>
              <XCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Encryption Status</p>
                <p className="text-2xl font-bold">Active</p>
              </div>
              <Lock className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="authentication" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="authentication">Authentication</TabsTrigger>
          <TabsTrigger value="encryption">Encryption</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="audit">Audit Log</TabsTrigger>
        </TabsList>

        <TabsContent value="authentication" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Authentication Methods List */}
            <Card>
              <CardHeader>
                <CardTitle>Authentication Methods</CardTitle>
                <CardDescription>
                  Manage authentication methods and settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {authMethods.map((method) => (
                  <div
                    key={method.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      activeMethod === method.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setActiveMethod(method.id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getMethodIcon(method.type)}
                        <h3 className="font-medium">{method.name}</h3>
                        <Badge variant={method.enabled ? 'default' : 'outline'}>
                          {method.enabled ? 'Active' : 'Inactive'}
                        </Badge>
                        {method.required && (
                          <Badge variant="destructive" className="text-xs">
                            Required
                          </Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {(method.metrics.successRate * 100).toFixed(0)}% success
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Type:</span>
                        <div className="font-medium capitalize">{method.type.replace('_', ' ')}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Avg Response:</span>
                        <div className="font-medium">{method.metrics.avgResponseTime}s</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Failed Attempts:</span>
                        <div className="font-medium">{method.metrics.failedAttempts}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Last Used:</span>
                        <div className="font-medium">
                          {method.metrics.lastUsed ? new Date(method.metrics.lastUsed).toLocaleDateString() : 'Never'}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Method Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Method Configuration</CardTitle>
                <CardDescription>
                  {activeMethodData ? 'Configure authentication method settings' : 'Select a method to configure'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeMethodData ? (
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="method-name">Method Name</Label>
                      <Input
                        id="method-name"
                        value={activeMethodData.name}
                        onChange={(e) => updateAuthMethod(activeMethodData.id, { name: e.target.value })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable Method</Label>
                        <p className="text-sm text-muted-foreground">
                          Allow users to authenticate with this method
                        </p>
                      </div>
                      <Switch
                        checked={activeMethodData.enabled}
                        onCheckedChange={(checked) => updateAuthMethod(activeMethodData.id, { enabled: checked })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Required Method</Label>
                        <p className="text-sm text-muted-foreground">
                          This method must be used for authentication
                        </p>
                      </div>
                      <Switch
                        checked={activeMethodData.required}
                        onCheckedChange={(checked) => updateAuthMethod(activeMethodData.id, { required: checked })}
                      />
                    </div>

                    <Separator />

                    {/* Method-specific configuration */}
                    {activeMethodData.type === 'password' && (
                      <div className="space-y-4">
                        <div>
                          <Label>Minimum Password Length</Label>
                          <Input
                            type="number"
                            value={activeMethodData.config.minLength || 8}
                            onChange={(e) => updateAuthMethod(activeMethodData.id, {
                              config: { ...activeMethodData.config, minLength: Number(e.target.value) }
                            })}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <Label>Require Special Characters</Label>
                          <Switch
                            checked={activeMethodData.config.requireSpecial || false}
                            onCheckedChange={(checked) => updateAuthMethod(activeMethodData.id, {
                              config: { ...activeMethodData.config, requireSpecial: checked }
                            })}
                          />
                        </div>
                      </div>
                    )}

                    {activeMethodData.type === '2fa' && (
                      <div className="space-y-4">
                        <div>
                          <Label>Issuer Name</Label>
                          <Input
                            value={activeMethodData.config.issuer || ''}
                            onChange={(e) => updateAuthMethod(activeMethodData.id, {
                              config: { ...activeMethodData.config, issuer: e.target.value }
                            })}
                            placeholder="Trading System"
                          />
                        </div>

                        <Button onClick={test2FA} disabled={testing2FA} variant="outline" className="w-full">
                          <Smartphone className="h-4 w-4 mr-2" />
                          {testing2FA ? 'Testing...' : 'Test 2FA Setup'}
                        </Button>
                      </div>
                    )}

                    {activeMethodData.type === 'api_key' && (
                      <div className="space-y-4">
                        <div>
                          <Label>Key Expiration (days)</Label>
                          <Input
                            type="number"
                            value={activeMethodData.config.expireDays || 365}
                            onChange={(e) => updateAuthMethod(activeMethodData.id, {
                              config: { ...activeMethodData.config, expireDays: Number(e.target.value) }
                            })}
                          />
                        </div>

                        <div>
                          <Label>Hashing Algorithm</Label>
                          <Select
                            value={activeMethodData.config.algorithm || 'SHA-256'}
                            onValueChange={(value) => updateAuthMethod(activeMethodData.id, {
                              config: { ...activeMethodData.config, algorithm: value }
                            })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="SHA-256">SHA-256</SelectItem>
                              <SelectItem value="SHA-512">SHA-512</SelectItem>
                              <SelectItem value="bcrypt">bcrypt</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label>Generated API Key</Label>
                          <div className="flex">
                            <Input
                              value={generateAPIKey()}
                              readOnly
                              className="font-mono text-xs"
                            />
                            <Button variant="outline" size="sm" className="ml-2">
                              <Eye className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}

                    <Separator />

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Success Rate:</span>
                        <div className="font-medium">{(activeMethodData.metrics.successRate * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Response Time:</span>
                        <div className="font-medium">{activeMethodData.metrics.avgResponseTime}s</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Failed Attempts:</span>
                        <div className="font-medium">{activeMethodData.metrics.failedAttempts}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Last Used:</span>
                        <div className="font-medium">
                          {activeMethodData.metrics.lastUsed ? new Date(activeMethodData.metrics.lastUsed).toLocaleString() : 'Never'}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Key className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Select an authentication method to configure</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="encryption" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Encryption Settings</CardTitle>
                <CardDescription>
                  Configure encryption algorithms and key management
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Encryption</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable data encryption for security
                    </p>
                  </div>
                  <Switch
                    checked={encryptionSettings.enabled}
                    onCheckedChange={(checked) => setEncryptionSettings(prev => ({ ...prev, enabled: checked }))}
                  />
                </div>

                <div>
                  <Label htmlFor="algorithm">Encryption Algorithm</Label>
                  <Select
                    value={encryptionSettings.algorithm}
                    onValueChange={(value: any) => setEncryptionSettings(prev => ({ ...prev, algorithm: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AES-256">AES-256</SelectItem>
                      <SelectItem value="AES-128">AES-128</SelectItem>
                      <SelectItem value="RSA-2048">RSA-2048</SelectItem>
                      <SelectItem value="ChaCha20-Poly1305">ChaCha20-Poly1305</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Encrypt at Rest</Label>
                    <p className="text-sm text-muted-foreground">
                      Encrypt stored data
                    </p>
                  </div>
                  <Switch
                    checked={encryptionSettings.encryptAtRest}
                    onCheckedChange={(checked) => setEncryptionSettings(prev => ({ ...prev, encryptAtRest: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Encrypt in Transit</Label>
                    <p className="text-sm text-muted-foreground">
                      Encrypt transmitted data
                    </p>
                  </div>
                  <Switch
                    checked={encryptionSettings.encryptInTransit}
                    onCheckedChange={(checked) => setEncryptionSettings(prev => ({ ...prev, encryptInTransit: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Hardware Security Module</Label>
                    <p className="text-sm text-muted-foreground">
                      Use HSM for key storage
                    </p>
                  </div>
                  <Switch
                    checked={encryptionSettings.enableHsm}
                    onCheckedChange={(checked) => setEncryptionSettings(prev => ({ ...prev, enableHsm: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Management</CardTitle>
                <CardDescription>
                  Manage encryption keys and rotation
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Key Rotation</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically rotate encryption keys
                    </p>
                  </div>
                  <Switch
                    checked={encryptionSettings.keyRotation}
                    onCheckedChange={(checked) => setEncryptionSettings(prev => ({ ...prev, keyRotation: checked }))}
                  />
                </div>

                <div>
                  <Label>Rotation Period: {encryptionSettings.keyRotationPeriod} days</Label>
                  <Slider
                    value={[encryptionSettings.keyRotationPeriod]}
                    onValueChange={([value]) => setEncryptionSettings(prev => ({ ...prev, keyRotationPeriod: value }))}
                    min={30}
                    max={365}
                    step={30}
                    className="mt-2"
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Key Status</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Master Key</div>
                      <div className="font-medium flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Active
                      </div>
                    </div>
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Key Age</div>
                      <div className="font-medium">45 days</div>
                    </div>
                  </div>
                </div>

                <Alert>
                  <Lock className="h-4 w-4" />
                  <AlertDescription>
                    Next key rotation scheduled in {Math.max(0, encryptionSettings.keyRotationPeriod - 45)} days
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="policies" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Session & Access Policies</CardTitle>
                <CardDescription>
                  Configure session management and access controls
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Session Timeout: {securitySettings.sessionTimeout} minutes</Label>
                  <Slider
                    value={[securitySettings.sessionTimeout]}
                    onValueChange={([value]) => setSecuritySettings(prev => ({ ...prev, sessionTimeout: value }))}
                    min={5}
                    max={120}
                    step={5}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Max Failed Attempts: {securitySettings.maxFailedAttempts}</Label>
                  <Slider
                    value={[securitySettings.maxFailedAttempts]}
                    onValueChange={([value]) => setSecuritySettings(prev => ({ ...prev, maxFailedAttempts: value }))}
                    min={3}
                    max={10}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Lockout Duration: {securitySettings.lockoutDuration} minutes</Label>
                  <Slider
                    value={[securitySettings.lockoutDuration]}
                    onValueChange={([value]) => setSecuritySettings(prev => ({ ...prev, lockoutDuration: value }))}
                    min={5}
                    max={60}
                    step={5}
                    className="mt-2"
                  />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enforce Strong Passwords</Label>
                    <p className="text-sm text-muted-foreground">
                      Require complex passwords
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.enforceStrongPasswords}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, enforceStrongPasswords: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require 2FA</Label>
                    <p className="text-sm text-muted-foreground">
                      Mandate two-factor authentication
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.require2FA}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, require2FA: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Remember Me</Label>
                    <p className="text-sm text-muted-foreground">
                      Allow persistent sessions
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.allowRememberMe}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, allowRememberMe: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Network Security</CardTitle>
                <CardDescription>
                  Configure IP filtering and geo-blocking
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable IP Whitelist</Label>
                    <p className="text-sm text-muted-foreground">
                      Only allow connections from trusted IPs
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.enableIPWhitelist}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, enableIPWhitelist: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Geo-blocking</Label>
                    <p className="text-sm text-muted-foreground">
                      Block access from specific regions
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.enableGeoBlocking}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, enableGeoBlocking: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Audit Logging</Label>
                    <p className="text-sm text-muted-foreground">
                      Log all security events
                    </p>
                  </div>
                  <Switch
                    checked={securitySettings.enableAuditLogging}
                    onCheckedChange={(checked) => setSecuritySettings(prev => ({ ...prev, enableAuditLogging: checked }))}
                  />
                </div>

                <Separator />

                <div>
                  <Label htmlFor="trusted-ips">Trusted IP Addresses</Label>
                  <Textarea
                    id="trusted-ips"
                    value={securitySettings.trustedIPs.join('\n')}
                    onChange={(e) => {
                      const ips = e.target.value.split('\n').filter(ip => ip.trim());
                      setSecuritySettings(prev => ({ ...prev, trustedIPs: ips }));
                    }}
                    placeholder="Enter IP addresses (one per line)"
                    rows={3}
                  />
                </div>

                <div>
                  <Label htmlFor="blocked-ips">Blocked IP Addresses</Label>
                  <Textarea
                    id="blocked-ips"
                    value={securitySettings.blockedIPs.join('\n')}
                    onChange={(e) => {
                      const ips = e.target.value.split('\n').filter(ip => ip.trim());
                      setSecuritySettings(prev => ({ ...prev, blockedIPs: ips }));
                    }}
                    placeholder="Enter IP addresses to block (one per line)"
                    rows={3}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="audit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Audit Log</CardTitle>
              <CardDescription>
                Monitor security events and authentication attempts
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {securityAudit.map((event) => (
                  <div key={event.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getEventIcon(event.event)}
                        <span className="font-medium">{event.event.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                        {getResultBadge(event.result)}
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {new Date(event.timestamp).toLocaleString()}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">User:</span>
                        <div className="font-medium">{event.user}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">IP Address:</span>
                        <div className="font-medium">{event.ip}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Result:</span>
                        <div className="font-medium">{event.result}</div>
                      </div>
                    </div>
                    
                    <div className="mt-2 text-sm text-muted-foreground">
                      {event.details}
                    </div>
                  </div>
                ))}

                {securityAudit.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No audit events recorded</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SecurityConfig;