import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings, 
  Key, 
  Wifi, 
  WifiOff, 
  CheckCircle, 
  XCircle, 
  RefreshCw,
  Eye,
  EyeOff,
  Globe,
  Shield,
  Clock
} from 'lucide-react';
import { toast } from 'sonner';
import { useConfigStore } from '@/stores/configStore';

interface Broker {
  id: string;
  name: string;
  displayName: string;
  logo?: string;
  isEnabled: boolean;
  environment: 'paper' | 'live' | 'sandbox';
  credentials: {
    apiKey: string;
    secretKey: string;
    additionalFields?: Record<string, string>;
  };
  settings: {
    baseUrl?: string;
    gatewayUrl?: string;
    port?: number;
    timeout?: number;
    maxRetries?: number;
  };
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  lastConnected?: Date;
  latency?: number;
  error?: string;
}

const BrokerConfig: React.FC = () => {
  const { config, updateConfig, validateConfig } = useConfigStore();
  const [brokers, setBrokers] = useState<Broker[]>([
    {
      id: 'alpaca',
      name: 'alpaca',
      displayName: 'Alpaca Markets',
      isEnabled: false,
      environment: 'paper',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        baseUrl: 'https://paper-api.alpaca.markets',
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'binance',
      name: 'binance',
      displayName: 'Binance',
      isEnabled: false,
      environment: 'sandbox',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        baseUrl: 'https://testnet.binance.vision',
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'ibkr',
      name: 'ibkr',
      displayName: 'Interactive Brokers',
      isEnabled: false,
      environment: 'paper',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        gatewayUrl: 'localhost',
        port: 7497,
        timeout: 60000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'degiro',
      name: 'degiro',
      displayName: 'DEGIRO',
      isEnabled: false,
      environment: 'paper',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'trading212',
      name: 'trading212',
      displayName: 'Trading 212',
      isEnabled: false,
      environment: 'paper',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        baseUrl: 'https://live.trading212.com/api/v0',
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'xtb',
      name: 'xtb',
      displayName: 'XTB',
      isEnabled: false,
      environment: 'demo',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        baseUrl: 'https://demo.xtb.com',
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    },
    {
      id: 'traderepublic',
      name: 'traderepublic',
      displayName: 'Trade Republic',
      isEnabled: false,
      environment: 'paper',
      credentials: {
        apiKey: '',
        secretKey: ''
      },
      settings: {
        timeout: 30000,
        maxRetries: 3
      },
      status: 'disconnected'
    }
  ]);

  const [selectedBroker, setSelectedBroker] = useState<string>('alpaca');
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [testingConnection, setTestingConnection] = useState<string | null>(null);

  const currentBroker = brokers.find(b => b.id === selectedBroker);

  const updateBroker = (brokerId: string, updates: Partial<Broker>) => {
    setBrokers(prev => prev.map(broker => 
      broker.id === brokerId ? { ...broker, ...updates } : broker
    ));
  };

  const testConnection = async (brokerId: string) => {
    const broker = brokers.find(b => b.id === brokerId);
    if (!broker || !broker.credentials.apiKey || !broker.credentials.secretKey) {
      toast.error('Please fill in API credentials first');
      return;
    }

    setTestingConnection(brokerId);
    
    try {
      // Simulate connection test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Simulate random latency
      const latency = Math.floor(Math.random() * 200) + 50;
      
      updateBroker(brokerId, {
        status: 'connected',
        lastConnected: new Date(),
        latency,
        error: undefined
      });
      
      toast.success(`Connected to ${broker.displayName} successfully`);
    } catch (error) {
      updateBroker(brokerId, {
        status: 'error',
        error: 'Connection failed. Please check your credentials.'
      });
      
      toast.error(`Failed to connect to ${broker.displayName}`);
    } finally {
      setTestingConnection(null);
    }
  };

  const toggleSecrets = (brokerId: string) => {
    setShowSecrets(prev => ({
      ...prev,
      [brokerId]: !prev[brokerId]
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'testing':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <WifiOff className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      case 'testing':
        return 'bg-blue-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-matrix-green">Broker Configuration</h1>
          <p className="text-matrix-green/70 mt-1">Configure your broker connections and API credentials</p>
        </div>
      </div>

      {/* Broker Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {brokers.map((broker) => (
          <Card 
            key={broker.id} 
            className={`border-matrix-green/20 bg-black/40 cursor-pointer transition-all hover:border-matrix-green/40 ${
              selectedBroker === broker.id ? 'ring-2 ring-matrix-green/50' : ''
            }`}
            onClick={() => setSelectedBroker(broker.id)}
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-matrix-green/20 rounded-lg flex items-center justify-center">
                    <Globe className="h-4 w-4 text-matrix-green" />
                  </div>
                  <div>
                    <h3 className="font-medium text-matrix-green">{broker.displayName}</h3>
                    <p className="text-xs text-matrix-green/60">{broker.name}</p>
                  </div>
                </div>
                <Badge 
                  variant="outline" 
                  className={`text-xs ${getStatusColor(broker.status)}/20 border-current`}
                >
                  <div className="flex items-center gap-1">
                    {getStatusIcon(broker.status)}
                    {broker.status}
                  </div>
                </Badge>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/60">Environment</span>
                  <span className="text-matrix-green capitalize">{broker.environment}</span>
                </div>
                
                {broker.latency && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-matrix-green/60">Latency</span>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3 text-matrix-green/60" />
                      <span className="text-matrix-green">{broker.latency}ms</span>
                    </div>
                  </div>
                )}
                
                {broker.lastConnected && (
                  <div className="text-xs text-matrix-green/50">
                    Last connected: {broker.lastConnected.toLocaleTimeString()}
                  </div>
                )}
                
                {broker.error && (
                  <Alert className="mt-2 py-2">
                    <AlertDescription className="text-xs text-red-400">
                      {broker.error}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Selected Broker Configuration */}
      {currentBroker && (
        <Card className="border-matrix-green/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-matrix-green flex items-center gap-2">
              <Settings className="h-5 w-5" />
              {currentBroker.displayName} Configuration
            </CardTitle>
            <CardDescription className="text-matrix-green/70">
              Configure API credentials and connection settings
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="credentials" className="space-y-4">
              <TabsList className="bg-black/60 border border-matrix-green/20">
                <TabsTrigger value="credentials" className="data-[state=active]:bg-matrix-green/20">
                  <Key className="h-4 w-4 mr-2" />
                  Credentials
                </TabsTrigger>
                <TabsTrigger value="settings" className="data-[state=active]:bg-matrix-green/20">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </TabsTrigger>
                <TabsTrigger value="advanced" className="data-[state=active]:bg-matrix-green/20">
                  <Shield className="h-4 w-4 mr-2" />
                  Advanced
                </TabsTrigger>
              </TabsList>

              <TabsContent value="credentials" className="space-y-4">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-matrix-green">Enable Broker</Label>
                      <p className="text-xs text-matrix-green/60">Enable this broker for trading</p>
                    </div>
                    <Switch
                      checked={currentBroker.isEnabled}
                      onCheckedChange={(checked) => updateBroker(currentBroker.id, { isEnabled: checked })}
                    />
                  </div>

                  <div className="space-y-0.5">
                    <Label className="text-matrix-green">Environment</Label>
                    <Select
                      value={currentBroker.environment}
                      onValueChange={(value: 'paper' | 'live' | 'sandbox' | 'demo') => 
                        updateBroker(currentBroker.id, { environment: value })
                      }
                    >
                      <SelectTrigger className="bg-black/40 border-matrix-green/20">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="paper">Paper Trading</SelectItem>
                        <SelectItem value="sandbox">Sandbox</SelectItem>
                        <SelectItem value="demo">Demo</SelectItem>
                        <SelectItem value="live">Live Trading</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">API Key</Label>
                    <div className="relative">
                      <Input
                        type={showSecrets[currentBroker.id]?.apiKey ? 'text' : 'password'}
                        value={currentBroker.credentials.apiKey}
                        onChange={(e) => updateBroker(currentBroker.id, {
                          credentials: { ...currentBroker.credentials, apiKey: e.target.value }
                        })}
                        placeholder="Enter your API key"
                        className="bg-black/40 border-matrix-green/20 pr-10"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                        onClick={() => toggleSecrets(currentBroker.id)}
                      >
                        {showSecrets[currentBroker.id]?.apiKey ? (
                          <EyeOff className="h-4 w-4 text-matrix-green/60" />
                        ) : (
                          <Eye className="h-4 w-4 text-matrix-green/60" />
                        )}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Secret Key</Label>
                    <div className="relative">
                      <Input
                        type={showSecrets[currentBroker.id]?.secretKey ? 'text' : 'password'}
                        value={currentBroker.credentials.secretKey}
                        onChange={(e) => updateBroker(currentBroker.id, {
                          credentials: { ...currentBroker.credentials, secretKey: e.target.value }
                        })}
                        placeholder="Enter your secret key"
                        className="bg-black/40 border-matrix-green/20 pr-10"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                        onClick={() => toggleSecrets(currentBroker.id)}
                      >
                        {showSecrets[currentBroker.id]?.secretKey ? (
                          <EyeOff className="h-4 w-4 text-matrix-green/60" />
                        ) : (
                          <Eye className="h-4 w-4 text-matrix-green/60" />
                        )}
                      </Button>
                    </div>
                  </div>

                  {/* Broker-specific additional fields */}
                  {currentBroker.id === 'ibkr' && (
                    <div className="space-y-2">
                      <Label className="text-matrix-green">IB Gateway</Label>
                      <Input
                        value={currentBroker.settings.gatewayUrl || ''}
                        onChange={(e) => updateBroker(currentBroker.id, {
                          settings: { ...currentBroker.settings, gatewayUrl: e.target.value }
                        })}
                        placeholder="localhost"
                        className="bg-black/40 border-matrix-green/20"
                      />
                    </div>
                  )}

                  {currentBroker.id === 'binance' && (
                    <div className="space-y-2">
                      <Label className="text-matrix-green">Base URL</Label>
                      <Input
                        value={currentBroker.settings.baseUrl || ''}
                        onChange={(e) => updateBroker(currentBroker.id, {
                          settings: { ...currentBroker.settings, baseUrl: e.target.value }
                        })}
                        placeholder="https://testnet.binance.vision"
                        className="bg-black/40 border-matrix-green/20"
                      />
                    </div>
                  )}

                  <div className="pt-4 space-y-3">
                    <Button
                      onClick={() => testConnection(currentBroker.id)}
                      disabled={testingConnection === currentBroker.id || !currentBroker.credentials.apiKey || !currentBroker.credentials.secretKey}
                      className="w-full"
                    >
                      {testingConnection === currentBroker.id ? (
                        <>
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Testing Connection...
                        </>
                      ) : (
                        <>
                          <Wifi className="h-4 w-4 mr-2" />
                          Test Connection
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="settings" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-matrix-green">Timeout (ms)</Label>
                    <Input
                      type="number"
                      value={currentBroker.settings.timeout || 30000}
                      onChange={(e) => updateBroker(currentBroker.id, {
                        settings: { ...currentBroker.settings, timeout: parseInt(e.target.value) }
                      })}
                      className="bg-black/40 border-matrix-green/20"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Max Retries</Label>
                    <Input
                      type="number"
                      value={currentBroker.settings.maxRetries || 3}
                      onChange={(e) => updateBroker(currentBroker.id, {
                        settings: { ...currentBroker.settings, maxRetries: parseInt(e.target.value) }
                      })}
                      className="bg-black/40 border-matrix-green/20"
                    />
                  </div>

                  {currentBroker.id === 'ibkr' && (
                    <div className="space-y-2">
                      <Label className="text-matrix-green">Port</Label>
                      <Input
                        type="number"
                        value={currentBroker.settings.port || 7497}
                        onChange={(e) => updateBroker(currentBroker.id, {
                          settings: { ...currentBroker.settings, port: parseInt(e.target.value) }
                        })}
                        className="bg-black/40 border-matrix-green/20"
                      />
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="advanced" className="space-y-4">
                <div className="text-center py-8 text-matrix-green/50">
                  Advanced settings will be available here
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BrokerConfig;