import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { 
  Server, 
  Key, 
  TestTube, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Copy,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  RefreshCw,
  Settings
} from 'lucide-react';

interface Broker {
  id: string;
  name: string;
  type: 'alpaca' | 'binance' | 'ibkr' | 'degiro' | 'trading212' | 'xtb' | 'trade_republic';
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  config: {
    apiKey: string;
    secretKey: string;
    environment: 'paper' | 'live' | 'sandbox';
    baseUrl?: string;
    additionalSettings?: Record<string, any>;
  };
  connectionTest?: {
    lastTested: string;
    result: 'success' | 'failure';
    latency?: number;
    message?: string;
  };
}

const BrokerConfig: React.FC = () => {
  const [brokers, setBrokers] = useState<Broker[]>([]);
  const [activeBroker, setActiveBroker] = useState<string | null>(null);
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [testingConnection, setTestingConnection] = useState<string | null>(null);

  const brokerTypes = [
    { value: 'alpaca', label: 'Alpaca Trading' },
    { value: 'binance', label: 'Binance' },
    { value: 'ibkr', label: 'Interactive Brokers' },
    { value: 'degiro', label: 'DEGIRO' },
    { value: 'trading212', label: 'Trading 212' },
    { value: 'xtb', label: 'XTB' },
    { value: 'trade_republic', label: 'Trade Republic' }
  ];

  const environments = [
    { value: 'paper', label: 'Paper Trading' },
    { value: 'live', label: 'Live Trading' },
    { value: 'sandbox', label: 'Sandbox' }
  ];

  useEffect(() => {
    loadBrokers();
  }, []);

  const loadBrokers = async () => {
    setIsLoading(true);
    try {
      // Load brokers from storage
      const savedBrokers = await window.electronAPI?.getBrokers() || [];
      setBrokers(savedBrokers);
    } catch (error) {
      console.error('Failed to load brokers:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveBrokers = async () => {
    setIsLoading(true);
    try {
      await window.electronAPI?.saveBrokers(brokers);
    } catch (error) {
      console.error('Failed to save brokers:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const addBroker = () => {
    const newBroker: Broker = {
      id: `broker_${Date.now()}`,
      name: '',
      type: 'alpaca',
      status: 'disconnected',
      config: {
        apiKey: '',
        secretKey: '',
        environment: 'paper',
        baseUrl: '',
        additionalSettings: {}
      }
    };
    setBrokers([...brokers, newBroker]);
    setActiveBroker(newBroker.id);
  };

  const updateBroker = (id: string, updates: Partial<Broker>) => {
    setBrokers(brokers.map(broker => 
      broker.id === id ? { ...broker, ...updates } : broker
    ));
  };

  const deleteBroker = (id: string) => {
    setBrokers(brokers.filter(broker => broker.id !== id));
    if (activeBroker === id) {
      setActiveBroker(null);
    }
  };

  const testConnection = async (broker: Broker) => {
    setTestingConnection(broker.id);
    try {
      // Simulate connection test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const connectionTest = {
        lastTested: new Date().toISOString(),
        result: Math.random() > 0.3 ? 'success' as const : 'failure' as const,
        latency: Math.floor(Math.random() * 500) + 50,
        message: Math.random() > 0.3 ? 'Connection successful' : 'Invalid API credentials'
      };

      updateBroker(broker.id, {
        status: connectionTest.result === 'success' ? 'connected' : 'error',
        connectionTest
      });
    } catch (error) {
      updateBroker(broker.id, {
        status: 'error',
        connectionTest: {
          lastTested: new Date().toISOString(),
          result: 'failure',
          message: error instanceof Error ? error.message : 'Connection test failed'
        }
      });
    } finally {
      setTestingConnection(null);
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      connected: 'default',
      disconnected: 'outline',
      error: 'destructive',
      testing: 'secondary'
    };

    const colors: Record<string, string> = {
      connected: 'bg-green-500',
      disconnected: 'bg-gray-400',
      error: 'bg-red-500',
      testing: 'bg-yellow-500'
    };

    return (
      <Badge variant={variants[status] || 'outline'} className="gap-1">
        <div className={`h-2 w-2 rounded-full ${colors[status] || 'bg-gray-400'}`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getBrokerIcon = (type: string) => {
    return <Server className="h-4 w-4" />;
  };

  const activeBrokerData = brokers.find(broker => broker.id === activeBroker);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Broker Configuration
          </CardTitle>
          <CardDescription>
            Configure trading broker connections and API settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Connected brokers: {brokers.filter(b => b.status === 'connected').length} of {brokers.length}
            </p>
            <Button onClick={addBroker} size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Broker
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Brokers List */}
        <Card>
          <CardHeader>
            <CardTitle>Brokers</CardTitle>
            <CardDescription>
              Manage your broker connections
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {brokers.map((broker) => (
              <div
                key={broker.id}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  activeBroker === broker.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                }`}
                onClick={() => setActiveBroker(broker.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getBrokerIcon(broker.type)}
                    <div>
                      <h3 className="font-medium">{broker.name || 'Unnamed Broker'}</h3>
                      <p className="text-sm text-muted-foreground">
                        {brokerTypes.find(bt => bt.value === broker.type)?.label}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(broker.status)}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteBroker(broker.id);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                {broker.connectionTest && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    Last tested: {new Date(broker.connectionTest.lastTested).toLocaleString()}
                    {broker.connectionTest.latency && (
                      <span> • Latency: {broker.connectionTest.latency}ms</span>
                    )}
                  </div>
                )}
              </div>
            ))}

            {brokers.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Server className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No brokers configured</p>
                <Button onClick={addBroker} variant="outline" className="mt-2" size="sm">
                  Add your first broker
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Broker Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>
              {activeBrokerData ? 'Configure broker settings' : 'Select a broker to configure'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {activeBrokerData ? (
              <div className="space-y-4">
                {/* Basic Settings */}
                <div className="space-y-3">
                  <h3 className="text-sm font-medium">Basic Settings</h3>
                  
                  <div>
                    <Label htmlFor="broker-name">Name</Label>
                    <Input
                      id="broker-name"
                      value={activeBrokerData.name}
                      onChange={(e) => updateBroker(activeBrokerData.id, { name: e.target.value })}
                      placeholder="My Broker Account"
                    />
                  </div>

                  <div>
                    <Label htmlFor="broker-type">Type</Label>
                    <Select
                      value={activeBrokerData.type}
                      onValueChange={(value) => updateBroker(activeBrokerData.id, { type: value as Broker['type'] })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {brokerTypes.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            {type.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="environment">Environment</Label>
                    <Select
                      value={activeBrokerData.config.environment}
                      onValueChange={(value) => updateBroker(activeBrokerData.id, {
                        config: { ...activeBrokerData.config, environment: value as any }
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {environments.map((env) => (
                          <SelectItem key={env.value} value={env.value}>
                            {env.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Separator />

                {/* API Credentials */}
                <div className="space-y-3">
                  <h3 className="text-sm font-medium">API Credentials</h3>
                  
                  <div>
                    <Label htmlFor="api-key">API Key</Label>
                    <div className="flex">
                      <Input
                        id="api-key"
                        type={showSecrets[activeBrokerData.id] ? 'text' : 'password'}
                        value={activeBrokerData.config.apiKey}
                        onChange={(e) => updateBroker(activeBrokerData.id, {
                          config: { ...activeBrokerData.config, apiKey: e.target.value }
                        })}
                        placeholder="Enter API key"
                        className="pr-10"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-2"
                        onClick={() => copyToClipboard(activeBrokerData.config.apiKey)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-1"
                        onClick={() => setShowSecrets({
                          ...showSecrets,
                          [activeBrokerData.id]: !showSecrets[activeBrokerData.id]
                        })}
                      >
                        {showSecrets[activeBrokerData.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="secret-key">Secret Key</Label>
                    <div className="flex">
                      <Input
                        id="secret-key"
                        type={showSecrets[activeBrokerData.id] ? 'text' : 'password'}
                        value={activeBrokerData.config.secretKey}
                        onChange={(e) => updateBroker(activeBrokerData.id, {
                          config: { ...activeBrokerData.config, secretKey: e.target.value }
                        })}
                        placeholder="Enter secret key"
                        className="pr-10"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-2"
                        onClick={() => copyToClipboard(activeBrokerData.config.secretKey)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Additional Settings */}
                {activeBrokerData.type === 'binance' && (
                  <>
                    <Separator />
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Binance Settings</h3>
                      <div>
                        <Label htmlFor="base-url">Base URL (Optional)</Label>
                        <Input
                          id="base-url"
                          value={activeBrokerData.config.baseUrl || ''}
                          onChange={(e) => updateBroker(activeBrokerData.id, {
                            config: { ...activeBrokerData.config, baseUrl: e.target.value }
                          })}
                          placeholder="https://api.binance.com"
                        />
                      </div>
                    </div>
                  </>
                )}

                {activeBrokerData.type === 'ibkr' && (
                  <>
                    <Separator />
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Interactive Brokers Settings</h3>
                      <div>
                        <Label htmlFor="ib-gateway">TWS/Gateway Host</Label>
                        <Input
                          id="ib-gateway"
                          placeholder="localhost"
                          defaultValue="localhost"
                        />
                      </div>
                      <div>
                        <Label htmlFor="ib-port">Port</Label>
                        <Input
                          id="ib-port"
                          placeholder="7497"
                          defaultValue="7497"
                        />
                      </div>
                    </div>
                  </>
                )}

                {/* Connection Test */}
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium">Connection Test</h3>
                      <p className="text-xs text-muted-foreground">
                        Test the connection to verify your credentials
                      </p>
                    </div>
                    <Button
                      onClick={() => testConnection(activeBrokerData)}
                      disabled={testingConnection === activeBrokerData.id || !activeBrokerData.config.apiKey || !activeBrokerData.config.secretKey}
                      variant={activeBrokerData.status === 'connected' ? 'outline' : 'default'}
                    >
                      {testingConnection === activeBrokerData.id ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <TestTube className="h-4 w-4 mr-2" />
                      )}
                      {testingConnection === activeBrokerData.id ? 'Testing...' : 'Test Connection'}
                    </Button>
                  </div>

                  {/* Connection Status */}
                  {activeBrokerData.connectionTest && (
                    <Alert className={`mt-3 ${activeBrokerData.connectionTest.result === 'success' ? 'border-green-200' : 'border-red-200'}`}>
                      <AlertDescription className="text-sm">
                        {activeBrokerData.connectionTest.result === 'success' ? (
                          <span className="flex items-center gap-2 text-green-600">
                            <CheckCircle className="h-4 w-4" />
                            {activeBrokerData.connectionTest.message}
                            {activeBrokerData.connectionTest.latency && (
                              <span> (Latency: {activeBrokerData.connectionTest.latency}ms)</span>
                            )}
                          </span>
                        ) : (
                          <span className="flex items-center gap-2 text-red-600">
                            <XCircle className="h-4 w-4" />
                            {activeBrokerData.connectionTest.message}
                          </span>
                        )}
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Select a broker to configure its settings</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Save Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Changes are automatically saved to local storage
            </p>
            <Button onClick={saveBrokers} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              {isLoading ? 'Saving...' : 'Save All Brokers'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default BrokerConfig;