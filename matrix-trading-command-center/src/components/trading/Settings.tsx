import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTradingStore, BrokerConnection } from '@/stores/tradingStore';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { StatusIndicator } from '@/components/ui/StatusIndicator';
import { 
  Settings as SettingsIcon, 
  Wifi, 
  Key, 
  Database, 
  Shield,
  Bell,
  Monitor,
  Palette,
  Clock,
  Save,
  RefreshCw,
  Trash2,
  Plus,
  Check,
  X,
  AlertCircle,
  Server
} from 'lucide-react';

export const Settings: React.FC = () => {
  const { brokerConnections, updateBrokerConnection } = useTradingStore();
  const [activeTab, setActiveTab] = useState('brokers');
  const [showBrokerForm, setShowBrokerForm] = useState(false);
  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  
  const [brokerForm, setBrokerForm] = useState({
    name: '',
    type: 'alpaca' as 'alpaca' | 'binance' | 'ibkr',
    apiKey: '',
    secretKey: '',
    environment: 'paper' as 'paper' | 'live'
  });

  const [systemSettings, setSystemSettings] = useState({
    autoRefresh: true,
    refreshInterval: 5000,
    darkMode: true,
    notifications: true,
    soundAlerts: false,
    riskAlerts: true,
    maxOrdersPerMinute: 60,
    defaultPositionSize: 1000,
    stopLossPercent: 5,
    takeProfitPercent: 10
  });

  const tabs = [
    { id: 'brokers', name: 'Brokers', icon: Wifi },
    { id: 'api', name: 'API Keys', icon: Key },
    { id: 'system', name: 'System', icon: Monitor },
    { id: 'notifications', name: 'Alerts', icon: Bell },
    { id: 'trading', name: 'Trading', icon: SettingsIcon }
  ];

  const handleAddBroker = () => {
    if (!brokerForm.name || !brokerForm.apiKey || !brokerForm.secretKey) return;

    const newBroker: BrokerConnection = {
      id: Date.now().toString(),
      name: brokerForm.name,
      status: 'disconnected',
      latency: 0,
      lastHeartbeat: Date.now(),
      capabilities: []
    };

    updateBrokerConnection(newBroker.id, newBroker);
    
    setBrokerForm({
      name: '',
      type: 'alpaca',
      apiKey: '',
      secretKey: '',
      environment: 'paper'
    });
    setShowBrokerForm(false);
  };

  const handleConnectBroker = (brokerId: string) => {
    updateBrokerConnection(brokerId, { status: 'connecting' });
    
    // Simulate connection process
    setTimeout(() => {
      const success = Math.random() > 0.2; // 80% success rate
      
      if (success) {
        updateBrokerConnection(brokerId, { 
          status: 'connected', 
          latency: Math.floor(Math.random() * 50) + 10,
          lastHeartbeat: Date.now()
        });
      } else {
        updateBrokerConnection(brokerId, { status: 'error' });
      }
    }, 2000);
  };

  const handleDisconnectBroker = (brokerId: string) => {
    updateBrokerConnection(brokerId, { status: 'disconnected', latency: 0 });
  };

  const toggleApiKeyVisibility = (brokerId: string) => {
    setShowApiKeys(prev => ({
      ...prev,
      [brokerId]: !prev[brokerId]
    }));
  };

  const maskApiKey = (key: string) => {
    if (key.length <= 8) return '*'.repeat(key.length);
    return key.substring(0, 4) + '*'.repeat(key.length - 8) + key.substring(key.length - 4);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'brokers':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-green-400">Broker Connections</h3>
              <MatrixButton
                onClick={() => setShowBrokerForm(true)}
                className="flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Add Broker
              </MatrixButton>
            </div>

            <div className="space-y-4">
              {brokerConnections.map((broker) => (
                <MatrixCard key={broker.id} glow={broker.status === 'connected'}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-gray-800 rounded-lg flex items-center justify-center">
                        <Server className="w-6 h-6 text-green-400" />
                      </div>
                      <div>
                        <div className="font-bold text-green-400">{broker.name}</div>
                        <div className="text-sm text-green-600">
                          {broker.capabilities.join(', ') || 'No capabilities configured'}
                        </div>
                        {broker.latency > 0 && (
                          <div className="text-xs text-green-600">
                            Latency: {broker.latency}ms
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <StatusIndicator 
                        status={
                          broker.status === 'connected' ? 'online' :
                          broker.status === 'connecting' ? 'warning' :
                          broker.status === 'error' ? 'error' : 'offline'
                        }
                      >
                        {broker.status.toUpperCase()}
                      </StatusIndicator>

                      <div className="flex gap-2">
                        {broker.status === 'connected' ? (
                          <MatrixButton
                            variant="secondary"
                            size="sm"
                            onClick={() => handleDisconnectBroker(broker.id)}
                          >
                            <X className="w-3 h-3 mr-1" />
                            Disconnect
                          </MatrixButton>
                        ) : (
                          <MatrixButton
                            size="sm"
                            onClick={() => handleConnectBroker(broker.id)}
                            disabled={broker.status === 'connecting'}
                          >
                            {broker.status === 'connecting' ? (
                              <>
                                <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                                Connecting
                              </>
                            ) : (
                              <>
                                <Check className="w-3 h-3 mr-1" />
                                Connect
                              </>
                            )}
                          </MatrixButton>
                        )}
                        
                        <MatrixButton variant="secondary" size="sm">
                          <SettingsIcon className="w-3 h-3" />
                        </MatrixButton>
                        
                        <MatrixButton variant="secondary" size="sm">
                          <Trash2 className="w-3 h-3" />
                        </MatrixButton>
                      </div>
                    </div>
                  </div>
                </MatrixCard>
              ))}
            </div>

            {/* Broker Form Modal */}
            <AnimatePresence>
              {showBrokerForm && (
                <motion.div
                  className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <motion.div
                    className="matrix-card p-6 w-full max-w-md mx-4"
                    initial={{ scale: 0.9, y: 20 }}
                    animate={{ scale: 1, y: 0 }}
                    exit={{ scale: 0.9, y: 20 }}
                  >
                    <h2 className="text-xl font-bold matrix-text-glow mb-4">
                      Add Broker Connection
                    </h2>

                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-green-400 mb-1">Broker Name</label>
                        <MatrixInput
                          placeholder="Alpaca Trading"
                          value={brokerForm.name}
                          onChange={(e) => setBrokerForm({ ...brokerForm, name: e.target.value })}
                        />
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-1">Broker Type</label>
                        <select
                          value={brokerForm.type}
                          onChange={(e) => setBrokerForm({ ...brokerForm, type: e.target.value as any })}
                          className="matrix-input w-full px-3 py-2 rounded"
                        >
                          <option value="alpaca">Alpaca</option>
                          <option value="binance">Binance</option>
                          <option value="ibkr">Interactive Brokers</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-1">Environment</label>
                        <select
                          value={brokerForm.environment}
                          onChange={(e) => setBrokerForm({ ...brokerForm, environment: e.target.value as any })}
                          className="matrix-input w-full px-3 py-2 rounded"
                        >
                          <option value="paper">Paper Trading</option>
                          <option value="live">Live Trading</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-1">API Key</label>
                        <MatrixInput
                          type="password"
                          placeholder="Enter API key..."
                          value={brokerForm.apiKey}
                          onChange={(e) => setBrokerForm({ ...brokerForm, apiKey: e.target.value })}
                        />
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-1">Secret Key</label>
                        <MatrixInput
                          type="password"
                          placeholder="Enter secret key..."
                          value={brokerForm.secretKey}
                          onChange={(e) => setBrokerForm({ ...brokerForm, secretKey: e.target.value })}
                        />
                      </div>
                    </div>

                    <div className="flex gap-2 mt-6">
                      <MatrixButton
                        onClick={handleAddBroker}
                        className="flex-1"
                        disabled={!brokerForm.name || !brokerForm.apiKey || !brokerForm.secretKey}
                      >
                        Add Broker
                      </MatrixButton>
                      <MatrixButton
                        variant="secondary"
                        onClick={() => setShowBrokerForm(false)}
                      >
                        Cancel
                      </MatrixButton>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-bold text-green-400">API Key Management</h3>
            
            <div className="space-y-4">
              {brokerConnections.map((broker) => (
                <MatrixCard key={`api-${broker.id}`}>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-bold text-green-400">{broker.name}</div>
                        <div className="text-sm text-green-600">API Configuration</div>
                      </div>
                      <StatusIndicator 
                        status={broker.status === 'connected' ? 'online' : 'offline'}
                      >
                        {broker.status.toUpperCase()}
                      </StatusIndicator>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-green-400 mb-1">API Key</label>
                        <div className="flex gap-2">
                          <MatrixInput
                            type={showApiKeys[broker.id] ? 'text' : 'password'}
                            value="AKIAIOSFODNN7EXAMPLE"
                            readOnly
                            className="font-mono"
                          />
                          <MatrixButton
                            variant="secondary"
                            size="sm"
                            onClick={() => toggleApiKeyVisibility(broker.id)}
                          >
                            {showApiKeys[broker.id] ? <X className="w-4 h-4" /> : <Key className="w-4 h-4" />}
                          </MatrixButton>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm text-green-400 mb-1">Secret Key</label>
                        <div className="flex gap-2">
                          <MatrixInput
                            type={showApiKeys[broker.id] ? 'text' : 'password'}
                            value="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                            readOnly
                            className="font-mono"
                          />
                          <MatrixButton
                            variant="secondary"
                            size="sm"
                            onClick={() => toggleApiKeyVisibility(broker.id)}
                          >
                            {showApiKeys[broker.id] ? <X className="w-4 h-4" /> : <Key className="w-4 h-4" />}
                          </MatrixButton>
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <MatrixButton size="sm">
                        <RefreshCw className="w-3 h-3 mr-1" />
                        Rotate Keys
                      </MatrixButton>
                      <MatrixButton variant="secondary" size="sm">
                        <Save className="w-3 h-3 mr-1" />
                        Save
                      </MatrixButton>
                    </div>
                  </div>
                </MatrixCard>
              ))}
            </div>
          </div>
        );

      case 'system':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-bold text-green-400">System Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MatrixCard title="Display" glow>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Dark Mode</span>
                    <div className="w-12 h-6 bg-green-800 rounded-full p-1">
                      <div className="w-4 h-4 bg-green-400 rounded-full ml-auto"></div>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">Theme</label>
                    <select
                      value="matrix"
                      className="matrix-input w-full px-3 py-2 rounded"
                    >
                      <option value="matrix">Matrix Green</option>
                      <option value="cyber">Cyber Blue</option>
                      <option value="neon">Neon Pink</option>
                    </select>
                  </div>
                </div>
              </MatrixCard>

              <MatrixCard title="Performance" glow>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Auto Refresh</span>
                    <div className={`w-12 h-6 rounded-full p-1 ${
                      systemSettings.autoRefresh ? 'bg-green-800' : 'bg-gray-800'
                    }`}>
                      <div className={`w-4 h-4 bg-green-400 rounded-full transition-transform ${
                        systemSettings.autoRefresh ? 'translate-x-6' : ''
                      }`}></div>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Refresh Interval ({systemSettings.refreshInterval}ms)
                    </label>
                    <input
                      type="range"
                      min="1000"
                      max="30000"
                      step="1000"
                      value={systemSettings.refreshInterval}
                      onChange={(e) => setSystemSettings({
                        ...systemSettings,
                        refreshInterval: parseInt(e.target.value)
                      })}
                      className="w-full"
                    />
                  </div>
                </div>
              </MatrixCard>

              <MatrixCard title="Data" glow>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Save Local Data</span>
                    <div className="w-12 h-6 bg-green-800 rounded-full p-1">
                      <div className="w-4 h-4 bg-green-400 rounded-full ml-auto"></div>
                    </div>
                  </div>
                  
                  <div className="text-sm text-green-600">
                    Storage Used: 45.2 MB / 1 GB
                  </div>
                  
                  <MatrixButton variant="secondary" size="sm" className="w-full">
                    <Database className="w-3 h-3 mr-1" />
                    Clear Cache
                  </MatrixButton>
                </div>
              </MatrixCard>

              <MatrixCard title="Security" glow>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Encryption</span>
                    <Shield className="w-5 h-5 text-green-400" />
                  </div>
                  
                  <div className="text-sm text-green-600">
                    All data encrypted with AES-256
                  </div>
                  
                  <MatrixButton variant="secondary" size="sm" className="w-full">
                    <Key className="w-3 h-3 mr-1" />
                    Change Master Password
                  </MatrixButton>
                </div>
              </MatrixCard>
            </div>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-bold text-green-400">Notification Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MatrixCard title="Alert Types" glow>
                <div className="space-y-4">
                  {[
                    { key: 'notifications', label: 'Desktop Notifications', icon: Bell },
                    { key: 'soundAlerts', label: 'Sound Alerts', icon: AlertCircle },
                    { key: 'riskAlerts', label: 'Risk Alerts', icon: Shield }
                  ].map(({ key, label, icon: Icon }) => (
                    <div key={key} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4 text-green-400" />
                        <span className="text-green-400">{label}</span>
                      </div>
                      <div className={`w-12 h-6 rounded-full p-1 ${
                        systemSettings[key as keyof typeof systemSettings] ? 'bg-green-800' : 'bg-gray-800'
                      }`}>
                        <div className={`w-4 h-4 bg-green-400 rounded-full transition-transform ${
                          systemSettings[key as keyof typeof systemSettings] ? 'translate-x-6' : ''
                        }`}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </MatrixCard>

              <MatrixCard title="Alert Conditions" glow>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      P&L Alert Threshold ($)
                    </label>
                    <MatrixInput
                      type="number"
                      value="500"
                      placeholder="500"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Risk Utilization Alert (%)
                    </label>
                    <MatrixInput
                      type="number"
                      value="80"
                      placeholder="80"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Connection Timeout (seconds)
                    </label>
                    <MatrixInput
                      type="number"
                      value="30"
                      placeholder="30"
                      className="w-full"
                    />
                  </div>
                </div>
              </MatrixCard>
            </div>
          </div>
        );

      case 'trading':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-bold text-green-400">Trading Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MatrixCard title="Order Management" glow>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Max Orders Per Minute
                    </label>
                    <MatrixInput
                      type="number"
                      value={systemSettings.maxOrdersPerMinute}
                      onChange={(e) => setSystemSettings({
                        ...systemSettings,
                        maxOrdersPerMinute: parseInt(e.target.value)
                      })}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Default Position Size ($)
                    </label>
                    <MatrixInput
                      type="number"
                      value={systemSettings.defaultPositionSize}
                      onChange={(e) => setSystemSettings({
                        ...systemSettings,
                        defaultPositionSize: parseInt(e.target.value)
                      })}
                      className="w-full"
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Confirm Orders</span>
                    <div className="w-12 h-6 bg-green-800 rounded-full p-1">
                      <div className="w-4 h-4 bg-green-400 rounded-full ml-auto"></div>
                    </div>
                  </div>
                </div>
              </MatrixCard>

              <MatrixCard title="Risk Management" glow>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Default Stop Loss (%)
                    </label>
                    <MatrixInput
                      type="number"
                      value={systemSettings.stopLossPercent}
                      onChange={(e) => setSystemSettings({
                        ...systemSettings,
                        stopLossPercent: parseFloat(e.target.value)
                      })}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm text-green-400 mb-2">
                      Default Take Profit (%)
                    </label>
                    <MatrixInput
                      type="number"
                      value={systemSettings.takeProfitPercent}
                      onChange={(e) => setSystemSettings({
                        ...systemSettings,
                        takeProfitPercent: parseFloat(e.target.value)
                      })}
                      className="w-full"
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-green-400">Auto Risk Check</span>
                    <div className="w-12 h-6 bg-green-800 rounded-full p-1">
                      <div className="w-4 h-4 bg-green-400 rounded-full ml-auto"></div>
                    </div>
                  </div>
                </div>
              </MatrixCard>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold matrix-text-glow text-green-400">
            SETTINGS & CONFIGURATION
          </h1>
          <p className="text-green-600 mt-1">Configure your trading environment</p>
        </div>
        <MatrixButton className="flex items-center gap-2">
          <Save className="w-4 h-4" />
          Save All Changes
        </MatrixButton>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <MatrixButton
              key={tab.id}
              variant={activeTab === tab.id ? 'primary' : 'secondary'}
              onClick={() => setActiveTab(tab.id)}
              className="flex items-center gap-2"
            >
              <Icon className="w-4 h-4" />
              {tab.name}
            </MatrixButton>
          );
        })}
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {renderTabContent()}
      </motion.div>
    </div>
  );
};