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
  Brain, 
  Zap, 
  Cpu, 
  Database,
  Key,
  TestTube,
  Settings,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  DollarSign,
  Clock,
  BarChart3
} from 'lucide-react';

interface LLMProvider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'local' | 'custom';
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  apiKey: string;
  model: string;
  endpoint?: string;
  config: {
    temperature: number;
    maxTokens: number;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;
  };
  usage: {
    requestsToday: number;
    tokensToday: number;
    costToday: number;
    limitReached: boolean;
  };
  lastTested?: string;
}

interface AISettings {
  enabled: boolean;
  tradingEnabled: boolean;
  analysisEnabled: boolean;
  predictiveEnabled: boolean;
  autoOptimize: boolean;
  realTimeLearning: boolean;
  confidenceThreshold: number;
  maxRequestsPerHour: number;
  fallbackStrategy: string;
}

interface AIConfiguration {
  providers: LLMProvider[];
  settings: AISettings;
  models: {
    sentiment: string;
    technical: string;
    fundamental: string;
    news: string;
  };
  prompts: {
    analysis: string;
    decision: string;
    risk: string;
    news: string;
  };
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    avgResponseTime: number;
    totalRequests: number;
    successRate: number;
  };
}

const AIConfig: React.FC = () => {
  const [configuration, setConfiguration] = useState<AIConfiguration>({
    providers: [
      {
        id: 'openai_default',
        name: 'OpenAI GPT-4',
        type: 'openai',
        status: 'connected',
        apiKey: '',
        model: 'gpt-4',
        config: {
          temperature: 0.1,
          maxTokens: 2048,
          topP: 0.9,
          frequencyPenalty: 0.0,
          presencePenalty: 0.0
        },
        usage: {
          requestsToday: 127,
          tokensToday: 45632,
          costToday: 12.45,
          limitReached: false
        }
      }
    ],
    settings: {
      enabled: true,
      tradingEnabled: false,
      analysisEnabled: true,
      predictiveEnabled: true,
      autoOptimize: true,
      realTimeLearning: false,
      confidenceThreshold: 0.7,
      maxRequestsPerHour: 1000,
      fallbackStrategy: 'technical'
    },
    models: {
      sentiment: 'gpt-4',
      technical: 'gpt-4',
      fundamental: 'gpt-4',
      news: 'gpt-3.5-turbo'
    },
    prompts: {
      analysis: 'Analyze the current market conditions and provide insights...',
      decision: 'Based on the analysis, make a trading decision...',
      risk: 'Evaluate the risk of this potential trade...',
      news: 'Analyze the impact of this news on market sentiment...'
    },
    metrics: {
      accuracy: 0.847,
      precision: 0.892,
      recall: 0.823,
      f1Score: 0.856,
      avgResponseTime: 1.23,
      totalRequests: 15420,
      successRate: 0.987
    }
  });

  const [activeProvider, setActiveProvider] = useState<string>('openai_default');
  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [isTesting, setIsTesting] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const modelOptions = {
    openai: [
      { value: 'gpt-4', label: 'GPT-4' },
      { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
      { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
    ],
    anthropic: [
      { value: 'claude-3-opus', label: 'Claude 3 Opus' },
      { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
      { value: 'claude-3-haiku', label: 'Claude 3 Haiku' }
    ],
    local: [
      { value: 'llama-2-70b', label: 'Llama 2 70B' },
      { value: 'llama-2-13b', label: 'Llama 2 13B' },
      { value: 'mistral-7b', label: 'Mistral 7B' }
    ],
    custom: [
      { value: 'custom-model', label: 'Custom Model' }
    ]
  };

  useEffect(() => {
    loadAIConfiguration();
  }, []);

  const loadAIConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getAIConfig();
      if (config) {
        setConfiguration(config);
      }
    } catch (error) {
      console.error('Failed to load AI configuration:', error);
    }
  };

  const saveAIConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.saveAIConfig(configuration);
    } catch (error) {
      console.error('Failed to save AI configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const addProvider = () => {
    const newProvider: LLMProvider = {
      id: `provider_${Date.now()}`,
      name: 'New Provider',
      type: 'openai',
      status: 'disconnected',
      apiKey: '',
      model: 'gpt-3.5-turbo',
      config: {
        temperature: 0.1,
        maxTokens: 2048,
        topP: 0.9,
        frequencyPenalty: 0.0,
        presencePenalty: 0.0
      },
      usage: {
        requestsToday: 0,
        tokensToday: 0,
        costToday: 0,
        limitReached: false
      }
    };
    setConfiguration(prev => ({
      ...prev,
      providers: [...prev.providers, newProvider]
    }));
    setActiveProvider(newProvider.id);
  };

  const updateProvider = (id: string, updates: Partial<LLMProvider>) => {
    setConfiguration(prev => ({
      ...prev,
      providers: prev.providers.map(p => p.id === id ? { ...p, ...updates } : p)
    }));
  };

  const deleteProvider = (id: string) => {
    setConfiguration(prev => ({
      ...prev,
      providers: prev.providers.filter(p => p.id !== id)
    }));
    if (activeProvider === id && configuration.providers.length > 1) {
      setActiveProvider(configuration.providers[0].id);
    }
  };

  const testProvider = async (provider: LLMProvider) => {
    setIsTesting(provider.id);
    try {
      // Simulate API test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const success = Math.random() > 0.2;
      updateProvider(provider.id, {
        status: success ? 'connected' : 'error',
        lastTested: new Date().toISOString()
      });
    } catch (error) {
      updateProvider(provider.id, {
        status: 'error',
        lastTested: new Date().toISOString()
      });
    } finally {
      setIsTesting(null);
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

  const activeProviderData = configuration.providers.find(p => p.id === activeProvider);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI & LLM Configuration
          </CardTitle>
          <CardDescription>
            Configure AI models, LLM providers, and machine learning settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={configuration.settings.enabled}
                  onCheckedChange={(checked) => setConfiguration(prev => ({
                    ...prev,
                    settings: { ...prev.settings, enabled: checked }
                  }))}
                />
                <Label>AI System {configuration.settings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant={configuration.settings.tradingEnabled ? 'default' : 'outline'}>
                Trading: {configuration.settings.tradingEnabled ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            <Button onClick={saveAIConfiguration} disabled={saving}>
              <Settings className="h-4 w-4 mr-2" />
              {saving ? 'Saving...' : 'Save Configuration'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* AI Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Accuracy</p>
                <p className="text-2xl font-bold">{(configuration.metrics.accuracy * 100).toFixed(1)}%</p>
              </div>
              <BarChart3 className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">F1 Score</p>
                <p className="text-2xl font-bold">{configuration.metrics.f1Score.toFixed(3)}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
                <p className="text-2xl font-bold">{configuration.metrics.avgResponseTime}s</p>
              </div>
              <Clock className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold">{(configuration.metrics.successRate * 100).toFixed(1)}%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="providers" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="providers">Providers</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="prompts">Prompts</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="providers" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Providers List */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>LLM Providers</CardTitle>
                    <CardDescription>
                      Manage AI service providers and APIs
                    </CardDescription>
                  </div>
                  <Button onClick={addProvider} size="sm">
                    Add Provider
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {configuration.providers.map((provider) => (
                  <div
                    key={provider.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      activeProvider === provider.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setActiveProvider(provider.id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4" />
                        <h3 className="font-medium">{provider.name}</h3>
                        {getStatusBadge(provider.status)}
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            testProvider(provider);
                          }}
                          disabled={isTesting === provider.id}
                        >
                          <TestTube className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteProvider(provider.id);
                          }}
                        >
                          <Key className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Type:</span>
                        <div className="font-medium capitalize">{provider.type}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Model:</span>
                        <div className="font-medium">{provider.model}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Requests Today:</span>
                        <div className="font-medium">{provider.usage.requestsToday}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Cost Today:</span>
                        <div className="font-medium">${provider.usage.costToday.toFixed(2)}</div>
                      </div>
                    </div>

                    {provider.lastTested && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        Last tested: {new Date(provider.lastTested).toLocaleString()}
                      </div>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Provider Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Provider Configuration</CardTitle>
                <CardDescription>
                  {activeProviderData ? 'Configure provider settings and parameters' : 'Select a provider to configure'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeProviderData ? (
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="provider-name">Provider Name</Label>
                      <Input
                        id="provider-name"
                        value={activeProviderData.name}
                        onChange={(e) => updateProvider(activeProviderData.id, { name: e.target.value })}
                      />
                    </div>

                    <div>
                      <Label htmlFor="provider-type">Type</Label>
                      <Select
                        value={activeProviderData.type}
                        onValueChange={(value: any) => updateProvider(activeProviderData.id, { type: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="anthropic">Anthropic</SelectItem>
                          <SelectItem value="local">Local Model</SelectItem>
                          <SelectItem value="custom">Custom API</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label htmlFor="api-key">API Key</Label>
                      <div className="flex">
                        <Input
                          id="api-key"
                          type={showApiKeys[activeProviderData.id] ? 'text' : 'password'}
                          value={activeProviderData.apiKey}
                          onChange={(e) => updateProvider(activeProviderData.id, { apiKey: e.target.value })}
                          placeholder="Enter API key"
                          className="pr-10"
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          className="ml-2"
                          onClick={() => setShowApiKeys({
                            ...showApiKeys,
                            [activeProviderData.id]: !showApiKeys[activeProviderData.id]
                          })}
                        >
                          {showApiKeys[activeProviderData.id] ? 'Hide' : 'Show'}
                        </Button>
                      </div>
                    </div>

                    <div>
                      <Label htmlFor="model">Model</Label>
                      <Select
                        value={activeProviderData.model}
                        onValueChange={(value) => updateProvider(activeProviderData.id, { model: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {(modelOptions[activeProviderData.type] || []).map((model) => (
                            <SelectItem key={model.value} value={model.value}>
                              {model.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {activeProviderData.type === 'custom' && (
                      <div>
                        <Label htmlFor="endpoint">Custom Endpoint</Label>
                        <Input
                          id="endpoint"
                          value={activeProviderData.endpoint || ''}
                          onChange={(e) => updateProvider(activeProviderData.id, { endpoint: e.target.value })}
                          placeholder="https://api.example.com/v1/chat/completions"
                        />
                      </div>
                    )}

                    <Separator />

                    <div>
                      <h3 className="text-sm font-medium mb-3">Model Parameters</h3>
                      <div className="space-y-3">
                        <div>
                          <Label>Temperature: {activeProviderData.config.temperature}</Label>
                          <Slider
                            value={[activeProviderData.config.temperature]}
                            onValueChange={([value]) => updateProvider(activeProviderData.id, {
                              config: { ...activeProviderData.config, temperature: value }
                            })}
                            min={0}
                            max={2}
                            step={0.1}
                            className="mt-2"
                          />
                        </div>

                        <div>
                          <Label>Max Tokens: {activeProviderData.config.maxTokens}</Label>
                          <Slider
                            value={[activeProviderData.config.maxTokens]}
                            onValueChange={([value]) => updateProvider(activeProviderData.id, {
                              config: { ...activeProviderData.config, maxTokens: value }
                            })}
                            min={100}
                            max={8192}
                            step={100}
                            className="mt-2"
                          />
                        </div>

                        <div>
                          <Label>Top P: {activeProviderData.config.topP}</Label>
                          <Slider
                            value={[activeProviderData.config.topP]}
                            onValueChange={([value]) => updateProvider(activeProviderData.id, {
                              config: { ...activeProviderData.config, topP: value }
                            })}
                            min={0}
                            max={1}
                            step={0.1}
                            className="mt-2"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Select a provider to configure its settings</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Model Configuration</CardTitle>
              <CardDescription>
                Configure which models to use for different AI tasks
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="sentiment-model">Sentiment Analysis</Label>
                  <Select
                    value={configuration.models.sentiment}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      models: { ...prev.models, sentiment: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions.openai.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                          {model.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="technical-model">Technical Analysis</Label>
                  <Select
                    value={configuration.models.technical}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      models: { ...prev.models, technical: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions.openai.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                          {model.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="fundamental-model">Fundamental Analysis</Label>
                  <Select
                    value={configuration.models.fundamental}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      models: { ...prev.models, fundamental: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions.openai.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                          {model.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="news-model">News Analysis</Label>
                  <Select
                    value={configuration.models.news}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      models: { ...prev.models, news: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions.openai.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                          {model.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="prompts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Prompts</CardTitle>
              <CardDescription>
                Configure prompts used for different AI analysis tasks
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="analysis-prompt">Market Analysis Prompt</Label>
                <Textarea
                  id="analysis-prompt"
                  value={configuration.prompts.analysis}
                  onChange={(e) => setConfiguration(prev => ({
                    ...prev,
                    prompts: { ...prev.prompts, analysis: e.target.value }
                  }))}
                  placeholder="Enter the prompt for market analysis..."
                  rows={4}
                />
              </div>

              <div>
                <Label htmlFor="decision-prompt">Trading Decision Prompt</Label>
                <Textarea
                  id="decision-prompt"
                  value={configuration.prompts.decision}
                  onChange={(e) => setConfiguration(prev => ({
                    ...prev,
                    prompts: { ...prev.prompts, decision: e.target.value }
                  }))}
                  placeholder="Enter the prompt for trading decisions..."
                  rows={4}
                />
              </div>

              <div>
                <Label htmlFor="risk-prompt">Risk Assessment Prompt</Label>
                <Textarea
                  id="risk-prompt"
                  value={configuration.prompts.risk}
                  onChange={(e) => setConfiguration(prev => ({
                    ...prev,
                    prompts: { ...prev.prompts, risk: e.target.value }
                  }))}
                  placeholder="Enter the prompt for risk assessment..."
                  rows={4}
                />
              </div>

              <div>
                <Label htmlFor="news-prompt">News Impact Prompt</Label>
                <Textarea
                  id="news-prompt"
                  value={configuration.prompts.news}
                  onChange={(e) => setConfiguration(prev => ({
                    ...prev,
                    prompts: { ...prev.prompts, news: e.target.value }
                  }))}
                  placeholder="Enter the prompt for news analysis..."
                  rows={4}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>AI System Settings</CardTitle>
                <CardDescription>
                  Configure AI system behavior
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Trading Decisions</Label>
                    <p className="text-sm text-muted-foreground">
                      Allow AI to make trading decisions
                    </p>
                  </div>
                  <Switch
                    checked={configuration.settings.tradingEnabled}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, tradingEnabled: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Analysis</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable market analysis features
                    </p>
                  </div>
                  <Switch
                    checked={configuration.settings.analysisEnabled}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, analysisEnabled: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Predictive Analytics</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable price prediction models
                    </p>
                  </div>
                  <Switch
                    checked={configuration.settings.predictiveEnabled}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, predictiveEnabled: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Auto Optimization</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically optimize AI parameters
                    </p>
                  </div>
                  <Switch
                    checked={configuration.settings.autoOptimize}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, autoOptimize: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Real-time Learning</Label>
                    <p className="text-sm text-muted-foreground">
                      Learn from trading results in real-time
                    </p>
                  </div>
                  <Switch
                    checked={configuration.settings.realTimeLearning}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, realTimeLearning: checked }
                    }))}
                  />
                </div>

                <Separator />

                <div>
                  <Label>Confidence Threshold: {configuration.settings.confidenceThreshold}</Label>
                  <Slider
                    value={[configuration.settings.confidenceThreshold]}
                    onValueChange={([value]) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, confidenceThreshold: value }
                    }))}
                    min={0.1}
                    max={1}
                    step={0.1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Max Requests Per Hour</Label>
                  <Input
                    type="number"
                    value={configuration.settings.maxRequestsPerHour}
                    onChange={(e) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, maxRequestsPerHour: Number(e.target.value) }
                    }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Fallback Strategy</CardTitle>
                <CardDescription>
                  Strategy to use when AI is unavailable
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div>
                  <Label htmlFor="fallback-strategy">Fallback Strategy</Label>
                  <Select
                    value={configuration.settings.fallbackStrategy}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      settings: { ...prev.settings, fallbackStrategy: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="technical">Technical Analysis Only</SelectItem>
                      <SelectItem value="conservative">Conservative Trading</SelectItem>
                      <SelectItem value="hold">Hold Positions</SelectItem>
                      <SelectItem value="close">Close All Positions</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Alert className="mt-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    The fallback strategy will be used when AI providers are unavailable or when confidence levels are below the threshold.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>
                  AI system performance statistics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Accuracy</div>
                    <div className="text-2xl font-bold">{(configuration.metrics.accuracy * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Precision</div>
                    <div className="text-2xl font-bold">{(configuration.metrics.precision * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Recall</div>
                    <div className="text-2xl font-bold">{(configuration.metrics.recall * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">F1 Score</div>
                    <div className="text-2xl font-bold">{configuration.metrics.f1Score.toFixed(3)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Usage Statistics</CardTitle>
                <CardDescription>
                  API usage and cost tracking
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Total Requests</span>
                    <span className="font-medium">{configuration.metrics.totalRequests.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Success Rate</span>
                    <span className="font-medium">{(configuration.metrics.successRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Avg Response Time</span>
                    <span className="font-medium">{configuration.metrics.avgResponseTime}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Today's Cost</span>
                    <span className="font-medium">
                      ${configuration.providers.reduce((sum, p) => sum + p.usage.costToday, 0).toFixed(2)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AIConfig;