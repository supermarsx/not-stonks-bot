import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  TrendingUp, 
  BarChart3, 
  Zap, 
  Settings,
  Play,
  Pause,
  Plus,
  Trash2,
  Copy,
  FileText,
  Brain,
  Target,
  Activity,
  AlertCircle
} from 'lucide-react';

interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'momentum' | 'mean_reversion' | 'arbitrage' | 'trend_following' | 'pairs_trading' | 'ai_ml' | 'custom';
  status: 'active' | 'inactive' | 'paused' | 'testing';
  parameters: Record<string, any>;
  settings: {
    enabled: boolean;
    autoStart: boolean;
    maxPositionSize: number;
    riskLevel: 'low' | 'medium' | 'high';
    confidence: number;
    lookbackPeriod: number;
  };
  performance: {
    totalTrades: number;
    winRate: number;
    avgReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
  createdAt: string;
  lastModified: string;
}

interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  type: string;
  defaultParameters: Record<string, any>;
  riskProfile: 'conservative' | 'moderate' | 'aggressive';
  expectedReturn: string;
  riskLevel: string;
}

const StrategyConfig: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [templates] = useState<StrategyTemplate[]>([
    {
      id: 'momentum_template',
      name: 'Momentum Strategy',
      description: 'Trend-following strategy based on price momentum',
      type: 'momentum',
      defaultParameters: {
        period: 20,
        threshold: 0.02,
        maxHoldPeriod: 30
      },
      riskProfile: 'moderate',
      expectedReturn: '8-12%',
      riskLevel: 'Medium'
    },
    {
      id: 'mean_reversion_template',
      name: 'Mean Reversion',
      description: 'Strategy based on price returning to average',
      type: 'mean_reversion',
      defaultParameters: {
        lookback: 50,
        stdThreshold: 2.0,
        holdPeriod: 5
      },
      riskProfile: 'conservative',
      expectedReturn: '6-10%',
      riskLevel: 'Low'
    },
    {
      id: 'arbitrage_template',
      name: 'Arbitrage Strategy',
      description: 'Risk-free arbitrage between correlated instruments',
      type: 'arbitrage',
      defaultParameters: {
        correlationThreshold: 0.8,
        minSpread: 0.01,
        maxExposure: 0.05
      },
      riskProfile: 'conservative',
      expectedReturn: '3-8%',
      riskLevel: 'Low'
    },
    {
      id: 'ai_ml_template',
      name: 'AI/ML Strategy',
      description: 'Machine learning based trading strategy',
      type: 'ai_ml',
      defaultParameters: {
        modelType: 'random_forest',
        features: ['price', 'volume', 'rsi', 'macd'],
        predictionHorizon: 1,
        retrainFrequency: 'daily'
      },
      riskProfile: 'aggressive',
      expectedReturn: '15-25%',
      riskLevel: 'High'
    }
  ]);

  const [activeStrategy, setActiveStrategy] = useState<string | null>(null);
  const [showTemplateModal, setShowTemplateModal] = useState(false);
  const [newStrategyName, setNewStrategyName] = useState('');

  const strategyTypes = [
    { value: 'momentum', label: 'Momentum Strategy' },
    { value: 'mean_reversion', label: 'Mean Reversion' },
    { value: 'arbitrage', label: 'Arbitrage' },
    { value: 'trend_following', label: 'Trend Following' },
    { value: 'pairs_trading', label: 'Pairs Trading' },
    { value: 'ai_ml', label: 'AI/ML Strategy' },
    { value: 'custom', label: 'Custom Strategy' }
  ];

  const riskLevels = [
    { value: 'low', label: 'Low Risk' },
    { value: 'medium', label: 'Medium Risk' },
    { value: 'high', label: 'High Risk' }
  ];

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const savedStrategies = await window.electronAPI?.getStrategies() || [];
      setStrategies(savedStrategies);
    } catch (error) {
      console.error('Failed to load strategies:', error);
    }
  };

  const saveStrategies = async () => {
    try {
      await window.electronAPI?.saveStrategies(strategies);
    } catch (error) {
      console.error('Failed to save strategies:', error);
    }
  };

  const createStrategy = (template: StrategyTemplate, customName: string) => {
    const newStrategy: Strategy = {
      id: `strategy_${Date.now()}`,
      name: customName,
      description: template.description,
      type: template.type as Strategy['type'],
      status: 'inactive',
      parameters: { ...template.defaultParameters },
      settings: {
        enabled: true,
        autoStart: false,
        maxPositionSize: 0.1,
        riskLevel: template.riskProfile as any,
        confidence: 0.7,
        lookbackPeriod: 20
      },
      performance: {
        totalTrades: 0,
        winRate: 0,
        avgReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0
      },
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString()
    };

    setStrategies([...strategies, newStrategy]);
    setActiveStrategy(newStrategy.id);
    setShowTemplateModal(false);
    setNewStrategyName('');
  };

  const updateStrategy = (id: string, updates: Partial<Strategy>) => {
    setStrategies(strategies.map(strategy => 
      strategy.id === id 
        ? { ...strategy, ...updates, lastModified: new Date().toISOString() }
        : strategy
    ));
  };

  const deleteStrategy = (id: string) => {
    setStrategies(strategies.filter(strategy => strategy.id !== id));
    if (activeStrategy === id) {
      setActiveStrategy(null);
    }
  };

  const duplicateStrategy = (strategy: Strategy) => {
    const duplicated: Strategy = {
      ...strategy,
      id: `strategy_${Date.now()}`,
      name: `${strategy.name} (Copy)`,
      status: 'inactive',
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString(),
      performance: {
        totalTrades: 0,
        winRate: 0,
        avgReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0
      }
    };

    setStrategies([...strategies, duplicated]);
    setActiveStrategy(duplicated.id);
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      active: 'default',
      inactive: 'outline',
      paused: 'secondary',
      testing: 'secondary'
    };

    const colors: Record<string, string> = {
      active: 'bg-green-500',
      inactive: 'bg-gray-400',
      paused: 'bg-yellow-500',
      testing: 'bg-blue-500'
    };

    return (
      <Badge variant={variants[status] || 'outline'} className="gap-1">
        <div className={`h-2 w-2 rounded-full ${colors[status] || 'bg-gray-400'}`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'momentum':
      case 'trend_following':
        return <TrendingUp className="h-4 w-4" />;
      case 'mean_reversion':
        return <BarChart3 className="h-4 w-4" />;
      case 'arbitrage':
        return <Target className="h-4 w-4" />;
      case 'ai_ml':
        return <Brain className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const activeStrategyData = strategies.find(strategy => strategy.id === activeStrategy);

  const StrategyTemplateModal = () => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <Card className="w-full max-w-4xl max-h-[80vh] overflow-auto">
        <CardHeader>
          <CardTitle>Create New Strategy</CardTitle>
          <CardDescription>
            Choose a template and configure your strategy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <Label htmlFor="strategy-name">Strategy Name</Label>
            <Input
              id="strategy-name"
              value={newStrategyName}
              onChange={(e) => setNewStrategyName(e.target.value)}
              placeholder="Enter strategy name"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {templates.map((template) => (
              <Card 
                key={template.id}
                className="cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => newStrategyName && createStrategy(template, newStrategyName)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(template.type)}
                      <CardTitle className="text-base">{template.name}</CardTitle>
                    </div>
                    <Badge variant={template.riskProfile === 'conservative' ? 'default' : 
                                   template.riskProfile === 'moderate' ? 'secondary' : 'destructive'}>
                      {template.riskProfile}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <CardDescription>{template.description}</CardDescription>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Expected Return:</span>
                      <div className="font-medium">{template.expectedReturn}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Risk Level:</span>
                      <div className="font-medium">{template.riskLevel}</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-sm text-muted-foreground">Default Parameters:</span>
                    <div className="bg-gray-50 p-2 rounded text-xs font-mono">
                      {JSON.stringify(template.defaultParameters, null, 2)}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setShowTemplateModal(false)}>
              Cancel
            </Button>
            <Button 
              onClick={() => {
                // Create custom strategy
                const customStrategy: Strategy = {
                  id: `strategy_${Date.now()}`,
                  name: newStrategyName || 'Custom Strategy',
                  description: 'User-defined custom strategy',
                  type: 'custom',
                  status: 'inactive',
                  parameters: {},
                  settings: {
                    enabled: true,
                    autoStart: false,
                    maxPositionSize: 0.1,
                    riskLevel: 'medium',
                    confidence: 0.7,
                    lookbackPeriod: 20
                  },
                  performance: {
                    totalTrades: 0,
                    winRate: 0,
                    avgReturn: 0,
                    sharpeRatio: 0,
                    maxDrawdown: 0
                  },
                  createdAt: new Date().toISOString(),
                  lastModified: new Date().toISOString()
                };
                setStrategies([...strategies, customStrategy]);
                setActiveStrategy(customStrategy.id);
                setShowTemplateModal(false);
                setNewStrategyName('');
              }}
            >
              Create Custom Strategy
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Strategy Configuration
          </CardTitle>
          <CardDescription>
            Configure trading strategies and their parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-between items-center">
            <div className="flex gap-4">
              <p className="text-sm text-muted-foreground">
                Active strategies: {strategies.filter(s => s.status === 'active').length} of {strategies.length}
              </p>
              <p className="text-sm text-muted-foreground">
                Total P&L: ${strategies.reduce((sum, s) => sum + s.performance.avgReturn, 0).toFixed(2)}
              </p>
            </div>
            <Button onClick={() => setShowTemplateModal(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Strategy
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Strategies List */}
        <Card>
          <CardHeader>
            <CardTitle>Strategies</CardTitle>
            <CardDescription>
              Manage your trading strategies
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {strategies.map((strategy) => (
              <div
                key={strategy.id}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  activeStrategy === strategy.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                }`}
                onClick={() => setActiveStrategy(strategy.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getTypeIcon(strategy.type)}
                    <h3 className="font-medium">{strategy.name}</h3>
                    {getStatusBadge(strategy.status)}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        duplicateStrategy(strategy);
                      }}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteStrategy(strategy.id);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <div className="font-medium capitalize">{strategy.type.replace('_', ' ')}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Risk Level:</span>
                    <div className="font-medium capitalize">{strategy.settings.riskLevel}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Trades:</span>
                    <div className="font-medium">{strategy.performance.totalTrades}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Win Rate:</span>
                    <div className="font-medium">{strategy.performance.winRate.toFixed(1)}%</div>
                  </div>
                </div>

                <div className="mt-2 text-xs text-muted-foreground">
                  Modified: {new Date(strategy.lastModified).toLocaleString()}
                </div>
              </div>
            ))}

            {strategies.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <TrendingUp className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No strategies configured</p>
                <Button onClick={() => setShowTemplateModal(true)} variant="outline" className="mt-2" size="sm">
                  Create your first strategy
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Strategy Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>
              {activeStrategyData ? 'Configure strategy settings and parameters' : 'Select a strategy to configure'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {activeStrategyData ? (
              <Tabs defaultValue="basic" className="space-y-4">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="basic">Basic</TabsTrigger>
                  <TabsTrigger value="parameters">Parameters</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                </TabsList>

                <TabsContent value="basic" className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <Label htmlFor="strategy-name-edit">Name</Label>
                      <Input
                        id="strategy-name-edit"
                        value={activeStrategyData.name}
                        onChange={(e) => updateStrategy(activeStrategyData.id, { name: e.target.value })}
                      />
                    </div>

                    <div>
                      <Label htmlFor="strategy-type">Type</Label>
                      <Select
                        value={activeStrategyData.type}
                        onValueChange={(value) => updateStrategy(activeStrategyData.id, { type: value as Strategy['type'] })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {strategyTypes.map((type) => (
                            <SelectItem key={type.value} value={type.value}>
                              {type.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label htmlFor="strategy-description">Description</Label>
                      <Textarea
                        id="strategy-description"
                        value={activeStrategyData.description}
                        onChange={(e) => updateStrategy(activeStrategyData.id, { description: e.target.value })}
                        placeholder="Strategy description"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Status</Label>
                        <p className="text-sm text-muted-foreground">
                          {activeStrategyData.status === 'active' ? 'Strategy is currently running' : 
                           activeStrategyData.status === 'paused' ? 'Strategy is paused' :
                           'Strategy is not running'}
                        </p>
                      </div>
                      <Button
                        variant={activeStrategyData.status === 'active' ? 'destructive' : 'default'}
                        size="sm"
                        onClick={() => {
                          const newStatus = activeStrategyData.status === 'active' ? 'inactive' : 'active';
                          updateStrategy(activeStrategyData.id, { status: newStatus });
                        }}
                      >
                        {activeStrategyData.status === 'active' ? (
                          <>
                            <Pause className="h-4 w-4 mr-2" />
                            Stop
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4 mr-2" />
                            Start
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="parameters" className="space-y-4">
                  <div className="space-y-4">
                    {Object.entries(activeStrategyData.parameters).map(([key, value]) => (
                      <div key={key}>
                        <Label htmlFor={`param-${key}`}>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</Label>
                        <Input
                          id={`param-${key}`}
                          value={String(value)}
                          onChange={(e) => {
                            const newParams = {
                              ...activeStrategyData.parameters,
                              [key]: e.target.value
                            };
                            updateStrategy(activeStrategyData.id, { parameters: newParams });
                          }}
                        />
                      </div>
                    ))}

                    <Separator />

                    <div>
                      <Label htmlFor="new-param-name">Add Parameter</Label>
                      <div className="flex gap-2">
                        <Input
                          id="new-param-name"
                          placeholder="Parameter name"
                        />
                        <Button variant="outline" size="sm">
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="settings" className="space-y-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable Strategy</Label>
                        <p className="text-sm text-muted-foreground">
                          Allow this strategy to trade
                        </p>
                      </div>
                      <Switch
                        checked={activeStrategyData.settings.enabled}
                        onCheckedChange={(checked) => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            enabled: checked
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto Start</Label>
                        <p className="text-sm text-muted-foreground">
                          Automatically start strategy on system startup
                        </p>
                      </div>
                      <Switch
                        checked={activeStrategyData.settings.autoStart}
                        onCheckedChange={(checked) => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            autoStart: checked
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                      />
                    </div>

                    <div>
                      <Label>Risk Level</Label>
                      <Select
                        value={activeStrategyData.settings.riskLevel}
                        onValueChange={(value: 'low' | 'medium' | 'high') => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            riskLevel: value
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {riskLevels.map((level) => (
                            <SelectItem key={level.value} value={level.value}>
                              {level.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Max Position Size: {activeStrategyData.settings.maxPositionSize}%</Label>
                      <Slider
                        value={[activeStrategyData.settings.maxPositionSize * 100]}
                        onValueChange={([value]) => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            maxPositionSize: value / 100
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                        max={100}
                        step={1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label>Confidence Threshold: {activeStrategyData.settings.confidence}</Label>
                      <Slider
                        value={[activeStrategyData.settings.confidence]}
                        onValueChange={([value]) => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            confidence: value
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                        min={0}
                        max={1}
                        step={0.1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label>Lookback Period: {activeStrategyData.settings.lookbackPeriod} days</Label>
                      <Slider
                        value={[activeStrategyData.settings.lookbackPeriod]}
                        onValueChange={([value]) => {
                          const newSettings = {
                            ...activeStrategyData.settings,
                            lookbackPeriod: value
                          };
                          updateStrategy(activeStrategyData.id, { settings: newSettings });
                        }}
                        min={1}
                        max={365}
                        step={1}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="performance" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Total Trades</div>
                      <div className="text-2xl font-bold">{activeStrategyData.performance.totalTrades}</div>
                    </div>
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Win Rate</div>
                      <div className="text-2xl font-bold">{activeStrategyData.performance.winRate.toFixed(1)}%</div>
                    </div>
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Avg Return</div>
                      <div className="text-2xl font-bold">${activeStrategyData.performance.avgReturn.toFixed(2)}</div>
                    </div>
                    <div className="p-3 border rounded">
                      <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
                      <div className="text-2xl font-bold">{activeStrategyData.performance.sharpeRatio.toFixed(2)}</div>
                    </div>
                  </div>

                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Max Drawdown</div>
                    <div className="text-xl font-bold">{activeStrategyData.performance.maxDrawdown.toFixed(2)}%</div>
                  </div>

                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Performance metrics are updated in real-time while the strategy is running.
                    </AlertDescription>
                  </Alert>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Select a strategy to configure its settings</p>
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
            <Button onClick={saveStrategies}>
              <FileText className="h-4 w-4 mr-2" />
              Save All Strategies
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Template Modal */}
      {showTemplateModal && <StrategyTemplateModal />}
    </div>
  );
};

export default StrategyConfig;