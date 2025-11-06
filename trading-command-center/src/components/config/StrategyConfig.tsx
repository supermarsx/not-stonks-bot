import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { 
  Plus, 
  Settings, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Brain,
  Copy,
  Trash2,
  Play,
  Pause,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';
import { toast } from 'sonner';
import { useConfigStore } from '@/stores/configStore';

interface Strategy {
  id: string;
  name: string;
  type: 'momentum' | 'mean-reversion' | 'arbitrage' | 'ai-ml' | 'scalping' | 'swing' | 'position';
  isEnabled: boolean;
  autoStart: boolean;
  riskProfile: 'conservative' | 'moderate' | 'aggressive';
  parameters: Record<string, any>;
  settings: {
    maxPositionSize: number;
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
  status: 'active' | 'paused' | 'stopped';
  lastRun?: Date;
}

interface StrategyTemplate {
  type: string;
  name: string;
  description: string;
  defaultParameters: Record<string, any>;
  riskProfiles: {
    conservative: Record<string, any>;
    moderate: Record<string, any>;
    aggressive: Record<string, any>;
  };
}

const StrategyConfig: React.FC = () => {
  const { config, updateConfig, validateConfig } = useConfigStore();
  const [strategies, setStrategies] = useState<Strategy[]>([
    {
      id: '1',
      name: 'RSI Mean Reversion',
      type: 'mean-reversion',
      isEnabled: true,
      autoStart: true,
      riskProfile: 'moderate',
      parameters: {
        rsiPeriod: 14,
        rsiOversold: 30,
        rsiOverbought: 70,
        stopLoss: 2,
        takeProfit: 4
      },
      settings: {
        maxPositionSize: 10000,
        confidence: 0.7,
        lookbackPeriod: 100
      },
      performance: {
        totalTrades: 156,
        winRate: 0.68,
        avgReturn: 0.024,
        sharpeRatio: 1.45,
        maxDrawdown: 0.08
      },
      status: 'active',
      lastRun: new Date(Date.now() - 3600000)
    },
    {
      id: '2',
      name: 'Momentum Breakout',
      type: 'momentum',
      isEnabled: true,
      autoStart: false,
      riskProfile: 'aggressive',
      parameters: {
        maPeriod: 20,
        breakoutThreshold: 0.02,
        volumeThreshold: 1.5,
        stopLoss: 3,
        takeProfit: 6
      },
      settings: {
        maxPositionSize: 5000,
        confidence: 0.6,
        lookbackPeriod: 50
      },
      performance: {
        totalTrades: 89,
        winRate: 0.61,
        avgReturn: 0.035,
        sharpeRatio: 1.23,
        maxDrawdown: 0.12
      },
      status: 'paused',
      lastRun: new Date(Date.now() - 7200000)
    },
    {
      id: '3',
      name: 'AI Price Predictor',
      type: 'ai-ml',
      isEnabled: false,
      autoStart: false,
      riskProfile: 'conservative',
      parameters: {
        modelType: 'lstm',
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001,
        predictionHorizon: 24
      },
      settings: {
        maxPositionSize: 20000,
        confidence: 0.8,
        lookbackPeriod: 200
      },
      performance: {
        totalTrades: 45,
        winRate: 0.73,
        avgReturn: 0.042,
        sharpeRatio: 1.67,
        maxDrawdown: 0.06
      },
      status: 'stopped'
    }
  ]);

  const [selectedStrategy, setSelectedStrategy] = useState<string>('1');
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newStrategy, setNewStrategy] = useState<Partial<Strategy>>({
    name: '',
    type: 'momentum',
    riskProfile: 'moderate'
  });

  const strategyTemplates: StrategyTemplate[] = [
    {
      type: 'momentum',
      name: 'Momentum Trading',
      description: 'Follow market trends and momentum',
      defaultParameters: {
        maPeriod: 20,
        breakoutThreshold: 0.02,
        volumeThreshold: 1.5
      },
      riskProfiles: {
        conservative: { maPeriod: 30, breakoutThreshold: 0.015, volumeThreshold: 2.0 },
        moderate: { maPeriod: 20, breakoutThreshold: 0.02, volumeThreshold: 1.5 },
        aggressive: { maPeriod: 15, breakoutThreshold: 0.025, volumeThreshold: 1.2 }
      }
    },
    {
      type: 'mean-reversion',
      name: 'Mean Reversion',
      description: 'Trade based on price returning to average',
      defaultParameters: {
        rsiPeriod: 14,
        rsiOversold: 30,
        rsiOverbought: 70
      },
      riskProfiles: {
        conservative: { rsiOversold: 25, rsiOverbought: 75 },
        moderate: { rsiOversold: 30, rsiOverbought: 70 },
        aggressive: { rsiOversold: 35, rsiOverbought: 65 }
      }
    },
    {
      type: 'arbitrage',
      name: 'Arbitrage',
      description: 'Exploit price differences across markets',
      defaultParameters: {
        minSpread: 0.001,
        maxLatency: 100,
        minVolume: 10000
      },
      riskProfiles: {
        conservative: { minSpread: 0.002, maxLatency: 50 },
        moderate: { minSpread: 0.001, maxLatency: 100 },
        aggressive: { minSpread: 0.0005, maxLatency: 200 }
      }
    },
    {
      type: 'ai-ml',
      name: 'AI/ML Strategy',
      description: 'Machine learning based trading',
      defaultParameters: {
        modelType: 'lstm',
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001
      },
      riskProfiles: {
        conservative: { epochs: 50, learningRate: 0.0001, confidence: 0.9 },
        moderate: { epochs: 100, learningRate: 0.001, confidence: 0.8 },
        aggressive: { epochs: 200, learningRate: 0.01, confidence: 0.7 }
      }
    }
  ];

  const currentStrategy = strategies.find(s => s.id === selectedStrategy);

  const updateStrategy = (strategyId: string, updates: Partial<Strategy>) => {
    setStrategies(prev => prev.map(strategy => 
      strategy.id === strategyId ? { ...strategy, ...updates } : strategy
    ));
  };

  const createStrategy = () => {
    if (!newStrategy.name || !newStrategy.type) {
      toast.error('Please fill in all required fields');
      return;
    }

    const template = strategyTemplates.find(t => t.type === newStrategy.type);
    const riskProfile = newStrategy.riskProfile || 'moderate';
    
    const strategy: Strategy = {
      id: Date.now().toString(),
      name: newStrategy.name!,
      type: newStrategy.type as any,
      isEnabled: false,
      autoStart: false,
      riskProfile,
      parameters: {
        ...template?.defaultParameters,
        ...template?.riskProfiles[riskProfile]
      },
      settings: {
        maxPositionSize: 10000,
        confidence: 0.7,
        lookbackPeriod: 100
      },
      performance: {
        totalTrades: 0,
        winRate: 0,
        avgReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0
      },
      status: 'stopped'
    };

    setStrategies(prev => [...prev, strategy]);
    setNewStrategy({ name: '', type: 'momentum', riskProfile: 'moderate' });
    setShowCreateDialog(false);
    toast.success(`Strategy "${strategy.name}" created successfully`);
  };

  const duplicateStrategy = (strategyId: string) => {
    const strategy = strategies.find(s => s.id === strategyId);
    if (!strategy) return;

    const duplicate: Strategy = {
      ...strategy,
      id: Date.now().toString(),
      name: `${strategy.name} (Copy)`,
      isEnabled: false,
      status: 'stopped'
    };

    setStrategies(prev => [...prev, duplicate]);
    toast.success('Strategy duplicated successfully');
  };

  const deleteStrategy = (strategyId: string) => {
    setStrategies(prev => prev.filter(s => s.id !== strategyId));
    if (selectedStrategy === strategyId) {
      setSelectedStrategy(strategies[0]?.id || '');
    }
    toast.success('Strategy deleted successfully');
  };

  const toggleStrategy = (strategyId: string) => {
    const strategy = strategies.find(s => s.id === strategyId);
    if (!strategy) return;

    const newStatus = strategy.status === 'active' ? 'paused' : 'active';
    updateStrategy(strategyId, { status: newStatus });
    toast.success(`Strategy ${newStatus === 'active' ? 'started' : 'paused'}`);
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'momentum':
        return <TrendingUp className="h-4 w-4" />;
      case 'mean-reversion':
        return <TrendingDown className="h-4 w-4" />;
      case 'arbitrage':
        return <BarChart3 className="h-4 w-4" />;
      case 'ai-ml':
        return <Brain className="h-4 w-4" />;
      default:
        return <Settings className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500';
      case 'paused':
        return 'bg-yellow-500';
      case 'stopped':
        return 'bg-gray-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-matrix-green">Strategy Configuration</h1>
          <p className="text-matrix-green/70 mt-1">Configure and manage your trading strategies</p>
        </div>
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button className="bg-matrix-green text-black hover:bg-matrix-green/80">
              <Plus className="h-4 w-4 mr-2" />
              Create Strategy
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-black border-matrix-green/20">
            <DialogHeader>
              <DialogTitle className="text-matrix-green">Create New Strategy</DialogTitle>
              <DialogDescription className="text-matrix-green/70">
                Choose a strategy type and configure its parameters
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label className="text-matrix-green">Strategy Name</Label>
                <Input
                  value={newStrategy.name || ''}
                  onChange={(e) => setNewStrategy(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Enter strategy name"
                  className="bg-black/40 border-matrix-green/20"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-matrix-green">Strategy Type</Label>
                <Select
                  value={newStrategy.type}
                  onValueChange={(value) => setNewStrategy(prev => ({ ...prev, type: value as any }))}
                >
                  <SelectTrigger className="bg-black/40 border-matrix-green/20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {strategyTemplates.map(template => (
                      <SelectItem key={template.type} value={template.type}>
                        <div className="flex items-center gap-2">
                          {getTypeIcon(template.type)}
                          {template.name}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-matrix-green">Risk Profile</Label>
                <Select
                  value={newStrategy.riskProfile}
                  onValueChange={(value) => setNewStrategy(prev => ({ ...prev, riskProfile: value as any }))}
                >
                  <SelectTrigger className="bg-black/40 border-matrix-green/20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="conservative">Conservative</SelectItem>
                    <SelectItem value="moderate">Moderate</SelectItem>
                    <SelectItem value="aggressive">Aggressive</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex gap-2 pt-4">
                <Button onClick={createStrategy} className="flex-1">
                  Create Strategy
                </Button>
                <Button variant="outline" onClick={() => setShowCreateDialog(false)} className="flex-1">
                  Cancel
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Strategy Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategies.map((strategy) => (
          <Card 
            key={strategy.id} 
            className={`border-matrix-green/20 bg-black/40 cursor-pointer transition-all hover:border-matrix-green/40 ${
              selectedStrategy === strategy.id ? 'ring-2 ring-matrix-green/50' : ''
            }`}
            onClick={() => setSelectedStrategy(strategy.id)}
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-matrix-green/20 rounded-lg flex items-center justify-center">
                    {getTypeIcon(strategy.type)}
                  </div>
                  <div>
                    <h3 className="font-medium text-matrix-green">{strategy.name}</h3>
                    <p className="text-xs text-matrix-green/60 capitalize">{strategy.type.replace('-', ' ')}</p>
                  </div>
                </div>
                <Badge 
                  variant="outline" 
                  className={`text-xs ${getStatusColor(strategy.status)}/20 border-current`}
                >
                  {strategy.status}
                </Badge>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/60">Risk Profile</span>
                  <span className="text-matrix-green capitalize">{strategy.riskProfile}</span>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/60">Win Rate</span>
                  <span className="text-matrix-green">{(strategy.performance.winRate * 100).toFixed(1)}%</span>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/60">Trades</span>
                  <span className="text-matrix-green">{strategy.performance.totalTrades}</span>
                </div>
                
                {strategy.lastRun && (
                  <div className="text-xs text-matrix-green/50">
                    Last run: {strategy.lastRun.toLocaleString()}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-1 mt-3 pt-3 border-t border-matrix-green/10">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleStrategy(strategy.id);
                  }}
                  className="h-8 px-2"
                >
                  {strategy.status === 'active' ? (
                    <Pause className="h-3 w-3" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    duplicateStrategy(strategy.id);
                  }}
                  className="h-8 px-2"
                >
                  <Copy className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteStrategy(strategy.id);
                  }}
                  className="h-8 px-2 text-red-400 hover:text-red-300"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Selected Strategy Configuration */}
      {currentStrategy && (
        <Card className="border-matrix-green/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-matrix-green flex items-center gap-2">
              <Settings className="h-5 w-5" />
              {currentStrategy.name} Configuration
            </CardTitle>
            <CardDescription className="text-matrix-green/70">
              Configure parameters and settings for this strategy
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="parameters" className="space-y-4">
              <TabsList className="bg-black/60 border border-matrix-green/20">
                <TabsTrigger value="parameters" className="data-[state=active]:bg-matrix-green/20">
                  <Settings className="h-4 w-4 mr-2" />
                  Parameters
                </TabsTrigger>
                <TabsTrigger value="settings" className="data-[state=active]:bg-matrix-green/20">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Settings
                </TabsTrigger>
                <TabsTrigger value="performance" className="data-[state=active]:bg-matrix-green/20">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Performance
                </TabsTrigger>
              </TabsList>

              <TabsContent value="parameters" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(currentStrategy.parameters).map(([key, value]) => (
                    <div key={key} className="space-y-2">
                      <Label className="text-matrix-green capitalize">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </Label>
                      <Input
                        type={typeof value === 'number' ? 'number' : 'text'}
                        value={value}
                        onChange={(e) => {
                          const newValue = typeof value === 'number' ? 
                            parseFloat(e.target.value) : e.target.value;
                          updateStrategy(currentStrategy.id, {
                            parameters: {
                              ...currentStrategy.parameters,
                              [key]: newValue
                            }
                          });
                        }}
                        className="bg-black/40 border-matrix-green/20"
                      />
                    </div>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="settings" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-matrix-green">Enabled</Label>
                      <p className="text-xs text-matrix-green/60">Enable this strategy</p>
                    </div>
                    <Switch
                      checked={currentStrategy.isEnabled}
                      onCheckedChange={(checked) => updateStrategy(currentStrategy.id, { isEnabled: checked })}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-matrix-green">Auto Start</Label>
                      <p className="text-xs text-matrix-green/60">Start automatically on system boot</p>
                    </div>
                    <Switch
                      checked={currentStrategy.autoStart}
                      onCheckedChange={(checked) => updateStrategy(currentStrategy.id, { autoStart: checked })}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Max Position Size</Label>
                    <Input
                      type="number"
                      value={currentStrategy.settings.maxPositionSize}
                      onChange={(e) => updateStrategy(currentStrategy.id, {
                        settings: {
                          ...currentStrategy.settings,
                          maxPositionSize: parseFloat(e.target.value)
                        }
                      })}
                      className="bg-black/40 border-matrix-green/20"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Confidence Threshold</Label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={currentStrategy.settings.confidence}
                      onChange={(e) => updateStrategy(currentStrategy.id, {
                        settings: {
                          ...currentStrategy.settings,
                          confidence: parseFloat(e.target.value)
                        }
                      })}
                      className="bg-black/40 border-matrix-green/20"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Lookback Period</Label>
                    <Input
                      type="number"
                      value={currentStrategy.settings.lookbackPeriod}
                      onChange={(e) => updateStrategy(currentStrategy.id, {
                        settings: {
                          ...currentStrategy.settings,
                          lookbackPeriod: parseInt(e.target.value)
                        }
                      })}
                      className="bg-black/40 border-matrix-green/20"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-matrix-green">Risk Profile</Label>
                    <Select
                      value={currentStrategy.riskProfile}
                      onValueChange={(value: 'conservative' | 'moderate' | 'aggressive') => 
                        updateStrategy(currentStrategy.id, { riskProfile: value })
                      }
                    >
                      <SelectTrigger className="bg-black/40 border-matrix-green/20">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="conservative">Conservative</SelectItem>
                        <SelectItem value="moderate">Moderate</SelectItem>
                        <SelectItem value="aggressive">Aggressive</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="performance" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <Card className="border-matrix-green/20 bg-black/20">
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-matrix-green">{currentStrategy.performance.totalTrades}</div>
                      <div className="text-sm text-matrix-green/60">Total Trades</div>
                    </CardContent>
                  </Card>

                  <Card className="border-matrix-green/20 bg-black/20">
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-matrix-green">{(currentStrategy.performance.winRate * 100).toFixed(1)}%</div>
                      <div className="text-sm text-matrix-green/60">Win Rate</div>
                    </CardContent>
                  </Card>

                  <Card className="border-matrix-green/20 bg-black/20">
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-matrix-green">{(currentStrategy.performance.avgReturn * 100).toFixed(2)}%</div>
                      <div className="text-sm text-matrix-green/60">Avg Return</div>
                    </CardContent>
                  </Card>

                  <Card className="border-matrix-green/20 bg-black/20">
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-matrix-green">{currentStrategy.performance.sharpeRatio.toFixed(2)}</div>
                      <div className="text-sm text-matrix-green/60">Sharpe Ratio</div>
                    </CardContent>
                  </Card>

                  <Card className="border-matrix-green/20 bg-black/20">
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-matrix-green">{(currentStrategy.performance.maxDrawdown * 100).toFixed(1)}%</div>
                      <div className="text-sm text-matrix-green/60">Max Drawdown</div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default StrategyConfig;