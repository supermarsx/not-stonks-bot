import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  LogOut, 
  TrendingUp, 
  TrendingDown, 
  Clock,
  Target,
  AlertTriangle,
  Settings,
  Plus,
  Trash2,
  Copy,
  BarChart3,
  Zap,
  Shield,
  DollarSign,
  Percent
} from 'lucide-react';

interface ExitStrategy {
  id: string;
  name: string;
  description: string;
  type: 'stop_loss' | 'take_profit' | 'trailing_stop' | 'time_based' | 'profit_target' | 'correlation_exit' | 'custom';
  status: 'active' | 'inactive' | 'testing';
  priority: number;
  config: {
    enabled: boolean;
    percentage?: number;
    absolute?: number;
    timePeriod?: number;
    trailingDistance?: number;
    minProfit?: number;
    maxLoss?: number;
    conditions?: string;
  };
  performance: {
    totalExits: number;
    avgExitTime: number;
    successRate: number;
    avgProfit: number;
    avgLoss: number;
  };
  combinations: string[];
  createdAt: string;
}

const ExitConfig: React.FC = () => {
  const [exitStrategies, setExitStrategies] = useState<ExitStrategy[]>([
    {
      id: 'stop_loss_default',
      name: 'Stop Loss 5%',
      description: 'Standard 5% stop loss to limit downside risk',
      type: 'stop_loss',
      status: 'active',
      priority: 1,
      config: {
        enabled: true,
        percentage: 5
      },
      performance: {
        totalExits: 45,
        avgExitTime: 2.3,
        successRate: 0.75,
        avgProfit: -2.1,
        avgLoss: -5.0
      },
      combinations: ['take_profit_default'],
      createdAt: new Date().toISOString()
    },
    {
      id: 'take_profit_default',
      name: 'Take Profit 10%',
      description: 'Take profit at 10% gain to secure profits',
      type: 'take_profit',
      status: 'active',
      priority: 2,
      config: {
        enabled: true,
        percentage: 10
      },
      performance: {
        totalExits: 32,
        avgExitTime: 4.7,
        successRate: 0.68,
        avgProfit: 10.2,
        avgLoss: 0
      },
      combinations: [],
      createdAt: new Date().toISOString()
    },
    {
      id: 'trailing_stop_default',
      name: 'Trailing Stop 3%',
      description: 'Trailing stop that follows price upwards',
      type: 'trailing_stop',
      status: 'active',
      priority: 3,
      config: {
        enabled: true,
        trailingDistance: 3
      },
      performance: {
        totalExits: 28,
        avgExitTime: 6.1,
        successRate: 0.71,
        avgProfit: 8.7,
        avgLoss: -3.2
      },
      combinations: [],
      createdAt: new Date().toISOString()
    }
  ]);

  const [activeStrategy, setActiveStrategy] = useState<string>('stop_loss_default');
  const [showTemplates, setShowTemplates] = useState(false);
  const [editingCombination, setEditingCombination] = useState<string | null>(null);

  const strategyTypes = [
    { value: 'stop_loss', label: 'Stop Loss', icon: TrendingDown, description: 'Exit when loss reaches threshold' },
    { value: 'take_profit', label: 'Take Profit', icon: TrendingUp, description: 'Exit when profit reaches target' },
    { value: 'trailing_stop', label: 'Trailing Stop', icon: BarChart3, description: 'Dynamic stop that follows price' },
    { value: 'time_based', label: 'Time Based', icon: Clock, description: 'Exit after specified time period' },
    { value: 'profit_target', label: 'Profit Target', icon: Target, description: 'Exit at specific profit level' },
    { value: 'correlation_exit', label: 'Correlation Exit', icon: Zap, description: 'Exit based on correlation changes' },
    { value: 'custom', label: 'Custom', icon: Settings, description: 'Custom exit logic' }
  ];

  const exitTemplates = [
    {
      id: 'conservative_exit',
      name: 'Conservative Exit',
      description: 'Safe exit strategy with tight risk control',
      config: {
        stopLoss: 3,
        takeProfit: 6,
        trailingStop: 2
      }
    },
    {
      id: 'aggressive_exit',
      name: 'Aggressive Exit',
      description: 'Higher risk/reward exit strategy',
      config: {
        stopLoss: 8,
        takeProfit: 15,
        trailingStop: 5
      }
    },
    {
      id: 'swing_exit',
      name: 'Swing Trading Exit',
      description: 'Optimized for swing trading timeframes',
      config: {
        stopLoss: 4,
        takeProfit: 12,
        timeBased: 10 // days
      }
    },
    {
      id: 'scalping_exit',
      name: 'Scalping Exit',
      description: 'Quick entry/exit strategy',
      config: {
        stopLoss: 1,
        takeProfit: 2,
        timeBased: 1 // hour
      }
    }
  ];

  useEffect(() => {
    loadExitStrategies();
  }, []);

  const loadExitStrategies = async () => {
    try {
      const saved = await window.electronAPI?.getExitStrategies();
      if (saved) {
        setExitStrategies(saved);
      }
    } catch (error) {
      console.error('Failed to load exit strategies:', error);
    }
  };

  const saveExitStrategies = async () => {
    try {
      await window.electronAPI?.saveExitStrategies(exitStrategies);
    } catch (error) {
      console.error('Failed to save exit strategies:', error);
    }
  };

  const createExitStrategy = (template?: any) => {
    const newStrategy: ExitStrategy = {
      id: `exit_${Date.now()}`,
      name: 'New Exit Strategy',
      description: 'Custom exit strategy',
      type: 'stop_loss',
      status: 'inactive',
      priority: exitStrategies.length + 1,
      config: {
        enabled: true,
        percentage: 5
      },
      performance: {
        totalExits: 0,
        avgExitTime: 0,
        successRate: 0,
        avgProfit: 0,
        avgLoss: 0
      },
      combinations: [],
      createdAt: new Date().toISOString()
    };

    if (template) {
      newStrategy.name = template.name;
      newStrategy.description = template.description;
      // Apply template configuration
      Object.keys(template.config).forEach(key => {
        if (key === 'stopLoss') {
          newStrategy.config.percentage = template.config[key];
        } else if (key === 'timeBased') {
          newStrategy.config.timePeriod = template.config[key];
        }
      });
    }

    setExitStrategies([...exitStrategies, newStrategy]);
    setActiveStrategy(newStrategy.id);
  };

  const updateExitStrategy = (id: string, updates: Partial<ExitStrategy>) => {
    setExitStrategies(strategies => 
      strategies.map(strategy => 
        strategy.id === id ? { ...strategy, ...updates } : strategy
      )
    );
  };

  const deleteExitStrategy = (id: string) => {
    setExitStrategies(strategies => 
      strategies.filter(strategy => strategy.id !== id)
    );
    if (activeStrategy === id) {
      const remaining = exitStrategies.filter(s => s.id !== id);
      setActiveStrategy(remaining.length > 0 ? remaining[0].id : '');
    }
  };

  const duplicateExitStrategy = (strategy: ExitStrategy) => {
    const duplicated: ExitStrategy = {
      ...strategy,
      id: `exit_${Date.now()}`,
      name: `${strategy.name} (Copy)`,
      status: 'inactive',
      performance: {
        totalExits: 0,
        avgExitTime: 0,
        successRate: 0,
        avgProfit: 0,
        avgLoss: 0
      }
    };
    setExitStrategies([...exitStrategies, duplicated]);
    setActiveStrategy(duplicated.id);
  };

  const toggleStrategyStatus = (strategy: ExitStrategy) => {
    const newStatus = strategy.status === 'active' ? 'inactive' : 'active';
    updateExitStrategy(strategy.id, { status: newStatus });
  };

  const addCombination = (strategyId: string, combinationId: string) => {
    const strategy = exitStrategies.find(s => s.id === strategyId);
    if (strategy && !strategy.combinations.includes(combinationId)) {
      updateExitStrategy(strategyId, {
        combinations: [...strategy.combinations, combinationId]
      });
    }
  };

  const removeCombination = (strategyId: string, combinationId: string) => {
    const strategy = exitStrategies.find(s => s.id === strategyId);
    if (strategy) {
      updateExitStrategy(strategyId, {
        combinations: strategy.combinations.filter(id => id !== combinationId)
      });
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      active: 'default',
      inactive: 'outline',
      testing: 'secondary'
    };

    const colors: Record<string, string> = {
      active: 'bg-green-500',
      inactive: 'bg-gray-400',
      testing: 'bg-yellow-500'
    };

    return (
      <Badge variant={variants[status] || 'outline'} className="gap-1">
        <div className={`h-2 w-2 rounded-full ${colors[status] || 'bg-gray-400'}`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const getTypeIcon = (type: string) => {
    const strategyType = strategyTypes.find(st => st.value === type);
    return strategyType ? <strategyType.icon className="h-4 w-4" /> : <LogOut className="h-4 w-4" />;
  };

  const activeStrategyData = exitStrategies.find(s => s.id === activeStrategy);

  const ExitTemplatesModal = () => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <Card className="w-full max-w-2xl max-h-[80vh] overflow-auto">
        <CardHeader>
          <CardTitle>Exit Strategy Templates</CardTitle>
          <CardDescription>
            Choose a pre-configured exit strategy template
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {exitTemplates.map((template) => (
            <Card 
              key={template.id}
              className="cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => {
                createExitStrategy(template);
                setShowTemplates(false);
              }}
            >
              <CardHeader className="pb-3">
                <CardTitle className="text-base">{template.name}</CardTitle>
                <CardDescription>{template.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  {Object.entries(template.config).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground capitalize">
                        {key.replace(/([A-Z])/g, ' $1').trim()}:
                      </span>
                      <div className="font-medium">{value}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
          
          <div className="flex justify-end">
            <Button variant="outline" onClick={() => setShowTemplates(false)}>
              Cancel
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
            <LogOut className="h-5 w-5" />
            Exit Strategy Configuration
          </CardTitle>
          <CardDescription>
            Configure exit strategies and risk management rules
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-between items-center">
            <div className="flex gap-4">
              <p className="text-sm text-muted-foreground">
                Active strategies: {exitStrategies.filter(s => s.status === 'active').length} of {exitStrategies.length}
              </p>
              <p className="text-sm text-muted-foreground">
                Total exits: {exitStrategies.reduce((sum, s) => sum + s.performance.totalExits, 0)}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setShowTemplates(true)} size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Use Template
              </Button>
              <Button onClick={() => createExitStrategy()} size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Create Custom
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Exit Strategies List */}
        <Card>
          <CardHeader>
            <CardTitle>Exit Strategies</CardTitle>
            <CardDescription>
              Manage your exit strategy portfolio
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {exitStrategies.map((strategy) => (
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
                        duplicateExitStrategy(strategy);
                      }}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteExitStrategy(strategy.id);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mb-2">{strategy.description}</p>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <div className="font-medium capitalize">{strategy.type.replace('_', ' ')}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Priority:</span>
                    <div className="font-medium">{strategy.priority}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Total Exits:</span>
                    <div className="font-medium">{strategy.performance.totalExits}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Success Rate:</span>
                    <div className="font-medium">{(strategy.performance.successRate * 100).toFixed(1)}%</div>
                  </div>
                </div>

                {strategy.combinations.length > 0 && (
                  <div className="mt-2">
                    <span className="text-xs text-muted-foreground">Combinations: </span>
                    <Badge variant="outline" className="text-xs">
                      {strategy.combinations.length} strategy{strategy.combinations.length !== 1 ? 'ies' : ''}
                    </Badge>
                  </div>
                )}
              </div>
            ))}

            {exitStrategies.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <LogOut className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No exit strategies configured</p>
                <Button onClick={() => setShowTemplates(true)} variant="outline" className="mt-2" size="sm">
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
              {activeStrategyData ? 'Configure strategy parameters and settings' : 'Select a strategy to configure'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {activeStrategyData ? (
              <Tabs defaultValue="basic" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="basic">Basic</TabsTrigger>
                  <TabsTrigger value="parameters">Parameters</TabsTrigger>
                  <TabsTrigger value="combinations">Combinations</TabsTrigger>
                </TabsList>

                <TabsContent value="basic" className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <Label htmlFor="strategy-name">Name</Label>
                      <Input
                        id="strategy-name"
                        value={activeStrategyData.name}
                        onChange={(e) => updateExitStrategy(activeStrategyData.id, { name: e.target.value })}
                      />
                    </div>

                    <div>
                      <Label htmlFor="strategy-type">Type</Label>
                      <Select
                        value={activeStrategyData.type}
                        onValueChange={(value) => updateExitStrategy(activeStrategyData.id, { type: value as any })}
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
                        onChange={(e) => updateExitStrategy(activeStrategyData.id, { description: e.target.value })}
                        placeholder="Strategy description"
                      />
                    </div>

                    <div>
                      <Label htmlFor="priority">Priority (1 = highest)</Label>
                      <Input
                        id="priority"
                        type="number"
                        min="1"
                        value={activeStrategyData.priority}
                        onChange={(e) => updateExitStrategy(activeStrategyData.id, { priority: Number(e.target.value) })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable Strategy</Label>
                        <p className="text-sm text-muted-foreground">
                          Allow this strategy to trigger exits
                        </p>
                      </div>
                      <Switch
                        checked={activeStrategyData.config.enabled}
                        onCheckedChange={(checked) => updateExitStrategy(activeStrategyData.id, {
                          config: { ...activeStrategyData.config, enabled: checked }
                        })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Status</Label>
                        <p className="text-sm text-muted-foreground">
                          {activeStrategyData.status === 'active' ? 'Strategy is active' : 'Strategy is inactive'}
                        </p>
                      </div>
                      <Button
                        variant={activeStrategyData.status === 'active' ? 'outline' : 'default'}
                        size="sm"
                        onClick={() => toggleStrategyStatus(activeStrategyData)}
                      >
                        {activeStrategyData.status === 'active' ? 'Deactivate' : 'Activate'}
                      </Button>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="parameters" className="space-y-4">
                  <div className="space-y-4">
                    {activeStrategyData.type === 'stop_loss' && (
                      <div>
                        <Label>Stop Loss Percentage: {activeStrategyData.config.percentage}%</Label>
                        <Slider
                          value={[activeStrategyData.config.percentage || 0]}
                          onValueChange={([value]) => updateExitStrategy(activeStrategyData.id, {
                            config: { ...activeStrategyData.config, percentage: value }
                          })}
                          min={0.5}
                          max={20}
                          step={0.5}
                          className="mt-2"
                        />
                      </div>
                    )}

                    {activeStrategyData.type === 'take_profit' && (
                      <div>
                        <Label>Take Profit Percentage: {activeStrategyData.config.percentage}%</Label>
                        <Slider
                          value={[activeStrategyData.config.percentage || 0]}
                          onValueChange={([value]) => updateExitStrategy(activeStrategyData.id, {
                            config: { ...activeStrategyData.config, percentage: value }
                          })}
                          min={1}
                          max={50}
                          step={0.5}
                          className="mt-2"
                        />
                      </div>
                    )}

                    {activeStrategyData.type === 'trailing_stop' && (
                      <div>
                        <Label>Trailing Distance: {activeStrategyData.config.trailingDistance}%</Label>
                        <Slider
                          value={[activeStrategyData.config.trailingDistance || 0]}
                          onValueChange={([value]) => updateExitStrategy(activeStrategyData.id, {
                            config: { ...activeStrategyData.config, trailingDistance: value }
                          })}
                          min={1}
                          max={15}
                          step={0.5}
                          className="mt-2"
                        />
                      </div>
                    )}

                    {activeStrategyData.type === 'time_based' && (
                      <div>
                        <Label>Time Period: {activeStrategyData.config.timePeriod} days</Label>
                        <Slider
                          value={[activeStrategyData.config.timePeriod || 0]}
                          onValueChange={([value]) => updateExitStrategy(activeStrategyData.id, {
                            config: { ...activeStrategyData.config, timePeriod: value }
                          })}
                          min={1}
                          max={365}
                          step={1}
                          className="mt-2"
                        />
                      </div>
                    )}

                    {activeStrategyData.type === 'profit_target' && (
                      <div className="space-y-4">
                        <div>
                          <Label>Minimum Profit: ${activeStrategyData.config.minProfit}</Label>
                          <Input
                            type="number"
                            value={activeStrategyData.config.minProfit || 0}
                            onChange={(e) => updateExitStrategy(activeStrategyData.id, {
                              config: { ...activeStrategyData.config, minProfit: Number(e.target.value) }
                            })}
                          />
                        </div>
                        <div>
                          <Label>Maximum Loss: ${activeStrategyData.config.maxLoss}</Label>
                          <Input
                            type="number"
                            value={activeStrategyData.config.maxLoss || 0}
                            onChange={(e) => updateExitStrategy(activeStrategyData.id, {
                              config: { ...activeStrategyData.config, maxLoss: Number(e.target.value) }
                            })}
                          />
                        </div>
                      </div>
                    )}

                    {activeStrategyData.type === 'custom' && (
                      <div>
                        <Label htmlFor="conditions">Custom Conditions</Label>
                        <Textarea
                          id="conditions"
                          value={activeStrategyData.config.conditions || ''}
                          onChange={(e) => updateExitStrategy(activeStrategyData.id, {
                            config: { ...activeStrategyData.config, conditions: e.target.value }
                          })}
                          placeholder="Enter custom exit conditions..."
                          rows={4}
                        />
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="combinations" className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium mb-3">Strategy Combinations</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Combine this strategy with others for enhanced exits
                    </p>

                    {/* Current Combinations */}
                    {activeStrategyData.combinations.length > 0 && (
                      <div className="space-y-2">
                        <Label>Current Combinations</Label>
                        {activeStrategyData.combinations.map(comboId => {
                          const combo = exitStrategies.find(s => s.id === comboId);
                          return combo ? (
                            <div key={comboId} className="flex items-center justify-between p-2 border rounded">
                              <span className="text-sm">{combo.name}</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => removeCombination(activeStrategyData.id, comboId)}
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          ) : null;
                        })}
                      </div>
                    )}

                    <Separator className="my-4" />

                    {/* Available Strategies */}
                    <div>
                      <Label>Add Combination</Label>
                      <Select
                        onValueChange={(value) => {
                          if (value && value !== activeStrategyData.id) {
                            addCombination(activeStrategyData.id, value);
                          }
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a strategy to combine with..." />
                        </SelectTrigger>
                        <SelectContent>
                          {exitStrategies
                            .filter(s => s.id !== activeStrategyData.id && !activeStrategyData.combinations.includes(s.id))
                            .map(strategy => (
                              <SelectItem key={strategy.id} value={strategy.id}>
                                {strategy.name}
                              </SelectItem>
                            ))
                          }
                        </SelectContent>
                      </Select>
                    </div>

                    {exitStrategies.filter(s => s.id !== activeStrategyData.id && !activeStrategyData.combinations.includes(s.id)).length === 0 && (
                      <p className="text-sm text-muted-foreground">No additional strategies available for combination.</p>
                    )}
                  </div>
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

      {/* Performance Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Overview</CardTitle>
          <CardDescription>
            Exit strategy performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 border rounded text-center">
              <DollarSign className="h-8 w-8 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold">
                ${exitStrategies.reduce((sum, s) => sum + s.performance.avgProfit, 0).toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Profit</div>
            </div>
            <div className="p-4 border rounded text-center">
              <TrendingDown className="h-8 w-8 text-red-500 mx-auto mb-2" />
              <div className="text-2xl font-bold">
                ${exitStrategies.reduce((sum, s) => sum + s.performance.avgLoss, 0).toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Loss</div>
            </div>
            <div className="p-4 border rounded text-center">
              <Clock className="h-8 w-8 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold">
                {exitStrategies.reduce((sum, s) => sum + s.performance.avgExitTime, 0) / exitStrategies.length || 0}h
              </div>
              <div className="text-sm text-muted-foreground">Avg Exit Time</div>
            </div>
            <div className="p-4 border rounded text-center">
              <BarChart3 className="h-8 w-8 text-purple-500 mx-auto mb-2" />
              <div className="text-2xl font-bold">
                {((exitStrategies.reduce((sum, s) => sum + s.performance.successRate, 0) / exitStrategies.length) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Success Rate</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Changes are automatically saved to local storage
            </p>
            <Button onClick={saveExitStrategies}>
              <Settings className="h-4 w-4 mr-2" />
              Save All Strategies
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Templates Modal */}
      {showTemplates && <ExitTemplatesModal />}
    </div>
  );
};

export default ExitConfig;