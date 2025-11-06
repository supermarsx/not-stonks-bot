import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  DollarSign,
  Percent,
  Clock,
  Activity,
  Target,
  Zap,
  AlertCircle,
  CheckCircle,
  Settings,
  TrendingUp
} from 'lucide-react';

interface RiskLimits {
  dailyLossLimit: number;
  maxDrawdownPercent: number;
  maxPositionSize: number;
  maxLeverage: number;
  maxCorrelation: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxPositions: number;
}

interface RiskSettings {
  enabled: boolean;
  realTimeMonitoring: boolean;
  automaticShutdown: boolean;
  alertThresholds: {
    drawdown: number;
    loss: number;
    correlation: number;
    volatility: number;
  };
  circuitBreakers: {
    enabled: boolean;
    haltThreshold: number;
    resumeThreshold: number;
    duration: number;
  };
  positionSizing: {
    method: 'fixed' | 'percentage' | 'kelly' | 'volatility';
    baseSize: number;
    riskPerTrade: number;
    maxExposure: number;
  };
}

interface RiskMonitoring {
  currentDrawdown: number;
  dailyPnL: number;
  totalExposure: number;
  activePositions: number;
  riskScore: number;
  alerts: Array<{
    id: string;
    type: 'warning' | 'error' | 'info';
    message: string;
    timestamp: string;
    acknowledged: boolean;
  }>;
}

const RiskConfig: React.FC = () => {
  const [riskSettings, setRiskSettings] = useState<RiskSettings>({
    enabled: true,
    realTimeMonitoring: true,
    automaticShutdown: false,
    alertThresholds: {
      drawdown: 10,
      loss: 5,
      correlation: 0.8,
      volatility: 0.25
    },
    circuitBreakers: {
      enabled: true,
      haltThreshold: 20,
      resumeThreshold: 10,
      duration: 300 // 5 minutes
    },
    positionSizing: {
      method: 'percentage',
      baseSize: 10000,
      riskPerTrade: 0.02,
      maxExposure: 0.1
    }
  });

  const [riskLimits, setRiskLimits] = useState<RiskLimits>({
    dailyLossLimit: 5000,
    maxDrawdownPercent: 15,
    maxPositionSize: 0.1,
    maxLeverage: 2,
    maxCorrelation: 0.8,
    stopLossPercent: 5,
    takeProfitPercent: 10,
    maxPositions: 20
  });

  const [riskMonitoring, setRiskMonitoring] = useState<RiskMonitoring>({
    currentDrawdown: 2.3,
    dailyPnL: 150.75,
    totalExposure: 0.08,
    activePositions: 8,
    riskScore: 3.2,
    alerts: [
      {
        id: '1',
        type: 'warning',
        message: 'High correlation detected between positions',
        timestamp: new Date().toISOString(),
        acknowledged: false
      }
    ]
  });

  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadRiskConfiguration();
  }, []);

  const loadRiskConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getRiskConfig();
      if (config) {
        setRiskSettings(config.settings || riskSettings);
        setRiskLimits(config.limits || riskLimits);
        setRiskMonitoring(config.monitoring || riskMonitoring);
      }
    } catch (error) {
      console.error('Failed to load risk configuration:', error);
    }
  };

  const saveRiskConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.saveRiskConfig({
        settings: riskSettings,
        limits: riskLimits,
        monitoring: riskMonitoring
      });
    } catch (error) {
      console.error('Failed to save risk configuration:', error);
    } finally {
      setSaving(false);
      setIsEditing(false);
    }
  };

  const acknowledgeAlert = (alertId: string) => {
    setRiskMonitoring(prev => ({
      ...prev,
      alerts: prev.alerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    }));
  };

  const clearAlert = (alertId: string) => {
    setRiskMonitoring(prev => ({
      ...prev,
      alerts: prev.alerts.filter(alert => alert.id !== alertId)
    }));
  };

  const getRiskLevel = (score: number) => {
    if (score <= 2) return { level: 'Low', color: 'text-green-600', variant: 'default' as const };
    if (score <= 4) return { level: 'Medium', color: 'text-yellow-600', variant: 'secondary' as const };
    return { level: 'High', color: 'text-red-600', variant: 'destructive' as const };
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  const updateRiskSetting = (path: string, value: any) => {
    const keys = path.split('.');
    setRiskSettings(prev => {
      const updated = { ...prev };
      let current: any = updated;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current[keys[i]] = { ...current[keys[i]] };
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return updated;
    });
    setIsEditing(true);
  };

  const updateRiskLimit = (path: string, value: any) => {
    setRiskLimits(prev => ({ ...prev, [path]: value }));
    setIsEditing(true);
  };

  const testCircuitBreaker = () => {
    // Simulate circuit breaker test
    setRiskMonitoring(prev => ({
      ...prev,
      alerts: [
        ...prev.alerts,
        {
          id: Date.now().toString(),
          type: 'info',
          message: 'Circuit breaker test initiated',
          timestamp: new Date().toISOString(),
          acknowledged: false
        }
      ]
    }));
  };

  const riskLevel = getRiskLevel(riskMonitoring.riskScore);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Risk Management Configuration
          </CardTitle>
          <CardDescription>
            Configure risk limits, monitoring, and protection mechanisms
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={riskSettings.enabled}
                  onCheckedChange={(checked) => updateRiskSetting('enabled', checked)}
                />
                <Label>Risk Management {riskSettings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant={riskLevel.variant} className={riskLevel.color}>
                Risk Level: {riskLevel.level}
              </Badge>
            </div>
            <div className="flex gap-2">
              {isEditing && (
                <Button variant="outline" onClick={loadRiskConfiguration}>
                  Reset Changes
                </Button>
              )}
              <Button onClick={saveRiskConfiguration} disabled={!isEditing || saving}>
                <Settings className="h-4 w-4 mr-2" />
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Overview Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Current Drawdown</p>
                <p className="text-2xl font-bold">{riskMonitoring.currentDrawdown.toFixed(1)}%</p>
              </div>
              <TrendingDown className="h-8 w-8 text-red-500" />
            </div>
            <Progress value={riskMonitoring.currentDrawdown} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Daily P&L</p>
                <p className={`text-2xl font-bold ${riskMonitoring.dailyPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${riskMonitoring.dailyPnL.toFixed(2)}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Exposure</p>
                <p className="text-2xl font-bold">{(riskMonitoring.totalExposure * 100).toFixed(1)}%</p>
              </div>
              <Percent className="h-8 w-8 text-purple-500" />
            </div>
            <Progress value={riskMonitoring.totalExposure * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Positions</p>
                <p className="text-2xl font-bold">{riskMonitoring.activePositions}</p>
                <p className="text-xs text-muted-foreground">Max: {riskLimits.maxPositions}</p>
              </div>
              <Activity className="h-8 w-8 text-orange-500" />
            </div>
            <Progress value={(riskMonitoring.activePositions / riskLimits.maxPositions) * 100} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="limits" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="limits">Risk Limits</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="limits" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Portfolio Limits</CardTitle>
                <CardDescription>
                  Global portfolio risk limits
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Daily Loss Limit ($)</Label>
                  <Input
                    type="number"
                    value={riskLimits.dailyLossLimit}
                    onChange={(e) => updateRiskLimit('dailyLossLimit', Number(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Maximum acceptable daily loss
                  </p>
                </div>

                <div>
                  <Label>Max Drawdown: {riskLimits.maxDrawdownPercent}%</Label>
                  <Slider
                    value={[riskLimits.maxDrawdownPercent]}
                    onValueChange={([value]) => updateRiskLimit('maxDrawdownPercent', value)}
                    min={5}
                    max={50}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Max Leverage</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={riskLimits.maxLeverage}
                    onChange={(e) => updateRiskLimit('maxLeverage', Number(e.target.value))}
                  />
                </div>

                <div>
                  <Label>Max Correlation</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={riskLimits.maxCorrelation}
                    onChange={(e) => updateRiskLimit('maxCorrelation', Number(e.target.value))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Position Limits</CardTitle>
                <CardDescription>
                  Individual position risk controls
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Max Position Size: {(riskLimits.maxPositionSize * 100).toFixed(1)}%</Label>
                  <Slider
                    value={[riskLimits.maxPositionSize * 100]}
                    onValueChange={([value]) => updateRiskLimit('maxPositionSize', value / 100)}
                    min={1}
                    max={50}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Max Positions: {riskLimits.maxPositions}</Label>
                  <Slider
                    value={[riskLimits.maxPositions]}
                    onValueChange={([value]) => updateRiskLimit('maxPositions', value)}
                    min={1}
                    max={100}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Stop Loss: {riskLimits.stopLossPercent}%</Label>
                  <Slider
                    value={[riskLimits.stopLossPercent]}
                    onValueChange={([value]) => updateRiskLimit('stopLossPercent', value)}
                    min={1}
                    max={20}
                    step={0.5}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Take Profit: {riskLimits.takeProfitPercent}%</Label>
                  <Slider
                    value={[riskLimits.takeProfitPercent]}
                    onValueChange={([value]) => updateRiskLimit('takeProfitPercent', value)}
                    min={1}
                    max={50}
                    step={0.5}
                    className="mt-2"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Monitoring Settings</CardTitle>
                <CardDescription>
                  Real-time monitoring and alerts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Real-time Monitoring</Label>
                    <p className="text-sm text-muted-foreground">
                      Monitor risks continuously
                    </p>
                  </div>
                  <Switch
                    checked={riskSettings.realTimeMonitoring}
                    onCheckedChange={(checked) => updateRiskSetting('realTimeMonitoring', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Automatic Shutdown</Label>
                    <p className="text-sm text-muted-foreground">
                      Stop trading when limits exceeded
                    </p>
                  </div>
                  <Switch
                    checked={riskSettings.automaticShutdown}
                    onCheckedChange={(checked) => updateRiskSetting('automaticShutdown', checked)}
                  />
                </div>

                <Separator />

                <div>
                  <Label>Drawdown Alert Threshold</Label>
                  <Slider
                    value={[riskSettings.alertThresholds.drawdown]}
                    onValueChange={([value]) => updateRiskSetting('alertThresholds.drawdown', value)}
                    min={5}
                    max={25}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Daily Loss Alert Threshold</Label>
                  <Slider
                    value={[riskSettings.alertThresholds.loss]}
                    onValueChange={([value]) => updateRiskSetting('alertThresholds.loss', value)}
                    min={1}
                    max={10}
                    step={0.5}
                    className="mt-2"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Circuit Breakers</CardTitle>
                <CardDescription>
                  Automatic trading halt mechanisms
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Circuit Breakers</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatic trading halt system
                    </p>
                  </div>
                  <Switch
                    checked={riskSettings.circuitBreakers.enabled}
                    onCheckedChange={(checked) => updateRiskSetting('circuitBreakers.enabled', checked)}
                  />
                </div>

                {riskSettings.circuitBreakers.enabled && (
                  <>
                    <div>
                      <Label>Halt Threshold: {riskSettings.circuitBreakers.haltThreshold}%</Label>
                      <Slider
                        value={[riskSettings.circuitBreakers.haltThreshold]}
                        onValueChange={([value]) => updateRiskSetting('circuitBreakers.haltThreshold', value)}
                        min={10}
                        max={50}
                        step={1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label>Resume Threshold: {riskSettings.circuitBreakers.resumeThreshold}%</Label>
                      <Slider
                        value={[riskSettings.circuitBreakers.resumeThreshold]}
                        onValueChange={([value]) => updateRiskSetting('circuitBreakers.resumeThreshold', value)}
                        min={5}
                        max={25}
                        step={1}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label>Halt Duration: {riskSettings.circuitBreakers.duration / 60} minutes</Label>
                      <Slider
                        value={[riskSettings.circuitBreakers.duration / 60]}
                        onValueChange={([value]) => updateRiskSetting('circuitBreakers.duration', value * 60)}
                        min={1}
                        max={60}
                        step={1}
                        className="mt-2"
                      />
                    </div>

                    <Button onClick={testCircuitBreaker} variant="outline" size="sm" className="w-full">
                      Test Circuit Breaker
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Position Sizing</CardTitle>
              <CardDescription>
                Configure how position sizes are calculated
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label>Sizing Method</Label>
                  <Select
                    value={riskSettings.positionSizing.method}
                    onValueChange={(value: any) => updateRiskSetting('positionSizing.method', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixed Size</SelectItem>
                      <SelectItem value="percentage">Percentage of Portfolio</SelectItem>
                      <SelectItem value="kelly">Kelly Criterion</SelectItem>
                      <SelectItem value="volatility">Volatility Based</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Base Size ($)</Label>
                  <Input
                    type="number"
                    value={riskSettings.positionSizing.baseSize}
                    onChange={(e) => updateRiskSetting('positionSizing.baseSize', Number(e.target.value))}
                  />
                </div>

                <div>
                  <Label>Risk Per Trade: {(riskSettings.positionSizing.riskPerTrade * 100).toFixed(1)}%</Label>
                  <Slider
                    value={[riskSettings.positionSizing.riskPerTrade * 100]}
                    onValueChange={([value]) => updateRiskSetting('positionSizing.riskPerTrade', value / 100)}
                    min={0.1}
                    max={10}
                    step={0.1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Max Exposure: {(riskSettings.positionSizing.maxExposure * 100).toFixed(1)}%</Label>
                  <Slider
                    value={[riskSettings.positionSizing.maxExposure * 100]}
                    onValueChange={([value]) => updateRiskSetting('positionSizing.maxExposure', value / 100)}
                    min={1}
                    max={50}
                    step={1}
                    className="mt-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Current Status</CardTitle>
                <CardDescription>
                  Real-time risk metrics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Risk Score</span>
                  <Badge variant={riskLevel.variant}>{riskMonitoring.riskScore.toFixed(1)} / 10</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span>Portfolio Heat</span>
                  <Badge variant="outline">
                    {(riskMonitoring.totalExposure * 100).toFixed(1)}% / {riskSettings.positionSizing.maxExposure * 100}%
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span>Drawdown Status</span>
                  <Badge variant={riskMonitoring.currentDrawdown > riskLimits.maxDrawdownPercent ? 'destructive' : 'default'}>
                    {riskMonitoring.currentDrawdown.toFixed(1)}% / {riskLimits.maxDrawdownPercent}%
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span>Daily P&L Status</span>
                  <Badge variant={riskMonitoring.dailyPnL < -riskLimits.dailyLossLimit ? 'destructive' : 'default'}>
                    ${riskMonitoring.dailyPnL.toFixed(2)} / -${riskLimits.dailyLossLimit}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk Metrics</CardTitle>
                <CardDescription>
                  Detailed risk analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Portfolio Beta</span>
                    <span>1.23</span>
                  </div>
                  <Progress value={75} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>VaR (95%)</span>
                    <span>$2,450</span>
                  </div>
                  <Progress value={60} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Sharpe Ratio</span>
                    <span>1.45</span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Sortino Ratio</span>
                    <span>1.78</span>
                  </div>
                  <Progress value={90} className="h-2" />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>System Health</CardTitle>
              <CardDescription>
                Risk management system status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Monitoring Active</p>
                  <p className="text-xs text-muted-foreground">Real-time tracking</p>
                </div>
                <div className="text-center">
                  <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Limits Valid</p>
                  <p className="text-xs text-muted-foreground">All within ranges</p>
                </div>
                <div className="text-center">
                  <AlertCircle className="h-8 w-8 text-yellow-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Warnings Active</p>
                  <p className="text-xs text-muted-foreground">{riskMonitoring.alerts.filter(a => !a.acknowledged).length} alerts</p>
                </div>
                <div className="text-center">
                  <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Circuit Breakers</p>
                  <p className="text-xs text-muted-foreground">Ready to activate</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Risk Alerts</CardTitle>
              <CardDescription>
                Recent risk management alerts and notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {riskMonitoring.alerts.length > 0 ? (
                  riskMonitoring.alerts.map((alert) => (
                    <div
                      key={alert.id}
                      className={`p-4 border rounded-lg ${alert.acknowledged ? 'opacity-60' : ''}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          {getAlertIcon(alert.type)}
                          <div className="flex-1">
                            <p className="font-medium">{alert.message}</p>
                            <p className="text-sm text-muted-foreground">
                              {new Date(alert.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          {!alert.acknowledged && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => acknowledgeAlert(alert.id)}
                            >
                              Acknowledge
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => clearAlert(alert.id)}
                          >
                            Clear
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No alerts at this time</p>
                    <p className="text-sm">Risk parameters are within acceptable limits</p>
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

export default RiskConfig;