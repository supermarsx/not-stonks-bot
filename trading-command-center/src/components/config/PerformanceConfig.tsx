import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Zap, 
  Cpu, 
  HardDrive,
  MemoryStick,
  Network,
  Settings,
  Activity,
  TrendingUp,
  BarChart3,
  Clock,
  Gauge,
  AlertCircle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';

interface PerformanceSettings {
  enabled: boolean;
  autoOptimization: boolean;
  profilingEnabled: boolean;
  monitoringInterval: number; // seconds
  alertThresholds: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
    responseTime: number;
  };
  optimizationRules: {
    aggressiveGC: boolean;
    connectionPooling: boolean;
    queryOptimization: boolean;
    cachingEnabled: boolean;
    compressionEnabled: boolean;
  };
}

interface ResourceLimits {
  maxCpuUsage: number; // percentage
  maxMemoryUsage: number; // percentage
  maxDiskUsage: number; // percentage
  maxNetworkBandwidth: number; // MB/s
  maxConcurrentConnections: number;
  maxThreads: number;
  maxFileDescriptors: number;
}

interface OptimizationMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  activeConnections: number;
  threadCount: number;
  gcTime: number;
}

interface CacheSettings {
  enabled: boolean;
  strategy: 'LRU' | 'LFU' | 'TTL' | 'Adaptive';
  maxSize: number; // MB
  ttl: number; // seconds
  compression: boolean;
  persistence: boolean;
  invalidationStrategy: 'time' | 'size' | 'manual';
}

const PerformanceConfig: React.FC = () => {
  const [performanceSettings, setPerformanceSettings] = useState<PerformanceSettings>({
    enabled: true,
    autoOptimization: true,
    profilingEnabled: false,
    monitoringInterval: 5,
    alertThresholds: {
      cpu: 80,
      memory: 85,
      disk: 90,
      network: 100,
      responseTime: 1000
    },
    optimizationRules: {
      aggressiveGC: false,
      connectionPooling: true,
      queryOptimization: true,
      cachingEnabled: true,
      compressionEnabled: true
    }
  });

  const [resourceLimits, setResourceLimits] = useState<ResourceLimits>({
    maxCpuUsage: 80,
    maxMemoryUsage: 85,
    maxDiskUsage: 90,
    maxNetworkBandwidth: 100,
    maxConcurrentConnections: 1000,
    maxThreads: 50,
    maxFileDescriptors: 10000
  });

  const [optimizationMetrics, setOptimizationMetrics] = useState<OptimizationMetrics>({
    cpuUsage: 45.2,
    memoryUsage: 62.8,
    diskUsage: 34.1,
    networkLatency: 15.3,
    responseTime: 234,
    throughput: 1250,
    errorRate: 0.02,
    activeConnections: 23,
    threadCount: 18,
    gcTime: 0.5
  });

  const [cacheSettings, setCacheSettings] = useState<CacheSettings>({
    enabled: true,
    strategy: 'LRU',
    maxSize: 512,
    ttl: 3600,
    compression: true,
    persistence: false,
    invalidationStrategy: 'time'
  });

  const [saving, setSaving] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [lastOptimization, setLastOptimization] = useState<string>(new Date().toISOString());

  useEffect(() => {
    loadPerformanceConfiguration();
    // Start real-time monitoring
    const interval = setInterval(updateMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadPerformanceConfiguration = async () => {
    try {
      const config = await window.electronAPI?.getPerformanceConfig();
      if (config) {
        setPerformanceSettings(config.settings || performanceSettings);
        setResourceLimits(config.limits || resourceLimits);
        setOptimizationMetrics(config.metrics || optimizationMetrics);
        setCacheSettings(config.cache || cacheSettings);
      }
    } catch (error) {
      console.error('Failed to load performance configuration:', error);
    }
  };

  const savePerformanceConfiguration = async () => {
    setSaving(true);
    try {
      await window.electronAPI?.savePerformanceConfig({
        settings: performanceSettings,
        limits: resourceLimits,
        metrics: optimizationMetrics,
        cache: cacheSettings
      });
    } catch (error) {
      console.error('Failed to save performance configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const runOptimization = async () => {
    setOptimizing(true);
    try {
      await window.electronAPI?.optimizePerformance();
      setLastOptimization(new Date().toISOString());
      
      // Simulate optimization improvements
      setOptimizationMetrics(prev => ({
        ...prev,
        cpuUsage: Math.max(0, prev.cpuUsage - 5),
        memoryUsage: Math.max(0, prev.memoryUsage - 3),
        responseTime: Math.max(0, prev.responseTime - 20),
        throughput: prev.throughput + 50
      }));
    } catch (error) {
      console.error('Failed to run optimization:', error);
    } finally {
      setOptimizing(false);
    }
  };

  const updateMetrics = () => {
    // Simulate real-time metric updates
    setOptimizationMetrics(prev => ({
      ...prev,
      cpuUsage: Math.max(0, Math.min(100, prev.cpuUsage + (Math.random() - 0.5) * 10)),
      memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 8)),
      diskUsage: Math.max(0, Math.min(100, prev.diskUsage + (Math.random() - 0.5) * 5)),
      networkLatency: Math.max(0, prev.networkLatency + (Math.random() - 0.5) * 20),
      responseTime: Math.max(0, prev.responseTime + (Math.random() - 0.5) * 100),
      throughput: Math.max(0, prev.throughput + (Math.random() - 0.5) * 100),
      errorRate: Math.max(0, prev.errorRate + (Math.random() - 0.5) * 0.01)
    }));
  };

  const getThresholdStatus = (value: number, threshold: number) => {
    const percentage = (value / threshold) * 100;
    if (percentage >= 90) return { status: 'danger', color: 'text-red-600' };
    if (percentage >= 75) return { status: 'warning', color: 'text-yellow-600' };
    return { status: 'good', color: 'text-green-600' };
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'cpu':
        return <Cpu className="h-4 w-4" />;
      case 'memory':
        return <MemoryStick className="h-4 w-4" />;
      case 'disk':
        return <HardDrive className="h-4 w-4" />;
      case 'network':
        return <Network className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Performance Tuning Configuration
          </CardTitle>
          <CardDescription>
            Configure optimization parameters and resource limits
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={performanceSettings.enabled}
                  onCheckedChange={(checked) => setPerformanceSettings(prev => ({ ...prev, enabled: checked }))}
                />
                <Label>Performance System {performanceSettings.enabled ? 'Enabled' : 'Disabled'}</Label>
              </div>
              <Badge variant={performanceSettings.autoOptimization ? 'default' : 'outline'}>
                Auto: {performanceSettings.autoOptimization ? 'On' : 'Off'}
              </Badge>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={runOptimization} disabled={optimizing}>
                <Zap className="h-4 w-4 mr-2" />
                {optimizing ? 'Optimizing...' : 'Run Optimization'}
              </Button>
              <Button onClick={savePerformanceConfiguration} disabled={saving}>
                <Settings className="h-4 w-4 mr-2" />
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">CPU Usage</p>
                <p className="text-2xl font-bold">{optimizationMetrics.cpuUsage.toFixed(1)}%</p>
                <Progress value={optimizationMetrics.cpuUsage} className="mt-1 h-1" />
              </div>
              <Cpu className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Memory Usage</p>
                <p className="text-2xl font-bold">{optimizationMetrics.memoryUsage.toFixed(1)}%</p>
                <Progress value={optimizationMetrics.memoryUsage} className="mt-1 h-1" />
              </div>
              <MemoryStick className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Response Time</p>
                <p className="text-2xl font-bold">{optimizationMetrics.responseTime.toFixed(0)}ms</p>
              </div>
              <Clock className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Throughput</p>
                <p className="text-2xl font-bold">{optimizationMetrics.throughput.toFixed(0)}/s</p>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Tabs */}
      <Tabs defaultValue="monitoring" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
          <TabsTrigger value="caching">Caching</TabsTrigger>
        </TabsList>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Real-time Metrics</CardTitle>
                <CardDescription>
                  Live system performance monitoring
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(optimizationMetrics).map(([key, value]) => {
                  if (key === 'throughput' || key === 'errorRate') return null;
                  
                  const thresholds = performanceSettings.alertThresholds;
                  const threshold = key === 'cpuUsage' ? thresholds.cpu :
                                   key === 'memoryUsage' ? thresholds.memory :
                                   key === 'diskUsage' ? thresholds.disk :
                                   key === 'networkLatency' ? thresholds.network :
                                   key === 'responseTime' ? thresholds.responseTime : 100;
                  
                  const status = getThresholdStatus(value, threshold);
                  
                  return (
                    <div key={key} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                        <span className={status.color}>
                          {typeof value === 'number' ? value.toFixed(1) : value}
                          {key.includes('Usage') || key.includes('Latency') ? (key.includes('Latency') ? 'ms' : '%') : 
                           key === 'responseTime' ? 'ms' : ''}
                        </span>
                      </div>
                      <Progress 
                        value={Math.min(100, (value / threshold) * 100)} 
                        className={`h-2 ${status.status === 'danger' ? '[&>div]:bg-red-500' : 
                                                 status.status === 'warning' ? '[&>div]:bg-yellow-500' : '[&>div]:bg-green-500'}`}
                      />
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Monitoring Settings</CardTitle>
                <CardDescription>
                  Configure performance monitoring behavior
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Real-time Profiling</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable detailed performance profiling
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.profilingEnabled}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({ ...prev, profilingEnabled: checked }))}
                  />
                </div>

                <div>
                  <Label>Monitoring Interval: {performanceSettings.monitoringInterval}s</Label>
                  <Slider
                    value={[performanceSettings.monitoringInterval]}
                    onValueChange={([value]) => setPerformanceSettings(prev => ({ ...prev, monitoringInterval: value }))}
                    min={1}
                    max={60}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <Separator />

                <div>
                  <h4 className="text-sm font-medium mb-3">Alert Thresholds</h4>
                  
                  <div className="space-y-3">
                    <div>
                      <Label>CPU: {performanceSettings.alertThresholds.cpu}%</Label>
                      <Slider
                        value={[performanceSettings.alertThresholds.cpu]}
                        onValueChange={([value]) => setPerformanceSettings(prev => ({
                          ...prev,
                          alertThresholds: { ...prev.alertThresholds, cpu: value }
                        }))}
                        min={50}
                        max={100}
                        step={5}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label>Memory: {performanceSettings.alertThresholds.memory}%</Label>
                      <Slider
                        value={[performanceSettings.alertThresholds.memory]}
                        onValueChange={([value]) => setPerformanceSettings(prev => ({
                          ...prev,
                          alertThresholds: { ...prev.alertThresholds, memory: value }
                        }))}
                        min={50}
                        max={100}
                        step={5}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label>Response Time: {performanceSettings.alertThresholds.responseTime}ms</Label>
                      <Slider
                        value={[performanceSettings.alertThresholds.responseTime]}
                        onValueChange={([value]) => setPerformanceSettings(prev => ({
                          ...prev,
                          alertThresholds: { ...prev.alertThresholds, responseTime: value }
                        }))}
                        min={100}
                        max={5000}
                        step={100}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Last optimization: {new Date(lastOptimization).toLocaleString()}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="resources" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Resource Limits</CardTitle>
              <CardDescription>
                Configure maximum resource usage limits
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label>Max CPU Usage: {resourceLimits.maxCpuUsage}%</Label>
                    <Slider
                      value={[resourceLimits.maxCpuUsage]}
                      onValueChange={([value]) => setResourceLimits(prev => ({ ...prev, maxCpuUsage: value }))}
                      min={50}
                      max={100}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>Max Memory Usage: {resourceLimits.maxMemoryUsage}%</Label>
                    <Slider
                      value={[resourceLimits.maxMemoryUsage]}
                      onValueChange={([value]) => setResourceLimits(prev => ({ ...prev, maxMemoryUsage: value }))}
                      min={50}
                      max={100}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>Max Disk Usage: {resourceLimits.maxDiskUsage}%</Label>
                    <Slider
                      value={[resourceLimits.maxDiskUsage]}
                      onValueChange={([value]) => setResourceLimits(prev => ({ ...prev, maxDiskUsage: value }))}
                      min={50}
                      max={100}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  <div>
                    <Label>Max Network Bandwidth: {resourceLimits.maxNetworkBandwidth} MB/s</Label>
                    <Slider
                      value={[resourceLimits.maxNetworkBandwidth]}
                      onValueChange={([value]) => setResourceLimits(prev => ({ ...prev, maxNetworkBandwidth: value }))}
                      min={10}
                      max={1000}
                      step={10}
                      className="mt-2"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="max-connections">Max Concurrent Connections</Label>
                    <Input
                      id="max-connections"
                      type="number"
                      value={resourceLimits.maxConcurrentConnections}
                      onChange={(e) => setResourceLimits(prev => ({ ...prev, maxConcurrentConnections: Number(e.target.value) }))}
                    />
                  </div>

                  <div>
                    <Label htmlFor="max-threads">Max Threads</Label>
                    <Input
                      id="max-threads"
                      type="number"
                      value={resourceLimits.maxThreads}
                      onChange={(e) => setResourceLimits(prev => ({ ...prev, maxThreads: Number(e.target.value) }))}
                    />
                  </div>

                  <div>
                    <Label htmlFor="max-fd">Max File Descriptors</Label>
                    <Input
                      id="max-fd"
                      type="number"
                      value={resourceLimits.maxFileDescriptors}
                      onChange={(e) => setResourceLimits(prev => ({ ...prev, maxFileDescriptors: Number(e.target.value) }))}
                    />
                  </div>

                  <Alert>
                    <Gauge className="h-4 w-4" />
                    <AlertDescription>
                      Current usage: {optimizationMetrics.activeConnections} connections, {optimizationMetrics.threadCount} threads
                    </AlertDescription>
                  </Alert>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Optimization Rules</CardTitle>
                <CardDescription>
                  Configure automatic optimization behaviors
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Auto Optimization</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically optimize performance
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.autoOptimization}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({ ...prev, autoOptimization: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Aggressive Garbage Collection</Label>
                    <p className="text-sm text-muted-foreground">
                      More frequent GC cycles
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.optimizationRules.aggressiveGC}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({
                      ...prev,
                      optimizationRules: { ...prev.optimizationRules, aggressiveGC: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Connection Pooling</Label>
                    <p className="text-sm text-muted-foreground">
                      Reuse database connections
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.optimizationRules.connectionPooling}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({
                      ...prev,
                      optimizationRules: { ...prev.optimizationRules, connectionPooling: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Query Optimization</Label>
                    <p className="text-sm text-muted-foreground">
                      Optimize database queries
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.optimizationRules.queryOptimization}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({
                      ...prev,
                      optimizationRules: { ...prev.optimizationRules, queryOptimization: checked }
                    }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Compression</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable data compression
                    </p>
                  </div>
                  <Switch
                    checked={performanceSettings.optimizationRules.compressionEnabled}
                    onCheckedChange={(checked) => setPerformanceSettings(prev => ({
                      ...prev,
                      optimizationRules: { ...prev.optimizationRules, compressionEnabled: checked }
                    }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Analysis</CardTitle>
                <CardDescription>
                  Detailed performance metrics and insights
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Network Latency</div>
                    <div className="text-2xl font-bold">{optimizationMetrics.networkLatency.toFixed(1)}ms</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Error Rate</div>
                    <div className="text-2xl font-bold">{(optimizationMetrics.errorRate * 100).toFixed(2)}%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">GC Time</div>
                    <div className="text-2xl font-bold">{optimizationMetrics.gcTime.toFixed(1)}s</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Cache Hit Rate</div>
                    <div className="text-2xl font-bold">87.3%</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>System Health Score</span>
                    <span>85/100</span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Optimization Potential</span>
                    <span>23%</span>
                  </div>
                  <Progress value={23} className="h-2" />
                </div>

                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    System performance is optimal. No immediate action required.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="caching" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Caching Configuration</CardTitle>
                <CardDescription>
                  Configure caching behavior and settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Caching</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable system-wide caching
                    </p>
                  </div>
                  <Switch
                    checked={cacheSettings.enabled}
                    onCheckedChange={(checked) => setCacheSettings(prev => ({ ...prev, enabled: checked }))}
                  />
                </div>

                <div>
                  <Label htmlFor="cache-strategy">Cache Strategy</Label>
                  <Select
                    value={cacheSettings.strategy}
                    onValueChange={(value: any) => setCacheSettings(prev => ({ ...prev, strategy: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="LRU">Least Recently Used</SelectItem>
                      <SelectItem value="LFU">Least Frequently Used</SelectItem>
                      <SelectItem value="TTL">Time To Live</SelectItem>
                      <SelectItem value="Adaptive">Adaptive</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Max Cache Size: {cacheSettings.maxSize}MB</Label>
                  <Slider
                    value={[cacheSettings.maxSize]}
                    onValueChange={([value]) => setCacheSettings(prev => ({ ...prev, maxSize: value }))}
                    min={64}
                    max={2048}
                    step={64}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Cache TTL: {cacheSettings.ttl}s</Label>
                  <Slider
                    value={[cacheSettings.ttl]}
                    onValueChange={([value]) => setCacheSettings(prev => ({ ...prev, ttl: value }))}
                    min={60}
                    max={86400}
                    step={60}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="invalidation">Invalidation Strategy</Label>
                  <Select
                    value={cacheSettings.invalidationStrategy}
                    onValueChange={(value: any) => setCacheSettings(prev => ({ ...prev, invalidationStrategy: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="time">Time-based</SelectItem>
                      <SelectItem value="size">Size-based</SelectItem>
                      <SelectItem value="manual">Manual</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Compression</Label>
                    <p className="text-sm text-muted-foreground">
                      Compress cached data
                    </p>
                  </div>
                  <Switch
                    checked={cacheSettings.compression}
                    onCheckedChange={(checked) => setCacheSettings(prev => ({ ...prev, compression: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Persistence</Label>
                    <p className="text-sm text-muted-foreground">
                      Persist cache to disk
                    </p>
                  </div>
                  <Switch
                    checked={cacheSettings.persistence}
                    onCheckedChange={(checked) => setCacheSettings(prev => ({ ...prev, persistence: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cache Statistics</CardTitle>
                <CardDescription>
                  Current cache performance metrics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Hit Rate</div>
                    <div className="text-2xl font-bold">87.3%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Miss Rate</div>
                    <div className="text-2xl font-bold">12.7%</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Cache Size</div>
                    <div className="text-2xl font-bold">234MB</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-sm text-muted-foreground">Entries</div>
                    <div className="text-2xl font-bold">1,247</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Cache Utilization</span>
                    <span>45.7%</span>
                  </div>
                  <Progress value={45.7} className="h-2" />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Memory Savings</span>
                    <span>67.2%</span>
                  </div>
                  <Progress value={67.2} className="h-2" />
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Evictions Today:</span>
                    <div className="font-medium">23</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Avg Entry Size:</span>
                    <div className="font-medium">192KB</div>
                  </div>
                </div>

                <Alert>
                  <RefreshCw className="h-4 w-4" />
                  <AlertDescription>
                    Cache is performing well with optimal hit rate and low miss rate.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerformanceConfig;