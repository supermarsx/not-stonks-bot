import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings, 
  Save, 
  Upload, 
  Download, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  Play,
  FileText,
  Shield,
  Zap
} from 'lucide-react';
import { useConfigStore } from '@/stores/configStore';
import { toast } from 'sonner';

interface ConfigSection {
  id: string;
  name: string;
  status: 'valid' | 'invalid' | 'pending' | 'untested';
  description: string;
  component: React.ComponentType;
}

const ConfigManager: React.FC = () => {
  const {
    config,
    validationStatus,
    isLoading,
    isSaving,
    autoSave,
    setAutoSave,
    validateConfig,
    saveConfig,
    loadConfig,
    importConfig,
    exportConfig,
    resetConfig
  } = useConfigStore();

  const [activeTab, setActiveTab] = useState('overview');
  const [validationProgress, setValidationProgress] = useState(0);
  const [isValidating, setIsValidating] = useState(false);

  // Default configuration sections
  const configSections: ConfigSection[] = [
    {
      id: 'brokers',
      name: 'Brokers',
      status: validationStatus.brokers || 'pending',
      description: 'Configure broker connections and API credentials',
      component: () => <div>Broker Configuration</div>
    },
    {
      id: 'strategies',
      name: 'Strategies',
      status: validationStatus.strategies || 'pending',
      description: 'Define trading strategies and parameters',
      component: () => <div>Strategy Configuration</div>
    },
    {
      id: 'risk',
      name: 'Risk Management',
      status: validationStatus.risk || 'pending',
      description: 'Set risk limits and position sizing rules',
      component: () => <div>Risk Configuration</div>
    },
    {
      id: 'ai',
      name: 'AI & ML',
      status: validationStatus.ai || 'pending',
      description: 'Configure AI models and machine learning parameters',
      component: () => <div>AI Configuration</div>
    },
    {
      id: 'notifications',
      name: 'Notifications',
      status: validationStatus.notifications || 'pending',
      description: 'Set up alerts and notification channels',
      component: () => <div>Notification Configuration</div>
    },
    {
      id: 'security',
      name: 'Security',
      status: validationStatus.security || 'pending',
      description: 'Security settings and authentication',
      component: () => <div>Security Configuration</div>
    },
    {
      id: 'performance',
      name: 'Performance',
      status: validationStatus.performance || 'pending',
      description: 'Performance monitoring and optimization',
      component: () => <div>Performance Configuration</div>
    },
    {
      id: 'logging',
      name: 'Logging',
      status: validationStatus.logging || 'pending',
      description: 'Configure logging levels and outputs',
      component: () => <div>Logging Configuration</div>
    }
  ];

  // Calculate overall configuration progress
  const getOverallStatus = () => {
    const statuses = configSections.map(section => section.status);
    const validCount = statuses.filter(status => status === 'valid').length;
    const totalCount = statuses.length;
    
    if (validCount === totalCount) return 'valid';
    if (validCount > 0) return 'partial';
    return 'invalid';
  };

  const overallStatus = getOverallStatus();
  const progress = (configSections.filter(s => s.status === 'valid').length / configSections.length) * 100;

  const handleValidate = async () => {
    setIsValidating(true);
    setValidationProgress(0);
    
    try {
      // Simulate validation progress
      for (let i = 0; i <= 100; i += 10) {
        setValidationProgress(i);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      await validateConfig();
      toast.success('Configuration validated successfully');
    } catch (error) {
      toast.error('Configuration validation failed');
    } finally {
      setIsValidating(false);
      setValidationProgress(0);
    }
  };

  const handleSave = async () => {
    try {
      await saveConfig();
      toast.success('Configuration saved successfully');
    } catch (error) {
      toast.error('Failed to save configuration');
    }
  };

  const handleExport = async () => {
    try {
      await exportConfig('json');
      toast.success('Configuration exported successfully');
    } catch (error) {
      toast.error('Failed to export configuration');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'valid':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'invalid':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <RefreshCw className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'valid':
        return 'bg-green-500';
      case 'invalid':
        return 'bg-red-500';
      case 'pending':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-matrix-green">Configuration Manager</h1>
          <p className="text-matrix-green/70 mt-1">Manage your trading system configuration</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoSave(!autoSave)}
            className={autoSave ? 'bg-green-500/20 border-green-500' : ''}
          >
            <Zap className="h-4 w-4 mr-2" />
            Auto-save: {autoSave ? 'ON' : 'OFF'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleValidate}
            disabled={isValidating}
          >
            {isValidating ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <CheckCircle className="h-4 w-4 mr-2" />
            )}
            Validate
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            disabled={isSaving}
          >
            {isSaving ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button
            variant="outline"
            size="sm"
          >
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
        </div>
      </div>

      {/* Configuration Overview */}
      <Card className="border-matrix-green/20 bg-black/40">
        <CardHeader>
          <CardTitle className="text-matrix-green flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Configuration Overview
          </CardTitle>
          <CardDescription className="text-matrix-green/70">
            Current configuration status and progress
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-matrix-green/80">Overall Status</span>
            <div className="flex items-center gap-2">
              {getStatusIcon(overallStatus)}
              <span className="text-sm font-medium text-matrix-green capitalize">{overallStatus}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-matrix-green/80">Progress</span>
              <span className="text-matrix-green">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {isValidating && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-matrix-green/80">Validating...</span>
                <span className="text-matrix-green">{validationProgress}%</span>
              </div>
              <Progress value={validationProgress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Configuration Sections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {configSections.map((section) => (
          <Card key={section.id} className="border-matrix-green/20 bg-black/40 hover:border-matrix-green/40 transition-colors cursor-pointer"
                onClick={() => setActiveTab(section.id)}>
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-medium text-matrix-green">{section.name}</h3>
                <Badge variant="outline" className={`text-xs ${getStatusColor(section.status)}/20 border-current`}>
                  <div className="flex items-center gap-1">
                    {getStatusIcon(section.status)}
                    {section.status}
                  </div>
                </Badge>
              </div>
              <p className="text-xs text-matrix-green/70">{section.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Configuration Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-black/60 border border-matrix-green/20">
          <TabsTrigger value="overview" className="data-[state=active]:bg-matrix-green/20">
            <FileText className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="advanced" className="data-[state=active]:bg-matrix-green/20">
            <Settings className="h-4 w-4 mr-2" />
            Advanced
          </TabsTrigger>
          <TabsTrigger value="tools" className="data-[state=active]:bg-matrix-green/20">
            <Shield className="h-4 w-4 mr-2" />
            Tools
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-matrix-green/20 bg-black/40">
              <CardHeader>
                <CardTitle className="text-matrix-green">Quick Actions</CardTitle>
                <CardDescription className="text-matrix-green/70">
                  Common configuration tasks
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start" variant="outline">
                  <Play className="h-4 w-4 mr-2" />
                  Start Trading System
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Validate All
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Save className="h-4 w-4 mr-2" />
                  Save Configuration
                </Button>
              </CardContent>
            </Card>

            <Card className="border-matrix-green/20 bg-black/40">
              <CardHeader>
                <CardTitle className="text-matrix-green">System Health</CardTitle>
                <CardDescription className="text-matrix-green/70">
                  Configuration validation results
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {configSections.map((section) => (
                  <div key={section.id} className="flex items-center justify-between">
                    <span className="text-sm text-matrix-green/80">{section.name}</span>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(section.status)}
                      <span className="text-xs text-matrix-green capitalize">{section.status}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4">
          <Card className="border-matrix-green/20 bg-black/40">
            <CardHeader>
              <CardTitle className="text-matrix-green">Advanced Configuration</CardTitle>
              <CardDescription className="text-matrix-green/70">
                Fine-tune system behavior and performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-matrix-green/50">
                Advanced configuration options will be displayed here
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tools" className="space-y-4">
          <Card className="border-matrix-green/20 bg-black/40">
            <CardHeader>
              <CardTitle className="text-matrix-green">Configuration Tools</CardTitle>
              <CardDescription className="text-matrix-green/70">
                Import, export, backup, and maintenance tools
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button className="w-full justify-start" variant="outline">
                <Upload className="h-4 w-4 mr-2" />
                Import Configuration
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Export Configuration
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Backup Configuration
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <FileText className="h-4 w-4 mr-2" />
                Load Template
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ConfigManager;