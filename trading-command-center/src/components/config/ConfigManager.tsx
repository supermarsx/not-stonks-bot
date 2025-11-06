import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Settings, 
  Save, 
  RefreshCw, 
  Download, 
  Upload, 
  Shield, 
  Brain, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Server,
  Bell,
  LogOut,
  Database,
  Zap,
  FileText,
  Eye,
  RotateCcw
} from 'lucide-react';

import BrokerConfig from './BrokerConfig';
import StrategyConfig from './StrategyConfig';
import RiskConfig from './RiskConfig';
import AIConfig from './AIConfig';
import ExitConfig from './ExitConfig';
import LoggingConfig from './LoggingConfig';
import NotificationConfig from './NotificationConfig';
import SecurityConfig from './SecurityConfig';
import PerformanceConfig from './PerformanceConfig';
import ConfigIO from './ConfigIO';
import ConfigValidator from './ConfigValidator';
import ConfigBackup from './ConfigBackup';

interface ConfigSection {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
  component: React.ComponentType<any>;
  required: boolean;
  status: 'valid' | 'invalid' | 'pending' | 'untested';
  lastModified?: string;
}

interface ConfigStatus {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  lastValidated: string;
}

const ConfigManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [configSections] = useState<ConfigSection[]>([
    {
      id: 'brokers',
      title: 'Broker Configuration',
      description: 'Configure trading broker connections and API settings',
      icon: Server,
      component: BrokerConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'strategies',
      title: 'Strategy Configuration',
      description: 'Set up trading strategies and parameters',
      icon: TrendingUp,
      component: StrategyConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'risk',
      title: 'Risk Management',
      description: 'Configure risk limits, thresholds, and monitoring',
      icon: Shield,
      component: RiskConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'ai',
      title: 'AI & LLM Settings',
      description: 'Configure AI models, LLM providers, and parameters',
      icon: Brain,
      component: AIConfig,
      required: false,
      status: 'pending'
    },
    {
      id: 'exit',
      title: 'Exit Strategies',
      description: 'Configure exit strategies and parameters',
      icon: LogOut,
      component: ExitConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'logging',
      title: 'Logging & Monitoring',
      description: 'Configure log levels, destinations, and monitoring',
      icon: FileText,
      component: LoggingConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'notifications',
      title: 'Notifications',
      description: 'Configure alert channels and notification templates',
      icon: Bell,
      component: NotificationConfig,
      required: false,
      status: 'pending'
    },
    {
      id: 'security',
      title: 'Security & Authentication',
      description: 'Configure authentication methods and security settings',
      icon: Shield,
      component: SecurityConfig,
      required: true,
      status: 'pending'
    },
    {
      id: 'performance',
      title: 'Performance Tuning',
      description: 'Configure optimization parameters and resource limits',
      icon: Zap,
      component: PerformanceConfig,
      required: false,
      status: 'pending'
    }
  ]);

  const [configStatus, setConfigStatus] = useState<ConfigStatus>({
    isValid: false,
    errors: [],
    warnings: [],
    lastValidated: new Date().toISOString()
  });

  const [isLoading, setIsLoading] = useState(false);
  const [unsavedChanges, setUnsavedChanges] = useState<string[]>([]);
  const [autoSave, setAutoSave] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Load configuration on component mount
  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    setIsLoading(true);
    try {
      // Load configuration from storage
      const config = await window.electronAPI?.loadConfig();
      // Update sections with loaded config
      console.log('Configuration loaded');
    } catch (error) {
      console.error('Failed to load configuration:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveConfiguration = async () => {
    setIsLoading(true);
    try {
      // Save configuration to storage
      await window.electronAPI?.saveConfig();
      setUnsavedChanges([]);
    } catch (error) {
      console.error('Failed to save configuration:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const validateConfiguration = async () => {
    setIsLoading(true);
    try {
      // Trigger validation
      const status = await ConfigValidator.validateAll();
      setConfigStatus(status);
    } catch (error) {
      console.error('Failed to validate configuration:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'valid':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'invalid':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return <Eye className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      valid: 'default',
      invalid: 'destructive',
      pending: 'secondary',
      untested: 'outline'
    };

    return (
      <Badge variant={variants[status] || 'outline'}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const OverviewTab = () => (
    <div className="space-y-6">
      {/* Configuration Status Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Configuration Overview
          </CardTitle>
          <CardDescription>
            Monitor and manage your trading system configuration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Configuration Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Configuration Progress</span>
              <span>{configSections.filter(s => s.status === 'valid').length}/{configSections.length}</span>
            </div>
            <Progress 
              value={(configSections.filter(s => s.status === 'valid').length / configSections.length) * 100} 
              className="h-2"
            />
          </div>

          {/* Validation Status */}
          <Alert className={configStatus.isValid ? 'border-green-200' : 'border-red-200'}>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              <div className="flex justify-between items-center">
                <span>
                  {configStatus.isValid 
                    ? 'Configuration is valid and ready for deployment' 
                    : 'Configuration has validation errors'
                  }
                </span>
                <span className="text-xs text-muted-foreground">
                  Last validated: {new Date(configStatus.lastValidated).toLocaleString()}
                </span>
              </div>
            </AlertDescription>
          </Alert>

          {/* Configuration Errors */}
          {configStatus.errors.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-red-600">Configuration Errors</h4>
              {configStatus.errors.map((error, index) => (
                <Alert key={index} variant="destructive">
                  <AlertDescription className="text-sm">{error}</AlertDescription>
                </Alert>
              ))}
            </div>
          )}

          {/* Configuration Warnings */}
          {configStatus.warnings.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-yellow-600">Configuration Warnings</h4>
              {configStatus.warnings.map((warning, index) => (
                <Alert key={index}>
                  <AlertDescription className="text-sm">{warning}</AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Configuration Sections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {configSections.map((section) => {
          const Icon = section.icon;
          return (
            <Card 
              key={section.id}
              className={`cursor-pointer transition-colors hover:bg-gray-50 ${
                unsavedChanges.includes(section.id) ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setActiveTab(section.id)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="h-5 w-5" />
                    <CardTitle className="text-sm">{section.title}</CardTitle>
                  </div>
                  <div className="flex items-center gap-2">
                    {section.required && (
                      <Badge variant="outline" className="text-xs">Required</Badge>
                    )}
                    {getStatusIcon(section.status)}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <CardDescription className="text-xs">
                  {section.description}
                </CardDescription>
                <div className="mt-2 flex items-center justify-between">
                  {getStatusBadge(section.status)}
                  {section.lastModified && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(section.lastModified).toLocaleDateString()}
                    </span>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Button 
              onClick={validateConfiguration} 
              disabled={isLoading}
              variant="outline"
              size="sm"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Validate All
            </Button>
            <Button 
              onClick={saveConfiguration} 
              disabled={isLoading}
              size="sm"
            >
              <Save className="h-4 w-4 mr-2" />
              Save Configuration
            </Button>
            <ConfigIO />
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const ConfigurationTabs = () => (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="advanced">Advanced</TabsTrigger>
        <TabsTrigger value="tools">Tools</TabsTrigger>
      </TabsList>

      <TabsContent value="overview">
        <OverviewTab />
      </TabsContent>

      <TabsContent value="advanced">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Configuration Sections</h2>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Settings className="h-4 w-4 mr-2" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced
            </Button>
          </div>

          <Tabs value={activeTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-3 lg:grid-cols-6">
              {configSections.map((section) => {
                const Icon = section.icon;
                return (
                  <TabsTrigger 
                    key={section.id} 
                    value={section.id}
                    className="flex items-center gap-1 text-xs"
                  >
                    <Icon className="h-3 w-3" />
                    <span className="hidden lg:inline">{section.title.split(' ')[0]}</span>
                  </TabsTrigger>
                );
              })}
            </TabsList>

            {configSections.map((section) => {
              const Component = section.component;
              return (
                <TabsContent key={section.id} value={section.id}>
                  <Component />
                </TabsContent>
              );
            })}
          </Tabs>
        </div>
      </TabsContent>

      <TabsContent value="tools">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ConfigIO />
          <ConfigValidator />
          <ConfigBackup />
        </div>
      </TabsContent>
    </Tabs>
  );

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Settings className="h-8 w-8" />
            Configuration Manager
          </h1>
          <p className="text-muted-foreground mt-1">
            Comprehensive configuration management for your trading system
          </p>
        </div>
        <div className="flex items-center gap-2">
          {unsavedChanges.length > 0 && (
            <Badge variant="outline" className="animate-pulse">
              {unsavedChanges.length} unsaved change{unsavedChanges.length !== 1 ? 's' : ''}
            </Badge>
          )}
          <Button 
            onClick={saveConfiguration} 
            disabled={isLoading || unsavedChanges.length === 0}
          >
            <Save className="h-4 w-4 mr-2" />
            Save
          </Button>
        </div>
      </div>

      {/* Auto-save Toggle */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">Auto-save</h3>
            <p className="text-sm text-muted-foreground">
              Automatically save configuration changes
            </p>
          </div>
          <Button
            variant={autoSave ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoSave(!autoSave)}
          >
            {autoSave ? 'On' : 'Off'}
          </Button>
        </div>
      </Card>

      {/* Configuration Interface */}
      <ConfigurationTabs />
    </div>
  );
};

export default ConfigManager;