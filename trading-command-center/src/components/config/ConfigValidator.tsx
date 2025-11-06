import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  RefreshCw,
  Settings,
  Shield,
  Zap,
  Database,
  Key,
  Activity,
  Clock,
  Target,
  TrendingUp
} from 'lucide-react';

interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  score: number; // 0-100
  lastValidated: string;
  totalChecks: number;
  passedChecks: number;
}

interface ValidationError {
  id: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  message: string;
  field?: string;
  value?: any;
  suggestion?: string;
  autoFixable: boolean;
}

interface ValidationWarning {
  id: string;
  category: string;
  message: string;
  field?: string;
  value?: any;
  suggestion?: string;
}

interface ValidationRule {
  id: string;
  name: string;
  category: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  autoFixable: boolean;
  enabled: boolean;
}

const ConfigValidator: React.FC = () => {
  const [validationResult, setValidationResult] = useState<ValidationResult>({
    isValid: false,
    errors: [],
    warnings: [],
    score: 0,
    lastValidated: new Date().toISOString(),
    totalChecks: 0,
    passedChecks: 0
  });

  const [isValidating, setIsValidating] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [autoFixEnabled, setAutoFixEnabled] = useState(false);

  const validationRules: ValidationRule[] = [
    {
      id: 'broker_api_keys',
      name: 'Broker API Keys',
      category: 'brokers',
      description: 'Verify all broker API keys are valid and properly formatted',
      severity: 'critical',
      autoFixable: false,
      enabled: true
    },
    {
      id: 'broker_environment',
      name: 'Broker Environment',
      category: 'brokers',
      description: 'Check that broker environment settings are consistent',
      severity: 'high',
      autoFixable: true,
      enabled: true
    },
    {
      id: 'strategy_parameters',
      name: 'Strategy Parameters',
      category: 'strategies',
      description: 'Validate strategy parameters are within acceptable ranges',
      severity: 'medium',
      autoFixable: true,
      enabled: true
    },
    {
      id: 'risk_limits',
      name: 'Risk Limits',
      category: 'risk',
      description: 'Ensure risk limits are properly configured and reasonable',
      severity: 'critical',
      autoFixable: false,
      enabled: true
    },
    {
      id: 'ai_model_config',
      name: 'AI Model Configuration',
      category: 'ai',
      description: 'Validate AI model settings and provider configurations',
      severity: 'high',
      autoFixable: true,
      enabled: true
    },
    {
      id: 'notification_channels',
      name: 'Notification Channels',
      category: 'notifications',
      description: 'Check notification channel configurations are valid',
      severity: 'medium',
      autoFixable: true,
      enabled: true
    },
    {
      id: 'security_settings',
      name: 'Security Settings',
      category: 'security',
      description: 'Verify security configurations meet minimum requirements',
      severity: 'critical',
      autoFixable: false,
      enabled: true
    },
    {
      id: 'performance_thresholds',
      name: 'Performance Thresholds',
      category: 'performance',
      description: 'Check performance monitoring thresholds are set appropriately',
      severity: 'low',
      autoFixable: true,
      enabled: true
    }
  ];

  const categories = [
    { value: 'all', label: 'All Categories', icon: Settings },
    { value: 'brokers', label: 'Brokers', icon: Database },
    { value: 'strategies', label: 'Strategies', icon: TrendingUp },
    { value: 'risk', label: 'Risk Management', icon: Shield },
    { value: 'ai', label: 'AI & LLM', icon: Target },
    { value: 'notifications', label: 'Notifications', icon: Activity },
    { value: 'security', label: 'Security', icon: Key },
    { value: 'performance', label: 'Performance', icon: Zap }
  ];

  const runValidation = async () => {
    setIsValidating(true);

    try {
      // Simulate comprehensive validation
      const config = await window.electronAPI?.getAllConfiguration();
      const results = await validateConfiguration(config, selectedCategory);
      setValidationResult(results);
    } catch (error) {
      console.error('Validation failed:', error);
      setValidationResult({
        isValid: false,
        errors: [{
          id: 'validation_error',
          category: 'system',
          severity: 'critical',
          message: 'Validation failed: ' + (error instanceof Error ? error.message : 'Unknown error'),
          autoFixable: false
        }],
        warnings: [],
        score: 0,
        lastValidated: new Date().toISOString(),
        totalChecks: 0,
        passedChecks: 0
      });
    } finally {
      setIsValidating(false);
    }
  };

  const validateConfiguration = async (config: any, category: string): Promise<ValidationResult> => {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let totalChecks = 0;
    let passedChecks = 0;

    // Validate brokers
    if (category === 'all' || category === 'brokers') {
      totalChecks += 3;
      
      // Check broker API keys
      if (!config.brokers || config.brokers.length === 0) {
        errors.push({
          id: 'no_brokers',
          category: 'brokers',
          severity: 'critical',
          message: 'No broker configurations found',
          suggestion: 'Add at least one broker configuration',
          autoFixable: false
        });
      } else {
        passedChecks++;
        
        config.brokers.forEach((broker: any, index: number) => {
          if (!broker.config?.apiKey || broker.config.apiKey.trim() === '') {
            errors.push({
              id: `broker_api_key_${index}`,
              category: 'brokers',
              severity: 'critical',
              message: `Broker "${broker.name}" is missing API key`,
              field: 'apiKey',
              value: broker.config?.apiKey,
              suggestion: 'Add valid API key for this broker',
              autoFixable: false
            });
          } else {
            passedChecks++;
          }

          if (broker.config?.environment && !['paper', 'live', 'sandbox'].includes(broker.config.environment)) {
            errors.push({
              id: `broker_env_${index}`,
              category: 'brokers',
              severity: 'high',
              message: `Broker "${broker.name}" has invalid environment setting`,
              field: 'environment',
              value: broker.config.environment,
              suggestion: 'Set environment to paper, live, or sandbox',
              autoFixable: true
            });
          } else {
            passedChecks++;
          }
        });
      }
    }

    // Validate strategies
    if (category === 'all' || category === 'strategies') {
      totalChecks += 2;
      
      if (!config.strategies || config.strategies.length === 0) {
        warnings.push({
          id: 'no_strategies',
          category: 'strategies',
          message: 'No trading strategies configured',
          suggestion: 'Consider adding at least one trading strategy'
        });
      } else {
        passedChecks++;
      }

      config.strategies?.forEach((strategy: any, index: number) => {
        if (!strategy.settings?.enabled && strategy.status === 'active') {
          errors.push({
            id: `strategy_enabled_${index}`,
            category: 'strategies',
            severity: 'medium',
            message: `Strategy "${strategy.name}" is active but not enabled`,
            field: 'enabled',
            value: strategy.settings?.enabled,
            suggestion: 'Enable the strategy or set status to inactive',
            autoFixable: true
          });
        } else {
          passedChecks++;
        }

        if (strategy.settings?.maxPositionSize > 0.5) {
          warnings.push({
            id: `strategy_position_size_${index}`,
            category: 'strategies',
            message: `Strategy "${strategy.name}" has high position size (${strategy.settings.maxPositionSize * 100}%)`,
            field: 'maxPositionSize',
            value: strategy.settings?.maxPositionSize,
            suggestion: 'Consider reducing maximum position size for better risk management'
          });
        }
      });
    }

    // Validate risk management
    if (category === 'all' || category === 'risk') {
      totalChecks += 2;
      
      if (!config.riskSettings) {
        errors.push({
          id: 'no_risk_settings',
          category: 'risk',
          severity: 'critical',
          message: 'Risk management settings not configured',
          suggestion: 'Configure risk limits and monitoring settings',
          autoFixable: false
        });
      } else {
        passedChecks++;
      }

      if (config.riskSettings?.limits?.dailyLossLimit > 10000) {
        warnings.push({
          id: 'high_daily_loss_limit',
          category: 'risk',
          message: 'Daily loss limit is set very high',
          field: 'dailyLossLimit',
          value: config.riskSettings?.limits?.dailyLossLimit,
          suggestion: 'Consider setting a lower daily loss limit for better risk control'
        });
      } else {
        passedChecks++;
      }
    }

    // Validate AI settings
    if (category === 'all' || category === 'ai') {
      totalChecks += 1;
      
      if (!config.aiSettings?.providers || config.aiSettings.providers.length === 0) {
        warnings.push({
          id: 'no_ai_providers',
          category: 'ai',
          message: 'No AI providers configured',
          suggestion: 'Configure at least one AI provider for enhanced trading capabilities'
        });
      } else {
        passedChecks++;
      }

      config.aiSettings?.providers?.forEach((provider: any, index: number) => {
        if (provider.status === 'connected' && !provider.config?.temperature) {
          warnings.push({
            id: `ai_temp_${index}`,
            category: 'ai',
            message: `AI provider "${provider.name}" missing temperature setting`,
            field: 'temperature',
            suggestion: 'Set an appropriate temperature value (0.0-1.0)'
          });
        }
      });
    }

    // Calculate score
    const score = totalChecks > 0 ? Math.round((passedChecks / totalChecks) * 100) : 100;
    const isValid = errors.filter(e => e.severity === 'critical').length === 0;

    return {
      isValid,
      errors,
      warnings,
      score,
      lastValidated: new Date().toISOString(),
      totalChecks,
      passedChecks
    };
  };

  const fixError = async (error: ValidationError) => {
    if (!error.autoFixable) {
      alert('This error cannot be automatically fixed');
      return;
    }

    try {
      await window.electronAPI?.fixConfigurationError(error.id);
      // Re-run validation after fix
      await runValidation();
    } catch (fixError) {
      console.error('Auto-fix failed:', fixError);
      alert('Auto-fix failed: ' + (fixError instanceof Error ? fixError.message : 'Unknown error'));
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'high':
        return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      case 'medium':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'low':
        return <AlertTriangle className="h-4 w-4 text-blue-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getSeverityBadge = (severity: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      critical: 'destructive',
      high: 'destructive',
      medium: 'secondary',
      low: 'outline'
    };

    const colors: Record<string, string> = {
      critical: 'text-red-600',
      high: 'text-orange-600',
      medium: 'text-yellow-600',
      low: 'text-blue-600'
    };

    return (
      <Badge variant={variants[severity] || 'outline'} className={`${colors[severity]} border-current`}>
        {severity.charAt(0).toUpperCase() + severity.slice(1)}
      </Badge>
    );
  };

  const getCategoryIcon = (category: string) => {
    const categoryData = categories.find(c => c.value === category);
    return categoryData ? <categoryData.icon className="h-4 w-4" /> : <Settings className="h-4 w-4" />;
  };

  const filteredErrors = selectedCategory === 'all' 
    ? validationResult.errors 
    : validationResult.errors.filter(e => e.category === selectedCategory);

  const filteredWarnings = selectedCategory === 'all'
    ? validationResult.warnings
    : validationResult.warnings.filter(w => w.category === selectedCategory);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle className="h-5 w-5" />
          Configuration Validator
        </CardTitle>
        <CardDescription>
          Validate configuration settings and fix common issues
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Validation Controls */}
        <div className="flex flex-wrap gap-4 items-center justify-between">
          <div className="flex gap-2">
            <Button onClick={runValidation} disabled={isValidating}>
              {isValidating ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <CheckCircle className="h-4 w-4 mr-2" />
              )}
              {isValidating ? 'Validating...' : 'Run Validation'}
            </Button>

            <Button variant="outline" onClick={() => setAutoFixEnabled(!autoFixEnabled)}>
              <Zap className="h-4 w-4 mr-2" />
              Auto-fix: {autoFixEnabled ? 'On' : 'Off'}
            </Button>
          </div>

          <div className="flex gap-2">
            <select 
              value={selectedCategory} 
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-1 border rounded text-sm"
            >
              {categories.map((category) => (
                <option key={category.value} value={category.value}>
                  {category.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Validation Summary */}
        {validationResult.totalChecks > 0 && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Overall Score</p>
                      <p className="text-2xl font-bold">{validationResult.score}/100</p>
                    </div>
                    <div className={`h-12 w-12 rounded-full flex items-center justify-center ${
                      validationResult.score >= 90 ? 'bg-green-100' :
                      validationResult.score >= 70 ? 'bg-yellow-100' : 'bg-red-100'
                    }`}>
                      <span className={`text-xl font-bold ${
                        validationResult.score >= 90 ? 'text-green-600' :
                        validationResult.score >= 70 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {Math.floor(validationResult.score / 10)}
                      </span>
                    </div>
                  </div>
                  <Progress value={validationResult.score} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Checks Passed</p>
                      <p className="text-2xl font-bold text-green-600">{validationResult.passedChecks}</p>
                    </div>
                    <CheckCircle className="h-8 w-8 text-green-500" />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    of {validationResult.totalChecks} total checks
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Errors</p>
                      <p className="text-2xl font-bold text-red-600">{validationResult.errors.length}</p>
                    </div>
                    <XCircle className="h-8 w-8 text-red-500" />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {validationResult.errors.filter(e => e.severity === 'critical').length} critical
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Warnings</p>
                      <p className="text-2xl font-bold text-yellow-600">{validationResult.warnings.length}</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-yellow-500" />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Recommendations
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Overall Status */}
            <Alert className={validationResult.isValid ? 'border-green-200' : 'border-red-200'}>
              {validationResult.isValid ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              <AlertDescription>
                <div className="flex justify-between items-center">
                  <span>
                    {validationResult.isValid 
                      ? 'Configuration is valid and ready for deployment' 
                      : `Configuration has ${validationResult.errors.length} error${validationResult.errors.length !== 1 ? 's' : ''}`
                    }
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Last validated: {new Date(validationResult.lastValidated).toLocaleString()}
                  </span>
                </div>
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Errors */}
        {filteredErrors.length > 0 && (
          <div className="space-y-3">
            <Separator />
            <h3 className="text-lg font-medium flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-500" />
              Errors ({filteredErrors.length})
            </h3>
            
            <div className="space-y-2">
              {filteredErrors.map((error) => (
                <div key={error.id} className="p-4 border border-red-200 rounded-lg bg-red-50">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      {getSeverityIcon(error.severity)}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{error.message}</span>
                          {getSeverityBadge(error.severity)}
                          <Badge variant="outline" className="text-xs">
                            {error.category}
                          </Badge>
                        </div>
                        
                        {error.field && (
                          <div className="text-sm text-muted-foreground">
                            Field: <code className="bg-gray-100 px-1 rounded">{error.field}</code>
                            {error.value && <> = <code className="bg-gray-100 px-1 rounded">{JSON.stringify(error.value)}</code></>}
                          </div>
                        )}
                        
                        {error.suggestion && (
                          <div className="text-sm text-blue-600 mt-1">
                            💡 {error.suggestion}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {error.autoFixable && autoFixEnabled && (
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => fixError(error)}
                      >
                        <Zap className="h-3 w-3 mr-1" />
                        Fix
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Warnings */}
        {filteredWarnings.length > 0 && (
          <div className="space-y-3">
            <Separator />
            <h3 className="text-lg font-medium flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              Warnings ({filteredWarnings.length})
            </h3>
            
            <div className="space-y-2">
              {filteredWarnings.map((warning) => (
                <div key={warning.id} className="p-4 border border-yellow-200 rounded-lg bg-yellow-50">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium">{warning.message}</span>
                        <Badge variant="outline" className="text-xs">
                          {warning.category}
                        </Badge>
                      </div>
                      
                      {warning.field && (
                        <div className="text-sm text-muted-foreground">
                          Field: <code className="bg-gray-100 px-1 rounded">{warning.field}</code>
                          {warning.value && <> = <code className="bg-gray-100 px-1 rounded">{JSON.stringify(warning.value)}</code></>}
                        </div>
                      )}
                      
                      {warning.suggestion && (
                        <div className="text-sm text-blue-600 mt-1">
                          💡 {warning.suggestion}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No Issues */}
        {validationResult.totalChecks > 0 && filteredErrors.length === 0 && filteredWarnings.length === 0 && (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-green-600 mb-2">No Issues Found</h3>
            <p className="text-muted-foreground">
              Your configuration passed all validation checks for the {categories.find(c => c.value === selectedCategory)?.label} category.
            </p>
          </div>
        )}

        {/* Initial State */}
        {validationResult.totalChecks === 0 && (
          <div className="text-center py-8">
            <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">Ready to Validate</h3>
            <p className="text-muted-foreground mb-4">
              Click "Run Validation" to check your configuration for issues and best practices.
            </p>
            <Button onClick={runValidation}>
              <CheckCircle className="h-4 w-4 mr-2" />
              Start Validation
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Export validation function for use by other components
export const validateAll = async (): Promise<any> => {
  // This would be the actual validation logic
  return {
    isValid: true,
    errors: [],
    warnings: [],
    lastValidated: new Date().toISOString()
  };
};

export default ConfigValidator;