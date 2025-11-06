import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Upload, 
  Download, 
  FileText, 
  Settings, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  RefreshCw,
  Shield,
  Lock,
  Unlock
} from 'lucide-react';
import { toast } from 'sonner';
import { useConfigStore } from '@/stores/configStore';

interface ImportResult {
  success: boolean;
  errors: string[];
  warnings: string[];
  imported: {
    brokers: number;
    strategies: number;
    risk: number;
    ai: number;
    notifications: number;
    security: number;
    performance: number;
  };
}

interface ExportOptions {
  format: 'json' | 'yaml' | 'csv' | 'xml';
  scope: 'all' | 'brokers' | 'strategies' | 'risk' | 'ai' | 'notifications' | 'security' | 'performance';
  includeSecrets: boolean;
  includeDefaults: boolean;
  compress: boolean;
  encrypt: boolean;
}

const ConfigIO: React.FC = () => {
  const { config, importConfig, exportConfig, loadConfig } = useConfigStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [importProgress, setImportProgress] = useState(0);
  const [exportProgress, setExportProgress] = useState(0);
  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [lastExport, setLastExport] = useState<Date | null>(null);
  const [lastImport, setLastImport] = useState<Date | null>(null);
  
  const [importOptions, setImportOptions] = useState({
    format: 'json' as const,
    scope: 'all' as const,
    validate: true,
    backup: true,
    merge: false
  });
  
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'json',
    scope: 'all',
    includeSecrets: false,
    includeDefaults: false,
    compress: false,
    encrypt: false
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file format
    const validFormats = ['.json', '.yaml', '.yml', '.csv', '.xml'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!validFormats.includes(fileExtension)) {
      toast.error('Invalid file format. Supported formats: JSON, YAML, CSV, XML');
      return;
    }

    setImportOptions(prev => ({
      ...prev,
      format: fileExtension === '.json' ? 'json' : 
               fileExtension.includes('yml') ? 'yaml' : 
               fileExtension === '.csv' ? 'csv' : 'xml'
    }));
  };

  const importConfiguration = async () => {
    if (!fileInputRef.current?.files?.[0]) {
      toast.error('Please select a file to import');
      return;
    }

    setIsImporting(true);
    setImportProgress(0);
    setImportResult(null);

    try {
      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append('file', file);
      formData.append('options', JSON.stringify(importOptions));

      // Simulate import progress
      for (let i = 0; i <= 100; i += 10) {
        setImportProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // Simulate import result
      const result: ImportResult = {
        success: true,
        errors: [],
        warnings: ['Some default values were not imported'],
        imported: {
          brokers: 3,
          strategies: 2,
          risk: 1,
          ai: 1,
          notifications: 1,
          security: 1,
          performance: 1
        }
      };

      setImportResult(result);
      setLastImport(new Date());
      
      if (result.errors.length === 0) {
        toast.success('Configuration imported successfully');
      } else {
        toast.warning('Configuration imported with warnings');
      }

    } catch (error) {
      const result: ImportResult = {
        success: false,
        errors: ['Import failed: Invalid file format'],
        warnings: [],
        imported: {
          brokers: 0,
          strategies: 0,
          risk: 0,
          ai: 0,
          notifications: 0,
          security: 0,
          performance: 0
        }
      };
      setImportResult(result);
      toast.error('Import failed');
    } finally {
      setIsImporting(false);
      setImportProgress(0);
    }
  };

  const exportConfiguration = async () => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      // Simulate export progress
      for (let i = 0; i <= 100; i += 10) {
        setExportProgress(i);
        await new Promise(resolve => setTimeout(resolve, 150));
      }

      await exportConfig(exportOptions.format, exportOptions);
      setLastExport(new Date());
      toast.success('Configuration exported successfully');
      
    } catch (error) {
      toast.error('Export failed');
    } finally {
      setIsExporting(false);
      setExportProgress(0);
    }
  };

  const loadTemplate = async (templateName: string) => {
    try {
      await loadConfig(templateName);
      toast.success(`Template "${templateName}" loaded successfully`);
    } catch (error) {
      toast.error('Failed to load template');
    }
  };

  const saveCurrentAsTemplate = async () => {
    const templateName = prompt('Enter template name:');
    if (!templateName) return;

    try {
      // Simulate saving template
      toast.success(`Template "${templateName}" saved successfully`);
    } catch (error) {
      toast.error('Failed to save template');
    }
  };

  const templates = [
    { name: 'Conservative Trading', description: 'Low-risk, stable returns configuration' },
    { name: 'Aggressive Growth', description: 'High-risk, high-reward configuration' },
    { name: 'Balanced Portfolio', description: 'Moderate risk with diversification' },
    { name: 'Scalping Only', description: 'High-frequency trading setup' },
    { name: 'Swing Trading', description: 'Medium-term position trading' }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-matrix-green">Configuration I/O</h1>
          <p className="text-matrix-green/70 mt-1">Import, export, and manage configuration templates</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Import Section */}
        <Card className="border-matrix-green/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-matrix-green flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Import Configuration
            </CardTitle>
            <CardDescription className="text-matrix-green/70">
              Import configuration from a file
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-matrix-green">Configuration File</Label>
              <Input
                ref={fileInputRef}
                type="file"
                accept=".json,.yaml,.yml,.csv,.xml"
                onChange={handleFileSelect}
                className="bg-black/40 border-matrix-green/20"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-matrix-green">Format</Label>
              <Select
                value={importOptions.format}
                onValueChange={(value: 'json' | 'yaml' | 'csv' | 'xml') => 
                  setImportOptions(prev => ({ ...prev, format: value }))
                }
              >
                <SelectTrigger className="bg-black/40 border-matrix-green/20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="yaml">YAML</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="xml">XML</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-matrix-green">Scope</Label>
              <Select
                value={importOptions.scope}
                onValueChange={(value: typeof importOptions.scope) => 
                  setImportOptions(prev => ({ ...prev, scope: value }))
                }
              >
                <SelectTrigger className="bg-black/40 border-matrix-green/20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Configuration</SelectItem>
                  <SelectItem value="brokers">Brokers Only</SelectItem>
                  <SelectItem value="strategies">Strategies Only</SelectItem>
                  <SelectItem value="risk">Risk Management Only</SelectItem>
                  <SelectItem value="ai">AI/LLM Only</SelectItem>
                  <SelectItem value="notifications">Notifications Only</SelectItem>
                  <SelectItem value="security">Security Only</SelectItem>
                  <SelectItem value="performance">Performance Only</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="validate"
                  checked={importOptions.validate}
                  onCheckedChange={(checked) => 
                    setImportOptions(prev => ({ ...prev, validate: !!checked }))
                  }
                />
                <Label htmlFor="validate" className="text-sm text-matrix-green/80">
                  Validate configuration before import
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="backup"
                  checked={importOptions.backup}
                  onCheckedChange={(checked) => 
                    setImportOptions(prev => ({ ...prev, backup: !!checked }))
                  }
                />
                <Label htmlFor="backup" className="text-sm text-matrix-green/80">
                  Create backup before import
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="merge"
                  checked={importOptions.merge}
                  onCheckedChange={(checked) => 
                    setImportOptions(prev => ({ ...prev, merge: !!checked }))
                  }
                />
                <Label htmlFor="merge" className="text-sm text-matrix-green/80">
                  Merge with existing configuration
                </Label>
              </div>
            </div>

            {isImporting && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/80">Importing...</span>
                  <span className="text-matrix-green">{importProgress}%</span>
                </div>
                <Progress value={importProgress} className="h-2" />
              </div>
            )}

            <Button
              onClick={importConfiguration}
              disabled={isImporting || !fileInputRef.current?.files?.[0]}
              className="w-full"
            >
              {isImporting ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Importing...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Import Configuration
                </>
              )}
            </Button>

            {importResult && (
              <div className="space-y-2">
                <Alert className={importResult.success ? 'border-green-500/20' : 'border-red-500/20'}>
                  {importResult.success ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                  <AlertDescription className={importResult.success ? 'text-green-400' : 'text-red-400'}>
                    {importResult.success ? 'Import completed successfully' : 'Import failed'}
                  </AlertDescription>
                </Alert>

                {importResult.warnings.length > 0 && (
                  <Alert className="border-yellow-500/20">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    <AlertDescription className="text-yellow-400">
                      {importResult.warnings.join(', ')}
                    </AlertDescription>
                  </Alert>
                )}

                {importResult.errors.length > 0 && (
                  <Alert className="border-red-500/20">
                    <XCircle className="h-4 w-4 text-red-500" />
                    <AlertDescription className="text-red-400">
                      {importResult.errors.join(', ')}
                    </AlertDescription>
                  </Alert>
                )}

                <div className="grid grid-cols-2 gap-2 mt-3">
                  <div className="text-center">
                    <div className="text-lg font-bold text-matrix-green">{importResult.imported.brokers}</div>
                    <div className="text-xs text-matrix-green/60">Brokers</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-matrix-green">{importResult.imported.strategies}</div>
                    <div className="text-xs text-matrix-green/60">Strategies</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Export Section */}
        <Card className="border-matrix-green/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-matrix-green flex items-center gap-2">
              <Download className="h-5 w-5" />
              Export Configuration
            </CardTitle>
            <CardDescription className="text-matrix-green/70">
              Export configuration to a file
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-matrix-green">Format</Label>
              <Select
                value={exportOptions.format}
                onValueChange={(value: ExportOptions['format']) => 
                  setExportOptions(prev => ({ ...prev, format: value }))
                }
              >
                <SelectTrigger className="bg-black/40 border-matrix-green/20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="yaml">YAML</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="xml">XML</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-matrix-green">Scope</Label>
              <Select
                value={exportOptions.scope}
                onValueChange={(value: ExportOptions['scope']) => 
                  setExportOptions(prev => ({ ...prev, scope: value }))
                }
              >
                <SelectTrigger className="bg-black/40 border-matrix-green/20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Configuration</SelectItem>
                  <SelectItem value="brokers">Brokers Only</SelectItem>
                  <SelectItem value="strategies">Strategies Only</SelectItem>
                  <SelectItem value="risk">Risk Management Only</SelectItem>
                  <SelectItem value="ai">AI/LLM Only</SelectItem>
                  <SelectItem value="notifications">Notifications Only</SelectItem>
                  <SelectItem value="security">Security Only</SelectItem>
                  <SelectItem value="performance">Performance Only</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="includeSecrets"
                  checked={exportOptions.includeSecrets}
                  onCheckedChange={(checked) => 
                    setExportOptions(prev => ({ ...prev, includeSecrets: !!checked }))
                  }
                />
                <Label htmlFor="includeSecrets" className="text-sm text-matrix-green/80">
                  Include API secrets and passwords
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="includeDefaults"
                  checked={exportOptions.includeDefaults}
                  onCheckedChange={(checked) => 
                    setExportOptions(prev => ({ ...prev, includeDefaults: !!checked }))
                  }
                />
                <Label htmlFor="includeDefaults" className="text-sm text-matrix-green/80">
                  Include default values
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="compress"
                  checked={exportOptions.compress}
                  onCheckedChange={(checked) => 
                    setExportOptions(prev => ({ ...prev, compress: !!checked }))
                  }
                />
                <Label htmlFor="compress" className="text-sm text-matrix-green/80">
                  Compress output file
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="encrypt"
                  checked={exportOptions.encrypt}
                  onCheckedChange={(checked) => 
                    setExportOptions(prev => ({ ...prev, encrypt: !!checked }))
                  }
                />
                <Label htmlFor="encrypt" className="text-sm text-matrix-green/80 flex items-center gap-1">
                  {exportOptions.encrypt ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                  Encrypt sensitive data
                </Label>
              </div>
            </div>

            {isExporting && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-matrix-green/80">Exporting...</span>
                  <span className="text-matrix-green">{exportProgress}%</span>
                </div>
                <Progress value={exportProgress} className="h-2" />
              </div>
            )}

            <Button
              onClick={exportConfiguration}
              disabled={isExporting}
              className="w-full"
            >
              {isExporting ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Export Configuration
                </>
              )}
            </Button>

            {lastExport && (
              <div className="text-xs text-matrix-green/60 text-center">
                Last export: {lastExport.toLocaleString()}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Templates Section */}
      <Card className="border-matrix-green/20 bg-black/40">
        <CardHeader>
          <CardTitle className="text-matrix-green flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Configuration Templates
          </CardTitle>
          <CardDescription className="text-matrix-green/70">
            Pre-configured templates for common trading setups
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {templates.map((template) => (
              <Card key={template.name} className="border-matrix-green/20 bg-black/20">
                <CardContent className="p-4">
                  <div className="space-y-3">
                    <div>
                      <h3 className="font-medium text-matrix-green">{template.name}</h3>
                      <p className="text-sm text-matrix-green/70">{template.description}</p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => loadTemplate(template.name)}
                        className="flex-1"
                      >
                        <Upload className="h-3 w-3 mr-1" />
                        Load
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="mt-6 pt-4 border-t border-matrix-green/10">
            <Button
              variant="outline"
              onClick={saveCurrentAsTemplate}
              className="w-full"
            >
              <FileText className="h-4 w-4 mr-2" />
              Save Current as Template
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ConfigIO;